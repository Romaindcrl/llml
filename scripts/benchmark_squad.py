"""Benchmark sur un VRAI jeu en ligne : SQuAD v1.1 (lecture-compréhension Wikipédia).

Notre système internalise les paragraphes dans les POIDS (LoRA, via Q/R extraites par
notre pipeline), puis répond aux VRAIES questions humaines de SQuAD (jamais vues à
l'entraînement). Comparé à RAG (BM25), compaction (résumé) et base.
Score : SQuAD-style (réponse de référence normalisée présente dans la prédiction).

Teste aussi le ROUTEUR sur des vraies questions : toutes factuelles -> doivent router
'recall'. On mesure le taux de mauvais routage (mots-clés vs LLM).

Logs par question (cuttable). Live : tail -f logs/benchmark_squad.log
"""

from __future__ import annotations

import os
import re
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import httpx  # noqa: E402

from m0 import d2l, rag as ragmod  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.rag import RAG  # noqa: E402

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
N_CONTEXTS = 8        # paragraphes distincts internalisés
Q_PER_CTX = 4         # questions SQuAD held-out par paragraphe
LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_squad.log")
L = os.path.join(_PROJ, "models", "lora")
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def _norm(s):
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def squad_hit(pred, golds):
    p = _norm(pred)
    return any(_norm(g) and _norm(g) in p for g in golds)


def load_squad():
    log("Téléchargement SQuAD dev-v1.1…")
    r = httpx.get(SQUAD_URL, timeout=60.0, headers={"User-Agent": "m0-squad/0.1"})
    r.raise_for_status()
    data = r.json()["data"]
    items = []  # (context, [questions...], [[golds]...])
    for article in data:
        for para in article["paragraphs"]:
            ctx = para["context"]
            qas = [qa for qa in para["qas"] if qa.get("answers")]
            if len(ctx) < 200 or len(qas) < Q_PER_CTX:
                continue
            qs = [qa["question"] for qa in qas[:Q_PER_CTX]]
            gs = [[a["text"] for a in qa["answers"]] for qa in qas[:Q_PER_CTX]]
            items.append((ctx, qs, gs))
            break  # un paragraphe par article -> sujets distincts
        if len(items) >= N_CONTEXTS:
            break
    return items


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== BENCHMARK SQuAD (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg)

    items = load_squad()
    log(f"[1/3] {len(items)} paragraphes · {sum(len(q) for _, q, _ in items)} questions held-out")

    rag = RAG(os.path.join(_PROJ, "logs", "rag_squad.txt")); rag.clear()
    train_qa = []
    for i, (ctx, _, _) in enumerate(items, 1):
        rag.add_document(ctx)
        qa = d2l.clean_and_balance(d2l.extract_qa(ctx, llm.generate, n=10), max_per_answer=2)
        train_qa += d2l.augment_pairs(qa, llm.generate, n_paraphrases=1)
        log(f"  [prep {i}/{len(items)}] +{len(qa)} Q/R extraites (corpus train={len(train_qa)})")
    train_qa = d2l.clean_and_balance(train_qa, max_per_answer=6)
    summary = llm.generate("Summarize these passages, keeping all names, numbers and facts:\n"
                           + "\n".join(c for c, _, _ in items), None)

    log("[2/3] Entraînement LoRA (internalise les paragraphes)")
    data = f"{_PROJ}/logs/squad_data"
    adapter = f"{L}/squad"
    n = d2l.build_chat_dataset(train_qa, data, repeat=4, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=4)
    iters = min(600, max(250, 10 * len(train_qa)))
    res = d2l.train_lora(cfg.mlx_model_path, data, adapter, iters=iters, num_layers=cfg.d2l_num_layers,
                         learning_rate=cfg.d2l_learning_rate, rank=16, python_exe=sys.executable,
                         log_file=LOG_PATH)
    log(f"  LoRA: ok={res['ok']} val_loss={res['val_loss']} ({n} lignes, {iters} iters)")

    flat = [(q, g) for _, qs, gs in items for q, g in zip(qs, gs)]
    nq = len(flat)
    R = {s: 0 for s in ("base", "RAG", "compaction", "ours")}

    def gen_ctx(q, ctx):
        p = q if ctx is None else f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer concisely:"
        return llm.generate(p, None)

    log("[3/3] Évaluation SQuAD-style")
    llm.set_adapter(None)
    for i, (q, g) in enumerate(flat, 1):
        b = squad_hit(gen_ctx(q, None), g)
        rr = squad_hit(gen_ctx(q, "\n".join(rag.topk(q, 4))), g)
        cc = squad_hit(gen_ctx(q, summary), g)
        R["base"] += b; R["RAG"] += rr; R["compaction"] += cc
        log(f"[q {i}/{nq}] base={int(b)} rag={int(rr)} comp={int(cc)}")
    llm.set_adapter(adapter)
    for i, (q, g) in enumerate(flat, 1):
        o = squad_hit(gen_ctx(q, None), g)
        R["ours"] += o
        log(f"[q {i}/{nq}] ours={int(o)}")

    # --- ROUTEUR sur de vraies questions (toutes factuelles -> 'recall' attendu)
    llm.set_adapter(None)
    kw_bad = [q for q, _ in flat if ragmod.classify(q, None) != "recall"]
    llm_bad = sum(1 for q in kw_bad if ragmod.classify(q, llm.generate) != "recall")
    log("")
    log(f"=== ROUTEUR sur {nq} vraies questions SQuAD (toutes 'recall' attendu) ===")
    log(f"mauvais routage : mots-clés = {len(kw_bad)}/{nq} ({len(kw_bad)/nq:.0%}) "
        f"-> dont LLM en sauve {len(kw_bad)-llm_bad}/{len(kw_bad)} "
        f"(LLM résiduel = {llm_bad}/{nq})")

    log("")
    log("=== RÉSULTATS SQuAD (exactitude, réponse présente) ===")
    log(f"{'méthode':12s} | score")
    for s in ("base", "RAG", "compaction", "ours"):
        log(f"{s:12s} | {R[s]/nq*100:4.0f}%  ({R[s]}/{nq})")
    log("ROUTÉ        | = ours (toutes les questions SQuAD routent 'recall' -> poids)")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
