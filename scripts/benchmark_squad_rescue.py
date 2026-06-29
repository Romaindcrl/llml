"""SQuAD — tentative de SAUVER les poids (run de base : ours=34% vs RAG 88%, base 59%).

Deux bras contre les 2 causes de l'échec, TOUS DEUX en format CHAT (leçon d2l : entraîner
en texte brut puis inférer via chat-template => mismatch destructeur) :
  - ours-QA++ : extraction Q/R EXHAUSTIVE (n=24/passage) + ancres renforcées (anti-oubli).
                Vise la COUVERTURE des faits que SQuAD pourrait demander.
  - ours-DOC  : recopie du passage en chat ("Parle-moi de {sujet}" -> contenu), pour
                internaliser TOUT le contenu (pas seulement nos Q/R), format-cohérent.
Réf : base + RAG recalculés sur les MÊMES 32 questions. ours-orig (run précédent) = 34%.
Logs par question (cuttable). Live : tail -f logs/benchmark_squad_rescue.log
"""

from __future__ import annotations

import os
import re
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import httpx  # noqa: E402

from m0 import d2l  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.rag import RAG, chunk_text  # noqa: E402

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
N_CONTEXTS, Q_PER_CTX = 8, 4
LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_squad_rescue.log")
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
    r = httpx.get(SQUAD_URL, timeout=60.0, headers={"User-Agent": "m0-squad/0.1"})
    r.raise_for_status()
    items = []
    for article in r.json()["data"]:
        title = article["title"].replace("_", " ")
        for para in article["paragraphs"]:
            ctx = para["context"]
            qas = [qa for qa in para["qas"] if qa.get("answers")]
            if len(ctx) < 200 or len(qas) < Q_PER_CTX:
                continue
            items.append((ctx, title, [qa["question"] for qa in qas[:Q_PER_CTX]],
                          [[a["text"] for a in qa["answers"]] for qa in qas[:Q_PER_CTX]]))
            break
        if len(items) >= N_CONTEXTS:
            break
    return items


def doc_pairs(ctx, title):
    """Pairs CHAT de recitation : (question sur le sujet -> morceau du passage)."""
    chunks = chunk_text(ctx)
    pairs = [(f"Que peux-tu me dire sur {title} ?", ctx)]
    for ch in chunks:
        pairs.append((f"Donne des informations factuelles sur {title}.", ch))
    return pairs


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== SQuAD RESCUE (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg)

    items = load_squad()
    flat = [(q, g) for _, _, qs, gs in items for q, g in zip(qs, gs)]
    nq = len(flat)
    log(f"[1/4] {len(items)} passages · {nq} questions held-out")

    rag = RAG(os.path.join(_PROJ, "logs", "rag_squad_r.txt")); rag.clear()
    for ctx, _, _, _ in items:
        rag.add_document(ctx)

    # bras QA++ : extraction EXHAUSTIVE
    qa = []
    for i, (ctx, _, _, _) in enumerate(items, 1):
        ex = d2l.clean_and_balance(d2l.extract_qa(ctx, llm.generate, n=24), max_per_answer=3)
        qa += d2l.augment_pairs(ex, llm.generate, n_paraphrases=1)
        log(f"  [QA++ {i}/{len(items)}] +{len(ex)} Q/R (total={len(qa)})")
    qa = d2l.clean_and_balance(qa, max_per_answer=8)

    # bras DOC : recitation du passage (chat)
    docp = []
    for ctx, title, _, _ in items:
        docp += doc_pairs(ctx, title)

    log("[2/4] Entraînement ours-QA++ (Q/R exhaustives + ancres renforcées)")
    d_qa, a_qa = f"{_PROJ}/logs/squad_qapp_data", f"{L}/squad_qapp"
    n1 = d2l.build_chat_dataset(qa, d_qa, repeat=4, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=6)
    it1 = min(700, max(300, 8 * len(qa)))
    r1 = d2l.train_lora(cfg.mlx_model_path, d_qa, a_qa, iters=it1, num_layers=cfg.d2l_num_layers,
                        learning_rate=cfg.d2l_learning_rate, rank=16, python_exe=sys.executable,
                        log_file=LOG_PATH)
    log(f"  QA++ : ok={r1['ok']} val_loss={r1['val_loss']} ({n1} lignes, {it1} iters)")

    log("[3/4] Entraînement ours-DOC (recopie passage en chat)")
    d_dc, a_dc = f"{_PROJ}/logs/squad_doc_data", f"{L}/squad_doc"
    n2 = d2l.build_chat_dataset(docp, d_dc, repeat=3, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=6)
    it2 = min(700, max(300, 8 * len(docp)))
    r2 = d2l.train_lora(cfg.mlx_model_path, d_dc, a_dc, iters=it2, num_layers=cfg.d2l_num_layers,
                        learning_rate=cfg.d2l_learning_rate, rank=16, python_exe=sys.executable,
                        log_file=LOG_PATH)
    log(f"  DOC : ok={r2['ok']} val_loss={r2['val_loss']} ({n2} lignes, {it2} iters)")

    def gen_ctx(q, ctx):
        p = q if ctx is None else f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer concisely:"
        return llm.generate(p, None)

    R = {s: 0 for s in ("base", "RAG", "ours-QA++", "ours-DOC")}
    log("[4/4] Évaluation SQuAD-style")
    llm.set_adapter(None)
    for i, (q, g) in enumerate(flat, 1):
        b = squad_hit(gen_ctx(q, None), g)
        rr = squad_hit(gen_ctx(q, "\n".join(rag.topk(q, 4))), g)
        R["base"] += b; R["RAG"] += rr
        log(f"[q {i}/{nq}] base={int(b)} rag={int(rr)}")
    llm.set_adapter(a_qa)
    for i, (q, g) in enumerate(flat, 1):
        o = squad_hit(gen_ctx(q, None), g); R["ours-QA++"] += o
        log(f"[q {i}/{nq}] ours-QA++={int(o)}")
    llm.set_adapter(a_dc)
    for i, (q, g) in enumerate(flat, 1):
        o = squad_hit(gen_ctx(q, None), g); R["ours-DOC"] += o
        log(f"[q {i}/{nq}] ours-DOC={int(o)}")

    log("")
    log("=== RÉSULTATS SQuAD — tentative de sauvetage des poids ===")
    log(f"{'méthode':12s} | score")
    log(f"{'base':12s} | {R['base']/nq*100:4.0f}%  ({R['base']}/{nq})")
    log(f"{'RAG':12s} | {R['RAG']/nq*100:4.0f}%  ({R['RAG']}/{nq})")
    log(f"{'ours-orig':12s} |   34%  (run précédent, Q/R n=10)")
    log(f"{'ours-QA++':12s} | {R['ours-QA++']/nq*100:4.0f}%  ({R['ours-QA++']}/{nq})")
    log(f"{'ours-DOC':12s} | {R['ours-DOC']/nq*100:4.0f}%  ({R['ours-DOC']}/{nq})")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
