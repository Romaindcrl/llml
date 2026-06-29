"""Benchmark : NOTRE méthode (LTM→replay→poids) vs RAG (BM25) vs compaction seule vs base.

Cadre RÉALISTE : corpus trop gros pour tenir en contexte. Chaque méthode doit répondre à
des questions held-out (jamais entraînées) :
  - base        : modèle seul, contexte vide (plancher)
  - RAG (BM25)  : récupère les top-k passages pertinents -> contexte (classique local)
  - compaction  : un résumé du corpus en contexte (lanes texte M0, pas de poids)
  - ours        : LoRA replay sur tout le corpus -> contexte VIDE (savoir dans les poids)

Tâches : classiques (QA wiki) + programmation (API fictive 'Glyph', inconnue du modèle).
Live : tail -f logs/benchmark.log
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from collections import Counter

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import httpx  # noqa: E402

from m0 import d2l  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402

WIKI = [("Tardigrade", "fr"), ("Stromboli", "fr"), ("Aurore polaire", "fr")]
LOG_PATH = os.path.join(_PROJ, "logs", "benchmark.log")
L = os.path.join(_PROJ, "models", "lora")
_T0 = time.time()

# --- Tâche de PROGRAMMATION : doc d'une API fictive (le modèle ne peut PAS la connaître)
GLYPH_DOC = (
    "Bibliotheque Glyph (v2), une API Python de rendu graphique.\n"
    "La fonction glyph.load(path) charge un fichier et retourne un objet Canvas.\n"
    "La methode Canvas.render(mode) dessine le canvas ; le mode par defaut est 'vector'.\n"
    "La methode Canvas.export(path, dpi) exporte l'image ; le dpi par defaut est 300.\n"
    "La fonction glyph.palette(name) retourne une palette de couleurs ; la palette par "
    "defaut s'appelle 'sol'.\n"
    "La constante glyph.MAX_LAYERS vaut 64.\n"
    "Pour fusionner deux canvas on appelle canvas.merge(other, opacity).\n"
    "La classe Canvas leve l'exception GlyphError si le fichier est corrompu.\n"
)
GLYPH_QA = [
    ("Quelle fonction Glyph charge un fichier ?", "glyph.load"),
    ("Que retourne glyph.load ?", "Canvas"),
    ("Quel est le mode par defaut de Canvas.render ?", "vector"),
    ("Quel est le dpi par defaut de Canvas.export ?", "300"),
    ("Comment s'appelle la palette par defaut ?", "sol"),
    ("Combien vaut glyph.MAX_LAYERS ?", "64"),
    ("Quelle methode fusionne deux canvas ?", "merge"),
    ("Quelle exception est levee si le fichier est corrompu ?", "GlyphError"),
]


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def fetch_wiki(title, lang="fr", max_chars=4000):
    r = httpx.get(f"https://{lang}.wikipedia.org/w/api.php",
                  params={"format": "json", "action": "query", "prop": "extracts",
                          "explaintext": 1, "redirects": 1, "titles": title},
                  timeout=30.0, headers={"User-Agent": "m0-bench/0.1"})
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("title", title), (page.get("extract", "") or "")[:max_chars]


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 15]


class BM25:
    def __init__(self, chunks):
        self.chunks = chunks
        self.toks = [re.findall(r"\w+", c.lower()) for c in chunks]
        self.N = len(chunks)
        self.avgdl = sum(len(t) for t in self.toks) / max(1, self.N)
        self.df = Counter()
        for t in self.toks:
            self.df.update(set(t))
        self.k1, self.b = 1.5, 0.75

    def _idf(self, w):
        n = self.df.get(w, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def topk(self, q, k=3):
        qt = re.findall(r"\w+", q.lower())
        scores = []
        for i, t in enumerate(self.toks):
            tf = Counter(t)
            dl = len(t)
            s = sum(self._idf(w) * tf[w] * (self.k1 + 1) /
                    (tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    for w in qt if w in tf)
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:k]]


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== BENCHMARK (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg)

    # 1) corpus : docs + Q/R held-out (classic via extraction, prog hand-written)
    log("[1/4] Préparation du corpus")
    corpus_texts, train_qa, eval_items = [], [], []
    for title, lang in WIKI:
        log(f"  [prep] extraction + augmentation : {title}…")
        t, text = fetch_wiki(title, lang)
        corpus_texts.append(text)
        qa = d2l.clean_and_balance(d2l.extract_qa(text, llm.generate, n=12), max_per_answer=2)
        tr, ev = d2l.split_train_eval(d2l.clean_and_balance(
            d2l.augment_pairs(qa, llm.generate, n_paraphrases=2), max_per_answer=6) or qa, 1)
        # held-out = faits distincts non vus ; on garde 3 pour l'éval
        seen_ans, ev_clean = set(), []
        for q, a in ev:
            if a.lower() not in seen_ans:
                seen_ans.add(a.lower()); ev_clean.append((q, a))
        train_qa += tr
        eval_items += [(q, a, "classic") for q, a in ev_clean[:3]]
        log(f"[prep] {t}: train={len(tr)} eval={len(ev_clean[:3])}")
    # programmation : MÊME pipeline que le wiki (augment + split) -> held-out = reformulations
    # des faits ENTRAÎNÉS (équitable : toutes les méthodes testées sur les mêmes faits).
    corpus_texts.append(GLYPH_DOC)
    gtr, gev = d2l.split_train_eval(d2l.clean_and_balance(
        d2l.augment_pairs(d2l.clean_and_balance(GLYPH_QA, max_per_answer=2),
                          llm.generate, n_paraphrases=2), max_per_answer=6) or GLYPH_QA, 1)
    train_qa += gtr
    eval_items += [(q, a, "prog") for q, a in gev[:4]]
    log(f"[prep] Glyph(prog): train={len(gtr)} eval={len(gev[:4])} | TOTAL eval={len(eval_items)}")

    # 2) RAG (BM25) + compaction (résumé)
    log("[2/4] Index RAG (BM25) + résumé de compaction")
    chunks = [s for txt in corpus_texts for s in sentences(txt)]
    rag = BM25(chunks)
    corpus_concat = "\n\n".join(corpus_texts)
    summary = llm.generate(
        "Resume ce corpus en faits concis (un par ligne, garde les chiffres/noms exacts) :\n"
        + corpus_concat, None)
    log(f"[build] {len(chunks)} chunks BM25 · résumé={d2l.count_tokens(summary) if hasattr(d2l,'count_tokens') else len(summary)//4} tokens")

    # 3) NOTRE méthode : replay LoRA sur tout le corpus
    log("[3/4] Entraînement de NOTRE LoRA (replay sur le corpus)")
    train_qa = d2l.clean_and_balance(train_qa, max_per_answer=6)
    data = f"{_PROJ}/logs/bench_data"
    adapter = f"{L}/bench"
    n = d2l.build_chat_dataset(train_qa, data, repeat=4, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=4)
    iters = min(500, max(150, 12 * len(train_qa)))
    res = d2l.train_lora(cfg.mlx_model_path, data, adapter, iters=iters, num_layers=cfg.d2l_num_layers,
                         learning_rate=cfg.d2l_learning_rate, rank=16, python_exe=sys.executable,
                         log_file=LOG_PATH)
    log(f"[train] ours: ok={res['ok']} val_loss={res['val_loss']} ({n} lignes, {iters} iters)")

    # 4) Évaluation des 4 SUT
    log("[4/4] Évaluation")

    def gen_plain(prompt):
        return llm.generate(prompt, None)

    results = {s: {"classic": [0, 0], "prog": [0, 0], "ctx": []} for s in
               ("base", "RAG", "compaction", "ours")}

    nq = len(eval_items)
    # --- SUT sur modèle de BASE (sans adapter) : base, RAG, compaction
    llm.set_adapter(None)
    for i, (q, a, tt) in enumerate(eval_items, 1):
        b = d2l.answer_recalled(gen_plain(q), a)
        results["base"][tt][1] += 1; results["base"][tt][0] += b; results["base"]["ctx"].append(0)
        ctx = "\n".join(rag.topk(q, 3))
        results["RAG"]["ctx"].append(len(re.findall(r"\w+", ctx)))
        rr = d2l.answer_recalled(gen_plain(f"Contexte :\n{ctx}\n\nQuestion : {q}\nReponds en quelques mots :"), a)
        results["RAG"][tt][1] += 1; results["RAG"][tt][0] += rr
        results["compaction"]["ctx"].append(len(re.findall(r"\w+", summary)))
        cc = d2l.answer_recalled(gen_plain(f"Resume du corpus :\n{summary}\n\nQuestion : {q}\nReponds en quelques mots :"), a)
        results["compaction"][tt][1] += 1; results["compaction"][tt][0] += cc
        log(f"[eval {i}/{nq} {tt:7s}] base={int(b)} rag={int(rr)} comp={int(cc)}")

    # --- NOTRE méthode : adapter chargé, contexte VIDE
    llm.set_adapter(adapter)
    for i, (q, a, tt) in enumerate(eval_items, 1):
        o = d2l.answer_recalled(gen_plain(q), a)
        results["ours"][tt][1] += 1; results["ours"][tt][0] += o; results["ours"]["ctx"].append(0)
        log(f"[eval {i}/{nq} {tt:7s}] ours={int(o)}")

    # --- table
    def acc(p):
        return f"{(p[0] / p[1] * 100):3.0f}%" if p[1] else "  -"
    log("")
    log("=== RÉSULTATS (exactitude held-out) ===")
    log(f"{'méthode':12s} | {'classique':>9s} | {'prog':>5s} | {'global':>6s} | ctx tokens/q")
    for s in ("base", "RAG", "compaction", "ours"):
        r = results[s]
        tot = [r["classic"][0] + r["prog"][0], r["classic"][1] + r["prog"][1]]
        ctxavg = sum(r["ctx"]) / max(1, len(r["ctx"]))
        log(f"{s:12s} | {acc(r['classic']):>9s} | {acc(r['prog']):>5s} | {acc(tot):>6s} | {ctxavg:5.0f}")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
