"""Benchmark RAG (BM25) vs GREP (recherche littérale) — pur retrieval, sans LLM.

Question pratique : pour retrouver LE bon passage, un RAG lexical (BM25) bat-il un simple
grep ? On teste sur SQuAD (vraies questions humaines, souvent reformulées -> le cas dur pour
le littéral). Corpus = phrases de N paragraphes ; gold(question) = phrase(s) contenant la
réponse. Métrique : taux où le gold est dans le top-k récupéré (hit@1, hit@3).

3 méthodes :
  - BM25       : m0.rag.RAG (IDF + normalisation de longueur).
  - grep-multi : score = nb de mots-de-contenu de la requête présents littéralement (ripgrep
                 généreux multi-termes).
  - grep-strict: une seule recherche sur le terme le PLUS distinctif de la requête (grep classique).
Live : tail -f logs/benchmark_rag_vs_grep.log
"""

from __future__ import annotations

import math
import os
import re
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import httpx  # noqa: E402

from m0.rag import RAG, _STOP, chunk_text  # noqa: E402

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
N_PARA = 20
LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_rag_vs_grep.log")
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def _words(s):
    return [w for w in re.findall(r"\w+", (s or "").lower()) if w not in _STOP and len(w) > 1]


def _norm(s):
    return " ".join(re.findall(r"\w+", (s or "").lower()))


def load_squad():
    r = httpx.get(SQUAD_URL, timeout=60.0, headers={"User-Agent": "m0-grep/0.1"})
    r.raise_for_status()
    items = []
    for article in r.json()["data"]:
        for para in article["paragraphs"]:
            qas = [qa for qa in para["qas"] if qa.get("answers")]
            if len(para["context"]) < 200 or not qas:
                continue
            items.append((para["context"],
                          [(qa["question"], [a["text"] for a in qa["answers"]]) for qa in qas[:3]]))
            break
        if len(items) >= N_PARA:
            break
    return items


def grep_multi(query, chunks, toks, k):
    qw = set(_words(query))
    scored = sorted(((sum(1 for w in qw if w in c.lower()), -i, i) for i, c in enumerate(chunks)),
                    reverse=True)
    return [chunks[i] for s, _, i in scored[:k] if s > 0]


def grep_strict(query, chunks, toks, df, n_docs, k):
    qw = _words(query)
    if not qw:
        return []
    term = max(qw, key=lambda w: math.log(1 + n_docs / (df.get(w, 0) + 1)))  # le plus rare
    hits = [c for c in chunks if term in c.lower()]
    return hits[:k]


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    log("=== BENCHMARK RAG (BM25) vs GREP — retrieval pur (SQuAD) ===")
    items = load_squad()
    chunks, questions = [], []
    for ctx, qas in items:
        sents = chunk_text(ctx)
        chunks += sents
        for q, answers in qas:
            gold = [s for s in sents if any(_norm(a) and _norm(a) in _norm(s) for a in answers)]
            if gold:
                questions.append((q, gold))
    log(f"corpus = {len(chunks)} phrases ({len(items)} paragraphes) ; {len(questions)} questions")

    rag = RAG()
    for ctx, _ in items:
        rag.add_document(ctx)
    toks = [set(_words(c)) for c in chunks]
    df = {}
    for t in toks:
        for w in t:
            df[w] = df.get(w, 0) + 1
    nd = len(chunks)

    res = {m: {1: 0, 3: 0} for m in ("BM25", "grep-multi", "grep-strict")}
    for q, gold in questions:
        goldset = set(gold)
        retr = {"BM25": rag.topk(q, 3), "grep-multi": grep_multi(q, chunks, toks, 3),
                "grep-strict": grep_strict(q, chunks, toks, df, nd, 3)}
        for m, got in retr.items():
            if got and got[0] in goldset:
                res[m][1] += 1
            if any(g in goldset for g in got):
                res[m][3] += 1

    n = len(questions)
    log("")
    log("=== RÉSULTATS — le bon passage est-il récupéré ? ===")
    log(f"{'méthode':12s} | hit@1 | hit@3")
    for m in ("BM25", "grep-multi", "grep-strict"):
        log(f"{m:12s} | {res[m][1]/n*100:4.0f}% | {res[m][3]/n*100:4.0f}%")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
