#!/usr/bin/env python3
"""Lot 6bis — dérive un corpus GELÉ en 5 cycles chronologiques à partir de
eval/lot4b/tech_docs.jsonl (déterministe, zéro web). Chaque bloc = une section
de version « ## X.Y.Z … Released on DATE … ». Blocs triés par date, répartis en
5 cycles. Chaque QA (17) est assignée au cycle qui introduit SA version (parsée
depuis l'énoncé). Sortie : eval/lot6bis/frozen_corpus.jsonl (1 ligne/cycle)."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "lot4b", "tech_docs.jsonl")
OUT = os.path.join(HERE, "frozen_corpus.jsonl")
N_CYCLES = 5


def split_versions(doc_id, text):
    """Découpe le changelog en blocs de version avec leur date."""
    # sections commençant par "## X.Y.Z"
    parts = re.split(r"\n## (\d+\.\d+\.\d+)\n", "\n" + text)
    blocks = []
    # parts = [préambule, ver1, corps1, ver2, corps2, ...]
    for i in range(1, len(parts), 2):
        ver = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        m = re.search(r"Released on (\d{4}-\d{2}-\d{2})", body)
        date = m.group(1) if m else "9999-99-99"
        header = f"## {doc_id} {ver}\n"
        blocks.append({"doc": doc_id, "version": ver, "date": date,
                       "text": header + body.strip()})
    return blocks


def main():
    docs = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    blocks, qa = [], []
    for d in docs:
        blocks += split_versions(d["doc_id"], d["text"])
        for q in d["questions"]:
            m = re.search(r"(\d+\.\d+\.\d+)", q["q"])
            qa.append({"qid": q["qid"], "q": q["q"], "answer": q["answer"],
                       "doc": d["doc_id"], "version": m.group(1) if m else None})

    blocks.sort(key=lambda b: (b["date"], b["doc"], b["version"]))
    # répartition en 5 cycles ~équilibrés en nombre de blocs, ordre chrono
    cycles = [[] for _ in range(N_CYCLES)]
    for i, b in enumerate(blocks):
        cycles[min(i * N_CYCLES // len(blocks), N_CYCLES - 1)].append(b)

    # version -> cycle
    ver2cyc = {}
    for ci, cb in enumerate(cycles, 1):
        for b in cb:
            ver2cyc[(b["doc"], b["version"])] = ci

    recs = []
    for ci, cb in enumerate(cycles, 1):
        qs = [q for q in qa if ver2cyc.get((q["doc"], q["version"])) == ci]
        dates = [b["date"] for b in cb]
        recs.append({
            "cycle": ci,
            "date_range": f"{min(dates)}..{max(dates)}" if dates else "",
            "versions": [f"{b['doc']} {b['version']}" for b in cb],
            "blocks": [b["text"] for b in cb],
            "questions": [{"qid": q["qid"], "q": q["q"], "answer": q["answer"]} for q in qs],
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_q = sum(len(r["questions"]) for r in recs)
    print(f"-> {OUT}")
    for r in recs:
        chars = sum(len(b) for b in r["blocks"])
        print(f"  cycle {r['cycle']} [{r['date_range']}] {len(r['blocks'])} blocs "
              f"({chars} chars ~{chars // 4} tok), {len(r['questions'])} QA: "
              f"{[q['qid'] for q in r['questions']]}")
    print(f"TOTAL {len(recs)} cycles, {n_q} QA")
    assert n_q == 17, f"attendu 17 QA, obtenu {n_q}"


if __name__ == "__main__":
    main()
