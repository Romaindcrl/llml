#!/usr/bin/env python3
"""Lot 4 — prépare le corpus QuALITY (CDC §4.2) pour le test mémoire closed-book.

Tire 15 article_ids UNIQUES avec seed=42 sur la liste TRIÉE des article_ids du
split dev (algorithme figé dans eval/scripts/draw_quality_docs.py, pré-enregistré),
puis émet un docs.jsonl : une ligne par document sélectionné, avec son texte et
ses questions à choix multiples NATIVES (rédigées par des humains — pas de
génération, pas de validation requise ; indépendance forte préservée).

Schéma QuALITY v1.0.1 (htmlstripped dev) : une ligne JSON par (article, set),
champs `article_id`, `title`, `article` (texte nu), `questions` = liste de
{question, options:[4], gold_label (1-indexé), difficult, ...}.

Usage :
  python eval/lot4/prep_quality.py \
      --data QuALITY.v1.0.1.htmlstripped.dev \
      --out eval/lot4/quality_docs.jsonl -k 15 --limit 5
"""

import argparse
import json
import random


def draw_ids(rows, k):
    ids = sorted({str(r["article_id"]) for r in rows})
    return sorted(random.Random(42).sample(ids, min(k, len(ids))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="QuALITY dev jsonl (htmlstripped)")
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=15, help="docs tirés seed=42")
    ap.add_argument("--limit", type=int, default=0,
                    help="n premiers docs (MVP budget) ; 0 = tous les k")
    a = ap.parse_args()

    rows = []
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    chosen = draw_ids(rows, a.k)
    if a.limit:
        chosen = chosen[: a.limit]
    chosen_set = set(chosen)

    # regroupe toutes les questions par article_id (un article peut apparaître
    # sur plusieurs lignes/sets)
    by_id = {}
    for r in rows:
        aid = str(r["article_id"])
        if aid not in chosen_set:
            continue
        d = by_id.setdefault(aid, {"doc_id": aid, "title": r.get("title", ""),
                                   "text": r.get("article", ""), "questions": []})
        if not d["text"]:
            d["text"] = r.get("article", "")
        for q in r.get("questions", []):
            gold = q.get("gold_label")
            opts = q.get("options") or []
            if not gold or len(opts) < 2:
                continue  # test set sans label, ou question malformée
            d["questions"].append({
                "qid": str(q.get("question_unique_id", "")),
                "q": q.get("question", ""),
                "options": opts,
                "gold_idx": int(gold) - 1,  # QuALITY gold_label est 1-indexé
                "difficult": int(q.get("difficult", 0)),
            })

    n_out = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for aid in chosen:
            d = by_id.get(aid)
            if not d or not d["questions"] or not d["text"]:
                print(f"  ! doc {aid} ignoré (pas de texte/questions)")
                continue
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_out += 1
            print(f"  doc {aid}: {len(d['questions'])} questions, "
                  f"{len(d['text'].split())} mots — {d['title'][:50]}")
    print(f"-> {a.out} ({n_out} docs)")


if __name__ == "__main__":
    main()
