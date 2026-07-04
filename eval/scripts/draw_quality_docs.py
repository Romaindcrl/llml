#!/usr/bin/env python3
"""Tirage pré-enregistré des 15 documents QuALITY (CDC §4.2) — seed=42.

QuALITY (Pang et al.) : documents longs + QA à choix multiples, split dev.
On tire 15 article_ids UNIQUES avec random.seed(42) sur la liste TRIÉE des
article_ids du dev set — déterministe et vérifiable. Le tirage est exécuté au
Lot 4 (téléchargement du dataset requis) mais l'algorithme est figé ici, avant
le pré-enregistrement.

Source des données : https://github.com/nyu-mll/quality (QuALITY.v1.0.1),
fichier dev (htmlstripped). Usage :
    python eval/scripts/draw_quality_docs.py --data QuALITY.v1.0.1.htmlstripped.dev
"""

import argparse
import json
import random


def draw(path: str, k: int = 15) -> list[str]:
    articles: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            articles.setdefault(str(row["article_id"]), row.get("title", ""))
    ids = sorted(articles)
    rng = random.Random(42)
    chosen = sorted(rng.sample(ids, k))
    for aid in chosen:
        print(f"{aid}\t{articles[aid]}")
    return chosen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="QuALITY dev jsonl (htmlstripped)")
    ap.add_argument("-k", type=int, default=15)
    args = ap.parse_args()
    draw(args.data, args.k)
