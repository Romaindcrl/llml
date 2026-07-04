#!/usr/bin/env python3
"""Tirage pré-enregistré des 6 domaines MMLU-Pro (CDC §4.1) — seed=42, figé AVANT
tout run et publié dans l'issue de pré-enregistrement.

Les 14 catégories officielles de MMLU-Pro (TIGER-Lab/MMLU-Pro), triées puis
échantillonnées avec random.seed(42) — déterministe et vérifiable par quiconque.
"""

import random

CATEGORIES = sorted([
    "biology", "business", "chemistry", "computer science", "economics",
    "engineering", "health", "history", "law", "math", "philosophy",
    "physics", "psychology", "other",
])

if __name__ == "__main__":
    rng = random.Random(42)
    drawn = sorted(rng.sample(CATEGORIES, 6))
    print("seed=42 -> 6 domaines MMLU-Pro :")
    for d in drawn:
        print(" -", d)
