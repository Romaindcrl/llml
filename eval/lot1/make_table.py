#!/usr/bin/env python3
"""Génère results/tables/lot1_baselines.md depuis scores_master.csv (CDC §3.4).

ICs 95% :
  - lignes lm-eval : stderr fourni par le harness -> IC = value ± 1.96·stderr ;
  - lignes EvalPlus (pass@1, N connu) : bootstrap binomial 10 000 resamples
    (seed 42) sur la proportion — équivalent au bootstrap par item puisque le
    score par item est 0/1 et que value·N est entier.

Le tableau affiche N partout (CDC §3.4). Les métriques rapportées :
  humaneval_plus/base, mbpp_plus/base (pass@1) ; gsm8k strict-match ;
  ifeval prompt_level_strict_acc ; mmlu_pro par domaine + moyenne pondérée.
"""

import argparse
import csv
import random
from collections import defaultdict

PRIMARY = {
    ("humaneval_base", "pass@1"): "HumanEval (base)",
    ("humaneval_plus", "pass@1"): "HumanEval+",
    ("mbpp_base", "pass@1"): "MBPP (base)",
    ("mbpp_plus", "pass@1"): "MBPP+",
    ("gsm8k", "exact_match[strict-match]"): "GSM8K (strict)",
    ("ifeval", "prompt_level_strict_acc"): "IFEval (prompt strict)",
}
MMLU_PREFIX = "mmlu_pro_"


def boot_ci(p: float, n: int, iters: int = 10_000, seed: int = 42):
    rng = random.Random(seed)
    k = round(p * n)
    stats = []
    for _ in range(iters):
        s = sum(1 for _ in range(n) if rng.random() < k / n)
        stats.append(s / n)
    stats.sort()
    return stats[int(0.025 * iters)], stats[int(0.975 * iters)]


def fmt(v, lo, hi):
    return f"{100*v:.1f} [{100*lo:.1f}, {100*hi:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/raw/lot1/scores_master.csv")
    ap.add_argument("--out", default="results/tables/lot1_baselines.md")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    cells = {}          # (model, quant, label) -> "score [lo, hi]"
    ns = {}             # (model, quant, label) -> n
    mmlu = defaultdict(list)  # (model, quant) -> [(domain, v, n, stderr)]
    for r in rows:
        key = (r["benchmark"], r["metric"])
        v = float(r["value"])
        n = int(float(r["n"])) if r["n"] else None
        mq = (r["model"], r["quant"])
        if key in PRIMARY:
            label = PRIMARY[key]
            if r["stderr"]:
                se = float(r["stderr"])
                lo, hi = v - 1.96 * se, v + 1.96 * se
            else:
                lo, hi = boot_ci(v, n)
            cells[(*mq, label)] = fmt(v, lo, hi)
            ns[(*mq, label)] = n
        elif r["benchmark"].startswith(MMLU_PREFIX) and r["metric"].startswith("exact_match"):
            mmlu[mq].append((r["benchmark"][len(MMLU_PREFIX):], v, n,
                             float(r["stderr"]) if r["stderr"] else None))

    for mq, doms in mmlu.items():
        tot = sum(n for _, _, n, _ in doms if n)
        if not tot:
            continue
        wmean = sum(v * n for _, v, n, _ in doms if n) / tot
        # IC de la moyenne pondérée : bootstrap binomial par domaine, agrégé
        rng = random.Random(42)
        stats = []
        for _ in range(10_000):
            acc = 0
            for _, v, n, _ in doms:
                k = round(v * n)
                acc += sum(1 for _ in range(50) if rng.random() < k / n) / 50 * n
            stats.append(acc / tot)
        stats.sort()
        cells[(*mq, "MMLU-Pro (6 dom., pondéré)")] = fmt(
            wmean, stats[250], stats[9750])
        ns[(*mq, "MMLU-Pro (6 dom., pondéré)")] = tot
        for d, v, n, se in sorted(doms):
            lo, hi = (v - 1.96 * se, v + 1.96 * se) if se else boot_ci(v, n)
            cells[(*mq, f"· mmlu_pro {d}")] = fmt(v, lo, hi)
            ns[(*mq, f"· mmlu_pro {d}")] = n

    col_order = [("M1", "bf16"), ("M1", "8bit"), ("M2", "8bit"),
                 ("M3", "8bit"), ("M4", "8bit")]
    col_present = [c for c in col_order if any((c[0], c[1]) == (m, q)
                   for (m, q, _) in cells)]
    labels = list(PRIMARY.values()) + ["MMLU-Pro (6 dom., pondéré)"] + sorted(
        {l for (_, _, l) in cells if l.startswith("· mmlu_pro")})

    lines = [
        "# Lot 1 — Baselines C0 (modèles de base seuls)",
        "",
        "Scores en % avec IC 95% [lo, hi] (bootstrap 10 000 resamples / ±1.96·stderr",
        "lm-eval). N par benchmark : HumanEval+ 164 · MBPP+ 378 · GSM8K 1319 ·",
        "IFEval 541 · MMLU-Pro 6 domaines (N par domaine dans le CSV maître).",
        "Configs gelées : greedy, seed 42, harness officiels — voir issue #1 et",
        "AGENTS.md (3 gates déclenchées/résolues documentées).",
        "",
        "| Benchmark | " + " | ".join(f"{m} {q}" for m, q in col_present) + " |",
        "|" + "---|" * (len(col_present) + 1),
    ]
    for label in labels:
        row = [label]
        for m, q in col_present:
            row.append(cells.get((m, q, label), "—"))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", f"Source : `{a.csv}` (une ligne par cellule, horodatée, avec le",
              "fichier brut d'origine). Runs d'investigation conservés dans",
              "`results/raw/lot1/*/` (gsm8k chat-template, mmlu_pro tronqué).", ""]

    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"-> {a.out}")
    for line in lines[7:]:
        print(line)


if __name__ == "__main__":
    main()
