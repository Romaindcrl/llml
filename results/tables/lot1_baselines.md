# Lot 1 — Baselines C0 (modèles de base seuls)

Scores en % avec IC 95% [lo, hi] (bootstrap 10 000 resamples / ±1.96·stderr
lm-eval). N par benchmark : HumanEval+ 164 · MBPP+ 378 · GSM8K 1319 ·
IFEval 541 · MMLU-Pro 6 domaines (N par domaine dans le CSV maître).
Configs gelées : greedy, seed 42, harness officiels — voir issue #1 et
AGENTS.md (3 gates déclenchées/résolues documentées).

| Benchmark | M1 bf16 | M1 8bit | M3 8bit |
|---|---|---|---|
| HumanEval (base) | 81.7 [75.6, 87.8] | 81.7 [75.6, 87.8] | 64.6 [57.3, 72.0] |
| HumanEval+ | 76.2 [69.5, 82.3] | 78.0 [71.3, 84.1] | 59.8 [51.8, 67.1] |
| MBPP (base) | — | 81.5 [77.5, 85.4] | 72.5 [68.0, 77.0] |
| MBPP+ | — | 70.6 [65.9, 75.1] | 61.9 [56.9, 66.9] |
| GSM8K (strict) | 77.7 [75.5, 80.0] | 77.3 [75.0, 79.5] | 77.4 [75.1, 79.7] |
| IFEval (prompt strict) | 71.9 [68.1, 75.7] | 71.0 [67.2, 74.8] | — |
| MMLU-Pro (6 dom., pondéré) | 64.7 [59.1, 70.4] | — | — |
| · mmlu_pro biology | 72.5 [69.3, 75.8] | — | — |
| · mmlu_pro business | 62.2 [58.8, 65.6] | — | — |
| · mmlu_pro computer_science | 59.3 [54.5, 64.0] | — | — |
| · mmlu_pro economics | 68.5 [65.3, 71.6] | — | — |
| · mmlu_pro math | 69.8 [67.4, 72.2] | — | — |
| · mmlu_pro other | 52.3 [49.1, 55.5] | — | — |

Source : `results/raw/lot1/scores_master.csv` (une ligne par cellule, horodatée, avec le
fichier brut d'origine). Runs d'investigation conservés dans
`results/raw/lot1/*/` (gsm8k chat-template, mmlu_pro tronqué).
