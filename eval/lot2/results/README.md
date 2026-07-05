# Lot 2 — Claim C : boucle de vérification LLML (résultats)

Réplication publique du pilier **vérification** de LLML (draft → exécuter les
exemples documentés → réparer ≤2×, réparation adoptée seulement si elle passe
les exemples), scoring **100 % officiel EvalPlus** (tests cachés), deux bras
**appariés depuis le même draft**.

- Modèle : `Qwen/Qwen2.5-7B-Instruct` (M1), quant 8-bit (bitsandbytes), greedy.
- Bras `C0` (modèle nu) : draft one-shot.
- Bras `C0+verify` (LLML) : draft + boucle de vérification.
- Vérité-terrain offerte (CDC §4.4) : la boucle ne voit que les exemples de
  l'énoncé (doctests HumanEval, asserts MBPP), jamais les tests cachés.

## HumanEval+ — 164/164 (définitif)

| Bras | HumanEval (base) | HumanEval+ (plus, tests cachés) |
|------|------------------|----------------------------------|
| Modèle nu (C0)      | 136/164 = **82,9 %** | 129/164 = **78,7 %** |
| LLML (C0 + verify)  | 137/164 = **83,5 %** | 130/164 = **79,3 %** |
| **Δ**               | **+1 problème**      | **+1 problème (+0,6 pt)** |

- **1 problème corrigé** par la boucle (`HumanEval/19`), **0 régression**.
- Mécanique (méta) : 164 items, 76 avec exemples documentés / 88 sans. Sur les
  76, 62 drafts passent déjà les exemples ; **14 drafts échouent → réparation
  déclenchée ; 1 réparation adoptée** (passe les exemples) — et cette unique
  réparation passe aussi les tests cachés → gain net +1, aucune casse.

Lecture honnête : sur des tâches auto-contenues comme HumanEval, la mémoire de
poids ne peut rien apporter (rien à « savoir ») ; seul le pilier vérification
peut aider, et il le fait de façon **monotone** (ne dégrade jamais, récupère les
échecs rattrapables par les exemples de l'énoncé). C'est un gain réel mais
modeste, cohérent avec la théorie (self-repair façon Reflexion), et surtout
**sans aucune régression** — la propriété la plus importante d'un tel étage.

## MBPP+ — en cours

Même protocole, 378 problèmes, baseline de référence MBPP+ ≈ 70,6 %.

## Fichiers

- `humaneval_draft_samples.jsonl` — solutions bras nu (C0), 164 items.
- `humaneval_verified_samples.jsonl` — solutions bras LLML (C0+verify), 164 items.
- `humaneval_verify_meta.jsonl` — trace par item (nb exemples, réparations
  tentées, réparation adoptée).
- `tally_humaneval.json` — décompte final (pass/fail base & plus par bras,
  flips win/lose).

Reproduction : `eval/lot2/run_verify.py` (génération) + `eval/lot2/live_score.py`
(scoring officiel). Le scoreboard live est régénéré par
`eval/lot2/make_scoreboard.py`.
