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
peut aider, et aucune dégradation n'est observée sur ce run ; cela ne garantit pas
l'absence de régression sur de nouveaux problèmes. C'est un gain réel mais
modeste ; aucune régression n'a été observée sur cet échantillon.

## MBPP+ — 378/378 (définitif)

Même protocole, 378 problèmes. Ici les exemples offerts sont les `assert` de
l'énoncé EvalPlus (présents sur les 378 items, vs seulement 76/164 doctests sur
HumanEval), donc la boucle de vérification se déclenche bien plus souvent.

| Bras | MBPP (base) | MBPP+ (plus, tests cachés) |
|------|-------------|-----------------------------|
| Modèle nu (C0)      | 307/378 = **81,2 %** | 261/378 = **69,0 %** |
| LLML (C0 + verify)  | 312/378 = **82,5 %** | 264/378 = **69,8 %** |
| **Δ**               | **+5 problèmes**     | **+3 problèmes (+0,8 pt)** |

- **3 problèmes corrigés** par la boucle (`Mbpp/6`, `Mbpp/259`, `Mbpp/391`),
  **0 régression**.
- Mécanique (méta) : 378 items, tous avec `assert` d'exemple ; 316 drafts les
  passent déjà ; **62 drafts échouent → réparation déclenchée ; 5 réparations
  adoptées** (passent les asserts). Sur ces 5, **3 passent aussi les tests
  cachés** (gain net) et 2 passent les exemples sans changer le verdict caché
  (neutre). Toujours **aucune casse**.

## Bilan Claim C (les deux benchmarks)

| Benchmark | Modèle nu (plus) | LLML verify (plus) | Δ | Corrigés | Régressions |
|-----------|------------------|--------------------|----|----------|-------------|
| HumanEval+ (164) | 78,7 % | **79,3 %** | **+1** | 1 | **0** |
| MBPP+ (378) | 69,0 % | **69,8 %** | **+3** | 3 | **0** |
| **Cumulé (542)** | — | — | **+4** | **4** | **0** |

Sur **542 problèmes de code appariés**, la boucle de vérification de LLML
récupère **4 échecs** (rattrapables par les exemples de l'énoncé) et n'introduit
**aucune régression**. Le gain de capacité brute est modeste (attendu : sur des
tâches auto-contenues, seul l'étage vérification peut aider, pas la mémoire),
mais les exemples visibles ne garantissent pas la correction sur les tests
cachés. L'absence de régression observée ne constitue pas une propriété générale
de monotonie. Les fichiers ci-dessous sont des archives, pas une réexécution lors
de la revue de septembre ; voir le [statut de la campagne](../../results/CAMPAIGN_STATUS.md).

## Fichiers

HumanEval+ : `humaneval_draft_samples.jsonl` (nu), `humaneval_verified_samples.jsonl`
(LLML), `humaneval_verify_meta.jsonl` (trace par item), `tally_humaneval.json`.
MBPP+ : `mbpp_draft_samples.jsonl`, `mbpp_verified_samples.jsonl`,
`mbpp_verify_meta.jsonl`, `tally_mbpp.json` (mêmes formats).

Reproduction : `eval/lot2/run_verify.py` (génération) + `eval/lot2/live_score.py`
(scoring officiel). Le scoreboard live est régénéré par
`eval/lot2/make_scoreboard.py`.
