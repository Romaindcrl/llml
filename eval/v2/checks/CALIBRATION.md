# Calibration des checks v2 (CDC §2.3) — règles gelées

> Méthode : chaque règle est exécutée sur les 30 fonctions ORIGINALES
> reconstruites de son repo (dedent(header + expected_body)) — le code du repo
> est conforme par définition. Toute règle échouant sur >20% des originaux
> applicables est corrigée ou supprimée AVANT gel. Exécution :
> `python3 eval/v2/checks/calibrate.py` (2026-07-11, clones épinglés
> repos.lock.json, ruff cf. requirements-v2.lock, perl v5.38.2).

| Repo | Règles gelées | Adhérence des originaux | Règles >20% échec |
|---|---|---|---|
| curl | 20 (lint natif checksrc.pl) | 100,0% (465/465) | aucune |
| zulip | 33 (9 custom_check verbatim + 24 familles ruff) | 99,9% (720/721) | aucune |
| twisted | 18 (AST + struct) | 97,7% (296/303) | aucune |
| FreeRTOS-Kernel | 20 (struct/regex) | 100,0% (246/246) | aucune |
| tigerbeetle | 15 (struct/regex) | 100,0% (297/297) | aucune |
| **Total** | **106** | **99,4% (2024/2032)** | — |

Résidus <20% assumés (originaux non parfaits sur : zulip.ruff_N 1/30 — helper
`__`-préfixé toléré par leur CI ; twisted 7 instances réparties, aucune règle
au-dessus du seuil).
