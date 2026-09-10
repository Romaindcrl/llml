# Statut des campagnes et clôture des issues

Mise à jour du 10 septembre 2026, lors de la revue de la PR #2.

## Décisions

- **Issue #1 — campagne v1 terminée, résultats archivés avec limites.** La PR #2
  conserve les sorties disponibles et le rapport. La clôture ne signifie pas que
  toute la matrice préenregistrée a été exécutée ni que toutes ses hypothèses sont
  confirmées. Le résultat négatif sur le rappel factuel est conservé.
- **Issue #3 — ancien protocole v2 non poursuivi dans ce périmètre.** La génération
  de fonctions conformes à quatre dépôts sur GPU est distincte de la nouvelle piste
  de correcteur local. Les lots 2–5 et les hypothèses P/G/R n'ont pas de résultats
  disponibles. Clôture comme `not planned`, sans annoncer une réussite ni un échec
  du LoRA procédural.
- **PR #2 — intégration des archives et du backend CUDA expérimental.** Les
  corrections de maintenance ajoutées en septembre ne produisent pas de nouveaux
  scores et ne modifient pas les sorties historiques.

Les textes des préenregistrements restent accessibles dans les issues et aux
commits de gel ; ils ne sont pas réécrits après les mesures.

## Preuves disponibles et limites

| Ensemble | Faits vérifiables dans Git | Limite |
|---|---|---|
| Lot 2 | 542 paires draft/verified ; 6 modifications ; tallies : 4 gains sur tests cachés, 0 régression observée | Synthèses de scoring archivées ; absence de preuve générale de non-régression ou de gain humain |
| Lot intégré | 42 décisions de routage ; 12 tâches de génération, C0/C1 11/12 contre C2 9/12 | Petit échantillon ; ne remplace pas la matrice complète B1/B2 |
| Lots 4, 4b, 6bis | Sorties de rappel et résultats défavorables à la mémoire-poids sur les cas étudiés | Recette, modèles et corpus précis ; pas une impossibilité générale du fine-tuning |
| v2 | SHAs de cinq dépôts, listes de splits, règles, calibration et deux logs C0 | Tâches et générations détaillées absentes ; aucun résultat des lots 2–5 |

Les 536 sorties inchangées du Lot 2 incluent 88 tâches sans exemple offert,
378 drafts passant les exemples et 70 réparations rejetées. Elles ne constituent
pas 536 exemples validés « code correct, ne rien modifier ».

Artefacts v2 absents de la tête historique `2ebe76a` et de son historique inspecté :

- `eval/v2/tasks/*.jsonl` ;
- `results/v2/raw/gen_C0.jsonl`, `gen_C0_nowin.jsonl` ;
- `results/v2/raw/score_C0.jsonl`, `score_C0_nowin.jsonl`.

Les logs annoncent 150 tâches, dont 120 retenues dans le protocole ; les listes de
fichiers et les générateurs ne remplacent pas les sorties gelées. Le runner échoue
désormais avant chargement du modèle si les tâches ou les entrées d'un bras sont
absentes. Les futurs JSONL de ces emplacements publics ne sont plus ignorés par
Git. Aucun fichier manquant n'a été recréé ni présenté comme une archive retrouvée.

Le plan v1 prévoyait notamment une matrice multi-modèle plus large, une courbe
B3 sur dix cycles et des statistiques sur les différences appariées. Les petits
lots intégrés conservés ne suffisent pas à certifier ces objectifs. Le Lot 6bis
exécuté étudie une boucle de rappel factuel ; il ne doit pas être confondu avec la
mesure de tâches SDK de l'amendement initial de l'issue #1.

## Suite possible

L'audit du correcteur local a identifié dix incidents historiques SymPy, mais
aucune famille homogène avec cas sans modification et test final indépendant
prête pour l'entraînement. Les cinq cas du sous-système printing restent
hétérogènes et ne constituent pas une cible choisie. La qualification autorisée
examine d'abord Microsoft CodeReviewer, puis au maximum deux dépôts publics si
sa provenance est insuffisante. Plafonds : 90 minutes humaines, 60 minutes
d'extraction, aucun entraînement et aucune dépense.

Aucun entraînement, déploiement, achat de crédits ou poursuite de campagne GPU
n'est autorisé par cette clôture. Le design du correcteur et ses budgets restent
soumis aux étapes de validation convenues avec le propriétaire.

## Vérifications de maintenance (10 septembre)

- `python3 scripts/smoke.py` : 15 assertions réussies, backend mock.
- `python3 -m unittest discover -s tests` : 13 tests réussis ; état des adapters
  partagés, rollback de `/sleep`, sélection de format et rejet des entrées v2
  manquantes. Les poids et trainers sont remplacés par des doubles de test.
- Analyse syntaxique des 40 fichiers Python modifiés ou ajoutés par la PR : OK.
- `git diff --check` : OK.

Ces vérifications ne constituent pas un entraînement, un test CUDA/Metal ou une
réplication des performances. Les nouvelles protections sont de la maintenance,
pas l'implémentation du correcteur LoRA envisagé.

## Références

- [Issue #1](https://github.com/Romaindcrl/llml/issues/1), gel v1 `131527d`.
- [Issue #3](https://github.com/Romaindcrl/llml/issues/3), gel v2 `7d216a0`.
- [PR #2](https://github.com/Romaindcrl/llml/pull/2), tête historique `2ebe76a`.
- [Rapport v1](REPORT.md), [sorties Lot 2](../lot2/results/),
  [résultats intégrés](../lot_integrated/results/), [corpus v2](../v2/).
