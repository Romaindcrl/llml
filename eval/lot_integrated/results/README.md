# Test intégré — LLML au complet (résultats)

Le seul test qui exerce **tout le système ensemble**, pas les briques isolées :
le routeur `classify()` (m0/rag.py) décide **par requête** — rappel factuel →
mémoire-poids (LoRA replay `/sleep`), génération de code → base — sur un **flux
mixte**. M1 = Qwen2.5-7B-Instruct 8-bit.

- Corpus mémoire : 3 docs QuALITY internalisés en UN LoRA replay (114 faits,
  gate d'acquisition committée).
- Flux : **30 questions de rappel** (closed-book, choix multiple) + **12 tâches
  de génération** (HumanEval, pass@1 EvalPlus officiel).
- 4 configs : **C0** base nue · **C1** LLML complet (le routeur décide) ·
  **C2** mémoire toujours active (routeur désactivé, mode d'échec) ·
  **oracle** routage parfait (borne haute).

## Résultat

**Routage : 42/42 = 100 %** — le routeur aiguille chaque requête sans erreur
(30/30 rappels → mémoire, 12/12 générations → base).

| Config | Rappel (30) | Génération pass@1 (12) |
|--------|-------------|-------------------------|
| C0 — base nue            | 63,3 % | **91,7 %** |
| C1 — **LLML complet**    | 63,3 % | **91,7 %** |
| C2 — mémoire forcée      | 63,3 % | **75,0 %** |
| oracle — routage parfait | 63,3 % | 91,7 % |

## Lecture honnête

Trois enseignements, sans filtre :

1. **Le routeur gagne sa place.** Forcer la mémoire-poids sur toutes les requêtes
   (C2) fait chuter la génération de code **91,7 % → 75,0 %** (−17 pts). Le
   système avec routeur (C1) la **préserve à 91,7 %**. C'est la réplication
   publique de Claim B2 (le bench interne #16 « adapter always-on dégrade ») :
   l'étage mémoire EST nocif s'il est toujours actif, et le routeur existe
   précisément pour l'empêcher.

2. **Le routeur est fiable** : 100 % de routage correct ici, et **C1 = oracle**
   exactement (le routeur ne perd rien vs un routage parfait).

3. **Mais le système n'AJOUTE pas de rappel** : C1 = C0 = 63,3 % en rappel — la
   mémoire-poids ne relève pas le score de rappel (cohérent avec Claim A : sur ces
   questions QuALITY, la mémoire n'apporte rien). Le système « au complet » fait
   donc **aussi bien que la base, sans la dégrader**, et sa valeur ici est
   **défensive** (protéger la génération) plutôt qu'additive.

**Verdict :** l'intégration marche comme conçue — le routeur est le garde-fou qui
rend l'étage mémoire sûr (sans lui, −17 pts de code). Le gain net de *capacité*
reste porté par la vérification (Claim C) ; la mémoire-poids, sur ce corpus, est
neutre en rappel et coûteuse à activer partout — d'où l'intérêt du routeur.
Petits N (30 rappels, 12 codegen) : à lire comme une démonstration de mécanisme,
pas comme un chiffre de leaderboard.

## Fichiers

- `system_results.json` — routage + les 4 configs (rappel, génération pass@1).

Repro : `eval/lot_integrated/run_system.py` (routeur + mémoire replay + base,
4 configs) ; scoreboard `eval/lot_integrated/make_system_scoreboard.py`.
