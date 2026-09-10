# Lot 6bis — Pré-enregistrement : boucle d'apprentissage autonome (bench interne #12)

> **Statut : PRÉ-ENREGISTRÉ.** Ce fichier est committé **avant tout run**
> (règle de l'amendement #1, `AGENTS.md:124-128`). Les hypothèses H-D1..D4,
> le design et les critères de succès/échec sont figés ici ; tout écart au plan
> sera documenté explicitement dans le README de résultats, jamais silencieux.

## Ce qu'on teste

Les Lots 2/4/4b/intégré ont validé les **briques** et **une intégration
partielle one-shot** (routeur+mémoire+base, `lot_integrated`). Ce qui n'a
**jamais** tourné : la **boucle vivante en continu** — l'agent travaille →
son contexte **sature** → le contenu libéré est **auto-promu** en mémoire
long-terme → un cycle **`/sleep` gé** entraîne un LoRA → le **routeur** décide
par requête (rappel→poids / génération→base+RAG). C'est le bench interne #12.

Aucune brique nouvelle : le harness **compose les primitives réelles** du serveur
(`m0/agent.py:chat_turn` compaction+`on_compact`, `scripts/serve.py:_on_compact`,
`eval/lot4/run_memory.py:sleep_train` = réplique de `serve.py:_do_sleep`,
`m0/rag.py:classify`). Il ne réimplémente aucun scoring.

## Corpus GELÉ (zéro web live)

Snapshot committé = changelogs **post-cutoff** `uv` 0.11.22→0.11.26 et
`ruff` 0.15.17→0.15.20 (juin 2026), déjà dans `eval/lot4b/tech_docs.jsonl`.
Contrôle **C0-doit-échouer déjà prouvé** : base nue = **0/17** (Lot 4b). Toute
réussite est donc un gain réel, non un savoir pré-existant du modèle.

Le corpus est **slicé en 5 blocs chronologiques** (par date de version). Chaque
cycle *k* fait « lire » le bloc *k* à l'agent (suite de tours `chat_turn`)
jusqu'à saturation du contexte → auto-promotion. Les 17 QA factuelles sont
assignées au cycle qui introduit leur version. `/sleep` est **cumulatif** (ré-
entraîne sur toute la LTM accumulée) — c'est ce qui permet le test d'oubli.

## Paramètre explicite de faisabilité

`M0_COMPACT_TRIGGER` abaissé (défaut 4000 → **1200 tokens**) pour que chaque bloc
(~600–800 tokens) déclenche une vraie saturation sur ce corpus modeste. **Le
mécanisme est inchangé** : seul le seuil de déclenchement de la compaction
descend. Tous les autres hyperparamètres restent aux défauts LLML (gate 0.45,
`d2l_*` par défaut, rollback). Documenté ici, non silencieux.

## Configurations comparées (par cycle, sur les mêmes QA)

| Config | Politique |
|--------|-----------|
| **C0** | base nue (ni boucle, ni mémoire, ni RAG) — contrôle, = 0/17 attendu |
| **C1 = LLML intégré** | routeur `classify()` par requête : rappel→LoRA-mémoire seul, génération→base+RAG (fidèle à `serve.py`) |
| **RAG-oracle** | on force chaque QA factuelle dans base+RAG (ignore le routeur) — isole le coût de la politique « rappel→poids » |

## Hypothèses pré-enregistrées

- **H-D1 (le système apprend).** Au fil des cycles, le rappel du système
  intégré **C1** sur le matériel étudié dépasse significativement **C0** (0/17).
  *Seuil de succès : C1 poolé ≥ +20 pts sur C0.*
- **H-D2 (pas de régression sous apprentissage continu).** Une capacité de
  génération générale held-out (codegen HumanEval simple) reste intacte après
  5 cycles de `/sleep` — le routeur bascule sur base+RAG pour la génération.
  *Succès : gen pass@1 après cycle 5 ≥ gen pass@1 base, à ±1 item près.*
- **H-D3 (pas d'oubli catastrophique).** Le matériel appris au **cycle 1** reste
  rappelé (config C1) après les `/sleep` des cycles 2→5. *Succès : rappel des QA
  du cycle 1 mesuré après cycle 5 ≥ rappel au cycle 1, à −1 item près.*
- **H-D4 (l'intégration vs les parties).** La boucle complète C1 est comparée à
  **RAG-oracle**. *Question ouverte pré-enregistrée : si RAG-oracle ≫ C1, alors
  la politique de routage « rappel→poids » est le goulot, pas la boucle.*

## Prédictions honnêtes (au vu des Lots 4/4b)

Vu que la mémoire-poids ne rappelle quasiment rien (Lot 4 : 0/11 ; Lot 4b :
2/17 dont 2 coïncidences), on **s'attend** à :
- **H-D1** : plausiblement vraie **mais portée par le RAG** dans la boucle, pas
  par les poids. À confirmer.
- **H-D2** : vraie (le routeur protège la génération, cf. Lot intégré C1=oracle).
- **H-D3** : **incertaine** — si le rappel vient du RAG (persistant) il tient ;
  s'il devait venir des poids, il n'a jamais été là.
- **H-D4** : on **s'attend à RAG-oracle ≫ C1**, révélant que router les faits
  vers les poids (au lieu du RAG) est sous-optimal. Ce serait le résultat le
  plus informatif du lot.

## Critère de kill / lecture négative

Un résultat négatif (C1 ≈ C0, ou RAG-oracle ≫ C1) est **publié tel quel** : il
documente que la valeur du système vient du RAG + du routeur de protection, pas
de la consolidation en poids. Conforme à la politique « le négatif est publiable ».

## Reproductibilité

- Modèle : `Qwen/Qwen2.5-7B-Instruct` 8-bit (identique aux autres lots).
- Décodage : `temperature=0` (greedy).
- Harness : `eval/lot6bis/run_autoloop.py` ; corpus `eval/lot6bis/frozen_corpus.jsonl`
  (dérivé déterministe de `eval/lot4b/tech_docs.jsonl` par `build_corpus.py`).
- Résultats bruts : `eval/lot6bis/results/autoloop_results.jsonl` (1 ligne/cycle).
