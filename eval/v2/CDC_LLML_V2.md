# Cahier des charges — LLML v2 : mémoire procédurale

**Projet :** LLML (https://github.com/Romaindcrl/llml)
**Contexte :** l'évaluation publique v1 (PR #2) a réfuté le rappel factuel dans les poids (0/11 QuALITY, 2/17 artefacts) et validé le routeur (42/42, prévention de la dégradation 92→75%) ainsi que la boucle verify (+4/542, zéro régression). La gate /sleep v1 est invalidée : elle a commité un adapter (score 0.538) dont le rappel réel était nul — elle mesure la récitation de paraphrases, pas l'apprentissage.
**Thèse v2 :** la mémoire n'est pas une seule chose. Mémoire factuelle → RAG (mesuré gagnant en v1). Mémoire procédurale (conventions, style, procédures d'un domaine) → poids. Le routeur décide du lane et protège le modèle de base.
**Exécutant :** agent Claude Code sur machine GPU distante.
**Version :** 2.0 — juillet 2026

---

## 0. Instructions générales pour l'agent

- Lire ce document EN ENTIER avant de commencer, le relire au début de chaque lot. Lire aussi `results/REPORT.md` de la v1 et la PR #2 : la v2 s'appuie sur ces résultats et ses artefacts (adapters du Lot 4 v1 conservés — NE PAS les supprimer, ils servent de jeu de test au Lot 2).
- Créer/mettre à jour `AGENTS.md` à la racine : règles méthodologiques (§3), avancement des lots, journal des décisions.
- Lots strictement séquentiels. Critères d'acceptation validés avant de passer au suivant.
- **Aucun LLM juge, nulle part.** Toute métrique d'adhérence est déterministe (linters, règles AST, regex structurelles). C'est un principe produit, pas juste d'éval.
- En cas de choix ambigu : option la plus défavorable à LLML, décision loggée.
- Tout résultat en double : CSV brut dans `results/v2/raw/` + markdown dans `results/v2/tables/`.
- Budget global plafonné : **40h GPU** (4090 spot). Dépassement 2x sur un lot → arrêt, rapport d'étape.

---

## 1. Claims à valider

**Claim P — Mémoire procédurale.** Un adapter LoRA entraîné via `/sleep` sur les conventions d'un repo atteint une adhérence aux conventions **non-inférieure** au fichier de conventions placé en contexte (marge de non-infériorité : 5 points), pour **zéro token de conventions par requête**, sans dégrader les capacités générales.

**Claim G — Gate v2.** Une gate de commit reconstruite distingue apprentissage et récitation : elle **rejette** les adapters factuels de la v1 (récitation connue, rappel réel nul) et **accepte** les adapters procéduraux qui passent les checks déterministes.

**Claim R — Non-régression (héritée).** Le routeur maintient les scores généraux avec des adapters procéduraux chargés ; un adapter procédural always-on reproduit la dégradation (réplication v1 en régime procédural).

**Hors périmètre (délégué, pas abandonné) :** le lane factuel reste RAG par défaut. Ouvrir une issue GitHub `D2L as optional factual write backend` documentant le mécanisme (hypernetwork Sakana, arXiv 2602.15902) et les conditions d'intégration future. Aucune implémentation D2L dans cette version.

---

## 2. Matériel expérimental

### 2.1 Modèles

| ID | Modèle | Rôle |
|---|---|---|
| M1 | `Qwen/Qwen2.5-7B-Instruct` (8-bit, même révision que v1) | Primaire — obligatoire partout |
| M3 | `meta-llama/Llama-3.1-8B-Instruct` | Deuxième famille — Lots 4 et 5 uniquement (si budget) |

Quantization, template de chat, décodage : identiques au CDC v1 (§2.2, §3.2). Le harness v1 (`eval/`) est réutilisé tel quel pour le Lot 5.

### 2.2 Corpus : repos à conventions fortes

Sélectionner **5 repos publics** (3 minimum si pénurie) selon ces critères, tous vérifiables par script :
- Guide de style explicite (CONTRIBUTING.md, STYLE.md, config lint stricte, ou conventions documentées dans le README).
- Conventions **idiosyncratiques** : nommage inhabituel, structure de modules imposée, patterns d'erreur maison — pas juste "PEP8", que le modèle connaît déjà.
- Actifs (commit < 6 mois), ≥ 50 fichiers de code, licence permissive.
- Langages : au moins 2 langages différents sur l'ensemble des 5 repos.
- **Interdits :** tout repo de Romain (dont LLML lui-même), tout repo cité dans les données d'entraînement de la v1.

**Contrôle de contamination (obligatoire, symétrique du "C0-doit-échouer" v1) :** mesurer l'adhérence du modèle nu (C0) sur chaque repo AVANT tout entraînement. Si C0 ≥ 85% d'adhérence, le repo est écarté (rien à apprendre — le modèle connaît déjà ces conventions, probablement via pretraining) et remplacé. Liste des repos écartés publiée.

### 2.3 Checks d'adhérence déterministes

Pour chaque repo, compiler le guide de style en **≥ 15 règles vérifiables mécaniquement**, réparties en trois familles :
1. **Lint natif** : la config lint du repo lui-même, exécutée telle quelle.
2. **Règles AST** : nommage (casse, préfixes), structure (ordre des imports, taille max de fonction, présence de docstrings au format du repo), patterns imposés (gestion d'erreur, logging).
3. **Règles structurelles** : placement des fichiers, conventions de tests, format des messages/commentaires.

Les règles sont écrites en scripts (`eval/v2/checks/<repo>/`), **gelées et commitées avant tout entraînement** (même logique que le pré-enregistrement). Score d'adhérence d'une génération = % de règles applicables satisfaites.

### 2.4 Tâches held-out

Split par seed=42 : 80% des fichiers de chaque repo = corpus d'entraînement `/sleep`, 20% = held-out, jamais vus par l'adapter.
Une tâche = **réimplémenter une fonction masquée** d'un fichier held-out : le modèle reçoit le squelette du fichier, la signature et la docstring, et génère le corps. **30 tâches par repo**, tirées par seed, gelées dans le pré-enregistrement.
Métriques par tâche : score d'adhérence (checks §2.3), correction fonctionnelle quand le repo a des tests exécutables (pass/fail, sinon N/A — rapporter la couverture), tokens de prompt, latence.

### 2.5 Configurations

| Config | Description |
|---|---|
| C0 | Modèle nu, aucune information de conventions |
| C_ctx | Fichier de conventions complet dans le contexte (baseline à battre — coût en tokens comptabilisé) |
| C_lora | Adapter procédural du repo, routé, zéro conventions en contexte |
| C_both | Adapter + fichier en contexte (test de complémentarité) |
| C_wrong | Adapter du repo A sur les tâches du repo B (interférence croisée) |

---

## 3. Règles méthodologiques

Héritées du CDC v1 §3 en intégralité (pré-enregistrement par issue GitHub, décodage déterministe, versions pinnées, ICs bootstrap 95% + N affichés, publication totale, pas de sélection post-hoc, coûts comptabilisés). Ajouts v2 :

1. **Gel avant entraînement** : checks §2.3, tâches §2.4, seuils de la gate §5 — tout est commité et référencé dans l'issue de pré-enregistrement AVANT le premier run `/sleep`.
2. **Génération des paraphrases de la gate par famille étrangère** : tout jeu d'éval de la gate v2 est généré par M3 (famille Llama) quand l'adapter cible tourne sur M1, et inversement. Jamais la même famille des deux côtés.
3. **Hypothèses pré-enregistrées** :
   - H1 (Claim P) : adhérence C_lora ≥ C_ctx − 5 pts, avec ~0 token de conventions par requête.
   - H2 (Claim R) : C_lora routé ≈ C0 sur les benchmarks généraux (ICs chevauchants) ; C_lora always-on < C0 significativement.
   - H3 (Claim G) : la gate v2 rejette ≥ 90% des adapters factuels v1 et accepte ≥ 90% des adapters procéduraux validés par checks.
   - H4 (bonus, non bloquante) : C_both > C_ctx (l'adapter ajoute par-dessus le contexte).
   - H5 (contrôle) : C_wrong ≈ C0 (l'adapter encode bien le repo, pas du "code générique amélioré" — si C_wrong ≈ C_lora, l'adapter n'a rien appris de spécifique).

---

## 4. Lots d'exécution

### Lot 0 — Setup et lecture (2h GPU)
- Provisionner (même stack que v1), geler le commit LLML v2, `requirements.lock`.
- Lire REPORT.md v1 + PR #2. Vérifier la présence des adapters factuels v1 (jeu de test du Lot 2). S'ils manquent : les ré-entraîner à l'identique depuis les configs loggées v1 (les hyperparamètres sont dans la PR).
- Smoke test : chargement M1 + swap adapter + un run de 3 checks AST sur un repo témoin.
- ✅ Acceptation : environnement vert, artefacts v1 localisés ou régénérés, AGENTS.md initialisé.

### Lot 1 — Corpus et checks (4h GPU, surtout CPU)
- Sélection des 5 repos (§2.2) : proposer 8 candidats avec justification, **STOP checkpoint humain** — Romain en valide 5.
- Écrire les checks (§2.3), les geler. Générer les 150 tâches (§2.4), les geler.
- Contrôle de contamination : C0 sur les 150 tâches. Écarter/remplacer les repos ≥ 85%.
- Rédiger l'issue de pré-enregistrement (hypothèses H1–H5, seuils, repos, seeds). **STOP checkpoint humain** — Romain publie. Aucun run officiel avant.
- ✅ Acceptation : 5 repos validés et non contaminés, ≥ 75 règles gelées, 150 tâches gelées, issue publiée.

### Lot 2 — Gate v2 (5h GPU)
Construction :
- Jeu d'éval **paraphrase-disjoint** : questions/tâches générées par la famille étrangère (§3.2), avec contrainte de distance aux données d'entraînement.
- **Détecteur de récitation** : score composite = max overlap n-gram (n=8) entre la sortie et le corpus d'entraînement de l'adapter + similarité d'embedding (modèle d'embedding local, seuils pré-enregistrés). Au-dessus du seuil → la réponse est classée récitation, pas rappel.
- Pour les adapters procéduraux : la gate = checks déterministes §2.3 sur tâches held-out + non-régression rapide (sous-ensemble de 20 items HumanEval+).
Validation (le test décisif) :
- Rejouer les **adapters factuels v1** dans la gate v2 → attendu : rejet (H3). C'est un test à vérité terrain connue : on SAIT qu'ils récitent.
- Passer un adapter procédural du Lot 3 (anticipé sur 1 repo) → attendu : acceptation.
- ✅ Acceptation : matrice de confusion de la gate publiée, H3 tranchée. **Kill : si la gate v2 ne rejette pas les adapters v1, STOP** — le design de la gate est faux, tout /sleep autonome ultérieur serait aveugle. Rapport et arbitrage humain.

### Lot 3 — Entraînement des adapters procéduraux (8h GPU)
- `/sleep` retargeté : corpus = fichiers d'entraînement du repo + guide de style, objectif = complétion in-style (pas de QA factuelles). Hyperparamètres loggés, un adapter par repo, M1 (+ M3 si budget).
- Chaque adapter passe la gate v2 avant d'être considéré comme "commité".
- ✅ Acceptation : 5 adapters M1 commités par la gate, métadonnées complètes (données, config, scores gate).

### Lot 4 — Matrice principale Claim P (10h GPU)
- 150 tâches × configs C0/C_ctx/C_lora/C_both/C_wrong sur M1 (M3 : C0/C_ctx/C_lora si budget).
- Métriques : adhérence, correction fonctionnelle, tokens, latence. ICs bootstrap par repo et agrégés.
- ✅ Acceptation : H1, H4, H5 tranchées avec ICs. **Kill : si C_lora < C_ctx − 15 pts agrégé, le claim procédural est réfuté** → le résultat négatif devient le sujet du rapport (bis repetita, et c'est ok).

### Lot 5 — Non-régression Claim R (6h GPU)
- Harness v1 réutilisé : HumanEval+ complet + GSM8K sur M1, configs C0 / routé (adapters procéduraux chargés) / always-on (adapter du repo le plus éloigné du code de test).
- 5 cycles `/sleep` procéduraux successifs (un par repo), re-mesure après cycles 1, 3, 5 (courbe d'oubli, protocole v1 §4.3 allégé).
- ✅ Acceptation : H2 tranchée, courbe d'oubli publiée.

### Lot 6 — Stats, rapport, README v2 (CPU)
- `results/v2/REPORT.md` : hypothèses une par une, méthodo, limites, coût total.
- **Réécriture du README** : la nouvelle thèse (factuel→RAG mesuré, procédural→poids mesuré, routeur protecteur, gate qui distingue apprendre/réciter), les résultats v1 intégrés honnêtement (y compris la réfutation factuelle, en lien vers la PR #2), suppression de tout chiffre v1 invalidé (92→98 etc.), badges harness standard.
- `eval/v2/reproduce.sh` un-clic.
- ✅ Acceptation : REPORT relu, diff README proposé, chaque claim du README adossé à un tableau de résultats.

### Lot 7 — Intégration et publication (1h)
- PR unique v2. Issue D2L ouverte (§1). Mise à jour de l'issue de pré-enregistrement avec les verdicts.
- Préparer (sans publier) : draft de post r/LocalLLaMA + Show HN racontant l'arc complet — "j'ai construit une mémoire factuelle dans les poids, mon éval l'a réfutée, voici ce qui marche à la place et comment ma gate sait maintenant faire la différence entre apprendre et réciter". Publication = décision de Romain.
- ✅ Acceptation : PR ouverte, drafts livrés dans `docs/launch/`.

---

## 5. Kill criteria récapitulatifs

- Lot 1 : impossible de trouver 5 repos non contaminés → réduire à 3 et le documenter ; en dessous de 3, arbitrage humain.
- Lot 2 : gate v2 incapable de rejeter les adapters v1 → STOP global.
- Lot 4 : C_lora < C_ctx − 15 pts → claim P réfuté, pivot du rapport en résultat négatif.
- Lot 5 : régression significative en mode routé → claim central du système faux, STOP marketing, rapport.
- Budget : 40h GPU plafond.

## 6. Risques

| Risque | Mitigation |
|---|---|
| Le modèle connaît déjà les conventions (pretraining) | Contrôle C0 ≥ 85% → repo écarté (§2.2) |
| Checks trop faciles / lint ≈ formatage trivial | 3 familles de règles dont AST et structurelles ; gel avant entraînement |
| L'adapter apprend "du meilleur code" et pas "ce repo" | Config C_wrong (H5) |
| Adhérence ≠ code correct | Correction fonctionnelle mesurée quand tests exécutables ; couverture rapportée |
| Gate v2 sur-calibrée sur les artefacts v1 | Seuils pré-enregistrés avant le test de rejet ; matrice de confusion complète publiée |
| Petit N (5 repos, 150 tâches) | ICs partout, résultats présentés par repo ET agrégés, prétention limitée à "démonstration de mécanisme" |
| Spot préempté | Checkpoint par lot, CSV flushés après chaque run, sync repo |

## 7. Livrables

1. `eval/v2/` — checks gelés, tâches, scripts, `reproduce.sh`, `requirements.lock`.
2. `results/v2/` — raw CSV, tables, REPORT.md, matrice de confusion de la gate.
3. Gate v2 intégrée au code LLML (remplace la gate v1 dans `/sleep`).
4. Issue de pré-enregistrement v2 + issue D2L.
5. README v2 (diff proposé en PR).
6. `docs/launch/` — drafts de posts, non publiés.
