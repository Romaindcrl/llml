# Cahier des charges — Validation publique de LLML

**Projet :** LLML (https://github.com/Romaindcrl/llml)
**Objectif :** valider les claims du repo sur des benchmarks publics standards, avec une méthodologie irréprochable et intégralement reproductible.
**Exécutant :** agent Claude Code sur machine GPU distante (RunPod / Vast.ai).
**Version :** 1.0 — juillet 2026

---

## 0. Instructions générales pour l'agent

- Lire ce document EN ENTIER avant de commencer. Le relire au début de chaque lot.
- Créer un fichier `AGENTS.md` à la racine du workspace d'éval résumant : les règles méthodologiques (§3), l'état d'avancement des lots, et les décisions prises. Le tenir à jour après chaque lot.
- Les lots sont **strictement séquentiels**. Ne pas commencer un lot tant que les critères d'acceptation du lot précédent ne sont pas remplis.
- **Jamais de harness maison.** Toute évaluation passe par `lm-eval-harness` ou `EvalPlus`, versions pinnées. Si un benchmark n'est pas couvert par ces outils, utiliser le script officiel du benchmark, jamais une réimplémentation.
- Tout résultat est écrit en double : CSV brut dans `results/raw/` + tableau markdown dans `results/tables/`. Aucun chiffre ne vit uniquement dans un log.
- En cas d'ambiguïté sur un choix méthodologique : choisir l'option la plus défavorable à LLML et documenter la décision dans `AGENTS.md`. On préfère un résultat plus faible et inattaquable à un résultat flatteur et contestable.
- Budget temps GPU indicatif par lot indiqué à titre de garde-fou. Si un lot dépasse 2x son budget, s'arrêter et produire un rapport d'étape au lieu de continuer.

---

## 1. Les deux claims à valider

**Claim A — La mémoire-poids fonctionne.** Un document internalisé dans un adapter LoRA (via le pipeline `/sleep` de LLML) est restituable avec une fidélité comparable au document placé en contexte, pour un coût d'inférence inférieur (tokens, VRAM, latence).

**Claim B — Le système ne dégrade rien.** (B1) Le routeur de domaine préserve les capacités générales du modèle de base : les scores sur les benchmarks standards sont statistiquement équivalents avec et sans LLML actif. (B2) Un adapter forcé en mode always-on dégrade ces mêmes scores (réplication du résultat 8% du bench interne #16). (B3) Des cycles répétés de `/sleep` ne causent pas d'oubli catastrophique mesurable.

**Claim C (secondaire) — La boucle verify apporte un gain réel.** Le delta HumanEval du repo (92→98) doit être re-mesuré sur harness standard (EvalPlus) avant d'être cité.

Chaque run appartient à exactement un claim. Aucun benchmark "au cas où".

---

## 2. Matrice expérimentale

### 2.1 Modèles

| ID | Modèle (checkpoint HF exact) | Rôle | Notes |
|---|---|---|---|
| M1 | `Qwen/Qwen2.5-7B-Instruct` | Primaire (base actuelle de LLML) | Ne pas changer de version en cours d'éval |
| M2 | `Qwen/Qwen2.5-14B-Instruct` | Comparaison intra-famille pour le claim "7B ≈ 14B" | Même tokenizer/pretraining que M1 : seule comparaison de taille valide |
| M3 | `meta-llama/Llama-3.1-8B-Instruct` | Deuxième famille | Gated sur HF : token d'accès requis (fourni par Romain via variable d'env `HF_TOKEN`, jamais commité) |
| M4 | `google/gemma-2-9b-it` | Troisième famille / stress test | Architecture la plus différente ; sensible au fine-tuning → test le plus dur pour B3 |

**Exclusions volontaires (ne pas ajouter) :** modèles de raisonnement (R1-distill, Qwen3 thinking), MoE, variantes -Coder. Documenté au §7 du rapport final.

**Enregistrer pour chaque modèle :** hash de révision HF exact, config de quantization, template de chat utilisé.

### 2.2 Quantization

Politique unique : **8-bit (bitsandbytes) partout**, identique au setup LLML actuel. Un run de contrôle en bf16 sur M1 uniquement (Lot 1) pour quantifier l'écart quantization. Si l'écart bf16 vs 8-bit dépasse 2 points sur un benchmark, le signaler explicitement dans le rapport.

### 2.3 Configurations par run

| Config | Description |
|---|---|
| C0 | Modèle de base seul (baseline) |
| C1 | LLML complet : routeur actif, adapters chargés à la demande |
| C2 | Adapter forcé always-on (routeur désactivé) — réplication du mode de défaillance |
| C3 | RAG-only (retrieval LLML sans adapters) — pour Claim A uniquement |
| C4 | Contexte plein : document entier dans le prompt — baseline honnête de Claim A |

### 2.4 Cibles LoRA par architecture

⚠️ Les noms de modules diffèrent. Vérifier dans le code LLML que les `target_modules` ne sont pas hardcodés Qwen. Référence :
- Qwen2.5 / Llama 3.1 : `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Gemma 2 : mêmes noms de projections mais embeddings liés (tied) et logit soft-capping — vérifier la compatibilité PEFT avant tout run long, sur un train de 10 steps.

---

## 3. Règles méthodologiques (non négociables)

1. **Pré-enregistrement.** Avant tout run d'évaluation (fin du Lot 0), ouvrir une issue GitHub sur le repo LLML intitulée `Pre-registered evaluation plan — public benchmarks` contenant : la matrice §2, la liste exacte des benchmarks §4, les hypothèses chiffrées attendues, les critères de succès/échec. L'issue est créée par Romain (l'agent prépare le texte). Aucun run officiel avant que l'issue existe. Tout écart ultérieur au plan est documenté dans l'issue, jamais silencieux.
2. **Décodage déterministe.** Greedy (temperature=0) partout où le harness le permet. Si sampling requis : temperature et top_p fixés, seed=42, et 3 seeds (42/1337/2026) pour les benchmarks à variance (rapporter moyenne ± écart-type).
3. **Versions pinnées.** `requirements.lock` généré au Lot 0 (pip freeze). Versions exactes de lm-eval-harness, EvalPlus, transformers, peft, bitsandbytes, unsloth, torch, CUDA. Le commit hash de LLML utilisé est gelé au début du Lot 0 et ne change plus.
4. **Ns et intervalles de confiance.** Chaque tableau de résultats affiche N (nombre d'items) et un IC 95% bootstrap (10 000 resamples) pour toute comparaison entre configs. Une différence dont les IC se chevauchent est rapportée comme "non significative", point.
5. **Corpus externe pour Claim A.** Interdiction d'utiliser la spec LLML ou tout document écrit par Romain comme corpus d'internalisation. Corpus imposés au §4.2.
6. **Pas de sélection post-hoc.** Tous les runs lancés sont rapportés, y compris les échecs et les résultats défavorables. Un run invalide techniquement (OOM, crash) est relancé à l'identique et l'incident loggé.
7. **Coût comptabilisé.** Pour Claim A, chaque config rapporte : tokens de prompt moyens, VRAM pic, latence médiane de génération (par requête, même machine, même batch size=1). Le claim d'efficacité est un claim de coût, il se mesure.
8. **Publication totale.** Tous les scripts, configs, seeds et logs partent dans le repo (`eval/` + `results/`). Inclure la publication des scripts des benchs internes #14–15 actuellement "withheld".

---

## 4. Benchmarks

### 4.1 Claim B — Non-régression (via lm-eval-harness + EvalPlus)

| Benchmark | Outil | N | Configs | Modèles |
|---|---|---|---|---|
| HumanEval+ | EvalPlus | 164 | C0, C1, C2 | M1–M4 |
| MBPP+ | EvalPlus | 378 | C0, C1, C2 | M1, M3 |
| GSM8K (8-shot, strict-match) | lm-eval-harness | 1319 | C0, C1, C2 | M1–M4 |
| IFEval | lm-eval-harness | 541 | C0, C1 | M1, M3, M4 |
| MMLU-Pro (subset 6 domaines, seed fixe) | lm-eval-harness | ~1700 | C0, C1 | M1, M3 |

Le subset MMLU-Pro : tirer 6 domaines au hasard avec seed=42 AVANT le pré-enregistrement, les figer dans l'issue.

**Hypothèses pré-enregistrées :** C1 ≈ C0 (écart < 1 pt, IC chevauchants) ; C2 < C0 de façon significative sur au moins HumanEval+.

### 4.2 Claim A — Mémoire (corpus externes imposés)

**Corpus d'internalisation** (aucun écrit par Romain) :
- QuALITY (documents longs + QA associées, split dev) — 15 documents tirés avec seed=42.
- 3 documentations techniques publiques versionnées (choisir des projets que M1 connaît mal : bibliothèques niches post-2024, versions précises). Générer les QA de test avec un modèle tiers ≠ famille Qwen (utiliser M3), 20 questions factuelles par doc, vérifiées à la main par Romain avant les runs (checkpoint humain obligatoire).

| Test | Protocole | Configs |
|---|---|---|
| NIAH interne | Needles insérées dans les corpus internalisés, retrieval closed-book | C1 vs C3 vs C4 |
| QA closed-book | Questions QuALITY + QA docs techniques, document RETIRÉ du contexte | C1 vs C3 vs C4 vs C0 (contrôle : C0 doit échouer, sinon le doc était dans le pretraining → l'exclure) |
| Multi-hop | Sous-ensemble QuALITY nécessitant 2+ passages | C1 vs C4 |
| Coût | Tokens / VRAM / latence sur les mêmes requêtes | C1 vs C4 |

**Hypothèses pré-enregistrées :** C1 > C0 largement ; C1 atteint ≥ 80% du score de C4 (contexte plein) ; C1 bat C4 sur les trois métriques de coût. C3 (RAG) attendu ≥ C1 sur factuel pur — si c'est le cas, le dire tel quel, c'est cohérent avec le README.

### 4.3 Claim B3 — Protocole d'oubli

1. Mesurer C0 sur GSM8K + HumanEval+ (déjà fait Lot 1).
2. Exécuter 10 cycles `/sleep` consécutifs sur 10 documents distincts du corpus §4.2 (M1 et M4).
3. Re-mesurer en C1 après cycles 1, 5, 10.
4. Courbe score = f(nb cycles). Hypothèse : plat (pente non significative).

### 4.4 Claim C — Boucle verify

HumanEval+ et MBPP+ sur M1 : C0 sans verify vs C0 + verify. Documenter explicitement dans le rapport que le repair s'appuie sur les exemples de docstring (régime "vérité terrain offerte") — formulation à reprendre dans le README.

---

## 5. Lots d'exécution

### Lot 0 — Environnement, smoke tests, pré-enregistrement (budget : 2h GPU)
- Provisionner : 1× RTX 4090 24GB (ou A100 40GB si dispo au même tarif spot). CUDA 12.x, Python 3.11+.
- Cloner LLML, geler le commit hash. Installer, générer `requirements.lock`.
- Smoke test : M1 en 8-bit, 5 items HumanEval+ via EvalPlus, 5 items GSM8K via lm-eval-harness. Vérifier que le pipeline LLML (routeur + swap adapter) tourne sur la machine.
- Test PEFT 10 steps sur M4 (Gemma) pour valider les target_modules.
- Rédiger le texte de l'issue de pré-enregistrement → **STOP : checkpoint humain.** Romain publie l'issue. Reprise uniquement après confirmation.
- ✅ Acceptation : smoke tests verts sur les 4 modèles (chargement + 5 items), lock file commité, issue publiée.

### Lot 1 — Baselines C0 (budget : 8h GPU)
- Tous les benchmarks §4.1 en C0 sur M1–M4 (selon matrice). + run bf16 de contrôle sur M1.
- **Gate de sanité :** comparer chaque score C0 aux chiffres publiés (model cards, papers EvalPlus, leaderboards). Écart > 3 pts = STOP, investiguer le harness/template de chat avant de continuer. C'est ici que le "92% HumanEval" interne se confirme ou s'explique.
- ✅ Acceptation : tableau baselines complet avec ICs, écarts aux chiffres publiés documentés et expliqués.

### Lot 2 — Claim C, boucle verify (budget : 3h GPU)
- §4.4. Produire le tableau delta verify avec IC.
- ✅ Acceptation : delta mesuré sur harness standard, formulation "docstring examples" rédigée pour le README.

### Lot 3 — Claim B1/B2, matrice routing (budget : 10h GPU)
- §4.1 en C1 et C2. Pour C2 : utiliser l'adapter du corpus §4.2 le plus éloigné du code (un doc QuALITY littéraire) — c'est le pire cas honnête.
- ✅ Acceptation : trilogie C0/C1/C2 complète par modèle, hypothèses pré-enregistrées confrontées.

### Lot 4 — Claim A, mémoire (budget : 12h GPU, inclut les trainings /sleep)
- Internalisation des corpus §4.2 via le pipeline `/sleep` standard de LLML (recette du bench interne #9, hyperparamètres loggés).
- **STOP checkpoint humain** avant les runs : Romain valide les 60 QA générées.
- Runs §4.2 complets, métriques de coût incluses.
- ✅ Acceptation : les 4 tests × configs, avec le contrôle C0-doit-échouer appliqué (docs contaminés exclus et listés).

### Lot 5 — Claim B3, oubli (budget : 8h GPU)
- §4.3 sur M1 et M4.
- ✅ Acceptation : courbes d'oubli avec ICs, incidents Gemma documentés le cas échéant.

### Lot 6 — Ablations (budget : 5h GPU)
- Sur M1 uniquement : LLML sans-verify, LLML sans-routeur (≠ C2 : ici adapters jamais chargés), rang LoRA ÷2, sur les benchmarks où C1 a le plus d'effet.
- ✅ Acceptation : tableau d'ablations, contribution de chaque composant isolée.

### Lot 7 — Statistiques et rapport (budget : CPU only)
- Consolider tous les CSV. ICs bootstrap partout. Tests de significativité (permutation test) pour chaque comparaison pré-enregistrée.
- Rédiger `results/REPORT.md` : méthodo, tableaux, hypothèses confirmées/infirmées une par une, limites, coût total (heures GPU, €).
- Générer `eval/reproduce.sh` : un script unique qui rejoue toute la matrice depuis une machine vierge.
- ✅ Acceptation : REPORT.md relu, chaque claim du README actuel soit confirmé, soit reformulé, soit retiré — proposition de diff du README incluse.

### Lot 8 — Intégration repo (budget : 1h)
- PR unique : `eval/`, `results/`, scripts #14–15 publiés, README mis à jour, lien vers l'issue de pré-enregistrement, badge "evaluated with lm-eval-harness vX.Y / EvalPlus vZ".
- ✅ Acceptation : PR ouverte, CI verte si existante. Romain merge.

---

## 6. Kill criteria (arrêt anticipé du plan)

- Lot 1 : impossible de reproduire les baselines publiées à ±3 pts après investigation → le harness ou l'environnement est cassé, tout résultat ultérieur serait bruit. Arrêt + rapport.
- Lot 3 : si C1 < C0 de manière significative sur ≥ 2 benchmarks → le claim central de non-régression est faux. Arrêt du plan marketing, le résultat négatif devient LE sujet du rapport (et c'est publiable quand même).
- Lot 4 : si C1 < 50% de C4 sur la QA closed-book → la mémoire-poids ne tient pas sur corpus externe. Idem : résultat négatif documenté, pas caché.
- Budget global : plafond 60h GPU. Au-delà, rapport d'étape et arbitrage humain.

## 7. Risques identifiés

| Risque | Mitigation |
|---|---|
| Contamination pretraining (docs "internalisés" déjà connus du modèle) | Contrôle C0-doit-échouer systématique (§4.2) |
| Template de chat incorrect → baselines fausses | Gate Lot 1 vs chiffres publiés |
| target_modules hardcodés Qwen dans LLML | Test 10 steps Gemma au Lot 0 |
| Llama 3.1 gated | HF_TOKEN en variable d'env, jamais dans le repo |
| Variance des petits N (QA docs techniques) | 3 seeds + ICs, N affiché partout |
| Spot instance préemptée | Checkpointing par benchmark, résultats flushés en CSV après chaque run, sync vers le repo à chaque fin de lot |
| Coût dérivant | Budgets par lot + plafond global §6 |

## 8. Livrables finaux

1. `eval/` — tous les scripts, configs YAML, `reproduce.sh`, `requirements.lock`.
2. `results/raw/` (CSV) + `results/tables/` (markdown) + `results/REPORT.md`.
3. Issue GitHub de pré-enregistrement, mise à jour avec le verdict de chaque hypothèse.
4. PR de mise à jour du README (claims reformulés selon les résultats, scripts #14–15 publiés).
5. `AGENTS.md` à jour — journal de décisions complet.
