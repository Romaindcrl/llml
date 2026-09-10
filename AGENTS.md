# Instructions actuelles — maintenance de LLML

Les campagnes de juillet ci-dessous sont des archives. Leur statut actuel et la
portée des résultats sont décrits dans `eval/results/CAMPAIGN_STATUS.md`.

- Préserver les résultats bruts et les textes de préenregistrement ; distinguer
  mesures archivées, hypothèses et validations effectivement exécutées.
- Privilégier les vérifications locales sans modèle pour la maintenance.
- Les anciens budgets et plans GPU ne constituent pas une autorisation de
  provisionner, recharger, entraîner ou relancer une campagne.
- Le cadrage actuel autorise uniquement une qualification de corpus public :
  CodeReviewer d'abord, puis au plus deux dépôts publics si nécessaire ; plafonds
  90 minutes humaines, 60 minutes d'extraction, aucun entraînement ni dépense.
  Pas de design détaillé avant un corpus effectivement évaluable. Les commentaires
  correctifs et patches finaux des cas évalués restent hors de l'entrée du modèle.
- La demande courante du propriétaire prime sur les consignes historiques.

---

# AGENTS.md — Journal d'exécution de l'évaluation publique LLML

> Journal historique de l'évaluation. **v2 non poursuivie** : [`eval/v2/CDC_LLML_V2.md`](eval/v2/CDC_LLML_V2.md)
> (mémoire procédurale). La v1 ([`eval/CDC_EVAL_LLML.md`](eval/CDC_EVAL_LLML.md)) est
> close — verdicts dans [`eval/results/REPORT.md`](eval/results/REPORT.md) et la PR #2.
> Tenu à jour par l'agent après chaque lot. Relire le CDC en entier au début de chaque lot.

---

# ═══ v2 — Mémoire procédurale (CDC v2, juillet 2026) ═══

**Thèse v2** : factuel→RAG (mesuré gagnant v1), procédural→poids (à démontrer),
routeur protecteur (validé v1), gate v2 qui distingue apprendre/réciter.

## Règles v2 (CDC v2 §0 + §3, en sus des règles v1 ci-dessous)

- **Aucun LLM juge, nulle part** : adhérence = linters, règles AST, regex — déterministe.
- Gel avant entraînement : checks, tâches, seuils gate → commités + référencés dans
  l'issue de pré-enregistrement AVANT le premier `/sleep` v2.
- Paraphrases de la gate par **famille étrangère** (M3 génère pour un adapter M1, et vice-versa).
- Ambiguïté → option la plus défavorable à LLML, loggée ici.
- Budget plafond **40h GPU** ; dépassement 2x sur un lot → arrêt + rapport d'étape.

## État d'avancement des lots v2

| Lot | Contenu | Statut |
|---|---|---|
| 0 | Setup, lecture v1, artefacts v1, smoke | ✅ clos — SMOKE_V2_OK 184s (`results/v2/raw/smoke_v2.json`) ; déviation documentée : adapters v1 régénérés au Lot 2 (recette+corpus committés, budget GPU) |
| 1 | Corpus + checks + tâches + pré-enregistrement | ✅ clos — [issue #3 publiée](https://github.com/Romaindcrl/llml/issues/3) (délégation) ; 4 repos/120 tâches (TigerBeetle écarté, headroom 3,6), 106 règles calibrées 99,4%, C0 fenêtre-zéro archivé, sous-ensemble dur gelé |
| 2 | Gate v2 (kill : doit rejeter les adapters v1) | ⏸ |
| 3 | Entraînement adapters procéduraux | ⏸ |
| 4 | Matrice Claim P (C0/C_ctx/C_lora/C_both/C_wrong) | ⏸ |
| 5 | Non-régression Claim R + courbe d'oubli | ⏸ |
| 6 | Stats, REPORT v2, README v2 | ⏸ |
| 7 | PR v2 + issue D2L + drafts launch | ⏸ |

## Journal v2

- **2026-07-11 — Lot 1 CLOS, issue #3 publiée** (délégation explicite de Romain,
  checkpoint #2). Commit de gel 7d216a07. Corpus officiel 4 repos/120 tâches.
  Coût GPU Lot 0+1 : ~$2.75 (3 pods secure éphémères, community à sec, 2 resume
  refusés par hosts occupés → terminate+redeploy). Solde restant : $1.61 —
  **Lot 2 bloqué sur recharge (~$15-25) + HF_TOKEN (M3/Llama, paraphrases gate)**.
- **2026-07-11 — Lot 1, verdict contamination FINAL (critère headroom).** C0
  fenêtre-zéro : FreeRTOS 89,3% / curl 89,8% / tigerbeetle 96,4% / twisted 85,9%
  / zulip 78,3%. Le critère absolu ≥85% confondait règles d'absence et règles de
  conflit → critère amendé (headroom <10 pts = écarté), AVANT publication de
  l'issue. TigerBeetle écarté (3,6 pts). **Corpus officiel : 4 repos, 120
  tâches.** Sous-ensemble dur pré-enregistré = (règle×tâche) échouées par le
  C0_nowin archivé. Conflits dominants confirmés : if( x )/if(x), /* */ vs //,
  camelCase+quotes Twisted, ruff Q/G/N Zulip.
- **2026-07-11 — Lot 1, KILL CRITERION déclenché puis ARBITRAGE HUMAIN (fenêtre zéro).**
  Contrôle de contamination avec fenêtre de 60 lignes : 4/5 repos ≥85%
  (FreeRTOS 98,5%, TigerBeetle 97,4%, curl 93,5%, Twisted 87,4%, Zulip 82,5%).
  Diagnostic sur les sorties brutes : mimétisme in-context — le C0 réutilise
  les symboles locaux visibles dans la fenêtre (xStart, pxEnd…), style copié
  du fichier montré, pas (principalement) du pretraining. Décision de Romain :
  **tâches à FENÊTRE ZÉRO** (signature+docstring+chemin, sans voir le fichier)
  = le régime réel de la mémoire procédurale. Amendement intégré à l'issue de
  pré-enregistrement AVANT publication (aucun run officiel effectué). Le C0
  fenêtre-60 est archivé comme analyse exploratoire (results/v2/raw/gen_C0.jsonl).
- **2026-07-11 — Lot 0 CLOS.** Smoke OK en 184s sur pod 3jf5blbtd14mlv (secure
  $0.69/h, community à sec) : chargement M1 8-bit ✓, micro-LoRA 10 iters +
  swap load/unload ✓ (train_loss 3.654), moteur de checks AST sur témoin ✓.
  **Déviation loggée** : les adapters factuels v1 (détruits avec les pods) seront
  régénérés AU LOT 2 et non au Lot 0 — recette et corpus intégralement committés
  (run_memory.py::sleep_train + tech_docs.jsonl + frozen_corpus.jsonl), le
  ré-entraînement immédiat aurait consommé le solde restant ($2.81) requis pour
  le contrôle de contamination du Lot 1. Pod STOPPÉ (pas terminé : disque
  conservé ≈$0.20/j pour resume rapide — venv+cache modèle).
- **2026-07-11 — Lot 1, tâches gelées.** 150 tâches (30×5) + splits 80/20 seed 42
  committés. FreeRTOS : `portable/` réinclus (le cœur ne fait que 8 fichiers .c ;
  les ports suivent les mêmes conventions).
- **2026-07-11 — Lot 1, checkpoint humain #1 VALIDÉ.** Romain a choisi le combo
  recommandé parmi les 8 candidats vérifiés (`eval/v2/CANDIDATE_REPOS.md`) :
  **FreeRTOS-Kernel + curl + TigerBeetle + Twisted + Zulip** (C, Zig, Python,
  TypeScript — 4 langages). Réservistes si contamination C0≥85% : nginx, Godot,
  MicroPython, postgres, openssl. SHAs épinglés au premier clone (à consigner
  dans le pré-enregistrement).
- **2026-07-11 — Lot 0 ouvert.** CDC v2 reçu de Romain et commité (`eval/v2/CDC_LLML_V2.md`).
  État initial : solde RunPod **$3.67** (plafond CDC 40h GPU ≈ $14 community / $28 secure
  → recharge nécessaire avant Lots 2-5, signalé à Romain). **Artefacts v1 : les adapters
  factuels du Lot 4 v1 ont été détruits** avec les pods (terminaison demandée par Romain
  post-v1 pour stopper la facturation) → application du plan B prévu au CDC v2 Lot 0 :
  ré-entraînement à l'identique depuis `eval/lot4/run_memory.py::sleep_train` (recette
  complète loggée : gate_acq 0.45, lr 5e-5, r=16, α=20·r, 8 couches, iters=min(400,
  max(120, 25·n_facts)), paraphrases ×6) sur les mêmes corpus commités
  (`eval/lot4b/tech_docs.jsonl`, QuALITY re-téléchargeable). HF_TOKEN non disponible
  localement → à re-fournir par Romain pour M3 (Llama gated, requis Lot 2 §3.2).
  PR #2 v1 toujours ouverte (suivi passif conservé).

---

# ═══ v1 — Évaluation publique des claims (close) ═══

## Règles méthodologiques (résumé du CDC §3 — non négociables)

1. **Pré-enregistrement** : aucun run officiel avant que l'issue GitHub
   `Pre-registered evaluation plan — public benchmarks` soit publiée par Romain.
   Tout écart ultérieur est documenté dans l'issue.
2. **Décodage déterministe** : greedy (temperature=0) partout ; si sampling requis,
   seeds 42/1337/2026, moyenne ± écart-type.
3. **Versions pinnées** : `eval/requirements.lock` ; commit LLML gelé (ci-dessous).
4. **N + IC 95% bootstrap** (10 000 resamples) sur chaque comparaison ; IC qui se
   chevauchent = "non significatif", point.
5. **Corpus externes uniquement** pour le Claim A (jamais un texte de Romain).
6. **Pas de sélection post-hoc** : tous les runs lancés sont rapportés.
7. **Coût comptabilisé** (tokens, VRAM pic, latence médiane batch=1) pour le Claim A.
8. **Publication totale** : scripts, configs, seeds, logs dans `eval/` + `results/`.
9. En cas d'ambiguïté : choisir l'option **la plus défavorable à LLML** et documenter ici.

## État d'avancement des lots

| Lot | Contenu | Statut |
|---|---|---|
| 0 | Env GPU, port CUDA, smoke tests, pré-enregistrement | ✅ clos — [issue #1 publiée](https://github.com/Romaindcrl/llml/issues/1) (délégation explicite de Romain) ; M3/M4 + PEFT Gemma reportés au Lot 1 (HF_TOKEN), déviation déclarée dans l'issue §6.5 |
| 1 | Baselines C0 (M1–M4) + contrôle bf16 | 🟡 en cours — Phase A (M1 8-bit, 5 benchmarks) lancée ; M3/M4 dès HF_TOKEN |
| 2 | Claim C — boucle verify | ⏸ |
| 3 | Claim B1/B2 — matrice routing | ⏸ |
| 4 | Claim A — mémoire (corpus externes) | ⏸ (2e checkpoint humain : validation des 60 QA) |
| 5 | Claim B3 — oubli | ⏸ |
| 6 | Ablations | ⏸ |
| 6bis | Boucle d'apprentissage autonome (réplication publique du bench #12) — [amendement #1](https://github.com/Romaindcrl/llml/issues/1#issuecomment-4883894952) | ⏸ après Lot 6 |
| 7 | Stats + REPORT.md | ⏸ |
| 8 | PR d'intégration | ⏸ |

## Environnement gelé (Lot 0)

- **Commit LLML gelé** : `131527d1b52d0e8c2054e09e4e0f60b34a7b2815` (branche
  `claude/runpod-deployment-setup-4rvggq`) — le commit d'éval EST le commit qui
  contient le port CUDA (`m0/hf.py`, `m0/d2l_hf.py`) ; les chiffres publiés
  référencent ce hash.
- **Machine** : RunPod Community, 1× RTX 4090 24 GB, 0,34 $/h, image
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (torch 2.4.1+cu124).
- **Stack pinnée** : transformers 4.46.3 · peft 0.13.2 · bitsandbytes 0.44.1 ·
  accelerate 1.1.1 · datasets 3.1.0 · lm_eval 0.4.5 · evalplus 0.3.1
  (lock complet : `eval/requirements.lock`, généré sur le pod).
- **Caches** : `HF_HOME=/workspace/hf` (volume persistant du pod).

## Journal de décisions

- **2026-07-04 — Portage CUDA (préalable au Lot 0).** Le repo est MLX/Apple Silicon ;
  la machine d'éval est NVIDIA. Ajout de `m0/hf.py` (backend `hf` : transformers +
  bitsandbytes 8-bit + hot-swap peft, contrat identique à `MLXClient`) et
  `m0/d2l_hf.py` (entraînement `/sleep` via peft, mêmes données/retour que
  `d2l.train_lora`). `scripts/serve.py` : checks `isinstance(MLXClient)` remplacés
  par des checks de capacité (`hasattr set_adapter`) — l'ancien code dégradait
  silencieusement sur backend non-MLX.
- **2026-07-04 — target_modules pinnés** : `q,k,v,o,gate,up,down_proj` sur les 8
  dernières couches (les adapters MLX du repo incluaient les projections MLP —
  formes 18944×3584 citées dans `m0/lora_merge.py` ; cohérent avec le CDC §2.4 ;
  rien d'hardcodé Qwen — mêmes noms sur Llama 3.1 et Gemma 2, à valider par le
  test 10 steps M4). Équivalence d'échelle MLX→PEFT : `lora_alpha = 20 × rank`
  (MLX `scale=20`) — les défauts PEFT donneraient un delta ~10-20× plus faible.
- **2026-07-04 — unsloth non utilisé** : le CDC liste unsloth parmi les versions à
  pinner ; le port utilise peft vanilla (moins de dépendances, déterminisme) ;
  unsloth n'apparaît donc pas dans le lock — décision documentée ici.
- **2026-07-04 — Versions de stack** : pinnées sur un couple connu-compatible avec
  l'image CUDA 12.4/torch 2.4.1 du pod plutôt que "latest" (reproductibilité >
  nouveauté). La gate du Lot 1 (écart aux chiffres publiés > 3 pts = STOP)
  validera le harness.
- **2026-07-04 — Infra** : SSH sortant bloqué depuis l'environnement de contrôle →
  pod piloté par un serveur d'exécution HTTP (token) derrière le proxy HTTPS
  RunPod (`eval/runpod/agent_server.py` + `remote.py`). Community cloud on-demand
  (0,34 $/h) sans network-volume : les modèles se re-téléchargent en ~15 min si le
  pod est détruit (arbitrage coût). Checkpointing CSV après chaque run (CDC §7).
- **2026-07-04 — Tirage MMLU-Pro (seed=42, AVANT pré-enregistrement)** :
  `biology, business, computer science, economics, math, other`
  (`eval/scripts/draw_mmlu_domains.py`).
- **2026-07-04 — Constats d'honnêteté issus de la lecture du repo** (à re-vérifier
  publiquement, intégrés à l'issue de pré-enregistrement) :
  - le 92→98 HumanEval interne est mesuré sur les **40 premiers problèmes** (pas
    164) : granularité ±2,5 pts ; le run public utilise EvalPlus sur 164 + IC ;
  - le harness HumanEval interne (PRELUDE d'imports, détection `PASS`) est plus
    laxiste que l'officiel — attendre des scores publics ≤ internes ;
  - la boucle repair : sous greedy, le 2e essai renvoie un prompt identique →
    effectivement **1 réparation** ; répliqué tel quel ;
  - **B3 (oubli multi-cycles) n'a aucun benchmark interne** : pré-enregistré comme
    hypothèse OUVERTE, pas comme réplication ;
  - benchs #14–15 : scripts retenus car spec privée → l'éval publique reconstruit
    des tenants synthétiques publics (les chiffres ne sont pas comparables 1:1).
- **2026-07-04 — Choix de templates lm-eval gelés (Lot 1)** : `--apply_chat_template
  --fewshot_as_multiturn` partout (modèles instruct, méthodo type Open LLM
  Leaderboard v2) ; GSM8K 8-shot strict-match ; IFEval 0-shot ; MMLU-Pro subtasks
  `mmlu_pro_{biology,business,computer_science,economics,math,other}`. EvalPlus :
  génération via wrapper mince (evalplus 0.3.1 ne charge pas en 8-bit), scoring
  100% officiel ; validation croisée wrapper-vs-natif en bf16 prévue avant de
  publier tout chiffre HumanEval+.
- **2026-07-04 — Sémantique de swap** : le hot-swap peft garde les adapters
  résidents en VRAM (vs reload complet MLX côté `serve.py`) ; les latences de swap
  CUDA ne sont PAS comparables aux ~2 ms Apple-unified-memory — mesurées et
  rapportées séparément, jamais fusionnées avec les claims MLX.

- **2026-07-04 — GATE Lot 1 déclenchée et résolue (template GSM8K).** Premier
  résultat officiel (M1 bf16, gsm8k batché 29 min) : strict-match 27,4 % vs ~80
  attendu → STOP + investigation (CDC §5, risque « template incorrect »).
  Cause : avec chat template, le modèle répond dans son format RLHF, pas en
  `#### N`. Décision gelée : GSM8K SANS chat template (8-shot continuation,
  référence communautaire lm-eval) ; IFEval/MMLU-Pro AVEC (benchs instruct).
  Le run chat-template est conservé en raw comme run d'investigation
  (flexible-extract 73,7 % cohérent). Flotte relancée sur cette config.
- **2026-07-05 — GATE Lot 1 #2 (MMLU-Pro tronqué).** M1 bf16 : 20-30 %/domaine vs
  ~56 attendu. Diagnostic via log_samples : CoT tronqué avant « the answer is
  (X) » → extraction `[invalid]`. Fix : `--gen_kwargs max_gen_toks=1024` ;
  validation 20 items gatée avant le redo complet ; run tronqué conservé en
  `mmlu_pro_truncated_investigation`. Aussi : IFEval nécessitait `langdetect`/
  `immutabledict` (extra lm-eval non installé) — pins corrigés, redo chaîné.
- **2026-07-05 — Incident process (leçon)** : pod B terminé avant rapatriement
  des artefacts bruts MBPP bf16 (log evalplus + samples) — scores connus
  (80,7/69,8, CSV du pod cité en session) mais bruts perdus. Correctif : re-run
  complet de la cellule chaîné sur le pod A (aucun chiffre publié sans artefact
  brut). Règle durcie : checklist de rapatriement PAR BENCHMARK avant tout
  podTerminate.
- **2026-07-04 — Amendement #1 (Lot 6bis)** : sur demande de Romain, ajout de la
  réplication publique de la boucle d'apprentissage autonome (bench #12) —
  corpus web GELÉ (snapshot committé, injecté via research_fn, zéro web live),
  contrôle C0-doit-échouer, 5 cycles, hypothèses H-D1..D4 pré-enregistrées dans
  le commentaire d'amendement AVANT tout run. ~3-4 h GPU, après le Lot 6.
- **2026-07-04 — Flotte Lot 1** : 4× 4090 community identiques en parallèle
  (A=M1-8bit+M2, B=M1-bf16, C=M3, D=M4) ; même image/stack, un modèle par pod,
  machineId loggé ; « même machine » ne s'applique qu'aux coûts du Claim A (Lot 4).
  Déclaré dans le commentaire d'amendement de l'issue #1.

## Plan opérationnel Lot 1 (préparé, exécution après publication de l'issue)

- **C0 (baselines)** : lm-eval `--model hf` (load_in_8bit) et EvalPlus en direct —
  chemin standard, comparable aux model cards / leaderboard EvalPlus.
- **C1/C2 (LLML actif)** : les harness parlent à un serveur OpenAI-compatible mince
  (HFClient + routeur + adapters) via `local-chat-completions` / `--backend openai`.
  Jamais de scoring maison : seul le *serving* change.
- **Check d'équivalence obligatoire** avant tout chiffre C1/C2 : C0-via-serveur vs
  C0-direct sur M1 (HumanEval+ et GSM8K) doivent coïncider (sinon le chemin de
  serving confond la comparaison C1≈C0 — investiguer avant de continuer).
- Checkpointing : 1 CSV par (modèle, benchmark, config) écrit dès la fin du run,
  sync git à chaque fin de lot (pod community préemptible).
- Vitesse : générations 8-bit bnb lentes (~20-40 tok/s) ; si le Lot 1 dépasse 2×
  son budget (16 h GPU), STOP et rapport d'étape (CDC §0) — options : batch via
  lm-eval hf, ou GPU plus gros ponctuellement (arbitrage coût documenté).
- **Mesure smoke (Lot 0)** : GSM8K 8-shot en 8-bit batch=1 ≈ 20 s/item sur M1 →
  1319 items ≈ 7 h par (modèle, config) : impraticable à batch 1. Décision Lot 1 :
  `--batch_size auto` pour les C0 lm-eval (le batching ne change pas le greedy
  par item) ; pour C1/C2 via serveur, paralléliser côté harness si nécessaire.
  Le budget Lot 1 sera re-estimé après le premier run complet C0/M1 et journalisé.

## Suivi budget

| Poste | Montant |
|---|---|
| Crédits RunPod initiaux | 10,00 $ |
| Plafond visé (tout compris) | ≤ 50 $ |
| Budget CDC | 60 h GPU max (§6) ; à 0,34 $/h ≈ 20 $ |
| Dépensé (au dernier pointage) | voir section mise à jour à chaque fin de lot |

## Blocages courants

- `HF_TOKEN` requis pour M3 (`meta-llama/Llama-3.1-8B-Instruct`) et M4
  (`google/gemma-2-9b-it`) — licences à accepter sur le compte HF de Romain,
  token à fournir en variable d'env (jamais commité, CDC §7).
