# [DRAFT — à publier par Romain] Pre-registered evaluation plan v2 — procedural memory

> Brouillon de l'issue GitHub de pré-enregistrement (CDC v2 §3, checkpoint
> humain #2 du Lot 1). **Aucun run officiel des Lots 2-5 avant publication.**
> Le commit de gel référencé ci-dessous fixe checks, tâches, seuils et prompts.

## Thèse

La mémoire n'est pas une seule chose. **Factuel → RAG** (mesuré gagnant en v1,
PR #2 : rappel poids 0/11 QuALITY, 2/17 tech-docs, 0/17 en boucle). **Procédural
(conventions, style, patterns d'un repo) → poids** (à démontrer ici). Le routeur
décide du lane et protège le modèle de base (validé v1 : 42/42, préservation
92% vs 75% always-on).

## Claims

- **Claim P** : un adapter LoRA `/sleep` entraîné sur les conventions d'un repo
  atteint une adhérence **non-inférieure** au fichier de conventions en contexte
  (marge 5 pts), pour ~0 token de conventions par requête, sans dégrader les
  capacités générales.
- **Claim G** : la gate v2 distingue apprentissage et récitation — rejette les
  adapters factuels v1 (récitation à vérité terrain connue), accepte les
  adapters procéduraux validés par checks.
- **Claim R** : non-régression (héritée v1) en mode routé ; dégradation
  reproduite en always-on.

## Matériel gelé (au commit de cette issue)

- **Repos (5, checkpoint humain #1 validé)** : FreeRTOS-Kernel @9db704cd,
  curl @5c5334f8, tigerbeetle @97c7a8ef, twisted @fe27ab1d, zulip @5784ebe8
  (`eval/v2/repos.lock.json`). Réservistes si contamination : nginx, Godot,
  MicroPython, postgres, openssl (dossier : `eval/v2/CANDIDATE_REPOS.md`).
- **Tâches** : 150 (30×5), split fichiers 80/20 seed 42, corps 5-60 lignes,
  ≤2 tâches/fichier (`eval/v2/tasks/`, `eval/v2/splits/`, gén. `make_tasks.py`).
- **Checks** : ≥15 règles déterministes/repo (`eval/v2/checks/<repo>/rules.py`),
  3 familles (lint natif : checksrc.pl curl, custom_check+ruff-config Zulip ;
  AST ; structurel). **Calibration publiée** : chaque règle conservée passe
  ≥80% des reconstructions originales (`eval/v2/checks/CALIBRATION.md`).
- **Conventions C_ctx** : fichiers de style verbatim des repos
  (`eval/v2/conventions/`, 1,4k-7,5k tokens/repo — coût comptabilisé).
- **Prompt unique** (toutes configs, seul le bloc conventions varie) :
  `eval/v2/run_matrix.py::PROMPT`. Décodage greedy, max 512 tokens.
- **Scoring** : adhérence = % de règles applicables satisfaites sur la fonction
  RECONSTRUITE (header verbatim + corps généré, dédenté). Nettoyage
  déterministe pré-scoring : retrait des clôtures markdown si présentes.

## Décisions de design pré-enregistrées (avant tout run officiel)

1. **Zulip = Python uniquement** (l'adhérence TS exigerait une stack node ;
   le lane TS est retiré des tâches ET des checks).
2. **Docstring optionnelle** dans l'extraction Zulip (leur style ne l'impose
   pas), obligatoire chez Twisted (le leur l'impose).
3. **Correction fonctionnelle** : proxy = validité syntaxique de la fonction
   reconstruite (parse AST / compilabilité de l'extrait), déterministe et
   uniforme. L'exécution des suites de tests des 5 repos sur 750 générations
   dépasse le budget ; limite assumée et rapportée.
4. **Le run C0 du contrôle de contamination** (150 tâches, greedy, harness
   figé) **est réutilisé comme C0 officiel du Lot 4** — mêmes conditions,
   générations archivées avant ce pré-enregistrement, scoring post-hoc
   déterministe.
5. En cas d'ambiguïté : option la plus défavorable à LLML, journalisée
   (`AGENTS.md`).

## Contrôle de contamination (symétrique du « C0-doit-échouer » v1)

C0 mesuré AVANT tout entraînement sur les 150 tâches. **Repo écarté si
adhérence C0 ≥ 85%** (le modèle connaît déjà ces conventions) → remplacé par
un réserviste (nouveau cycle checks+tâches+contamination). Liste des écartés
publiée ici.

## Hypothèses (seuils figés)

- **H1 (Claim P)** : adhérence C_lora ≥ C_ctx − 5 pts (agrégé, IC bootstrap
  95%, 10 000 resamples), avec ~0 token de conventions/requête.
- **H2 (Claim R)** : C_lora routé ≈ C0 sur HumanEval+ et GSM8K (ICs
  chevauchants) ; adapter always-on < C0 significativement.
- **H3 (Claim G)** : la gate v2 rejette ≥ 90% des adapters factuels v1 ET
  accepte ≥ 90% des adapters procéduraux validés par checks.
- **H4 (bonus, non bloquante)** : C_both > C_ctx.
- **H5 (contrôle de spécificité)** : C_wrong ≈ C0. Si C_wrong ≈ C_lora,
  l'adapter n'a rien appris de spécifique au repo → Claim P non soutenu.

## Gate v2 (seuils figés avant le test de rejet)

- Jeu d'éval **paraphrase-disjoint** généré par la famille étrangère (M3
  Llama-3.1-8B pour un adapter M1, et inversement). Jamais la même famille
  des deux côtés.
- **Détecteur de récitation** : max-overlap n-gram (n=8) entre sortie et corpus
  d'entraînement de l'adapter + similarité d'embedding locale. Seuils figés :
  récitation si overlap-8 ≥ 0,35 OU similarité ≥ 0,92 vs plus-proche-voisin
  du corpus d'entraînement.
- Adapters procéduraux : gate = checks déterministes sur held-out + non-régression
  rapide (20 items HumanEval+, pass@1 ≥ base − 10 pts).
- **Test décisif (vérité terrain)** : les adapters factuels v1 — ré-entraînés à
  l'identique depuis les recettes/corpus committés v1 (les originaux ont été
  détruits avec les pods, déviation journalisée) — doivent être REJETÉS.
  **Kill : si la gate v2 ne les rejette pas → STOP global.**

## Kill criteria

- Lot 1 : < 3 repos non contaminés → arbitrage humain.
- Lot 2 : gate v2 ne rejette pas les adapters v1 → **STOP global**.
- Lot 4 : C_lora < C_ctx − 15 pts agrégé → Claim P réfuté, rapport pivote en
  résultat négatif (publié tel quel).
- Lot 5 : régression significative en mode routé → STOP marketing, rapport.
- Budget : 40h GPU plafond ; dépassement 2× sur un lot → arrêt + rapport.

## Environnement

M1 = Qwen/Qwen2.5-7B-Instruct 8-bit (même révision que v1), greedy partout,
versions pinnées (`eval/requirements.lock`, `eval/v2/requirements-v2.lock` :
ruff épinglé pour les checks Zulip). M3 = Llama-3.1-8B-Instruct (génération
des paraphrases de gate + Lots 4-5 si budget). Coûts (tokens, latence, VRAM)
comptabilisés par config.
