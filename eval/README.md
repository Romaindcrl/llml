# eval/ — Validation publique de LLML sur benchmarks standards

Workspace d'exécution du cahier des charges [`CDC_EVAL_LLML.md`](CDC_EVAL_LLML.md).
Journal de décisions et état d'avancement : [`../AGENTS.md`](../AGENTS.md).

## Structure

- `CDC_EVAL_LLML.md` — le cahier des charges (source de vérité méthodologique).
- `preregistration_issue.md` — texte de l'issue de pré-enregistrement (publiée par
  Romain avant tout run officiel — CDC §3.1).
- `setup_pod.sh` — installation de la stack pinnée sur le pod GPU.
- `requirements.lock` — pip freeze exact du pod (CDC §3.3).
- `runpod/` — outillage machine distante :
  - `runpod_api.py` — provisionnement (deploy/stop/resume/terminate/balance) ;
  - `agent_server.py` — serveur d'exécution tournant sur le pod (HTTPS via le
    proxy RunPod ; SSH sortant indisponible depuis l'environnement de contrôle) ;
  - `remote.py` — client (run/upload/download) ; état local `.pod.json` (gitignoré,
    contient le token d'exécution).
- `smoke/` — smoke tests du Lot 0 :
  - `smoke_llml_pipeline.py` — pipeline LLML complet sur CUDA (train /sleep →
    swap → rappel → routeur → non-régression locale) ;
  - `smoke_harness.py` — EvalPlus (5 items HumanEval+) + lm-eval (5 items GSM8K) ;
  - `smoke_models.py` — chargement M1–M4 + test PEFT 10 steps sur Gemma 2.
- `scripts/draw_mmlu_domains.py` — tirage pré-enregistré des 6 domaines MMLU-Pro
  (seed=42) : biology, business, computer science, economics, math, other.

Les résultats bruts vont dans `../results/raw/` (CSV/JSON), les tableaux dans
`../results/tables/` (markdown) — aucun chiffre ne vit uniquement dans un log.

## Backend CUDA

Le repo LLML est un prototype MLX (Apple Silicon) ; l'éval tourne sur NVIDIA.
Le port vit dans le repo principal :
- `m0/hf.py` — backend `hf` (transformers + bitsandbytes 8-bit + hot-swap peft),
  contrat identique à `MLXClient` (`chat`/`generate`/`set_adapter`) ;
- `m0/d2l_hf.py` — entraînement `/sleep` (peft), mêmes données et même contrat de
  retour que `d2l.train_lora` (recette MLX répliquée : r=16, α=20·r, 8 dernières
  couches, lr 5e-5, format chat, ancres + gate held-out).

Activation : `M0_BACKEND=hf M0_HF_MODEL=Qwen/Qwen2.5-7B-Instruct` (+
`M0_HF_QUANT=8bit|bf16`, `M0_HF_ADAPTER=...`).
