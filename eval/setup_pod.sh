#!/usr/bin/env bash
# Lot 0 — installation de l'environnement d'éval sur le pod RunPod.
# Image attendue : runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# (torch 2.4.1+cu124 préinstallé — le venv l'hérite via --system-site-packages).
# Usage (sur le pod) : bash eval/setup_pod.sh
set -euxo pipefail

export HF_HOME=/workspace/hf   # caches modèles sur le volume persistant

python3 -V
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

python3 -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/pip install -q -U pip
/workspace/venv/bin/pip install -q \
  'transformers==4.46.3' 'peft==0.13.2' 'bitsandbytes==0.44.1' 'accelerate==1.1.1' \
  'datasets==3.1.0' 'lm_eval==0.4.5' 'evalplus==0.3.1' \
  'fastapi==0.115.5' 'uvicorn==0.32.1' 'httpx' 'sentencepiece' 'protobuf'

/workspace/venv/bin/python -c 'import transformers, peft, bitsandbytes, lm_eval, evalplus; \
  print("STACK_OK", transformers.__version__, peft.__version__, bitsandbytes.__version__)'

# Lock file exact (CDC §3.3) — à commiter dans eval/requirements.lock.
/workspace/venv/bin/pip freeze > /workspace/requirements.lock
echo "lock -> /workspace/requirements.lock"
