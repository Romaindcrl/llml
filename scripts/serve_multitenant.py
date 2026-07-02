"""Serveur MULTI-TENANT (industrialisation du proto) — 1 base, N LoRA, routage par requête.

OpenAI-compatible. La base est chargée UNE fois ; chaque requête est routée vers l'adaptateur du
tenant (header `X-Tenant`, ou champ `tenant`, ou `model`) par VRAI hot-swap (`load_weights`, ~ms),
JAMAIS de rechargement de base. Auto-découverte des tenants compatibles dans models/lora/*.

Lancer :  M0_MLX_MODEL_PATH=models/qwen2.5-7b-it-mlx-8bit python scripts/serve_multitenant.py
Tester :  curl -s localhost:8001/v1/tenants
          curl -s localhost:8001/v1/chat/completions -H 'X-Tenant: VIS' \
               -d '{"messages":[{"role":"user","content":"Quel service email ?"}]}'

NB : ce proto SÉRIALISE les requêtes (un lock) car l'état adaptateur est partagé ; en production,
le batching multi-LoRA (vLLM / S-LoRA / Punica) sert des tenants différents dans le même batch.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import threading
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

BASE = os.environ.get("M0_MLX_MODEL_PATH", os.path.join(_PROJ, "models", "qwen2.5-7b-it-mlx-8bit"))
PORT = int(os.environ.get("M0_MT_PORT", "8001"))
LORA_DIR = os.path.join(_PROJ, "models", "lora")
# Joli nom d'affichage pour quelques tenants connus (sinon = nom du dossier)
PRETTY = {"vis_spec_v2": "VIS", "clientB": "HelpDeskPro"}


def discover_tenants(base_path):
    """Tenants = adaptateurs entraînés sur LA MÊME base, même rang/num_layers (hot-swappables)."""
    base_name = os.path.basename(base_path.rstrip("/"))
    tenants, ref = {}, None
    for d in sorted(glob.glob(os.path.join(LORA_DIR, "*"))):
        cfg_p = os.path.join(d, "adapter_config.json")
        saf = os.path.join(d, "adapters.safetensors")
        if not (os.path.isfile(cfg_p) and os.path.isfile(saf)):
            continue
        try:
            cfg = json.load(open(cfg_p, encoding="utf-8"))
        except Exception:
            continue
        if os.path.basename(str(cfg.get("model", "")).rstrip("/")) != base_name:
            continue
        sig = (cfg.get("num_layers"), cfg.get("lora_parameters", {}).get("rank"))
        if ref is None:
            ref = sig
        if sig != ref:
            continue  # config incompatible -> non hot-swappable sur la même structure
        name = PRETTY.get(os.path.basename(d), os.path.basename(d))
        tenants[name] = saf
    return tenants


class Engine:
    def __init__(self, base_path, tenants):
        from mlx_lm import load, generate as mlx_generate
        import mlx.core as mx
        self._gen = mlx_generate
        self._mx = mx
        try:
            from mlx_lm.sample_utils import make_sampler
            self._sampler = make_sampler(temp=0.0)
        except Exception:
            self._sampler = None
        self.tenants = tenants
        first = next(iter(tenants))
        t0 = time.monotonic()
        self.model, self.tok = load(base_path, adapter_path=os.path.dirname(tenants[first]))
        self.load_s = time.monotonic() - t0
        self.active = first
        self.lock = threading.Lock()
        print(f"[engine] base {os.path.basename(base_path)} chargée 1× en {self.load_s:.1f}s ; "
              f"tenants: {list(tenants)}", flush=True)

    def _swap(self, tenant):
        if tenant == self.active:
            return 0.0
        t = time.monotonic()
        self.model.load_weights(self.tenants[tenant], strict=False)
        self._mx.eval(self.model.parameters())
        self.active = tenant
        return time.monotonic() - t

    def chat(self, tenant, messages, max_tokens=120):
        with self.lock:
            dt = self._swap(tenant)
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            kw = {"max_tokens": max_tokens, "verbose": False}
            if self._sampler:
                kw["sampler"] = self._sampler
            out = self._gen(self.model, self.tok, prompt=prompt, **kw)
        return out.strip(), dt


ENGINE: Engine | None = None


def build_app():
    from fastapi import FastAPI, Header, Body
    from fastapi.responses import JSONResponse
    app = FastAPI(title="LLML multi-tenant")

    @app.get("/v1/tenants")
    def tenants():
        return {"object": "list", "data": [{"id": t} for t in ENGINE.tenants]}

    @app.get("/health")
    def health():
        return {"status": "ok", "base_load_s": ENGINE.load_s, "tenants": list(ENGINE.tenants)}

    @app.post("/v1/chat/completions")
    def chat(body: dict = Body(...), x_tenant: str | None = Header(default=None)):
        tenant = x_tenant or body.get("tenant") or body.get("model")
        if tenant not in ENGINE.tenants:
            return JSONResponse(status_code=400, content={
                "error": f"tenant inconnu: {tenant!r}. Disponibles: {list(ENGINE.tenants)} "
                         "(header 'X-Tenant', ou champ 'tenant'/'model')."})
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens", 120))
        out, dt = ENGINE.chat(tenant, messages, max_tokens)
        return {
            "id": f"chatcmpl-{int(time.time()*1000)}", "object": "chat.completion",
            "model": tenant, "x_swap_ms": round(dt * 1000, 1),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": out}}],
        }

    return app


def main():
    global ENGINE
    tenants = discover_tenants(BASE)
    if not tenants:
        print(f"Aucun adaptateur hot-swappable trouvé pour la base {BASE} dans {LORA_DIR}.")
        print("Entraîne au moins un LoRA sur cette base (ex. scripts/train_clientB.py).")
        return
    ENGINE = Engine(BASE, tenants)
    import uvicorn
    print(f"[serve] http://localhost:{PORT}  (POST /v1/chat/completions, header X-Tenant)", flush=True)
    uvicorn.run(build_app(), host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
