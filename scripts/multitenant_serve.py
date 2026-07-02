"""PROTOTYPE serveur MULTI-TENANT : 1 base en RAM, N adaptateurs LoRA, switch à la volée.

Prouve le concept « pas limité au local » : on charge la base qwen UNE seule fois, puis on route
chaque requête vers l'adaptateur du bon tenant en n'échangeant QUE les ~46 Mo de poids A,B
(`model.load_weights(..., strict=False)`) — JAMAIS de rechargement des 8 Go de base.

Tenants : A = VIS (models/lora/vis_spec_v2), B = HelpDeskPro (models/lora/clientB).
Démo : la MÊME question, routée vers A ou B, donne la réponse du BON tenant — depuis une seule base.
Usage : python scripts/multitenant_serve.py
"""

from __future__ import annotations

import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

BASE = os.path.join(_PROJ, "models", "qwen2.5-7b-it-mlx-8bit")
TENANTS = {
    "VIS": os.path.join(_PROJ, "models", "lora", "vis_spec_v2", "adapters.safetensors"),
    "HelpDeskPro": os.path.join(_PROJ, "models", "lora", "clientB", "adapters.safetensors"),
}


def _rss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)  # macOS: bytes
    except Exception:
        return float("nan")


def main():
    from mlx_lm import load, generate as mlx_generate
    import mlx.core as mx
    try:
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=0.0)
    except Exception:
        sampler = None

    print("=== PROTOTYPE serveur multi-tenant (1 base, N adaptateurs) ===\n", flush=True)

    # 1) BASE chargée UNE fois (wrappée LoRA + 1er adaptateur via load())
    t0 = time.monotonic()
    first = next(iter(TENANTS))
    model, tok = load(BASE, adapter_path=os.path.dirname(TENANTS[first]))
    t_load = time.monotonic() - t0
    print(f"[base chargée 1×] {os.path.basename(BASE)} + structure LoRA — {t_load:.1f}s, "
          f"RSS≈{_rss_gb():.1f} Go (adaptateur '{first}' actif)\n", flush=True)
    active = first

    def swap(tenant):
        nonlocal active
        if tenant == active:
            return 0.0
        t = time.monotonic()
        model.load_weights(TENANTS[tenant], strict=False)   # n'écrase QUE les poids A,B (~46 Mo)
        mx.eval(model.parameters())
        active = tenant
        return time.monotonic() - t

    def ask(tenant, q):
        dt = swap(tenant)
        prompt = tok.apply_chat_template([{"role": "user", "content": q}],
                                         tokenize=False, add_generation_prompt=True)
        kw = {"max_tokens": 40, "verbose": False}
        if sampler:
            kw["sampler"] = sampler
        out = mlx_generate(model, tok, prompt=prompt, **kw).strip().replace("\n", " ")
        tag = f"(swap {dt*1000:.0f}ms)" if dt else "(déjà actif)"
        print(f"  [{tenant:11s} {tag:>14s}] Q: {q}\n               → {out[:90]}", flush=True)
        return out

    # 2) Démo : MÊME question, tenant différent -> réponse du bon tenant, depuis 1 base
    print("--- même question, routée vers chaque tenant ---", flush=True)
    pairs = [
        "Quel service est utilisé pour envoyer les emails ?",
        "Quel est le code couleur de la marque ?",
        "Quelle technologie alimente la recherche ?",
    ]
    swaps = []
    for q in pairs:
        a = ask("VIS", q)
        b = ask("HelpDeskPro", q)
        swaps.append("VIS" + a[:30] + " | B" + b[:30])
        print(flush=True)

    # 3) Isolation : chaque tenant ne connaît QUE son spec
    print("--- isolation (chaque adaptateur ne connaît que SON tenant) ---", flush=True)
    ask("VIS", "Quel est le SLA par défaut d'un ticket P1 ?")          # concept HelpDeskPro -> VIS l'ignore
    ask("HelpDeskPro", "Quelle table stocke les photos d'observation ?")  # concept VIS -> HelpDeskPro l'ignore

    # 4) Métriques : coût du switch vs coût de la base
    print("\n--- métriques ---", flush=True)
    times = []
    for tn in ("VIS", "HelpDeskPro", "VIS", "HelpDeskPro", "VIS", "HelpDeskPro"):
        dt = swap(tn)            # tenants alternés -> chaque swap est réel
        if dt:
            times.append(dt)
    avg = sum(times) / max(len(times), 1) * 1000
    print(f"  base chargée 1× : {t_load:.1f}s", flush=True)
    print(f"  swap d'adaptateur (46 Mo) : ~{avg:.0f}ms en moyenne — vs recharger la base = {t_load:.1f}s", flush=True)
    print(f"  ratio : un switch coûte ~{t_load*1000/max(avg,1):.0f}× moins qu'un rechargement de base", flush=True)
    print("\n=> 1 base en RAM, des centaines de tenants possibles (46 Mo chacun), routés à la volée.", flush=True)
    print("=== FIN ===", flush=True)


if __name__ == "__main__":
    main()
