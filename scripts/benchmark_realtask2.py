"""Banc F2 — remplir le cahier par EXÉCUTION, avec DÉCOMPOSITION pour les bras LLML.

Diagnostic F : le style-LoRA (entraîné mono-fonction) s'effondre sur « module 4 fonctions »
(rigidité de format, 0-25%). Fix réaliste : le SYSTÈME décompose — une génération par fonction
(le format d'entraînement exact), assemblage du module, substitution des faits, exécution.
C'est ce qu'un agent fait de toute façon. Les bras in-context restent monolithiques (leur mode
optimal, mesuré en F : 62/78%). Même harnais comportemental que F (16 asserts/entité).
Live : tail -f logs/benchmark_realtask2.log
"""

from __future__ import annotations

import gc
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from scripts.benchmark_realtask import (  # noqa: E402
    ACTIONS4, AD7, AD14, B7, B14, extract_code, run_behavior,
)
from scripts.benchmark_project import FACTS, HELD, build_spec  # noqa: E402
from scripts.benchmark_spec_final import lookup_facts, substitute  # noqa: E402
from scripts.benchmark_spec_xl import ACTIONS  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client, MLXClient  # noqa: E402
from m0.rag import RAG  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_realtask2.log")
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def _purge():
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def gen_module_decomposed(llm, ent):
    """Une génération PAR fonction (format d'entraînement du style-LoRA), puis assemblage."""
    parts = []
    for act in ACTIONS4:
        fr = ACTIONS.get(act, act)
        p = (f"Implémente la fonction {act}_{ent} qui {fr} un {ent} à partir de payload.")
        out = extract_code(llm.generate(p, None))
        # ne garde que la (première) définition de fonction
        keep, started = [], False
        for ln in out.splitlines():
            if ln.startswith("def "):
                if started:
                    break
                started = True
            if started or ln.startswith(("import ", "from ")):
                keep.append(ln)
        parts.append("\n".join(keep) if keep else out)
    return "\n\n".join(parts)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    spec = build_spec()
    rag = RAG(os.path.join(_PROJ, "logs", "rag_realtask2.txt")); rag.clear(); rag.add_document(spec)
    log(f"=== BANC F2 — exécution réelle, LLML en mode DÉCOMPOSÉ (spec {count_tokens(spec)} tok) ===")
    R = {}

    for name, base, adapter in (("7B + LLML (décomposé)", B7, AD7), ("14B + LLML (décomposé)", B14, AD14)):
        if not os.path.isfile(os.path.join(adapter, "adapters.safetensors")):
            log(f"   ⚠️ adaptateur absent pour {name}"); continue
        cfg = Config.from_env(); cfg.backend = "mlx"; cfg.mlx_model_path = base
        llm = make_client(cfg); llm.set_adapter(adapter); llm.cfg.mlx_max_tokens = 320
        tot_p = tot_n = 0
        for ent in HELD:
            rm, ec = FACTS[ent]
            module = gen_module_decomposed(llm, ent)
            module = substitute(module, *lookup_facts(ent, rag)[:2])
            p, n = run_behavior(module, ent, rm, ec)
            tot_p += p; tot_n += n
            log(f"   [{name}] {ent} : {p}/{n}  (module {count_tokens(module)} tok, "
                f"extrait: {module.splitlines()[0][:60] if module.splitlines() else 'VIDE'!r})")
        R[name] = (tot_p, tot_n)
        del llm; MLXClient._cache.clear(); _purge()

    log("")
    log("=== RÉSULTAT F2 (rappels F : 7B+LLML mono 0/32 · 7B ctx 20/32 · 14B ctx 25/32 · 14B+LLML mono 8/32) ===")
    log(f"{'bras':24s} | réussite | ctx cahier")
    for k, (p, n) in R.items():
        log(f"{k:24s} | {p}/{n} ({p/n*100:3.0f}%) | 0 tok (poids)")
    log("")
    best = max((p / n for p, n in R.values()), default=0)
    if best >= 0.78:
        log("🟢 DÉCOMPOSITION VALIDÉE : le mode système (une fonction par génération, assemblage, "
            "substitution) rend LLML compétitif avec le cahier-en-contexte sur du LIVRABLE exécuté "
            "— à 0 token de cahier récurrent.")
    elif best >= 0.62:
        log("🟢 LLML décomposé rejoint le 7B-cahier-en-ctx (62%) — écart restant vs 14B ctx (78%).")
    else:
        log("🟠 la décomposition ne suffit pas — la rigidité du style-LoRA va plus loin que le format.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
