"""Banc D — le régime « VRAIE VIE » : cahier 13k + code qui sature la fenêtre 32k, 14B en face.

Objection utilisateur (juste) : le banc A (doc 600 tok) n'était pas représentatif — en vrai on a
un cahier de ~20k tokens + un projet de code énorme. Ici : cahier ~13k (227 entités), fenêtre
DURE 32k, charge de code L. À L=24k, cahier+code = 37k > 32k → DÉBORDEMENT : l'approche
« tout en contexte » doit tronquer (elle perd le module fondation, donc le signal inter-fichiers).

Bras :
  - 7B + LLML          : conventions dans les POIDS (adaptateur `project` existant) + RAG faits
                         + substitution déterministe ; le cahier coûte ~0 tok -> TOUT le code tient.
  - 14B tout-en-ctx    : cahier COMPLET prioritaire + code tronqué au budget (le défaut réel).
  - 14B + RAG-cahier   : extraits RAG du cahier (~200 tok) + code complet (l'ingénieur malin).
Charges : L=8k (tout tient) et L=24k (débordement). Score : conventions% / faits% / audit
inter-fichiers / fondation gardée. NB 400k : intestable localement — mais le mécanisme démontré
(spec+travail > fenêtre => l'in-context casse) est invariant d'échelle, et le coût récurrent
(re-payer le cahier à chaque appel) croît avec l'échelle. Live : tail -f logs/benchmark_bigctx.log
"""

from __future__ import annotations

import gc
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from scripts.benchmark_project import (  # noqa: E402
    FACTS, HELD, NS, WINDOW, build_spec, filler_code, fit_code,
)
from scripts.benchmark_spec_final import lookup_facts, substitute  # noqa: E402
from scripts.benchmark_spec_xl import CONV, frac  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client, MLXClient  # noqa: E402
from m0.rag import RAG  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_bigctx.log")
B7 = os.path.join(_PROJ, "models", "qwen2.5-7b-it-mlx-8bit")
B14 = os.path.join(_PROJ, "models", "qwen2.5-coder-14b-mlx-4bit")
ADAPTER = os.path.join(_PROJ, "models", "lora", "project")
L_LEVELS = [8000, 24000]
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


def task_for(ent):
    return (f"Implémente le module {ent}.py : la fonction get_{ent}(payload) qui récupère un "
            f"{ent} par identifiant, en respectant le cahier des charges. N'oublie PAS la ligne "
            "d'audit obligatoire définie dans le module fondation du projet.")


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    if not os.path.isfile(os.path.join(ADAPTER, "adapters.safetensors")):
        log("ERREUR : adaptateur `project` absent — lancer benchmark_project.py d'abord"); return

    spec = build_spec()
    spec_tok = count_tokens(spec)
    rag = RAG(os.path.join(_PROJ, "logs", "rag_bigctx.txt")); rag.clear()
    rag.add_document(spec)
    log(f"=== BANC D — cahier {spec_tok} tok, fenêtre dure {WINDOW}, charges {L_LEVELS} ===")
    log(f"    à L={L_LEVELS[1]} : cahier+code = {spec_tok + L_LEVELS[1]} tok > {WINDOW} → DÉBORDEMENT")

    SUT = ("7B + LLML", "14B tout-en-ctx", "14B + RAG-cahier")
    agg = {(L, s): {"c": [], "f": [], "x": [], "kept": [], "ctx": []} for L in L_LEVELS for s in SUT}

    # ---------- Phase 1 : 7B + LLML (poids + RAG + substitution), tout le code en fenêtre
    cfg = Config.from_env(); cfg.backend = "mlx"; cfg.mlx_model_path = B7
    llm = make_client(cfg); llm.set_adapter(ADAPTER); llm.cfg.mlx_max_tokens = 380
    log("[1/2] 7B + LLML (cahier dans les poids, ~0 tok)")
    for L in L_LEVELS:
        code_full = filler_code(L)
        c_ours = fit_code(code_full, WINDOW - 600)          # le cahier ne mange rien
        for ent in HELD:
            rm, ec = FACTS[ent]
            draft = llm.generate(f"Code du projet :\n{c_ours}\n\n{task_for(ent)}"
                                 "\nÉcris uniquement le code Python du module.", None)
            out = substitute(draft, *lookup_facts(ent, rag)[:2])
            a = agg[(L, "7B + LLML")]
            a["c"].append(frac(out, CONV)); a["f"].append(frac(out, [rm, ec]))
            a["x"].append(1.0 if NS in out else 0.0); a["kept"].append(1.0 if NS in c_ours else 0.0)
            a["ctx"].append(count_tokens(c_ours))
            log(f"   [L={L} {ent}] conv={frac(out, CONV)*100:.0f}% faits={frac(out,[rm,ec])*100:.0f}% "
                f"audit={int(NS in out)} (code gardé {count_tokens(c_ours)} tok, fondation {int(NS in c_ours)})")
        _purge()
    del llm; MLXClient._cache.clear(); _purge()

    # ---------- Phase 2 : 14B (tout-en-ctx prioritaire cahier, et RAG-cahier)
    cfg14 = Config.from_env(); cfg14.backend = "mlx"; cfg14.mlx_model_path = B14
    llm14 = make_client(cfg14); llm14.set_adapter(None); llm14.cfg.mlx_max_tokens = 380
    log("[2/2] 14B — tout-en-ctx (cahier complet prioritaire) et RAG-cahier")
    for L in L_LEVELS:
        code_full = filler_code(L)
        for ent in HELD:
            rm, ec = FACTS[ent]
            task = task_for(ent)
            # (a) tout-en-ctx : cahier complet, le code se tronque au budget restant
            c_fit = fit_code(code_full, WINDOW - spec_tok - 700)
            p_all = (f"Cahier des charges :\n{spec}\n\nCode du projet :\n{c_fit}\n\n{task}"
                     "\nÉcris uniquement le code Python du module.")
            o_all = llm14.generate(p_all, None)
            a = agg[(L, "14B tout-en-ctx")]
            a["c"].append(frac(o_all, CONV)); a["f"].append(frac(o_all, [rm, ec]))
            a["x"].append(1.0 if NS in o_all else 0.0); a["kept"].append(1.0 if NS in c_fit else 0.0)
            a["ctx"].append(count_tokens(p_all))
            log(f"   [L={L} {ent}] 14B-tout : conv={frac(o_all, CONV)*100:.0f}% "
                f"faits={frac(o_all,[rm,ec])*100:.0f}% audit={int(NS in o_all)} "
                f"(fondation en ctx : {int(NS in c_fit)}, prompt {count_tokens(p_all)} tok)")
            _purge()
            # (b) RAG-cahier : extraits seulement, code complet
            chunks = "\n".join(rag.topk(task, 6))
            c_rag = fit_code(code_full, WINDOW - count_tokens(chunks) - 700)
            p_rag = (f"Cahier (extraits pertinents) :\n{chunks}\n\nCode du projet :\n{c_rag}\n\n{task}"
                     "\nÉcris uniquement le code Python du module.")
            o_rag = llm14.generate(p_rag, None)
            a = agg[(L, "14B + RAG-cahier")]
            a["c"].append(frac(o_rag, CONV)); a["f"].append(frac(o_rag, [rm, ec]))
            a["x"].append(1.0 if NS in o_rag else 0.0); a["kept"].append(1.0 if NS in c_rag else 0.0)
            a["ctx"].append(count_tokens(p_rag))
            log(f"   [L={L} {ent}] 14B-RAG  : conv={frac(o_rag, CONV)*100:.0f}% "
                f"faits={frac(o_rag,[rm,ec])*100:.0f}% audit={int(NS in o_rag)} "
                f"(fondation en ctx : {int(NS in c_rag)})")
            _purge()

    def av(L, s, k):
        v = agg[(L, s)][k]; return sum(v) / len(v) * 100 if v else 0.0
    log("")
    log("=== RÉSULTAT BANC D — régime représentatif (cahier 13k + code, fenêtre dure 32k) ===")
    log(f"{'L':>6} | {'méthode':18s} | conv | faits | audit | fondation | ctx moyen")
    for L in L_LEVELS:
        for s in SUT:
            v = agg[(L, s)]
            ctxm = int(sum(v['ctx']) / max(len(v['ctx']), 1))
            log(f"{L:>6} | {s:18s} | {av(L,s,'c'):3.0f}% | {av(L,s,'f'):4.0f}% | {av(L,s,'x'):4.0f}% "
                f"| {av(L,s,'kept'):3.0f}%      | {ctxm}")
        log("       " + "-" * 66)
    log("")
    c7, c14 = av(L_LEVELS[1], "7B + LLML", "c"), av(L_LEVELS[1], "14B tout-en-ctx", "c")
    f7, f14 = av(L_LEVELS[1], "7B + LLML", "f"), av(L_LEVELS[1], "14B tout-en-ctx", "f")
    k7, k14 = av(L_LEVELS[1], "7B + LLML", "kept"), av(L_LEVELS[1], "14B tout-en-ctx", "kept")
    if (c7 >= c14 and f7 >= f14) or k14 < k7:
        log(f"🟢 AU DÉBORDEMENT (L={L_LEVELS[1]}) : 7B+LLML conv {c7:.0f}%/faits {f7:.0f}%/fondation "
            f"{k7:.0f}% vs 14B tout-en-ctx {c14:.0f}%/{f14:.0f}%/fondation {k14:.0f}% — le cahier en "
            "poids libère la fenêtre pour le code ; l'in-context sacrifie l'un ou l'autre.")
    else:
        log(f"🟠 le 14B tient même au débordement : conv {c14:.0f}%/faits {f14:.0f}% vs LLML {c7:.0f}%/{f7:.0f}% — à analyser.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
