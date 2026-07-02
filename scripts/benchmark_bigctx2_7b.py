"""D2 complément — les bras 7B manquants dans le régime legacy (piège d'imitation).

D2 avait : 7B+LLML, 14B tout-en-ctx, 14B+RAG, 14B+LLML. Il manque, pour le graphe et le
tableau : **7B tout-en-ctx** (cahier complet prioritaire + code tronqué) et **7B + RAG-cahier**
(extraits + code complet). Mêmes charges (8k / 24k=débordement), même scoring conv/faits/audit/
fondation. Live : tail -f logs/benchmark_bigctx2_7b.log
"""

from __future__ import annotations

import gc
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from scripts.benchmark_bigctx2 import B7, legacy_code, task_for  # noqa: E402
from scripts.benchmark_project import FACTS, HELD, NS, WINDOW, build_spec, fit_code  # noqa: E402
from scripts.benchmark_spec_xl import CONV, frac  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.rag import RAG  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_bigctx2_7b.log")
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


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    spec = build_spec(); spec_tok = count_tokens(spec)
    rag = RAG(os.path.join(_PROJ, "logs", "rag_bigctx2_7b.txt")); rag.clear(); rag.add_document(spec)
    log(f"=== D2 bras 7B manquants — legacy, cahier {spec_tok} tok, charges {L_LEVELS} ===")

    cfg = Config.from_env(); cfg.backend = "mlx"; cfg.mlx_model_path = B7
    llm = make_client(cfg); llm.set_adapter(None); llm.cfg.mlx_max_tokens = 380

    SUT = ("7B tout-en-ctx", "7B + RAG-cahier")
    agg = {(L, s): {"c": [], "f": [], "x": [], "kept": []} for L in L_LEVELS for s in SUT}

    for L in L_LEVELS:
        code = legacy_code(L)
        for ent in HELD:
            rm, ec = FACTS[ent]
            task = task_for(ent)
            # (a) tout-en-ctx : cahier complet prioritaire, code tronqué au budget
            c_fit = fit_code(code, WINDOW - spec_tok - 700)
            o = llm.generate(f"Cahier des charges :\n{spec}\n\nCode du projet :\n{c_fit}\n\n{task}"
                             "\nÉcris uniquement le code Python du module.", None)
            a = agg[(L, "7B tout-en-ctx")]
            a["c"].append(frac(o, CONV)); a["f"].append(frac(o, [rm, ec]))
            a["x"].append(1.0 if NS in o else 0.0); a["kept"].append(1.0 if NS in c_fit else 0.0)
            log(f"   [L={L} {ent}] 7B-tout : conv={frac(o, CONV)*100:.0f}% faits={frac(o,[rm,ec])*100:.0f}% "
                f"(fondation en ctx : {int(NS in c_fit)}, prompt ~{spec_tok + count_tokens(c_fit)} tok)")
            _purge()
            # (b) RAG-cahier : extraits + code complet
            chunks = "\n".join(rag.topk(task, 6))
            c_rag = fit_code(code, WINDOW - count_tokens(chunks) - 700)
            o = llm.generate(f"Cahier (extraits) :\n{chunks}\n\nCode du projet :\n{c_rag}\n\n{task}"
                             "\nÉcris uniquement le code Python du module.", None)
            a = agg[(L, "7B + RAG-cahier")]
            a["c"].append(frac(o, CONV)); a["f"].append(frac(o, [rm, ec]))
            a["x"].append(1.0 if NS in o else 0.0); a["kept"].append(1.0 if NS in c_rag else 0.0)
            log(f"   [L={L} {ent}] 7B-RAG  : conv={frac(o, CONV)*100:.0f}% faits={frac(o,[rm,ec])*100:.0f}%")
            _purge()

    def av(L, s, k):
        v = agg[(L, s)][k]; return sum(v) / len(v) * 100 if v else 0.0
    log("")
    log("=== RÉSULTAT — bras 7B, régime legacy (rappel : 7B+LLML = 100/100 partout) ===")
    log(f"{'L':>6} | {'méthode':16s} | conv | faits | audit | fondation")
    for L in L_LEVELS:
        for s in SUT:
            log(f"{L:>6} | {s:16s} | {av(L,s,'c'):3.0f}% | {av(L,s,'f'):4.0f}% | "
                f"{av(L,s,'x'):4.0f}% | {av(L,s,'kept'):3.0f}%")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
