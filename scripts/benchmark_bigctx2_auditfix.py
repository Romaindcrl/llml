"""D2-bis — clore l'audit : la passe de vérification LLML impose la ligne d'audit.

D2 a montré la rigidité : le style-LoRA ignore la convention d'audit venue du contexte
(audit 0% alors que la fondation est en fenêtre). Fix ARCHITECTURAL (pas un hack de banc) :
la étape-2 de LLML (vérification déterministe contre la mémoire) lit le module fondation
— que LLML est le seul à toujours garder (fondation 100%) — en extrait la vraie valeur
d'audit_tag, et insère la ligne après la docstring de chaque get_* si absente. La VALEUR
n'est jamais gravée dans les poids : elle vient du projet à l'exécution (inter-fichiers).

Bras : 7B+LLML et 14B+LLML (adaptateurs existants, AUCUN ré-entraînement), charges 8k/24k,
même scoring que D2. Attendu : conv/faits/audit/fondation = 100% partout.
Live : tail -f logs/benchmark_bigctx2_auditfix.log
"""

from __future__ import annotations

import gc
import os
import re
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from scripts.benchmark_bigctx2 import AD7, AD14, B7, B14, legacy_code, task_for  # noqa: E402
from scripts.benchmark_project import FACTS, HELD, NS, WINDOW, build_spec, fit_code  # noqa: E402
from scripts.benchmark_spec_final import lookup_facts, substitute  # noqa: E402
from scripts.benchmark_spec_xl import CONV, frac  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client, MLXClient  # noqa: E402
from m0.rag import RAG  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_bigctx2_auditfix.log")
L_LEVELS = [8000, 24000]
_T0 = time.time()
_AUDIT_RE = re.compile(r'audit_tag\s*=\s*"([^"]+)"')


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


def ensure_audit(code: str, project_code: str) -> str:
    """Étape-2 LLML : impose la convention d'audit de la FONDATION (valeur lue du projet).

    Extrait audit_tag du module fondation présent dans le contexte projet, puis insère
    `audit_tag = "<valeur>"` juste après la docstring de chaque fonction get_* qui ne l'a pas.
    Déterministe — même famille que substitute() (spec_final, validé 100/100).
    """
    m = _AUDIT_RE.search(project_code)
    if not m:
        return code
    tag = m.group(1)
    lines = code.splitlines()
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        out.append(ln); i += 1
        ds = ln.strip()
        if ds.startswith("def get_") and ds.endswith(":"):
            indent = (len(ln) - len(ln.lstrip())) + 4
            # saute la docstring éventuelle
            j = i
            if j < len(lines) and lines[j].strip().startswith(('"""', "'''")):
                q = lines[j].strip()[:3]
                if not (lines[j].strip().endswith(q) and len(lines[j].strip()) > 5):
                    j += 1
                    while j < len(lines) and q not in lines[j]:
                        j += 1
                j += 1
            # la ligne d'audit est-elle déjà là (dans les 3 lignes suivantes) ?
            nxt = "\n".join(lines[i:j + 3])
            if f'audit_tag = "{tag}"' not in nxt and "audit_tag" not in nxt:
                out.extend(lines[i:j])
                out.append(" " * indent + f'audit_tag = "{tag}"')
                i = j
    return "\n".join(out)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    spec = build_spec()
    rag = RAG(os.path.join(_PROJ, "logs", "rag_auditfix.txt")); rag.clear(); rag.add_document(spec)
    log("=== D2-bis — vérification LLML étendue à l'audit (aucun ré-entraînement) ===")

    ARMS = [("7B + LLML", B7, AD7), ("14B + LLML", B14, AD14)]
    agg = {(L, s): {"c": [], "f": [], "x": [], "kept": []} for L in L_LEVELS for s, _, _ in ARMS}

    for name, base, adapter in ARMS:
        if not os.path.isfile(os.path.join(adapter, "adapters.safetensors")):
            log(f"   ⚠️ adaptateur absent pour {name} — sauté"); continue
        cfg = Config.from_env(); cfg.backend = "mlx"; cfg.mlx_model_path = base
        llm = make_client(cfg); llm.set_adapter(adapter); llm.cfg.mlx_max_tokens = 380
        log(f"[{name}]")
        for L in L_LEVELS:
            code = legacy_code(L)
            c_fit = fit_code(code, WINDOW - 600)
            for ent in HELD:
                rm, ec = FACTS[ent]
                draft = llm.generate(f"Code du projet :\n{c_fit}\n\n{task_for(ent)}"
                                     "\nÉcris uniquement le code Python du module.", None)
                out = substitute(draft, *lookup_facts(ent, rag)[:2])
                out = ensure_audit(out, c_fit)                    # ← l'étape-2 étendue
                a = agg[(L, name)]
                a["c"].append(frac(out, CONV)); a["f"].append(frac(out, [rm, ec]))
                a["x"].append(1.0 if NS in out else 0.0)
                a["kept"].append(1.0 if NS in c_fit else 0.0)
                log(f"   [L={L} {ent}] conv={frac(out, CONV)*100:.0f}% faits={frac(out,[rm,ec])*100:.0f}% "
                    f"audit={int(NS in out)}")
            _purge()
        del llm; MLXClient._cache.clear(); _purge()

    def av(L, s, k):
        v = agg[(L, s)][k]; return sum(v) / len(v) * 100 if v else 0.0
    log("")
    log("=== RÉSULTAT D2-bis (rappel D2 : audit LLML = 0%) ===")
    log(f"{'L':>6} | {'méthode':12s} | conv | faits | audit | fondation")
    allok = True
    for L in L_LEVELS:
        for s, _, _ in ARMS:
            if agg[(L, s)]["c"]:
                log(f"{L:>6} | {s:12s} | {av(L,s,'c'):3.0f}% | {av(L,s,'f'):4.0f}% | "
                    f"{av(L,s,'x'):4.0f}% | {av(L,s,'kept'):3.0f}%")
                allok = allok and av(L, s, "x") == 100 and av(L, s, "c") == 100
    log("")
    log("🟢 AUDIT CLOS : conventions gravées (poids) + faits substitués (RAG) + convention "
        "inter-fichiers imposée par la vérification (valeur lue de la fondation) = 100% partout."
        if allok else "🟠 pas encore 100% partout — inspecter les sorties.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
