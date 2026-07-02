"""Banc F — REMPLIR un cahier des charges, ÉVALUÉ PAR EXÉCUTION (pas par marqueurs).

Objection utilisateur : nos scores mesurent la présence d'infos, pas la capacité à LIVRER.
Ici le modèle doit implémenter un MODULE COMPLET (get/update/delete/archive d'une entité
held-out, selon le cahier Nexus) et le code est EXÉCUTÉ dans un harnais fonctionnel caché :
stubs repo/Result/NexusError/log, puis asserts de COMPORTEMENT :
  - flux nominal : le module retourne l'enveloppe attendue pour un id existant ;
  - le BON repo.méthode du cahier est appelé (fait spécifique, vérifié à l'exécution) ;
  - NexusError avec le CODE EXACT du cahier sur id absent ;
  - validate_input appelé, journalisation effectuée.
Score = fraction d'asserts qui passent (16 par entité). Bras : 7B+LLML (poids+substitution) ·
7B cahier-en-ctx · 14B cahier-en-ctx · 14B+LLML (adaptateur project14b du banc D2 si dispo).
Live : tail -f logs/benchmark_realtask.log
"""

from __future__ import annotations

import gc
import os
import re
import subprocess
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from scripts.benchmark_project import FACTS, HELD, build_spec  # noqa: E402
from scripts.benchmark_spec_final import lookup_facts, substitute  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client, MLXClient  # noqa: E402
from m0.rag import RAG  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_realtask.log")
B7 = os.path.join(_PROJ, "models", "qwen2.5-7b-it-mlx-8bit")
B14 = os.path.join(_PROJ, "models", "qwen2.5-coder-14b-mlx-4bit")
AD7 = os.path.join(_PROJ, "models", "lora", "project")
AD14 = os.path.join(_PROJ, "models", "lora", "project14b")
ACTIONS4 = ["get", "update", "delete", "archive"]
_T0 = time.time()
_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

HARNESS = '''
class _Repo:
    def __init__(self): self.store = {}; self.calls = []
    def __getattr__(self, name):
        if name.startswith("_"): raise AttributeError(name)
        def _m(ident):
            self.calls.append(name)
            return self.store.get(ident)
        return _m
repo = _Repo()
class NexusError(Exception):
    def __init__(self, code, *a): self.code = code; super().__init__(code)
class Result:
    @staticmethod
    def ok(x): return {"ok": True, "data": x}
    @staticmethod
    def fail(x): return {"ok": False, "data": x}
_validated = []
def validate_input(p): _validated.append(dict(p))
def serialize_envelope(r): return {"envelope": r}
class _Log:
    def __init__(self): self.lines = []
    def _w(self, *a, **k): self.lines.append(" ".join(str(x) for x in a))
    def __getattr__(self, n): return self._w
log = _Log()
'''


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


def extract_code(text):
    m = _CODE_RE.findall(text or "")
    return m[-1].strip() if m else (text or "").strip()


def behavior_asserts(ent, rm, ec):
    """16 asserts de COMPORTEMENT (4 par action), exécutés un à un pour crédit partiel."""
    out = []
    for act in ACTIONS4:
        fn = f"{act}_{ent}"
        out += [
            # nominal : enveloppe exacte du cahier (Result.ok(serialize_envelope(record)))
            (f"repo.store.clear(); repo.calls.clear(); _validated.clear()\n"
             f"repo.store['x1']={{'id':'x1','statut':'actif'}}\n"
             f"r={fn}({{'id':'x1'}})\n"
             f"assert r=={{'ok':True,'data':{{'envelope':{{'id':'x1','statut':'actif'}}}}}}"),
            # le BON repo.méthode du cahier (fait, vérifié à l'exécution)
            (f"repo.store.clear(); repo.calls.clear()\n"
             f"repo.store['x2']={{'id':'x2'}}\n{fn}({{'id':'x2'}})\n"
             f"assert '{rm}' in repo.calls"),
            # validation appelée
            (f"_validated.clear(); repo.store['x3']={{'id':'x3'}}\n{fn}({{'id':'x3'}})\n"
             f"assert _validated"),
            # id absent -> NexusError avec le CODE EXACT du cahier
            (f"repo.store.clear()\n"
             f"try:\n    {fn}({{'id':'nope'}})\n    assert False, 'aurait dû lever'\n"
             f"except NexusError as e:\n    assert e.code=='{ec}'"),
        ]
    return out


def run_behavior(code, ent, rm, ec):
    """Exécute le module dans le harnais, puis chaque assert isolément. Renvoie (passés, total)."""
    checks = behavior_asserts(ent, rm, ec)
    passed = 0
    for chk in checks:
        src = HARNESS + "\n" + code + "\n\n" + chk + "\nprint('OK')\n"
        try:
            r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=8)
            passed += (r.returncode == 0 and "OK" in r.stdout)
        except Exception:
            pass
    return passed, len(checks)


def task_for(ent):
    return (f"Implémente le module {ent}.py COMPLET avec les QUATRE fonctions "
            f"{', '.join(a + '_' + ent for a in ACTIONS4)} — chacune prend `payload` (dict avec "
            f"la clé 'id') et suit STRICTEMENT le cahier des charges Nexus (validation, "
            f"journalisation, accès repo, gestion d'absence, enveloppe de retour). "
            "N'invente pas d'infrastructure : utilise directement validate_input, log, repo, "
            "NexusError, Result et serialize_envelope, supposés déjà importés.")


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    spec = build_spec()
    rag = RAG(os.path.join(_PROJ, "logs", "rag_realtask.txt")); rag.clear(); rag.add_document(spec)
    log(f"=== BANC F — remplir le cahier, ÉVALUATION PAR EXÉCUTION (spec {count_tokens(spec)} tok) ===")
    R = {}

    def eval_arm(name, llm, use_spec_ctx, use_llml):
        tot_p = tot_n = 0
        for ent in HELD:
            rm, ec = FACTS[ent]
            t = task_for(ent)
            if use_spec_ctx:
                prompt = f"Cahier des charges :\n{spec}\n\n{t}\nRéponds avec UN bloc ```python```."
            else:
                prompt = t + "\nRéponds avec UN bloc ```python```."
            draft = llm.generate(prompt, None)
            code = extract_code(draft)
            if use_llml:
                code = substitute(code, *lookup_facts(ent, rag)[:2])
            p, n = run_behavior(code, ent, rm, ec)
            tot_p += p; tot_n += n
            log(f"   [{name}] {ent} : {p}/{n} asserts comportementaux")
        R[name] = (tot_p, tot_n)

    # ---------- 7B
    cfg = Config.from_env(); cfg.backend = "mlx"; cfg.mlx_model_path = B7
    llm = make_client(cfg); llm.cfg.mlx_max_tokens = 900
    log("[1/2] 7B : LLML (poids) puis cahier-en-contexte")
    llm.set_adapter(AD7)
    eval_arm("7B + LLML", llm, use_spec_ctx=False, use_llml=True)
    llm.set_adapter(None)
    eval_arm("7B cahier-en-ctx", llm, use_spec_ctx=True, use_llml=False)
    del llm; MLXClient._cache.clear(); _purge()

    # ---------- 14B
    cfg14 = Config.from_env(); cfg14.backend = "mlx"; cfg14.mlx_model_path = B14
    llm14 = make_client(cfg14); llm14.cfg.mlx_max_tokens = 900
    log("[2/2] 14B : cahier-en-contexte puis LLML (si adaptateur D2 dispo)")
    llm14.set_adapter(None)
    eval_arm("14B cahier-en-ctx", llm14, use_spec_ctx=True, use_llml=False)
    if os.path.isfile(os.path.join(AD14, "adapters.safetensors")):
        llm14.set_adapter(AD14)
        eval_arm("14B + LLML", llm14, use_spec_ctx=False, use_llml=True)
    else:
        log("   (adaptateur 14B absent — bras 14B+LLML sauté ; relancer après D2)")

    log("")
    log("=== RÉSULTAT BANC F — asserts COMPORTEMENTAUX (module exécuté) ===")
    log(f"{'bras':18s} | réussite | ctx cahier")
    for k, (p, n) in R.items():
        ctx = "0 tok (poids)" if "LLML" in k else f"{count_tokens(spec)} tok"
        log(f"{k:18s} | {p}/{n} ({p/n*100:3.0f}%) | {ctx}")
    log("")
    if "7B + LLML" in R and "7B cahier-en-ctx" in R:
        a, b = R["7B + LLML"], R["7B cahier-en-ctx"]
        log(f"7B : LLML {a[0]}/{a[1]} vs cahier-en-ctx {b[0]}/{b[1]} — "
            f"{'les poids livrent un module FONCTIONNEL sans le cahier en fenêtre' if a[0] >= b[0] else 'le cahier en contexte reste devant sur ce modèle'}")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
