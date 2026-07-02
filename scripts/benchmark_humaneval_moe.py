"""HumanEval × MoE — le routeur sait-il quand N'UTILISER AUCUN expert ?

Le run principal mesure le coût de chaque adaptateur porté seul. Ici on teste le SYSTÈME MoE
complet sur des tâches HORS-DOMAINE (HumanEval = code générique, aucun des 3 experts n'est
pertinent) : le routeur — étendu d'une option GENERAL — doit envoyer vers le modèle NU.
Métriques : (1) taux de routage GENERAL (attendu ~100%), (2) pass@1 du système MoE
(attendu = celui du 7B nu → le MoE ne taxe pas la capacité générale).
Si le routeur force un expert, l'adaptateur hors-sujet peut coûter des points — c'est
exactement ce qu'on veut détecter. Live : tail -f logs/benchmark_humaneval_moe.log
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from scripts.benchmark_humaneval import CACHE, PRELUDE, run_tests  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_humaneval_moe.log")
_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_T0 = time.time()

EXPERTS = {
    "VIS": os.path.join(_PROJ, "models", "lora", "vis_spec_v2"),
    "HELPDESK": os.path.join(_PROJ, "models", "lora", "clientB"),
    "CORVEX": os.path.join(_PROJ, "models", "lora", "corvex_loop"),
}
ROUTER_PROMPT = (
    "Classifie la tâche dans UN de ces domaines et réponds par ce seul mot :\n"
    "- VIS : application d'inspection d'EPI (sessions, rapports, tenants Supabase)\n"
    "- HELPDESK : produit de support client (tickets, SLA, priorités)\n"
    "- CORVEX : SDK d'ingestion de frames (connexion, vaults)\n"
    "- GENERAL : tout le reste (code générique, algorithmes, questions sans rapport avec ces produits)\n\n"
    "Tâche : {q}\nDomaine :"
)


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
    probs = json.load(open(CACHE))
    log(f"=== HumanEval × MoE — routage hors-domaine ({len(probs)} problèmes) ===")
    cfg = Config.from_env(); cfg.backend = "mlx"
    llm = make_client(cfg)

    # [1] routage de tous les problèmes (modèle nu, option GENERAL disponible)
    llm.set_adapter(None); llm.cfg.mlx_max_tokens = 8
    routes = []
    for i, p in enumerate(probs):
        task_desc = p["prompt"].strip().splitlines()
        # description compacte : signature + 1re ligne de docstring
        sig = next((l for l in task_desc if l.strip().startswith("def ")), task_desc[0])
        doc = next((l.strip().strip('"\' ') for l in task_desc if '"""' in l or "'''" in l), "")
        raw = (llm.generate(ROUTER_PROMPT.format(q=f"{sig} — {doc}"[:200]), None) or "").upper()
        routed = next((e for e in list(EXPERTS) + ["GENERAL"] if e in raw), "GENERAL")
        routes.append(routed)
        if (i + 1) % 10 == 0:
            log(f"   …routage {i + 1}/{len(probs)}")
    from collections import Counter
    dist = Counter(routes)
    gen_rate = dist.get("GENERAL", 0) / len(probs) * 100
    log(f"   distribution : {dict(dist)} — GENERAL {gen_rate:.0f}%")

    # [2] génération selon la route (groupée par adaptateur pour limiter les swaps)
    llm.cfg.mlx_max_tokens = 512
    ok = 0
    order = ["GENERAL"] + list(EXPERTS)
    for grp in order:
        idxs = [i for i, r in enumerate(routes) if r == grp]
        if not idxs:
            continue
        llm.set_adapter(None if grp == "GENERAL" else EXPERTS[grp])
        for i in idxs:
            p = probs[i]
            prompt = ("Complete the following Python function. Reply with the COMPLETE function "
                      "(including the signature) in a single ```python block.\n\n```python\n"
                      + p["prompt"] + "```")
            out = llm.generate(prompt, None)
            m = _CODE_RE.findall(out or "")
            code = m[-1].strip() if m else (out or "").strip()
            if f"def {p['entry_point']}" not in code:
                code = p["prompt"] + "\n" + code
            ok += run_tests(code, p)
        log(f"   groupe {grp} : {len(idxs)} problème(s) traités (cumul pass={ok})")
        _purge()

    # référence : score du 7B nu depuis le run principal
    base = None
    try:
        for ln in open(os.path.join(_PROJ, "logs", "benchmark_humaneval.log"), encoding="utf-8"):
            if "[7B nu] pass@1" in ln:
                base = int(ln.split("=")[1].split("/")[0].strip())
    except Exception:
        pass

    log("")
    log("=== RÉSULTAT HumanEval × MoE ===")
    log(f"routage GENERAL (attendu ~100%) : {gen_rate:.0f}%  {dict(dist)}")
    log(f"MoE système  pass@1 : {ok}/{len(probs)} ({ok/len(probs)*100:.0f}%)")
    if base is not None:
        log(f"7B nu (réf.) pass@1 : {base}/{len(probs)} ({base/len(probs)*100:.0f}%)")
        d = ok - base
        if gen_rate >= 90 and abs(d) <= 2:
            log(f"🟢 Le MoE ne taxe PAS la capacité générale : le routeur écarte ses experts "
                f"hors-domaine ({gen_rate:.0f}% GENERAL) et le score reste celui du modèle nu ({d:+d}).")
        elif gen_rate < 90:
            log(f"🟠 Le routeur force des experts hors-domaine ({100-gen_rate:.0f}% des cas) — "
                f"impact : {d:+d} pts. L'option GENERAL doit être renforcée.")
        else:
            log(f"🟠 Routage OK mais écart {d:+d} pts — à inspecter.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
