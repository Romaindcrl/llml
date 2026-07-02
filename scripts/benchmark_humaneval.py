"""HumanEval (standard open source) — certifie le coût des adaptateurs sur la capacité générale.

Question : porter un LoRA-mémoire (spec projet) ou un LoRA-code coûte-t-il des points de
capacité de programmation GÉNÉRALE ? Juge neutre : OpenAI HumanEval (via HF datasets-server),
pass@1, exécution réelle des tests unitaires officiels.
Bras : 7B nu · 7B + LoRA-projet (la mémoire qu'un client porterait) · 7B + LoRA-code (distill C2).
Live : tail -f logs/benchmark_humaneval.log
"""

from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_humaneval.log")
CACHE = os.path.join(_PROJ, "logs", "humaneval_40.json")
N = 40
ARMS = [
    ("7B nu", None),
    ("7B + LoRA-projet", os.path.join(_PROJ, "models", "lora", "project")),
    ("7B + LoRA-code", os.path.join(_PROJ, "models", "lora", "code2_distill")),
]
PRELUDE = "from typing import List, Dict, Tuple, Optional, Any\nimport math\nimport re\n\n"
_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
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


def load_problems():
    if os.path.isfile(CACHE):
        return json.load(open(CACHE))
    import httpx
    url = ("https://datasets-server.huggingface.co/rows?dataset=openai%2Fopenai_humaneval"
           f"&config=openai_humaneval&split=test&offset=0&length={N}")
    r = httpx.get(url, timeout=60.0, headers={"User-Agent": "llml-he/0.1"})
    r.raise_for_status()
    rows = [x["row"] for x in r.json()["rows"]]
    json.dump(rows, open(CACHE, "w"))
    return rows


def run_tests(code, prob):
    src = PRELUDE + code + "\n\n" + prob["test"] + f"\n\ncheck({prob['entry_point']})\nprint('PASS')\n"
    try:
        r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=12)
        return r.returncode == 0 and "PASS" in r.stdout
    except Exception:
        return False


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    probs = load_problems()
    log(f"=== HumanEval pass@1 — {len(probs)} problèmes, exécution officielle ===")
    cfg = Config.from_env(); cfg.backend = "mlx"
    llm = make_client(cfg); llm.cfg.mlx_max_tokens = 512
    R = {}
    for name, adapter in ARMS:
        if adapter and not os.path.isfile(os.path.join(adapter, "adapters.safetensors")):
            log(f"[{name}] adaptateur absent — sauté"); continue
        llm.set_adapter(adapter)
        ok = 0
        for i, p in enumerate(probs):
            prompt = ("Complete the following Python function. Reply with the COMPLETE function "
                      "(including the signature) in a single ```python block.\n\n```python\n"
                      + p["prompt"] + "```")
            out = llm.generate(prompt, None)
            m = _CODE_RE.findall(out or "")
            code = m[-1].strip() if m else (out or "").strip()
            if f"def {p['entry_point']}" not in code:      # complétion sans signature
                code = p["prompt"] + "\n" + code
            ok += run_tests(code, p)
            if (i + 1) % 10 == 0:
                log(f"   …[{name}] {i + 1}/{len(probs)} (pass={ok})")
        R[name] = ok
        log(f"[{name}] pass@1 = {ok}/{len(probs)} ({ok/len(probs)*100:.0f}%)")
        _purge()

    log("")
    log("=== RÉSULTAT HumanEval (capacité de programmation générale) ===")
    base = R.get("7B nu", 0)
    for name, _ in ARMS:
        if name in R:
            d = R[name] - base
            log(f"{name:20s} | {R[name]}/{len(probs)} ({R[name]/len(probs)*100:3.0f}%) | Δ vs nu : {d:+d}")
    log("")
    proj = R.get("7B + LoRA-projet")
    if proj is not None:
        if abs(proj - base) <= 2:
            log(f"🟢 Porter la mémoire-projet ne coûte pas de capacité générale mesurable "
                f"({base}→{proj}/{len(probs)}).")
        elif proj < base:
            log(f"🟠 La mémoire-projet coûte {base - proj} pts de capacité générale — à documenter.")
        else:
            log(f"🟢 {proj - base:+d} pts (bruit probable).")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
