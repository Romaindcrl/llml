"""HumanEval × LLML-vérification — le 2-étapes appliqué au code (draft → exécuter → réparer).

La mémoire ne peut PAS améliorer HumanEval (tâches auto-contenues, rien à savoir — mesuré).
Mais le pilier VÉRIFICATION de LLML le peut, honnêtement : les docstrings contiennent des
exemples d'usage (doctests) ; le système génère, EXÉCUTE ces exemples (vérité-terrain disponible,
pas les tests cachés), et répare en cas d'échec (2 itérations max). Le score final est jugé sur
les tests OFFICIELS cachés, comme les autres bras. C'est le draft→vérif→fix de LLML avec
l'exécution comme vérificateur (cf. Reflexion ; notre §8 : self-repair 5→13/15).
Bras : 7B nu + boucle de vérification. Référence : 7B nu one-shot = 37/40 (92%).
Live : tail -f logs/benchmark_humaneval_repair.log
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

from scripts.benchmark_humaneval import CACHE, PRELUDE, run_tests  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_humaneval_repair.log")
_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_DOCTEST_RE = re.compile(r">>>\s*(.+?)\n\s*([^\s>][^\n]*)", re.MULTILINE)
MAX_REPAIRS = 2
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def extract_code(out, prob):
    m = _CODE_RE.findall(out or "")
    code = m[-1].strip() if m else (out or "").strip()
    if f"def {prob['entry_point']}" not in code:
        code = prob["prompt"] + "\n" + code
    return code


def docstring_examples(prob):
    """Extrait les doctests (appel, résultat attendu) de la docstring — vérité-terrain DISPONIBLE."""
    ex = []
    for call, expected in _DOCTEST_RE.findall(prob["prompt"]):
        call, expected = call.strip(), expected.strip()
        if not call or not expected or expected.startswith(">>>"):
            continue
        ex.append((call, expected))
    return ex[:5]


def check_examples(code, examples):
    """Exécute les exemples ; renvoie (tous_ok, premier_échec_descriptif)."""
    for call, expected in examples:
        src = (PRELUDE + code +
               f"\n\n_r = {call}\n_e = {expected}\nassert _r == _e, f'got {{_r!r}}, expected {{_e!r}}'\nprint('OK')\n")
        try:
            r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=10)
            if r.returncode != 0 or "OK" not in r.stdout:
                err = (r.stderr.strip().splitlines() or ["error"])[-1][:160]
                return False, f"`{call}` a échoué : {err}"
        except Exception as e:
            return False, f"`{call}` : {type(e).__name__}"
    return True, ""


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    probs = json.load(open(CACHE))
    log(f"=== HumanEval × vérification LLML (draft → exécuter les doctests → réparer ≤{MAX_REPAIRS}×) ===")
    cfg = Config.from_env(); cfg.backend = "mlx"
    llm = make_client(cfg); llm.set_adapter(None); llm.cfg.mlx_max_tokens = 512

    ok = repaired = no_ex = 0
    for i, p in enumerate(probs):
        base_prompt = ("Complete the following Python function. Reply with the COMPLETE function "
                       "(including the signature) in a single ```python block.\n\n```python\n"
                       + p["prompt"] + "```")
        code = extract_code(llm.generate(base_prompt, None), p)
        examples = docstring_examples(p)
        if not examples:
            no_ex += 1
        else:
            good, err = check_examples(code, examples)
            tries = 0
            while not good and tries < MAX_REPAIRS:
                tries += 1
                fix_prompt = (f"Your implementation below fails a documented example.\n\n```python\n{code}\n```\n\n"
                              f"Failure: {err}\n\nFix the function. Reply with the COMPLETE corrected "
                              "function in a single ```python block.")
                code2 = extract_code(llm.generate(fix_prompt, None), p)
                good2, err2 = check_examples(code2, examples)
                if good2 or tries == MAX_REPAIRS:
                    # n'adopte la réparation que si elle passe les exemples (sinon garde le draft)
                    if good2:
                        code = code2; repaired += 1
                    good, err = good2, err2
        ok += run_tests(code, p)          # jugé sur les tests OFFICIELS cachés
        if (i + 1) % 10 == 0:
            log(f"   …{i + 1}/{len(probs)} (pass={ok}, réparations adoptées={repaired})")

    n = len(probs)
    log("")
    log("=== RÉSULTAT — HumanEval pass@1 (tests officiels) ===")
    log(f"7B nu one-shot (réf. run principal) : 37/40 (92%)")
    log(f"7B + vérification LLML              : {ok}/{n} ({ok/n*100:.0f}%)  "
        f"[réparations adoptées : {repaired} ; sans doctest : {no_ex}]")
    d = ok - 37
    if d > 0:
        log(f"🟢 +{d} pts par la boucle de VÉRIFICATION (pas par la mémoire) : le système rend le "
            "modèle plus fiable — c'est le gain honnête et reproductible.")
    elif d == 0:
        log("🟠 pas de gain net — les échecs restants ne sont pas récupérables par les doctests.")
    else:
        log(f"🔴 {d} pts — la réparation a dégradé ; à inspecter.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
