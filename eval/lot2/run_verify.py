#!/usr/bin/env python3
"""Lot 2 — Claim C : boucle verify de LLML sur EvalPlus complet (M1).

Deux bras PAIRÉS depuis le même draft (delta propre, item par item) :
  - C0        : draft one-shot (prompt identique au script interne) ;
  - C0+verify : draft → exécuter les exemples DOCUMENTÉS du prompt → réparer
                (≤2 essais, réparation adoptée uniquement si elle passe les
                exemples) — réplication fidèle de benchmark_humaneval_repair.py
                (même _DOCTEST_RE, ≤5 exemples, même PRELUDE d'exécution, mêmes
                prompts draft/fix, greedy).

Régime « vérité terrain offerte » (CDC §4.4) : la boucle ne voit QUE les
exemples présents dans l'énoncé (doctests HumanEval, asserts MBPP), jamais les
tests cachés. Le scoring des DEUX bras est 100% officiel (evalplus.evaluate).
Sous greedy, un 2e essai de réparation sur le même échec renvoie le même prompt
→ effectivement 1 réparation distincte (documenté, comportement du script
interne conservé).

Usage (pod) : /workspace/venv/bin/python eval/lot2/run_verify.py \
    --model Qwen/Qwen2.5-7B-Instruct --dataset humaneval --quant 8bit \
    --outdir /workspace/results/lot2
Sorties : <outdir>/<ds>_draft_samples.jsonl, <ds>_verified_samples.jsonl,
          <ds>_verify_meta.jsonl (par item : exemples, réparations, adoptions).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

# Constantes RÉPLIQUÉES du harness interne (benchmark_humaneval{,_repair}.py)
PRELUDE = "from typing import List, Dict, Tuple, Optional, Any\nimport math\nimport re\n\n"
_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_DOCTEST_RE = re.compile(r">>>\s*(.+?)\n\s*([^\s>][^\n]*)", re.MULTILINE)
_ASSERT_RE = re.compile(r"^\s*(assert\s+.+)$", re.MULTILINE)
MAX_REPAIRS = 2

DRAFT_PROMPT = ("Complete the following Python function. Reply with the COMPLETE function "
                "(including the signature) in a single ```python block.\n\n```python\n{prompt}```")
FIX_PROMPT = ("Your implementation below fails a documented example.\n\n```python\n{code}\n```\n\n"
              "Failure: {err}\n\nFix the function. Reply with the COMPLETE corrected "
              "function in a single ```python block.")


def extract_code(out: str, prompt: str, entry_point: str) -> str:
    m = _CODE_RE.findall(out or "")
    code = m[-1].strip() if m else (out or "").strip()
    if f"def {entry_point}" not in code:
        code = prompt + "\n" + code
    return code


def documented_examples(prompt: str, dataset: str):
    """Exemples offerts par l'énoncé : doctests (HumanEval) ou asserts (MBPP)."""
    if dataset == "humaneval":
        ex = []
        for call, expected in _DOCTEST_RE.findall(prompt):
            call, expected = call.strip(), expected.strip()
            if call and expected and not expected.startswith(">>>"):
                ex.append(("doctest", call, expected))
        return ex[:5]
    # mbpp(+) : l'énoncé evalplus contient des asserts d'exemple
    return [("assert", a.strip(), "") for a in _ASSERT_RE.findall(prompt)][:5]


def check_examples(code: str, examples):
    for kind, a, b in examples:
        if kind == "doctest":
            src = (PRELUDE + code +
                   f"\n\n_r = {a}\n_e = {b}\nassert _r == _e, f'got {{_r!r}}, expected {{_e!r}}'\nprint('OK')\n")
        else:
            src = PRELUDE + code + f"\n\n{a}\nprint('OK')\n"
        try:
            r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                               text=True, timeout=10)
            if r.returncode != 0 or "OK" not in r.stdout:
                err = (r.stderr.strip().splitlines() or ["error"])[-1][:160]
                return False, f"`{a}` a échoué : {err}"
        except Exception as e:  # noqa: BLE001
            return False, f"`{a}` : {type(e).__name__}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--quant", choices=["8bit", "bf16"], default="8bit")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    args = ap.parse_args()

    _PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _PROJ)
    from m0.config import Config
    from m0.llm import make_client

    if args.dataset == "humaneval":
        from evalplus.data import get_human_eval_plus as get_problems
    else:
        from evalplus.data import get_mbpp_plus as get_problems
    problems = get_problems()

    cfg = Config.from_env()
    cfg.backend = "hf"
    cfg.hf_model_path = args.model
    cfg.hf_quant = args.quant
    cfg.temperature = 0.0
    cfg.mlx_max_tokens = args.max_new_tokens
    llm = make_client(cfg)
    llm.set_adapter(None)

    os.makedirs(args.outdir, exist_ok=True)
    p_draft = os.path.join(args.outdir, f"{args.dataset}_draft_samples.jsonl")
    p_ver = os.path.join(args.outdir, f"{args.dataset}_verified_samples.jsonl")
    p_meta = os.path.join(args.outdir, f"{args.dataset}_verify_meta.jsonl")
    done = set()
    if os.path.exists(p_meta):
        with open(p_meta, encoding="utf-8") as f:
            done = {json.loads(l)["task_id"] for l in f if l.strip()}
    todo = [t for t in problems if t not in done]
    print(f"{args.dataset}: {len(problems)} items, {len(done)} faits, {len(todo)} à faire",
          flush=True)

    t0 = time.time()
    fd = open(p_draft, "a", encoding="utf-8")
    fv = open(p_ver, "a", encoding="utf-8")
    fm = open(p_meta, "a", encoding="utf-8")
    for i, tid in enumerate(todo, 1):
        prob = problems[tid]
        prompt, entry = prob["prompt"], prob["entry_point"]
        draft = extract_code(llm.generate(DRAFT_PROMPT.format(prompt=prompt), None),
                             prompt, entry)
        code = draft
        examples = documented_examples(prompt, args.dataset)
        meta = {"task_id": tid, "n_examples": len(examples), "repairs_tried": 0,
                "repair_adopted": False}
        if examples:
            good, err = check_examples(code, examples)
            meta["draft_passes_examples"] = good
            tries = 0
            while not good and tries < MAX_REPAIRS:
                tries += 1
                meta["repairs_tried"] = tries
                code2 = extract_code(
                    llm.generate(FIX_PROMPT.format(code=code, err=err), None),
                    prompt, entry)
                good2, err2 = check_examples(code2, examples)
                if good2:
                    code = code2
                    meta["repair_adopted"] = True
                good, err = good2, err2
        fd.write(json.dumps({"task_id": tid, "solution": draft}) + "\n")
        fv.write(json.dumps({"task_id": tid, "solution": code}) + "\n")
        fm.write(json.dumps(meta) + "\n")
        for f in (fd, fv, fm):
            f.flush()
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)} ({el/i:.1f} s/item, ETA "
                  f"{el/i*(len(todo)-i)/60:.0f} min)", flush=True)
    print("VERIFY_GEN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
