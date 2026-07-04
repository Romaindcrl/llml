#!/usr/bin/env python3
"""Lot 0 — smoke test des harness publics (EvalPlus + lm-eval-harness) sur M1 8-bit.

But : vérifier que les DEUX harness officiels tournent bout-en-bout sur la machine
(génération + scoring), pas de mesurer un score (N=5).

  - HumanEval+ : 5 premiers problèmes, génération greedy via transformers 8-bit,
    scoring par `evalplus.evaluate` (jamais de harness maison — CDC §0).
  - GSM8K : lm_eval --tasks gsm8k --limit 5 (8-shot strict-match).

Usage (pod) : /workspace/venv/bin/python eval/smoke/smoke_harness.py \
    --model Qwen/Qwen2.5-7B-Instruct --out results/raw/lot0_smoke_harness.json
"""

import argparse
import json
import os
import subprocess
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ)


def gen_humaneval_samples(model: str, quant: str, n: int, out_dir: str) -> str:
    """Génère les complétions des n premiers problèmes HumanEval+ (greedy, chat)."""
    import torch
    from evalplus.data import get_human_eval_plus, write_jsonl
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model)
    kwargs = {"device_map": {"": 0}, "torch_dtype": torch.bfloat16}
    if quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    m = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    m.eval()

    problems = get_human_eval_plus()
    task_ids = list(problems)[:n]
    samples = []
    for tid in task_ids:
        prompt = problems[tid]["prompt"]
        msgs = [{"role": "user", "content":
                 "Complete the following Python function. Output ONLY the complete "
                 "function in a ```python code block.\n\n```python\n" + prompt + "\n```"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(m.device)
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=512, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code = _extract_code(raw, prompt)
        samples.append({"task_id": tid, "solution": code})
        print(f"  {tid} genere ({len(code)} chars)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "samples.jsonl")
    write_jsonl(path, samples)
    return path


def _extract_code(raw: str, prompt: str) -> str:
    """Bloc ```python``` si présent, sinon texte brut ; solution = code complet."""
    import re

    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", raw, re.DOTALL)
    code = m[0] if m else raw
    if "def " not in code:
        code = prompt + "\n" + code
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quant", default="8bit")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="results/raw/lot0_smoke_harness.json")
    ap.add_argument("--workdir", default="logs/smoke_harness")
    args = ap.parse_args()

    report: dict = {"model": args.model, "quant": args.quant}

    # --- EvalPlus / HumanEval+
    t0 = time.time()
    smp = gen_humaneval_samples(args.model, args.quant, args.n, args.workdir)
    proc = subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval",
         "--samples", smp],
        capture_output=True, text=True, timeout=1800)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
    report["evalplus"] = {
        "ok": proc.returncode == 0 and "pass@1" in (proc.stdout + proc.stderr),
        "n": args.n, "duration_s": round(time.time() - t0, 1), "output_tail": tail}
    print(tail, flush=True)

    # --- lm-eval / GSM8K (5 items, 8-shot)
    t0 = time.time()
    model_args = f"pretrained={args.model},trust_remote_code=False"
    if args.quant == "8bit":
        model_args += ",load_in_8bit=True"
    else:
        model_args += ",dtype=bfloat16"
    proc = subprocess.run(
        [sys.executable, "-m", "lm_eval", "--model", "hf",
         "--model_args", model_args,
         "--tasks", "gsm8k", "--num_fewshot", "8", "--limit", str(args.n),
         "--batch_size", "1", "--seed", "42",
         "--output_path", os.path.join(args.workdir, "lm_eval_out")],
        capture_output=True, text=True, timeout=3600)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-20:])
    report["lm_eval_gsm8k"] = {
        "ok": proc.returncode == 0 and "gsm8k" in (proc.stdout + proc.stderr),
        "n": args.n, "duration_s": round(time.time() - t0, 1), "output_tail": tail}
    print(tail, flush=True)

    report["ok"] = report["evalplus"]["ok"] and report["lm_eval_gsm8k"]["ok"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "model"},
                     default=str)[:500])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
