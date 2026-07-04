#!/usr/bin/env python3
"""Lot 1+ — génération des complétions HumanEval+/MBPP+ pour scoring EvalPlus.

Rôle strictement limité à la GÉNÉRATION (greedy, template chat du tokenizer,
8-bit bnb ou bf16) ; le scoring est 100% officiel (`evalplus.evaluate`).
Décision journalisée dans AGENTS.md : evalplus 0.3.1 ne charge pas de modèle
en 8-bit nativement ; un run de validation croisée wrapper-vs-natif est fait
en bf16 au Lot 1 (l'écart doit être nul ou quasi-nul, sinon STOP).

Reprise sur incident : les task_ids déjà présents dans le fichier de sortie
sont sautés (checkpointing par item, CDC §7).

Usage :
  python eval/lot1/gen_evalplus.py --model Qwen/Qwen2.5-7B-Instruct \
      --dataset humaneval --quant 8bit --out /workspace/results/he_m1_8bit.jsonl
"""

import argparse
import json
import os
import re
import time

PROMPT_TMPL = (
    "Complete the following Python function. Output ONLY the complete "
    "function in a ```python code block.\n\n```python\n{prompt}\n```"
)


def extract_code(raw: str, prompt: str) -> str:
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", raw, re.DOTALL)
    code = m[0] if m else raw
    if "def " not in code:
        code = prompt + "\n" + code
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--quant", choices=["8bit", "bf16"], default="8bit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    args = ap.parse_args()

    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    if args.dataset == "humaneval":
        from evalplus.data import get_human_eval_plus as get_problems
    else:
        from evalplus.data import get_mbpp_plus as get_problems

    problems = get_problems()
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["task_id"])
    todo = [tid for tid in problems if tid not in done]
    print(f"{args.dataset}: {len(problems)} problemes, {len(done)} deja faits, "
          f"{len(todo)} a generer", flush=True)
    if not todo:
        return 0

    tok = AutoTokenizer.from_pretrained(args.model)
    mcfg = AutoConfig.from_pretrained(args.model)
    kwargs = {"device_map": {"": 0}, "torch_dtype": torch.bfloat16,
              "low_cpu_mem_usage": True}
    if getattr(mcfg, "model_type", "") == "gemma2":
        kwargs["attn_implementation"] = "eager"
    if args.quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    revision = getattr(model.config, "_commit_hash", None)
    print(f"revision HF: {revision}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as fout:
        for i, tid in enumerate(todo, 1):
            prompt = problems[tid]["prompt"]
            msgs = [{"role": "user", "content": PROMPT_TMPL.format(prompt=prompt)}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
            raw = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            fout.write(json.dumps({"task_id": tid,
                                   "solution": extract_code(raw, prompt)}) + "\n")
            fout.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} ({el/i:.1f} s/item, ETA "
                      f"{el / i * (len(todo) - i) / 60:.0f} min)", flush=True)
    meta_path = args.out + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "revision": revision, "quant": args.quant,
                   "dataset": args.dataset, "greedy": True,
                   "prompt_template": PROMPT_TMPL,
                   "max_new_tokens": args.max_new_tokens}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
