#!/usr/bin/env python3
"""Lot 0 — smoke chargement M1–M4 (+ test PEFT 10 steps sur Gemma 2, CDC §2.4).

Pour chaque modèle de la matrice §2.1 : chargement 8-bit + une génération greedy
courte. M3 (Llama 3.1) et M4 (Gemma 2) sont gated sur HF : sautés proprement si
HF_TOKEN absent (statut "skipped_gated", à relancer une fois le token fourni).

Pour M4, en plus : entraînement PEFT de 10 pas via m0.d2l_hf pour valider les
target_modules et la compatibilité (embeddings liés + soft-capping de Gemma 2).

Usage (pod) : /workspace/venv/bin/python eval/smoke/smoke_models.py \
    --out results/raw/lot0_smoke_models.json [--only M1,M2]
"""

import argparse
import gc
import json
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ)

MODELS = {
    "M1": ("Qwen/Qwen2.5-7B-Instruct", False),
    "M2": ("Qwen/Qwen2.5-14B-Instruct", False),
    "M3": ("meta-llama/Llama-3.1-8B-Instruct", True),
    "M4": ("google/gemma-2-9b-it", True),
}


def try_load_generate(repo: str) -> dict:
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo)
    mcfg = AutoConfig.from_pretrained(repo)
    kwargs = {"device_map": {"": 0}, "torch_dtype": torch.bfloat16,
              "quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    if getattr(mcfg, "model_type", "") == "gemma2":
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    model.eval()
    msgs = [{"role": "user", "content": "What is 17 + 25? Answer with the number only."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    revision = getattr(model.config, "_commit_hash", None)
    mem = torch.cuda.max_memory_allocated() / 1e9
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {"ok": "42" in ans, "answer": ans.strip()[:60],
            "hf_revision": revision, "vram_peak_gb": round(mem, 1),
            "load_gen_s": round(time.time() - t0, 1)}


def gemma_peft_10steps(repo: str) -> dict:
    """10 pas d'entraînement LoRA sur M4 : valide target_modules + PEFT (CDC §2.4)."""
    from m0 import d2l, d2l_hf

    workdir = os.path.join(_PROJ, "logs", "smoke_gemma")
    data_dir = os.path.join(workdir, "data")
    pairs = [("Quel est le port du service Nimbus-Relay ?", "7431"),
             ("Qui maintient Nimbus-Relay ?", "Ilona Vertesi")]
    d2l.build_chat_dataset(pairs, data_dir, repeat=4,
                           anchors=d2l.ANCHOR_PAIRS[:6], anchor_repeat=1)
    res = d2l_hf.train_lora(repo, data_dir, os.path.join(workdir, "adapter"),
                            iters=10, num_layers=8, learning_rate=5e-5, rank=16,
                            log_file=os.path.join(workdir, "train.log"))
    return {"ok": res["ok"], "train_loss": res["train_loss"],
            "returncode": res["returncode"], "log_tail": res["log_tail"][-300:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw/lot0_smoke_models.json")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    has_token = bool(os.environ.get("HF_TOKEN"))

    report: dict = {}
    for mid, (repo, gated) in MODELS.items():
        if only and mid not in only:
            continue
        if gated and not has_token:
            report[mid] = {"ok": None, "status": "skipped_gated",
                           "note": "HF_TOKEN absent — a relancer avec le token"}
            print(f"{mid} ({repo}): SKIP (gated, pas de HF_TOKEN)", flush=True)
            continue
        print(f"{mid} ({repo}): chargement 8-bit…", flush=True)
        try:
            report[mid] = try_load_generate(repo)
        except Exception as e:  # noqa: BLE001
            report[mid] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
        print(f"  -> {report[mid]}", flush=True)

    if (not only or "M4" in only) and has_token and report.get("M4", {}).get("ok"):
        print("M4 : test PEFT 10 steps (target_modules)…", flush=True)
        try:
            report["M4_peft_10steps"] = gemma_peft_10steps(MODELS["M4"][0])
        except Exception as e:  # noqa: BLE001
            report["M4_peft_10steps"] = {"ok": False,
                                         "error": f"{type(e).__name__}: {e}"[:400]}
        print(f"  -> {report['M4_peft_10steps']}", flush=True)

    done = [v for v in report.values() if v.get("ok") is not None]
    report["ok"] = bool(done) and all(v["ok"] for v in done)
    report["skipped_gated"] = [k for k, v in report.items()
                               if isinstance(v, dict) and v.get("ok") is None]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
