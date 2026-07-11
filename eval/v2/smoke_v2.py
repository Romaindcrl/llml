#!/usr/bin/env python3
"""v2 Lot 0 — smoke test (CDC v2 Lot 0) :
  1. chargement M1 (Qwen2.5-7B-Instruct 8-bit),
  2. swap adapter (micro-LoRA 10 iters entraîné sur place, puis load/unload),
  3. 3 checks AST sur un repo témoin (le code m0/ de LLML fait office de témoin :
     on teste que le MOTEUR de checks tourne, pas une adhérence).

Usage (pod) :
  /workspace/venv/bin/python eval/v2/smoke_v2.py --outdir /workspace/results/v2
"""
import argparse
import glob
import json
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "eval", "v2", "checks"))

from m0 import d2l, d2l_hf                    # noqa: E402
from m0.config import Config                  # noqa: E402
from m0.llm import make_client                # noqa: E402
from ast_lib import DEMO_RULES, run_rules     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quant", default="8bit")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    report = {"model": a.model, "quant": a.quant, "steps": {}}
    t0 = time.time()

    cfg = Config.from_env()
    cfg.backend = "hf"
    cfg.hf_model_path = a.model
    cfg.hf_quant = a.quant
    cfg.temperature = 0.0
    llm = make_client(cfg)

    # 1) chargement + génération base
    llm.cfg.mlx_max_tokens = 24
    out = llm.generate("Say OK.", None)
    report["steps"]["load_generate"] = {"ok": bool(out and out.strip()), "out": (out or "")[:40]}
    print(f"[1/3] load+generate: {report['steps']['load_generate']}", flush=True)

    # 2) micro-LoRA (10 iters, 3 paires) puis swap load/unload
    work = os.path.join(a.outdir, "smoke_adapter")
    data_dir = os.path.join(work, "data")
    pairs = [("What is the smoke token?", "ZEPHYR-42"),
             ("Repeat the smoke token.", "ZEPHYR-42"),
             ("Smoke token, please.", "ZEPHYR-42")]
    d2l.build_chat_dataset(pairs, data_dir, repeat=2, anchors=None)
    llm.unload()
    res = d2l_hf.train_lora(a.model, data_dir, os.path.join(work, "adapter"),
                            iters=10, num_layers=4, learning_rate=5e-5, rank=8,
                            quant=a.quant, log_file=os.path.join(work, "train.log"))
    swap_ok = False
    if res.get("ok"):
        llm.set_adapter(os.path.join(work, "adapter"))
        _ = llm.generate("Say OK.", None)
        llm.set_adapter(None)
        _ = llm.generate("Say OK.", None)
        swap_ok = True
    report["steps"]["adapter_swap"] = {"train_ok": bool(res.get("ok")), "swap_ok": swap_ok,
                                       "train_loss": res.get("train_loss")}
    print(f"[2/3] micro-train+swap: {report['steps']['adapter_swap']}", flush=True)

    # 3) 3 checks AST sur repo témoin (m0/)
    files = sorted(glob.glob(os.path.join(_PROJ, "m0", "*.py")))[:5]
    checks = []
    for f in files:
        src = open(f, encoding="utf-8").read()
        r = run_rules(src, DEMO_RULES, path=f)
        checks.append({"file": os.path.basename(f), "adherence": r["adherence"],
                       "n_applicable": r["n_applicable"]})
    ok3 = len(checks) >= 3 and all(c["adherence"] is not None for c in checks)
    report["steps"]["ast_checks"] = {"ok": ok3, "files": checks}
    print(f"[3/3] AST checks sur {len(checks)} fichiers témoins: ok={ok3}", flush=True)

    report["elapsed_s"] = round(time.time() - t0)
    report["ok"] = all([report["steps"]["load_generate"]["ok"],
                        report["steps"]["adapter_swap"]["train_ok"],
                        report["steps"]["adapter_swap"]["swap_ok"], ok3])
    with open(os.path.join(a.outdir, "smoke_v2.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(("SMOKE_V2_OK" if report["ok"] else "SMOKE_V2_FAIL") + f" [{report['elapsed_s']}s]",
          flush=True)


if __name__ == "__main__":
    main()
