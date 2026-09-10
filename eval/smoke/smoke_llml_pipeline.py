#!/usr/bin/env python3
"""Lot 0 — smoke test du pipeline LLML sur CUDA (backend hf).

Vérifie sur la machine GPU que TOUTES les primitives dont l'éval a besoin tournent :
  1. chargement M1 8-bit + génération greedy de base ;
  2. entraînement /sleep (d2l_hf, subprocess peft) sur un micro-corpus de faits ;
  3. hot-swap d'adapter (set_adapter) + rappel closed-book (d2l.answer_recalled) ;
  4. routeur type MoE (classification par prompt, cf. benchmark_humaneval_moe) ;
  5. retour base nue : la génération de code reste intacte (non-régression locale) ;
  6. latence de swap mesurée.

Usage (sur le pod) :
  /workspace/venv/bin/python eval/smoke/smoke_llml_pipeline.py \
      --model Qwen/Qwen2.5-7B-Instruct --out results/raw/lot0_smoke_pipeline.json
"""

import argparse
import json
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ)

from m0 import d2l, d2l_hf  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402

# Micro-corpus de faits synthétiques (invention pure : aucun risque de contamination
# pretraining — le contrôle "C0 doit échouer" est structurellement garanti).
FACTS = [
    ("Quel est le port par defaut du service Nimbus-Relay ?", "7431"),
    ("Quel fichier configure le quota de Nimbus-Relay ?", "nimbus_quota.toml"),
    ("Qui maintient le module Nimbus-Relay ?", "Ilona Vertesi"),
    ("Quelle commande redemarre Nimbus-Relay ?", "nimctl bounce"),
    ("Quelle est la limite de connexions de Nimbus-Relay ?", "412"),
]

ROUTER_PROMPT = (
    "Tu es un routeur. Domaines disponibles :\n"
    "- NIMBUS : questions factuelles sur le produit Nimbus-Relay (config, quotas, mainteneur)\n"
    "- GENERAL : tout le reste (code generique, algorithmes, questions sans rapport)\n\n"
    "Reponds par UN SEUL mot (NIMBUS ou GENERAL) pour la requete suivante :\n{q}"
)

ROUTER_CASES = [
    ("Write a python function to reverse a linked list.", "GENERAL"),
    ("Quel est le port par defaut du service Nimbus-Relay ?", "NIMBUS"),
    ("Implement quicksort in Python.", "GENERAL"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quant", default="8bit")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--out", default="results/raw/lot0_smoke_pipeline.json")
    args = ap.parse_args()

    report: dict = {"model": args.model, "quant": args.quant, "steps": {}}
    t_all = time.time()

    cfg = Config.from_env()
    cfg.backend = "hf"
    cfg.hf_model_path = args.model
    cfg.hf_quant = args.quant
    cfg.temperature = 0.0
    cfg.mlx_max_tokens = 128
    llm = make_client(cfg)

    # -- 1. génération de base
    t0 = time.time()
    out = llm.generate("Combien font 17 + 25 ? Reponds par le nombre seul.", None)
    base_math_ok = "42" in out
    code_out = llm.generate(
        "Write a Python function `def add(a, b):` that returns a+b. Code only.", None)
    base_code_ok = "def add" in code_out and "return" in code_out
    report["steps"]["base_generation"] = {
        "ok": base_math_ok and base_code_ok, "math": out.strip()[:80],
        "load_plus_gen_s": round(time.time() - t0, 1)}

    # -- 2. dataset + train /sleep (recette serve.py: augmentation ignorée ici — smoke)
    workdir = os.path.join(_PROJ, "logs", "smoke_d2l")
    data_dir = os.path.join(workdir, "data")
    adapter_dir = os.path.join(workdir, "adapter")
    pairs = [(q, a) for q, a in FACTS]
    # petites paraphrases manuelles pour le held-out (pas d'appel LLM : déterminisme)
    aug = pairs + [(q.replace("Quel est", "Donne").replace("Quelle est", "Donne")
                    .replace("Quelle", "Donne la").replace("Qui", "Nomme qui"), a)
                   for q, a in pairs]
    train_pairs, eval_pairs = d2l.split_train_eval(aug, heldout_per_answer=1)
    n = d2l.build_chat_dataset(train_pairs, data_dir, repeat=4,
                               anchors=d2l.ANCHOR_PAIRS, anchor_repeat=2)
    t0 = time.time()
    res = d2l_hf.train_lora(
        args.model, data_dir, adapter_dir,
        iters=args.iters, num_layers=8, learning_rate=5e-5, rank=16,
        quant=args.quant,
        log_file=os.path.join(workdir, "train.log"),
    )
    report["steps"]["sleep_train"] = {
        "ok": res["ok"], "n_train_rows": n, "iters": args.iters,
        "train_loss": res["train_loss"], "val_loss": res["val_loss"],
        "duration_s": round(time.time() - t0, 1), "log_tail": res["log_tail"][-400:]}
    if not res["ok"]:
        report["ok"] = False
        _write(args.out, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    # -- 3. swap + rappel closed-book (held-out d'abord, sinon trivial)
    t0 = time.time()
    llm.set_adapter(adapter_dir)
    swap_in_s = time.time() - t0
    probe = eval_pairs or pairs
    recalled = sum(
        1 for q, a in probe if d2l.answer_recalled(llm.generate(q, None), a))
    report["steps"]["recall_with_adapter"] = {
        "ok": recalled >= max(1, len(probe) // 2),
        "recalled": f"{recalled}/{len(probe)}", "held_out": bool(eval_pairs),
        "first_swap_s": round(swap_in_s, 2)}

    # -- 4. routeur (adapter déchargé pour router : contrat benchmark_humaneval_moe)
    llm.set_adapter(None)
    routed_ok = 0
    routes = []
    for q, expected in ROUTER_CASES:
        raw = (llm.generate(ROUTER_PROMPT.format(q=q), None) or "").upper()
        got = "NIMBUS" if "NIMBUS" in raw else "GENERAL"
        routes.append({"q": q[:50], "expected": expected, "got": got})
        routed_ok += got == expected
    report["steps"]["router"] = {"ok": routed_ok == len(ROUTER_CASES), "routes": routes}

    # -- 5. base intacte après retour (mêmes prompts qu'en 1)
    out2 = llm.generate("Combien font 17 + 25 ? Reponds par le nombre seul.", None)
    code2 = llm.generate(
        "Write a Python function `def add(a, b):` that returns a+b. Code only.", None)
    report["steps"]["base_after_unload"] = {
        "ok": ("42" in out2) and ("def add" in code2),
        "identical_math": out2.strip() == out.strip()}

    # -- 6. latence de swap à chaud (adapter déjà résident)
    lat = []
    for _ in range(5):
        t0 = time.time()
        llm.set_adapter(adapter_dir)
        lat.append(time.time() - t0)
        t0 = time.time()
        llm.set_adapter(None)
        lat.append(time.time() - t0)
    report["steps"]["hot_swap_ms"] = {
        "ok": True, "median_ms": round(sorted(lat)[len(lat) // 2] * 1000, 1)}

    report["ok"] = all(s.get("ok") for s in report["steps"].values())
    report["total_s"] = round(time.time() - t_all, 1)
    _write(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
