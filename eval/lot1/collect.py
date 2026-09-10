#!/usr/bin/env python3
"""Collecte des scores Lot 1+ vers CSV (CDC §0 : aucun chiffre ne vit dans un log).

Sources :
  - lm-eval : fichier results_*.json (--output_path) -> une ligne par (task, metric)
  - evalplus : sortie texte de `evalplus.evaluate` -> pass@1 base et plus

Append dans le CSV cible (créé avec en-tête si absent). Idempotence : une clé
(model, quant, config, benchmark, metric) déjà présente n'est pas dupliquée.

Usage :
  collect.py lmeval  --json <results_*.json> --model M1 --quant 8bit --config C0 --csv scores.csv
  collect.py evalplus --log <evalplus_stdout.txt> --dataset humaneval \
      --model M1 --quant 8bit --config C0 --csv scores.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import time

FIELDS = ["timestamp", "model", "quant", "config", "benchmark", "metric",
          "value", "n", "stderr", "source"]


def _existing_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                keys.add((row["model"], row["quant"], row["config"],
                          row["benchmark"], row["metric"]))
    return keys


def _append(path, rows):
    keys = _existing_keys(path)
    new = [r for r in rows
           if (r["model"], r["quant"], r["config"], r["benchmark"], r["metric"])
           not in keys]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for r in new:
            w.writerow(r)
    print(f"{len(new)} lignes ajoutees ({len(rows) - len(new)} deja presentes) "
          f"-> {path}")


def collect_lmeval(args):
    paths = glob.glob(args.json) if any(c in args.json for c in "*?") else [args.json]
    rows = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for p in sorted(paths):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        nsamples = data.get("n-samples", {})
        for task, metrics in data.get("results", {}).items():
            n = (nsamples.get(task) or {}).get("effective")
            for key, val in metrics.items():
                if not isinstance(val, (int, float)) or key == "alias":
                    continue
                if "_stderr" in key or "," not in key:
                    continue
                name, filt = key.split(",", 1)
                # le filtre fait partie de la métrique (gsm8k: strict-match vs
                # flexible-extract donnent des scores très différents)
                metric = name if filt in ("none", "") else f"{name}[{filt}]"
                stderr = metrics.get(f"{name}_stderr,{filt}")
                rows.append({"timestamp": ts, "model": args.model,
                             "quant": args.quant, "config": args.config,
                             "benchmark": task, "metric": metric,
                             "value": round(float(val), 6), "n": n,
                             "stderr": (round(float(stderr), 6)
                                        if isinstance(stderr, (int, float)) else ""),
                             "source": os.path.basename(p)})
    _append(args.csv, rows)


_EP_RE = re.compile(r"^(humaneval|mbpp)\+?\s*(?:\(([^)]*)\))?\s*$", re.IGNORECASE)


def collect_evalplus(args):
    with open(args.log, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    rows = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    n = {"humaneval": 164, "mbpp": 378}[args.dataset]
    current = None
    for line in lines:
        m = _EP_RE.match(line)
        if m:
            label = (m.group(2) or "").lower()
            # "base + extra tests" contient AUSSI "base" : tester "extra" d'abord
            current = "plus" if ("extra" in label or line.split("(")[0].rstrip().endswith("+")) \
                      else "base"
            continue
        pm = re.match(r"pass@1:\s*([0-9.]+)", line)
        if pm and current:
            bench = f"{args.dataset}_plus" if current == "plus" else f"{args.dataset}_base"
            rows.append({"timestamp": ts, "model": args.model, "quant": args.quant,
                         "config": args.config, "benchmark": bench,
                         "metric": "pass@1", "value": float(pm.group(1)),
                         "n": n, "stderr": "", "source": os.path.basename(args.log)})
            current = None
    _append(args.csv, rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lm = sub.add_parser("lmeval")
    lm.add_argument("--json", required=True)
    ep = sub.add_parser("evalplus")
    ep.add_argument("--log", required=True)
    ep.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    for s in (lm, ep):
        s.add_argument("--model", required=True)
        s.add_argument("--quant", required=True)
        s.add_argument("--config", default="C0")
        s.add_argument("--csv", required=True)
    a = ap.parse_args()
    (collect_lmeval if a.cmd == "lmeval" else collect_evalplus)(a)
