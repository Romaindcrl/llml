#!/usr/bin/env python3
"""Scoring live tête-à-tête du Lot 2 (draft C0 vs verify C0+verify).

Score les problèmes DÉJÀ générés (sous-ensemble courant) avec le harness
OFFICIEL EvalPlus, sur exactement le même sous-ensemble pour les deux bras
(même timing, comparaison appariée). Les problèmes non encore générés sont
paddés en solution vide (échec immédiat) puis EXCLUS du décompte — seuls les
task_ids réellement générés dans les DEUX bras comptent.

Sort un JSON tally consommé par le dashboard. N'utilise que evalplus.evaluate
pour le verdict pass/fail (rien de maison) ; ce tally est PROVISOIRE — le
chiffre officiel reste le scoring final sur les 164/378 complets.

Usage : python eval/lot2/live_score.py --dataset humaneval \
    --dir /workspace/results/lot2 --out /workspace/results/lot2/tally_humaneval.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def load_solutions(path):
    d = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    d[r["task_id"]] = r["solution"]
    return d


def score_arm(full_task_ids, solutions, dataset, parallel):
    """Score l'arme via evalplus sur TOUS les task_ids (padding vide pour les
    non générés — evalplus refuse un sous-ensemble), renvoie
    {task_id: {'base': bool, 'plus': bool}}. L'appelant restreint le décompte
    aux task_ids réellement générés."""
    with tempfile.TemporaryDirectory() as td:
        smp = os.path.join(td, "samples.jsonl")
        with open(smp, "w", encoding="utf-8") as f:
            for tid in full_task_ids:
                f.write(json.dumps({"task_id": tid,
                                    "solution": solutions.get(tid, "")}) + "\n")
        subprocess.run(
            [sys.executable, "-m", "evalplus.evaluate", "--dataset", dataset,
             "--samples", smp, "--parallel", str(parallel)],
            capture_output=True, text=True, timeout=1800)
        res_path = smp.replace(".jsonl", "_eval_results.json")
        if not os.path.exists(res_path):
            return {}
        data = json.load(open(res_path, encoding="utf-8"))
        out = {}
        for tid, entries in data.get("eval", {}).items():
            e = entries[0] if isinstance(entries, list) and entries else entries
            out[tid] = {"base": e.get("base_status") == "pass",
                        "plus": e.get("plus_status") == "pass"}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=4)
    a = ap.parse_args()

    if a.dataset == "humaneval":
        from evalplus.data import get_human_eval_plus as get_problems
    else:
        from evalplus.data import get_mbpp_plus as get_problems
    problems = get_problems()
    full_ids = list(problems)
    total = len(full_ids)

    draft = load_solutions(os.path.join(a.dir, f"{a.dataset}_draft_samples.jsonl"))
    ver = load_solutions(os.path.join(a.dir, f"{a.dataset}_verified_samples.jsonl"))
    done = sorted(set(draft) & set(ver))  # scorés seulement si présents dans LES DEUX
    if not done:
        tally = {"dataset": a.dataset, "n_scored": 0, "n_total": total}
        json.dump(tally, open(a.out, "w"), indent=2)
        print(json.dumps(tally))
        return

    ds_full = score_arm(full_ids, draft, a.dataset, a.parallel)
    vs_full = score_arm(full_ids, ver, a.dataset, a.parallel)
    ds = {t: ds_full[t] for t in done if t in ds_full}
    vs = {t: vs_full[t] for t in done if t in vs_full}

    def tally_arm(scores):
        base_pass = sum(1 for t in done if scores.get(t, {}).get("base"))
        plus_pass = sum(1 for t in done if scores.get(t, {}).get("plus"))
        return {"base_pass": base_pass, "plus_pass": plus_pass,
                "base_fail": len(done) - base_pass, "plus_fail": len(done) - plus_pass}

    # problèmes où verify a changé le verdict (plus)
    flips_win = [t for t in done
                 if not ds.get(t, {}).get("plus") and vs.get(t, {}).get("plus")]
    flips_lose = [t for t in done
                  if ds.get(t, {}).get("plus") and not vs.get(t, {}).get("plus")]

    tally = {
        "dataset": a.dataset, "n_scored": len(done), "n_total": total,
        "baseline": tally_arm(ds), "llml": tally_arm(vs),
        "verify_wins": flips_win, "verify_regressions": flips_lose,
    }
    json.dump(tally, open(a.out, "w"), indent=2)
    print(json.dumps(tally))


if __name__ == "__main__":
    main()
