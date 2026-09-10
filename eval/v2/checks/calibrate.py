#!/usr/bin/env python3
"""v2 — calibration des règles d'adhérence sur les ORIGINAUX (CDC §2.3, gel Lot 1).

Auto-contrôle d'honnêteté : les règles sont exécutées sur les extraits ORIGINAUX
du repo — RECONSTRUITS exactement comme à l'éval :

    textwrap.dedent(header + "\\n" + expected_body)      (tâches gelées, mode officiel)

Un original est conforme par définition : toute règle qui échoue sur > 20 % des
reconstructions où elle s'applique est buguée ou trop stricte → corrigée ou
supprimée (journal : CALIBRATION.md).

Deux modes :
  * officiel   : les 30 tâches gelées de eval/v2/tasks/<repo>.jsonl ;
  * --train-sample N : N fonctions supplémentaires extraites des fichiers du
    split TRAIN (seed 42), même forme (fonction dédentée au niveau colonne 0)
    — volume statistique en plus des 30 tâches, jamais utilisé pour l'éval.

Usage :
  python3 eval/v2/checks/calibrate.py twisted [--train-sample 200] [-v]
  python3 eval/v2/checks/calibrate.py all
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import random
import re
import sys
import textwrap
from pathlib import Path

CHECKS = Path(__file__).resolve().parent
V2 = CHECKS.parent
DEFAULT_REPOS = "/tmp/claude-0/-home-user-llml/f9042f81-6531-534f-9c23-cc2ce7114935/scratchpad/repos"
FAIL_THRESHOLD = 0.80  # une règle < 80 % de pass sur les originaux applicables = rejetée

REPO_LANG = {"twisted": "python", "FreeRTOS-Kernel": "c", "tigerbeetle": "zig"}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- extraction de fonctions du TRAIN (mode supplémentaire) ---------------------------

def _extract_python(src: str):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    nested = set()
    for f in ast.walk(tree):
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(f):
                if sub is not f and isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nested.add(id(sub))
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and id(n) not in nested:
            seg = ast.get_source_segment(src, n)
            if seg and seg.count("\n") >= 5:
                out.append(textwrap.dedent(seg))
    return out


def _extract_c(src: str):
    lines = src.split("\n")
    out = []
    i = 0
    while i < len(lines):
        if lines[i].rstrip() == "{":
            j = i - 1
            sig = []
            while j >= 0 and lines[j].strip() and not lines[j].rstrip().endswith((";", "}", "\\")) \
                    and not lines[j].lstrip().startswith(("#", "/*", "*", "//")):
                sig.insert(0, lines[j])
                if lines[j] and not lines[j][0].isspace():
                    break
                j -= 1
            if sig and "(" in sig[0] and not sig[0][0].isspace() \
                    and not re.match(r"\s*(?:typedef|struct|union|enum)\b", sig[0]):
                depth, k = 0, i
                while k < len(lines):
                    depth += lines[k].count("{") - lines[k].count("}")
                    if depth == 0 and k > i:
                        break
                    if depth == 0 and k == i and lines[k].count("}"):
                        break
                    k += 1
                if k < len(lines) and k - i >= 4:
                    out.append("\n".join(sig + lines[i:k + 1]))
                i = k
        i += 1
    return out


def _extract_zig(src: str):
    # réutilise l'extracteur des règles tigerbeetle (même équilibrage d'accolades)
    tb = _load(CHECKS / "tigerbeetle" / "rules.py", "_tb_rules_for_extract")
    code = tb._code(src)
    raw_lines = src.split("\n")
    out = []
    for name, s, e, _body in tb._functions(code):
        if e - s >= 5:
            out.append(textwrap.dedent("\n".join(raw_lines[s:e + 1])))
    return out


_EXTRACTORS = {"python": _extract_python, "c": _extract_c, "zig": _extract_zig}
_GLOBS = {"python": "*.py", "c": "*.c", "zig": "*.zig"}


def train_extracts(repo: str, repos_dir: Path, n: int, seed: int = 42):
    split = json.loads((V2 / "splits" / f"{repo}.json").read_text())
    root = repos_dir / repo
    rng = random.Random(seed)
    files = list(split["train"])
    rng.shuffle(files)
    lang = REPO_LANG[repo]
    out = []
    for rel in files:
        p = root / rel
        if not p.exists():
            continue
        try:
            fns = _EXTRACTORS[lang](p.read_text(errors="replace"))
        except Exception:
            continue
        rng.shuffle(fns)
        for fn in fns[:3]:
            out.append((rel, fn))
        if len(out) >= n:
            break
    return out[:n]


# --- exécution -------------------------------------------------------------------------

def run(repo: str, repos_dir: Path, train_n: int, verbose: bool):
    ast_lib = _load(CHECKS / "ast_lib.py", "ast_lib")
    rules_mod = _load(CHECKS / repo / "rules.py", f"rules_{repo.replace('-', '_')}")
    rules = rules_mod.RULES
    print(f"\n=== {repo} — {len(rules)} règles ===")

    datasets = []
    tasks_path = V2 / "tasks" / f"{repo}.jsonl"
    rows = [json.loads(l) for l in tasks_path.read_text().splitlines() if l.strip()]
    recon = [(r["task_id"], r["file"],
              textwrap.dedent(r["header"] + "\n" + r["expected_body"])) for r in rows]
    datasets.append((f"tâches gelées ({len(recon)} reconstructions)",
                     [(tid, f, src) for tid, f, src in recon]))
    if train_n:
        extra = train_extracts(repo, repos_dir, train_n)
        datasets.append((f"train supplémentaire ({len(extra)} fonctions, seed 42)",
                         [(f"train#{i:03d}", rel, src) for i, (rel, src) in enumerate(extra)]))

    verdicts = {}
    for label, samples in datasets:
        stats = {r["id"]: [0, 0] for r in rules}  # applicable, passed
        adher = []
        fails = {r["id"]: [] for r in rules}
        for sid, relpath, src in samples:
            res = ast_lib.run_rules(src, rules, path=relpath)
            if res["adherence"] is not None:
                adher.append(res["adherence"])
            for it in res["results"]:
                if it["applicable"]:
                    stats[it["id"]][0] += 1
                    if it["passed"]:
                        stats[it["id"]][1] += 1
                    else:
                        fails[it["id"]].append((sid, it["detail"]))
        print(f"\n-- {label} --")
        print(f"{'règle':46s} {'appl.':>5s} {'pass':>5s} {'taux':>7s}")
        for r in rules:
            a, p = stats[r["id"]]
            rate = p / a if a else None
            mark = ""
            if a and rate < FAIL_THRESHOLD:
                mark = "  ✗ < 80 %"
            elif a == 0:
                mark = "  (jamais déclenchée)"
            print(f"{r['id']:46s} {a:5d} {p:5d} {f'{rate:7.1%}' if a else '    n/a'}{mark}")
            if label.startswith("tâches"):
                verdicts[r["id"]] = (a, p, rate)
            if verbose and fails[r["id"]]:
                for sid, det in fails[r["id"]][:4]:
                    print(f"    ✗ {sid}: {det}")
        micro_a = sum(v[0] for v in stats.values())
        micro_p = sum(v[1] for v in stats.values())
        print(f"{'AGRÉGÉ (micro)':46s} {micro_a:5d} {micro_p:5d} {micro_p/max(1,micro_a):7.1%}")
        print(f"adhérence moyenne par extrait : {sum(adher)/max(1,len(adher)):.1%} "
              f"(n={len(adher)})")
    bad = [rid for rid, (a, p, rate) in verdicts.items() if a and rate < FAIL_THRESHOLD]
    if bad:
        print(f"\n⚠ règles sous le seuil sur les tâches gelées : {bad}")
    return len(bad)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", choices=[*REPO_LANG, "all"])
    ap.add_argument("--repos-dir", default=DEFAULT_REPOS)
    ap.add_argument("--train-sample", type=int, default=0,
                    help="N fonctions du split train en plus des tâches gelées")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    repos = list(REPO_LANG) if args.repo == "all" else [args.repo]
    n_bad = 0
    for repo in repos:
        n_bad += run(repo, Path(args.repos_dir), args.train_sample, args.verbose)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
