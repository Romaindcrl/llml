#!/usr/bin/env python3
"""v2 Lot 1 — génère les 150 tâches held-out GELÉES (CDC v2 §2.4).

Par repo (repos.lock.json) : liste des fichiers de code du langage cible →
shuffle seed 42 → 80% train / 20% held-out. Une tâche = réimplémenter une
fonction masquée d'un fichier held-out : le modèle reçoit le squelette
(fenêtré si le fichier est long), la signature et la docstring/le commentaire,
et génère le corps. 30 tâches/repo tirées par seed, gelées en JSONL.

Sorties (committées = gel) :
  eval/v2/splits/<repo>.json   — listes train/heldout (chemins relatifs)
  eval/v2/tasks/<repo>.jsonl   — 30 tâches {task_id, file, func, signature,
                                  docstring, skeleton, expected_body, lines}

Usage : python3 eval/v2/make_tasks.py --repos-dir <dir des clones> [--repo <nom>]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = json.load(open(os.path.join(HERE, "repos.lock.json"), encoding="utf-8"))

EXT = {"c": (".c",), "zig": (".zig",), "python": (".py",), "python+ts": (".py", ".ts")}
# répertoires de code pertinents par repo (évite docs/, tests géants, vendored)
ROOTS = {
    "FreeRTOS-Kernel": ["."],
    "curl": ["lib", "src"],
    "tigerbeetle": ["src"],
    "twisted": ["src/twisted"],
    "zulip": ["zerver", "web/src"],
}
EXCLUDE_PAT = re.compile(r"(^|/)(test|tests|_test|third[_-]?party|vendor|examples?)(/|$)", re.I)
MIN_BODY, MAX_BODY = 5, 60          # lignes de corps masquable
WINDOW_BEFORE, WINDOW_AFTER = 60, 30  # fenêtrage du squelette (lignes)


def list_code_files(repo_dir: str, repo: str, lang: str) -> list[str]:
    exts = EXT[lang]
    out = []
    for root in ROOTS.get(repo, ["."]):
        base = os.path.join(repo_dir, root)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, repo_dir)
                if fn.endswith(exts) and not EXCLUDE_PAT.search(rel):
                    out.append(rel)
    return sorted(out)


# --- extracteurs de fonctions ---------------------------------------------------------

def funcs_python(src: str):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines()
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        body_start = node.body[0].end_lineno if isinstance(node.body[0], ast.Expr) else node.body[0].lineno - 1
        n_body = node.end_lineno - body_start
        if not (MIN_BODY <= n_body <= MAX_BODY):
            continue
        sig = lines[node.lineno - 1].strip()
        out.append({"name": node.name, "sig_line": node.lineno, "body_start": body_start + 1,
                    "end": node.end_lineno, "signature": sig, "docstring": doc[:400]})
    return out


_BRACE_FN = {
    "c":   re.compile(r"^[A-Za-z_][A-Za-z0-9_ \t\*\(\)]*\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*$"),
    "zig": re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    "ts":  re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("),
}


def funcs_braces(src: str, kind: str):
    """Extracteur pragmatique pour C/Zig/TS : signature détectée par regex,
    corps délimité par équilibrage d'accolades depuis la 1re '{'."""
    lines = src.splitlines()
    pat = _BRACE_FN[kind]
    out, i, n = [], 0, len(lines)
    while i < n:
        m = pat.match(lines[i])
        if not m or lines[i].rstrip().endswith(";"):
            i += 1
            continue
        # trouver la '{' d'ouverture (même ligne ou dans les 3 suivantes)
        j, depth, opened = i, 0, False
        while j < min(i + 4, n):
            if "{" in lines[j]:
                opened = True
                break
            j += 1
        if not opened:
            i += 1
            continue
        # équilibrage d'accolades (naïf : ignore les strings — acceptable, vérifié par bornes)
        end = j
        for k in range(j, n):
            depth += lines[k].count("{") - lines[k].count("}")
            if depth <= 0 and k > j:
                end = k
                break
        else:
            i += 1
            continue
        body_lines = end - j
        # commentaire au-dessus de la signature = "docstring"
        doc = []
        k = i - 1
        while k >= 0 and (lines[k].strip().startswith(("//", "*", "/*", "///")) or lines[k].strip() == "*/"):
            doc.insert(0, lines[k].strip())
            k -= 1
        if MIN_BODY <= body_lines <= MAX_BODY and depth <= 0:
            out.append({"name": m.group(1), "sig_line": i + 1, "body_start": j + 2,
                        "end": end + 1, "signature": lines[i].strip(),
                        "docstring": "\n".join(doc)[:400]})
        i = end + 1
    return out


def make_skeleton(src: str, f: dict, lang_kind: str) -> str:
    """Fichier fenêtré avec le corps de la fonction remplacé par un marqueur."""
    lines = src.splitlines()
    lo = max(0, f["sig_line"] - 1 - WINDOW_BEFORE)
    hi = min(len(lines), f["end"] + WINDOW_AFTER)
    marker = "        # <<< IMPLEMENT BODY >>>" if lang_kind == "python" else "    /* <<< IMPLEMENT BODY >>> */"
    out = []
    if lo > 0:
        out.append("... (fichier tronqué) ...")
    out += lines[lo:f["body_start"] - 1]
    out.append(marker)
    out += lines[f["end"]:hi]
    if hi < len(lines):
        out.append("... (fichier tronqué) ...")
    return "\n".join(out)


def gen_repo(repo: str, meta: dict, repos_dir: str, rng_seed: int = 42):
    repo_dir = os.path.join(repos_dir, repo)
    lang = meta["lang"]
    files = list_code_files(repo_dir, repo, lang)
    rng = random.Random(rng_seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * LOCK["split"]["train_frac"])
    train, heldout = sorted(shuffled[:n_train]), sorted(shuffled[n_train:])

    # candidats de tâches sur les held-out
    cands = []
    for rel in heldout:
        try:
            src = open(os.path.join(repo_dir, rel), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if rel.endswith(".py"):
            fs, kind = funcs_python(src), "python"
        elif rel.endswith(".zig"):
            fs, kind = funcs_braces(src, "zig"), "zig"
        elif rel.endswith(".ts"):
            fs, kind = funcs_braces(src, "ts"), "ts"
        else:
            fs, kind = funcs_braces(src, "c"), "c"
        for f in fs:
            cands.append((rel, kind, f, src))
    rng2 = random.Random(rng_seed + 1)
    rng2.shuffle(cands)
    picked, seen_files = [], set()
    for rel, kind, f, src in cands:               # ≤2 tâches par fichier, diversité d'abord
        if sum(1 for p in picked if p["file"] == rel) >= 2:
            continue
        body = "\n".join(src.splitlines()[f["body_start"] - 1:f["end"]])
        picked.append({
            "task_id": f"{repo}#{len(picked)+1:02d}", "repo": repo, "file": rel,
            "lang": kind, "func": f["name"], "signature": f["signature"],
            "docstring": f["docstring"], "skeleton": make_skeleton(src, f, kind),
            "expected_body": body, "sig_line": f["sig_line"], "end_line": f["end"],
        })
        seen_files.add(rel)
        if len(picked) >= LOCK["split"]["tasks_per_repo"]:
            break
    return train, heldout, picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-dir", required=True)
    ap.add_argument("--repo", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "splits"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "tasks"), exist_ok=True)
    for repo, meta in LOCK["repos"].items():
        if a.repo and repo != a.repo:
            continue
        train, heldout, tasks = gen_repo(repo, meta, a.repos_dir)
        with open(os.path.join(HERE, "splits", f"{repo}.json"), "w", encoding="utf-8") as f:
            json.dump({"sha": meta["sha"], "seed": 42, "train": train, "heldout": heldout}, f, indent=1)
        with open(os.path.join(HERE, "tasks", f"{repo}.jsonl"), "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        sk = [len(t["skeleton"]) for t in tasks]
        print(f"{repo}: {len(train)} train / {len(heldout)} heldout files, "
              f"{len(tasks)} tâches (skeleton méd {sorted(sk)[len(sk)//2] if sk else 0}c)")


if __name__ == "__main__":
    main()
