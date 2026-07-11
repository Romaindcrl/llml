#!/usr/bin/env python3
"""v2 — scoring d'adhérence des générations (déterministe, hors pod).

Pour chaque génération : nettoyage déterministe (pré-enregistré) → reconstruction
(header verbatim + corps) → checks du repo (rules.py) + validité syntaxique.

Nettoyage (ordre fixe) :
  1. retrait des clôtures markdown ```...``` (garde le contenu du premier bloc
     s'il existe, sinon texte entier) ;
  2. retrait du header répété si le modèle a re-généré la signature ;
  3. Python : dédentation du corps puis ré-indentation de 4 sous le header ;
     C/Zig : équilibrage final des accolades (ajout des '}' manquantes).

Sortie : results/v2/raw/score_<CONFIG>.jsonl + agrégats par repo.

Usage : python3 eval/v2/score_matrix.py --gen results/v2/raw/gen_C0.jsonl \
          [--out results/v2/raw/score_C0.jsonl]
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "checks"))
from ast_lib import run_rules  # noqa: E402

_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.S)


def clean_output(out: str, task: dict) -> str:
    m = _FENCE.search(out or "")
    body = m.group(1) if m else (out or "")
    body = body.strip("\n")
    # signature re-générée ? retire tout jusqu'à la fin de la 1re ligne du header
    first_sig = (task["header"].splitlines() or [""])[0].strip()
    if first_sig and first_sig[:30] in body[:400]:
        lines = body.splitlines()
        for i, l in enumerate(lines):
            if first_sig[:30] in l:
                # saute jusqu'à la fin du header re-généré (docstring incluse si python)
                j = i + 1
                if task["lang"] == "python":
                    # saute une éventuelle docstring re-générée
                    while j < len(lines) and (not lines[j].strip()
                                              or lines[j].lstrip().startswith(('"""', "'''", '#'))):
                        j += 1
                body = "\n".join(lines[j:])
                break
    return body


def reconstruct(task: dict, body: str) -> str:
    header = textwrap.dedent(task["header"])
    if task["lang"] == "python":
        b = textwrap.dedent(body)
        return header + "\n" + textwrap.indent(b, "    ") + "\n"
    rec = header + "\n" + body + "\n"
    depth = rec.count("{") - rec.count("}")
    if depth > 0:
        rec += "}\n" * depth
    return rec


def syntax_ok(task: dict, recon: str) -> bool:
    if task["lang"] == "python":
        try:
            ast.parse(recon)
            return True
        except SyntaxError:
            return False
    # C/Zig/TS : proxy = accolades équilibrées et non vides
    return recon.count("{") == recon.count("}") and recon.count("{") >= 1


def load_rules(repo: str):
    spec = importlib.util.spec_from_file_location(
        f"r_{repo}", os.path.join(HERE, "checks", repo, "rules.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.RULES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tasks = {}
    for p in sorted(os.listdir(os.path.join(HERE, "tasks"))):
        for line in open(os.path.join(HERE, "tasks", p), encoding="utf-8"):
            t = json.loads(line)
            tasks[t["task_id"]] = t
    rules_by_repo = {}
    out_path = a.out or a.gen.replace("gen_", "score_")
    agg = {}
    with open(out_path, "w", encoding="utf-8") as fout:
        for line in open(a.gen, encoding="utf-8"):
            g = json.loads(line)
            t = tasks[g["task_id"]]
            repo = t["repo"]
            if repo not in rules_by_repo:
                rules_by_repo[repo] = load_rules(repo)
            body = clean_output(g["output"], t)
            recon = reconstruct(t, body)
            r = run_rules(recon, rules_by_repo[repo], path=t["file"])
            ok = syntax_ok(t, recon)
            rec = {"task_id": g["task_id"], "repo": repo, "config": g["config"],
                   "adherence": r["adherence"], "n_applicable": r["n_applicable"],
                   "n_passed": r["n_passed"], "syntax_ok": ok,
                   "fails": [x["id"] for x in r["results"] if x["applicable"] and not x["passed"]],
                   "lat_ms": g.get("lat_ms"), "conv_chars": g.get("conv_chars", 0)}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            s = agg.setdefault(repo, {"p": 0, "a": 0, "syn": 0, "n": 0})
            s["p"] += r["n_passed"]; s["a"] += r["n_applicable"]
            s["syn"] += int(ok); s["n"] += 1
    print(f"→ {out_path}")
    tp = ta = 0
    for repo, s in sorted(agg.items()):
        adh = s["p"] / max(1, s["a"])
        tp += s["p"]; ta += s["a"]
        print(f"{repo:18s} adhérence {adh:6.1%} ({s['p']}/{s['a']})  "
              f"syntaxe {s['syn']}/{s['n']}")
    print(f"{'AGRÉGÉ':18s} adhérence {tp/max(1,ta):6.1%} ({tp}/{ta})")


if __name__ == "__main__":
    main()
