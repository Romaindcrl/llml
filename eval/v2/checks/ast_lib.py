#!/usr/bin/env python3
"""v2 — bibliothèque de checks d'adhérence DÉTERMINISTES (CDC v2 §2.3).

Trois familles : lint natif (exécuté à part), règles AST (ici), règles
structurelles (regex/chemins, ici aussi). Aucun LLM juge, nulle part.

Une règle = dict {id, family, desc, fn} où fn(tree, source, path) retourne
(applicable: bool, passed: bool, detail: str). Le score d'adhérence d'une
génération = % de règles APPLICABLES satisfaites.

Ce module ne contient que le moteur + 3 règles Python génériques de
démonstration pour le smoke du Lot 0. Les règles par repo (≥15/repo) vivent
dans eval/v2/checks/<repo>/rules.py et sont GELÉES avant tout entraînement.
"""
from __future__ import annotations

import ast
import re


def run_rules(source: str, rules: list[dict], path: str = "<gen>") -> dict:
    """Exécute une liste de règles sur un code source. Robuste aux erreurs de
    parse : source imparsable → toutes les règles AST applicable+failed
    (l'adhérence suppose du code syntaxiquement valide)."""
    results = []
    tree = None
    try:
        tree = ast.parse(source)
    except SyntaxError as e:  # code invalide : règles AST échouent, pas de crash
        for r in rules:
            if r["family"] == "ast":
                results.append({"id": r["id"], "applicable": True, "passed": False,
                                "detail": f"syntax error: {e}"})
            else:
                app, ok, det = r["fn"](None, source, path)
                results.append({"id": r["id"], "applicable": app, "passed": ok, "detail": det})
        return _tally(results)
    for r in rules:
        try:
            app, ok, det = r["fn"](tree, source, path)
        except Exception as e:  # noqa: BLE001 — une règle buguée ≠ un crash d'éval
            app, ok, det = True, False, f"rule error: {e}"
        results.append({"id": r["id"], "applicable": app, "passed": ok, "detail": det})
    return _tally(results)


def _tally(results: list[dict]) -> dict:
    applicable = [r for r in results if r["applicable"]]
    passed = [r for r in applicable if r["passed"]]
    return {
        "n_rules": len(results), "n_applicable": len(applicable), "n_passed": len(passed),
        "adherence": round(len(passed) / len(applicable), 4) if applicable else None,
        "results": results,
    }


# --- 3 règles Python de DÉMONSTRATION (smoke Lot 0 uniquement) -----------------------

def _rule_snake_case_functions(tree, source, path):
    if tree is None:
        return True, False, "unparseable"
    names = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not names:
        return False, False, "no functions"
    bad = [n for n in names if not re.fullmatch(r"_{0,2}[a-z][a-z0-9_]*_{0,2}", n)]
    return True, not bad, f"bad={bad[:5]}" if bad else f"{len(names)} fn ok"


def _rule_no_bare_except(tree, source, path):
    if tree is None:
        return True, False, "unparseable"
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    if not handlers:
        return False, False, "no except"
    bare = [n for n in handlers if n.type is None]
    return True, not bare, f"{len(bare)} bare except" if bare else f"{len(handlers)} handlers ok"


def _rule_module_docstring(tree, source, path):
    if tree is None:
        return True, False, "unparseable"
    return True, ast.get_docstring(tree) is not None, "module docstring"


DEMO_RULES = [
    {"id": "demo.snake_case_fn", "family": "ast", "desc": "fonctions en snake_case",
     "fn": _rule_snake_case_functions},
    {"id": "demo.no_bare_except", "family": "ast", "desc": "pas de except: nu",
     "fn": _rule_no_bare_except},
    {"id": "demo.module_docstring", "family": "ast", "desc": "docstring de module présente",
     "fn": _rule_module_docstring},
]
