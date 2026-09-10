#!/usr/bin/env python3
"""v2 — règles d'adhérence curl : wrapper du LINTER MAISON du repo
(scripts/checksrc.pl, exécuté tel quel = famille « lint natif », CDC §2.3.1).

Source des règles : docs/internals/CODE_STYLE.md + CHECKSRC.md du clone épinglé
(repos.lock.json, sha 5c5334f8). Une règle = une catégorie checksrc. Le linter
est exécuté UNE fois par extrait (cache) ; chaque règle lit les violations de
sa catégorie. Applicabilité par déclencheur : une catégorie n'est comptée que
si l'extrait contient la construction concernée (sinon un extrait qui n'emploie
pas la construction « passerait » trivialement et le score plafonne).

Env : LLML_V2_REPOS_DIR = dossier contenant les clones (défaut scratchpad).
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile

REPOS_DIR = os.environ.get(
    "LLML_V2_REPOS_DIR",
    "/tmp/claude-0/-home-user-llml/f9042f81-6531-534f-9c23-cc2ce7114935/scratchpad/repos")
CHECKSRC = os.path.join(REPOS_DIR, "curl", "scripts", "checksrc.pl")
_DOC = "curl/docs/internals/CODE_STYLE.md @5c5334f8 (linter: scripts/checksrc.pl)"

_CAT_RE = re.compile(r"warning: .*\(([A-Z]+)\)\s*$", re.M)
_cache: dict[str, dict] = {}


def _checksrc_categories(source: str) -> dict[str, int]:
    """Exécute checksrc.pl sur l'extrait, retourne {catégorie: n_violations}."""
    key = hashlib.sha1(source.encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    cats: dict[str, int] = {}
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
            f.write(source if source.endswith("\n") else source + "\n")
            tmp = f.name
        out = subprocess.run(["perl", CHECKSRC, tmp], capture_output=True,
                             text=True, timeout=30).stdout
        os.unlink(tmp)
        for m in _CAT_RE.finditer(out):
            cats[m.group(1)] = cats.get(m.group(1), 0) + 1
    except Exception as e:  # noqa: BLE001 — linter indisponible ≠ crash d'éval
        cats = {"_ERROR": 1, "_MSG": str(e)}  # type: ignore[dict-item]
    _cache[key] = cats
    return cats


# (catégorie, déclencheur d'applicabilité (regex sur l'extrait), description)
# Catégories FICHIER-niveau exclues (COPYRIGHT, INCLUDEDUP…) : hors périmètre extrait.
_CATS = [
    ("SPACEBEFOREPAREN", r"\b(if|while|for|switch|return)\b", "if(x) sans espace avant ("),
    ("CPPCOMMENTS",      r"//|/\*", "commentaires /* */ uniquement, jamais //"),
    ("BANNEDFUNC",       r"\b[a-z_]+\s*\(", "fonctions bannies (malloc, sprintf, strcpy…)"),
    ("TABS",             r".", "jamais de tabulation"),
    ("TRAILINGSPACE",    r".", "pas d'espace en fin de ligne"),
    ("LONGLINE",         r".", "lignes ≤ 79 colonnes"),
    ("INDENTATION",      r"\b(if|else|while|for)\b", "indentation 2 espaces"),
    ("BRACEPOS",         r"\{", "position d'accolade curl (fn: seule sur sa ligne)"),
    ("BRACEELSE",        r"\belse\b", "else sur sa propre ligne"),
    ("PARENBRACE",       r"\)\s*\{", "pas de ){ collés"),
    ("SPACEAFTERPAREN",  r"\(", "pas d'espace après ("),
    ("SPACEBEFORECLOSE", r"\)", "pas d'espace avant )"),
    ("COMMANOSPACE",     r",", "espace après virgule"),
    ("SEMINOSPACE",      r";", "pas d'espace avant ;"),
    ("EQUALSNULL",       r"[=!]=\s*NULL", "pas de == NULL (utiliser !ptr)"),
    ("NOTEQUALSZERO",    r"!=\s*0\b", "pas de != 0 en condition"),
    ("ASSIGNWITHINCONDITION", r"\bif\s*\(", "pas d'affectation dans une condition"),
    ("TYPEDEFSTRUCT",    r"\btypedef\b", "pas de typedef de struct"),
    ("RETURNNOSPACE",    r"\breturn\b", "return sans parenthèses collées"),
    ("EMPTYLINEBRACE",   r"\{", "pas de ligne vide avant accolade fermante"),
]


def _mk_rule(cat: str, trigger: str, desc: str) -> dict:
    trig_re = re.compile(trigger)

    def fn(tree, source, path, _cat=cat, _trig=trig_re):
        if not _trig.search(source):
            return False, False, "construction absente"
        cats = _checksrc_categories(source)
        if "_ERROR" in cats:
            return True, False, "checksrc indisponible"
        n = cats.get(_cat, 0)
        return True, n == 0, f"{n} violation(s) {_cat}" if n else "ok"

    return {"id": f"curl.{cat.lower()}", "family": "lint", "desc": desc,
            "source": _DOC, "fn": fn}


RULES = [_mk_rule(c, t, d) for c, t, d in _CATS]
