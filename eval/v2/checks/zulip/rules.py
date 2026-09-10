#!/usr/bin/env python3
"""v2 — règles d'adhérence Zulip (tâches PYTHON uniquement — décision Lot 1 :
le volet TS exigerait une stack node ; retiré et documenté au pré-enregistrement).

Deux sous-ensembles, tous « lint natif » (CDC §2.3.1) :
1. 9 patterns VERBATIM du linter maison (tools/linter_lib/custom_check.py
   @5784ebe8) + 1 adapté de leur semgrep — applicabilité par déclencheur.
2. 24 familles de règles ruff ISSUES DE LEUR CONFIG ([tool.ruff.lint].select de
   pyproject.toml @5784ebe8, strict + ruff-clean en CI), exécutées avec ruff
   épinglé (requirements-v2.lock) et LEUR pyproject via --config. Familles
   restreintes aux codes « body-safe » (pas de résolution inter-fichiers : F/I/
   ANN/TC/PYI exclues — la signature est FOURNIE au modèle, l'import n'est pas
   généré). Entrée attendue : fonction RECONSTRUITE (header verbatim + corps
   généré, dédentée) — cf. reconstruct() dans runner.

Env : LLML_V2_REPOS_DIR (clones épinglés).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess

REPOS_DIR = os.environ.get(
    "LLML_V2_REPOS_DIR",
    "/tmp/claude-0/-home-user-llml/f9042f81-6531-534f-9c23-cc2ce7114935/scratchpad/repos")
_CFG = os.path.join(REPOS_DIR, "zulip", "pyproject.toml")
_SRC = "zulip/tools/linter_lib/custom_check.py @5784ebe8"
_SRC_RUFF = "zulip/pyproject.toml [tool.ruff.lint].select @5784ebe8 (ruff épinglé, --config du repo)"

_LEGACY_SUBJECT = (
    "zerver/lib/topic.py", "zerver/lib/event_types.py", "zerver/lib/fix_unreads.py",
    "zerver/lib/email_mirror.py", "zerver/lib/email_notifications.py",
    "zerver/lib/send_email.py", "zerver/lib/typed_endpoint.py",
    "zerver/openapi/python_examples.py", "zerver/lib/email_mirror_server.py",
)

# --- 1) patterns verbatim du linter maison -----------------------------------------
_R = [
    ("msgid_var", r"msgid|MSGID", r"msgid|MSGID",
     'pas de variable "msgid" — message_id', _SRC, ()),
    ("subject_var", r"\bsubject\b|SUBJECT", r"subject|SUBJECT",
     "pas de variable subject — vocabulaire « topic »", _SRC, _LEGACY_SUBJECT),
    ("login_required", r"^[\t ]*(?!#)@login_required", r"login_required",
     "@login_required interdit — @zulip_login_required", _SRC, ()),
    ("type_comment_fn", r"^[\t ]*#[\t ]*type:", r"#\s*type:",
     "pas d'annotations de type en commentaire", _SRC, ()),
    ("type_comment_var", r"\S[\t ]*#[\t ]*type:(?![\t ]*ignore)", r"#\s*type:",
     "pas d'annotation de variable en commentaire", _SRC, ()),
    ("mock_missing_assert", r"\.(called(_once|_with|_once_with)?|not_called|has_calls)[(]",
     r"\bmock|\.called", 'mock : préfixer par "assert_"', _SRC, ()),
    ("assert_length", r"assertEqual[(]len[(][^\n ]*[)],", r"assertEqual",
     "assert_length plutôt que assertEqual(len(..)..)", _SRC, ()),
    ("naive_datetime", r"datetime\.(?:datetime\.)?(?:now|utcnow)\s*\(", r"datetime",
     "datetimes naïfs interdits — timezone_now()",
     "adapté de zulip/tools/semgrep-py.yml @5784ebe8", ()),
    ("pct_format_no_tuple", r"""["']\s*%\s*(?!\s*\()[A-Za-z_"']""", r"%",
     "format % : x % (y,) jamais x % y",
     "zulip docs/contributing/code-style.md @5784ebe8", ()),
]


def _mk(slug, pattern, trigger, desc, source, excludes):
    pat, trig = re.compile(pattern, re.M), re.compile(trigger)

    def fn(tree, src, path, _p=pat, _t=trig, _ex=excludes):
        if not (path or "").endswith(".py"):
            return False, False, "non-py"
        if _ex and any((path or "").endswith(e) for e in _ex):
            return False, False, "chemin exclu par la règle du repo"
        if not _t.search(src):
            return False, False, "construction absente"
        hits = _p.findall(src)
        return True, not hits, f"{len(hits)} occurrence(s) interdite(s)" if hits else "ok"

    return {"id": f"zulip.{slug}", "family": "lint", "desc": desc, "source": source, "fn": fn}


# --- 2) familles ruff de LEUR config (body-safe) ------------------------------------
_ZULIP_SELECT = {"ANN", "B", "C4", "COM", "DJ", "DTZ", "E", "EXE", "F", "FLY", "FURB",
                 "G", "I", "INT", "ISC", "LOG", "N", "PERF", "PGH", "PIE", "PL", "PYI",
                 "Q", "RSE", "RUF", "S", "SLOT", "SIM", "T10", "TC", "TID", "UP", "W", "YTT"}
_BODY_SAFE = {"B", "C4", "COM", "DTZ", "E", "FLY", "FURB", "G", "ISC", "LOG", "N", "PERF",
              "PGH", "PIE", "PL", "Q", "RSE", "RUF", "S", "SIM", "SLOT", "TID", "UP", "W"}
_FAMILIES = sorted(_ZULIP_SELECT & _BODY_SAFE)

_ruff_cache: dict[str, dict[str, int]] = {}


def _ruff_families(source: str, path: str) -> dict[str, int]:
    key = hashlib.sha1((source + "\0" + (path or "")).encode()).hexdigest()
    if key in _ruff_cache:
        return _ruff_cache[key]
    fams: dict[str, int] = {}
    try:
        p = subprocess.run(
            ["ruff", "check", "--config", _CFG, "--output-format", "json",
             "--stdin-filename", path or "zerver/snippet.py", "--no-cache", "-"],
            input=source if source.endswith("\n") else source + "\n",
            capture_output=True, text=True, timeout=60)
        for v in json.loads(p.stdout or "[]"):
            m = re.match(r"[A-Z]+", v.get("code") or "")
            if m:
                fams[m.group(0)] = fams.get(m.group(0), 0) + 1
        if "invalid-syntax" in (p.stdout or ""):
            fams["_SYNTAX"] = 1
    except Exception as e:  # noqa: BLE001
        fams["_ERROR"] = 1
        fams["_MSG"] = str(e)  # type: ignore[assignment]
    _ruff_cache[key] = fams
    return fams


def _mk_ruff(family: str) -> dict:
    def fn(tree, src, path, _f=family):
        if not (path or "").endswith(".py"):
            return False, False, "non-py"
        fams = _ruff_families(src, path)
        if "_ERROR" in fams:
            return True, False, "ruff indisponible"
        if "_SYNTAX" in fams:
            return True, False, "code invalide"
        n = fams.get(_f, 0)
        return True, n == 0, f"{n} violation(s) {_f}" if n else "ok"

    return {"id": f"zulip.ruff_{family}", "family": "lint",
            "desc": f"0 violation ruff famille {family} (config du repo)",
            "source": _SRC_RUFF, "fn": fn}


RULES = [_mk(*r) for r in _R] + [_mk_ruff(f) for f in _FAMILIES]
