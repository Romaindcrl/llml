#!/usr/bin/env python3
"""v2 — règles d'adhérence Twisted (familles « ast » + « struct », CDC §2.3.2-3).

Source unique : twisted/docs/development/coding-standard.rst @fe27ab1d (clone
épinglé, repos.lock). Chaque règle cite sa section. Entrée attendue : fonction
RECONSTRUITE (header verbatim + corps généré, dédentée) — un module Python
contenant un def top-level ; les règles marchent aussi sur un fichier entier.

Applicabilité par déclencheur : une règle dont la construction est absente de
l'extrait retourne (False, False, "construction absente") — cf. curl/zulip.
Règles écartées à la calibration : voir CALIBRATION.md (cb/eb, %-tuple, cookie
test-case-name, docstring-partout strict…), chiffres à l'appui.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize

_DOC = "twisted/docs/development/coding-standard.rst @fe27ab1d"

# snake_case pur : ≥2 segments minuscules séparés par _ (préfixes _ privés tolérés)
_SNAKE = re.compile(r"_{0,2}[a-z0-9]+(?:_[a-z0-9]+)+_?$")
_PASCAL = re.compile(r"_{0,2}[A-Z][A-Za-z0-9]*$")
_SPHINX = re.compile(r"(?m)^\s*:(?:param|type|returns?|rtype|raises?|ivar|cvar|var)\b")
_GOOGLE = re.compile(r"(?m)^\s*(?:Args|Returns|Raises|Yields|Attributes):\s*$")
_NUMPY = re.compile(r"(?m)^\s*Parameters\s*\n\s*-{4,}")


def _defs(tree):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _docstrings(tree):
    """[(node_docstring_expr, texte)] pour module/classes/fonctions."""
    out = []
    nodes = [tree] + [n for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for n in nodes:
        if n.body and isinstance(n.body[0], ast.Expr) and \
                isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
            out.append((n.body[0], n.body[0].value.value))
    return out


# --- naming ------------------------------------------------------------------------

def _r_fn_camelcase(tree, source, path):
    """§Methods/§Functions : mixedCase première lettre minuscule, pas de snake_case."""
    if tree is None:
        return True, False, "code invalide"
    names = [n.name for n in _defs(tree)]
    if not names:
        return False, False, "construction absente"
    bad = [n for n in names if _SNAKE.fullmatch(n)]
    return True, not bad, f"snake_case: {bad[:5]}" if bad else f"{len(names)} nom(s) ok"


def _r_local_camelcase(tree, source, path):
    """§Attributes (« named similarly to functions ») : locales en camelCase."""
    if tree is None:
        return True, False, "code invalide"
    bad, tot = [], 0
    for fn in _defs(tree):
        for n in ast.walk(fn):
            tgts = []
            if isinstance(n, ast.Assign):
                tgts = n.targets
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                tgts = [n.target]
            elif isinstance(n, ast.For):
                tgts = [n.target]
            elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                tgts = [n.optional_vars]
            for t in tgts:
                for e in ast.walk(t):
                    if isinstance(e, ast.Name):
                        tot += 1
                        if _SNAKE.fullmatch(e.id):
                            bad.append(e.id)
    if tot == 0:
        return False, False, "construction absente"
    return True, not bad, f"locales snake_case: {sorted(set(bad))[:5]}" if bad else f"{tot} locale(s) ok"


def _r_attr_camelcase(tree, source, path):
    """§Attributes : attributs self.x en camelCase, pas de snake_case."""
    if tree is None:
        return True, False, "code invalide"
    tot, bad = 0, []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and \
                isinstance(n.value, ast.Name) and n.value.id == "self":
            tot += 1
            if _SNAKE.fullmatch(n.attr):
                bad.append(n.attr)
    if tot == 0:
        return False, False, "construction absente"
    return True, not bad, f"attrs snake_case: {bad[:5]}" if bad else f"{tot} attr(s) ok"


def _r_no_name_mangling(tree, source, path):
    """§Attributes : jamais le « private » __x de Python — un seul underscore."""
    if tree is None:
        return True, False, "code invalide"
    tot, bad = 0, []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
            tot += 1
            if n.attr.startswith("__") and not n.attr.endswith("__"):
                bad.append(n.attr)
    if tot == 0:
        return False, False, "construction absente"
    return True, not bad, f"name mangling: {bad[:5]}" if bad else "pas de __attr"


def _r_class_pascal(tree, source, path):
    """§Classes : mixed case, première lettre majuscule."""
    if tree is None:
        return True, False, "code invalide"
    names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if not names:
        return False, False, "construction absente"
    bad = [n for n in names if not _PASCAL.fullmatch(n)]
    return True, not bad, f"classes non Pascal: {bad[:5]}" if bad else f"{len(names)} classe(s) ok"


def _r_interface_prefix(tree, source, path):
    """§Naming (usage repo, cf. IReactor) : les interfaces s'appellent I<Nom>."""
    if tree is None:
        return True, False, "code invalide"
    checked, bad = 0, []
    for c in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        bases = []
        for b in c.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)
        if any(x == "Interface" or re.fullmatch(r"I[A-Z]\w*", x) for x in bases):
            checked += 1
            if not re.fullmatch(r"_?I[A-Z]\w*", c.name):
                bad.append(c.name)
    if not checked:
        return False, False, "construction absente"
    return True, not bad, f"interfaces mal nommées: {bad}" if bad else f"{checked} interface(s) ok"


# --- docstrings epytext --------------------------------------------------------------

def _r_docstring_on_defs(tree, source, path):
    """§Docstrings : toute fonction/classe (hors closures imbriquées) documentée."""
    if tree is None:
        return True, False, "code invalide"
    nested = set()
    for f in _defs(tree):
        for sub in ast.walk(f):
            if sub is not f and isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested.add(id(sub))
    tops = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and id(n) not in nested]
    if not tops:
        return False, False, "construction absente"
    bad = [n.name for n in tops if ast.get_docstring(n) is None]
    return True, not bad, f"sans docstring: {bad[:5]}" if bad else f"{len(tops)} def(s) documenté(s)"


def _r_epytext_not_sphinx(tree, source, path):
    """§Docstrings : format epytext (@param…), jamais Sphinx/Google/NumPy."""
    if tree is None:
        return True, False, "code invalide"
    ds = [t for _, t in _docstrings(tree)]
    if not ds:
        return False, False, "construction absente"
    joined = "\n".join(ds)
    hits = []
    if _SPHINX.search(joined):
        hits.append("sphinx :param:")
    if _GOOGLE.search(joined):
        hits.append("google Args:")
    if _NUMPY.search(joined):
        hits.append("numpy")
    return True, not hits, "; ".join(hits) if hits else f"{len(ds)} docstring(s) epytext"


def _r_epytext_fields(tree, source, path):
    """§Docstrings : champs epytext bien formés — @param <nom>: / @raise <Exc>:."""
    if tree is None:
        return True, False, "code invalide"
    lines = []
    for _, t in _docstrings(tree):
        lines += [l.strip() for l in t.splitlines() if l.strip().startswith(("@param", "@raise", "@keyword"))]
    if not lines:
        return False, False, "construction absente"
    bad = [l for l in lines
           if not re.match(r"@(?:param|keyword)\s+\*{0,2}\w+\s*:|@raises?\s+[\w.]+\s*:", l)]
    return True, not bad, f"champs mal formés: {bad[:3]}" if bad else f"{len(lines)} champ(s) ok"


def _r_type_markup(tree, source, path):
    """§Docstrings : les types référencés via L{}/C{} (≥80 % des lignes @type/@rtype)."""
    if tree is None:
        return True, False, "code invalide"
    lines = []
    for _, t in _docstrings(tree):
        lines += [l.strip() for l in t.splitlines() if l.strip().startswith(("@type", "@rtype"))]
    if not lines:
        return False, False, "construction absente"
    ok = [l for l in lines if "L{" in l or "C{" in l]
    ratio = len(ok) / len(lines)
    return True, ratio >= 0.8, f"L{{}}/C{{}} sur {len(ok)}/{len(lines)} ligne(s) @type"


def _r_docstring_own_line(tree, source, path):
    """§Docstrings : triple quotes ouvrante ET fermante seules sur leur ligne."""
    if tree is None:
        return True, False, "code invalide"
    segs = []
    for node, _ in _docstrings(tree):
        seg = ast.get_source_segment(source, node)
        if seg:
            segs.append(seg)
    if not segs:
        return False, False, "construction absente"
    def ok(seg):
        ls = seg.splitlines()
        return (len(ls) >= 2
                and re.fullmatch(r'[rbuRBU]{0,2}("""|\'\'\')', ls[0].strip())
                and ls[-1].strip() in ('"""', "'''"))
    good = sum(1 for s in segs if ok(s))
    ratio = good / len(segs)
    return True, ratio >= 0.8, f"{good}/{len(segs)} docstring(s) avec quotes sur ligne propre"


# --- imports / idiomes ----------------------------------------------------------------

def _r_no_wildcard_import(tree, source, path):
    """§Modules : wildcard imports interdits."""
    if tree is None:
        return True, False, "code invalide"
    imps = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not imps:
        return False, False, "construction absente"
    bad = [n for n in imps if isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)]
    return True, not bad, f"{len(bad)} wildcard import(s)" if bad else f"{len(imps)} import(s) ok"


def _r_no_relative_import(tree, source, path):
    """§Modules : relative/sibling imports interdits."""
    if tree is None:
        return True, False, "code invalide"
    imps = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not imps:
        return False, False, "construction absente"
    bad = [n for n in imps if isinstance(n, ast.ImportFrom) and (n.level or 0) > 0]
    return True, not bad, f"{len(bad)} import(s) relatif(s)" if bad else f"{len(imps)} import(s) ok"


def _r_no_print(tree, source, path):
    """§Recommendations + usage repo (twisted.logger) : pas de print()."""
    if tree is None:
        return True, False, "code invalide"
    if "print" not in source:
        return False, False, "construction absente"
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"]
    return True, not calls, f"{len(calls)} print()" if calls else "print absent du code exécuté"


def _r_none_identity(tree, source, path):
    """Fallback PEP 8 (§Fallback) : comparaison à None par is/is not, pas ==/!=."""
    if tree is None:
        return True, False, "code invalide"
    tot, bad = 0, 0
    is_none = lambda x: isinstance(x, ast.Constant) and x.value is None  # noqa: E731
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            operands = [n.left] + list(n.comparators)
            for i, op in enumerate(n.ops):
                if is_none(operands[i]) or is_none(operands[i + 1]):
                    tot += 1
                    if isinstance(op, (ast.Eq, ast.NotEq)):
                        bad += 1
    if tot == 0:
        return False, False, "construction absente"
    return True, bad == 0, f"{bad}/{tot} comparaison(s) ==/!= None" if bad else f"{tot} comparaison(s) is None ok"


# --- struct (texte) -------------------------------------------------------------------

def _r_no_tabs(tree, source, path):
    """§Whitespace (style Black) : indentation en espaces, jamais de tab."""
    if not source.strip():
        return False, False, "source vide"
    bad = [i + 1 for i, l in enumerate(source.splitlines()) if l.startswith("\t")]
    return True, not bad, f"tabs lignes {bad[:5]}" if bad else "0 tab"


def _r_line_length(tree, source, path):
    """§Whitespace (style Black) : lignes ≤ 88 colonnes (URLs exemptées)."""
    lines = source.splitlines()
    if not lines:
        return False, False, "source vide"
    bad = [i + 1 for i, l in enumerate(lines)
           if len(l) > 88 and "http://" not in l and "https://" not in l]
    return True, not bad, f"{len(bad)} ligne(s) > 88c (ex. l.{bad[:3]})" if bad else "longueurs ok"


def _r_double_quotes(tree, source, path):
    """§Whitespace (style Black) : chaînes en double quotes."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except Exception:
        return True, False, "source non tokenizable"
    strs = [t.string for t in toks if t.type == tokenize.STRING]
    if not strs:
        return False, False, "construction absente"
    bad = []
    for s in strs:
        body = re.sub(r"^[rbufRBUF]{0,2}", "", s)
        if body.startswith("'") and '"' not in s:
            bad.append(s[:20])
    return True, not bad, f"single quotes: {bad[:3]}" if bad else f"{len(strs)} chaîne(s) ok"


RULES = [
    {"id": "twisted.fn_camelcase", "family": "ast",
     "desc": "méthodes/fonctions en camelCase (pas de snake_case)",
     "source": _DOC + " §Methods, §Functions", "fn": _r_fn_camelcase},
    {"id": "twisted.local_camelcase", "family": "ast",
     "desc": "variables locales en camelCase (pas de snake_case)",
     "source": _DOC + " §Attributes (named similarly to functions)", "fn": _r_local_camelcase},
    {"id": "twisted.attr_camelcase", "family": "ast",
     "desc": "attributs self.* en camelCase",
     "source": _DOC + " §Attributes", "fn": _r_attr_camelcase},
    {"id": "twisted.no_name_mangling", "family": "ast",
     "desc": "pas d'attribut __privé name-manglé (un seul underscore)",
     "source": _DOC + " §Attributes", "fn": _r_no_name_mangling},
    {"id": "twisted.class_pascalcase", "family": "ast",
     "desc": "classes en PascalCase",
     "source": _DOC + " §Classes", "fn": _r_class_pascal},
    {"id": "twisted.interface_i_prefix", "family": "ast",
     "desc": "interfaces nommées I<Nom>",
     "source": _DOC + " §Naming (IReactor)", "fn": _r_interface_prefix},
    {"id": "twisted.docstring_on_defs", "family": "ast",
     "desc": "docstring sur fonctions/classes (closures exemptées)",
     "source": _DOC + " §Docstrings", "fn": _r_docstring_on_defs},
    {"id": "twisted.epytext_not_sphinx", "family": "ast",
     "desc": "docstrings epytext — jamais :param:/Args:/NumPy",
     "source": _DOC + " §Docstrings (epytext)", "fn": _r_epytext_not_sphinx},
    {"id": "twisted.epytext_fields", "family": "ast",
     "desc": "champs @param/@raise bien formés (@param nom:)",
     "source": _DOC + " §Docstrings", "fn": _r_epytext_fields},
    {"id": "twisted.type_markup_l_c", "family": "ast",
     "desc": "types en L{}/C{} sur les lignes @type/@rtype",
     "source": _DOC + " §Docstrings (L{} pour les types)", "fn": _r_type_markup},
    {"id": "twisted.docstring_own_line_quotes", "family": "ast",
     "desc": "triple quotes de docstring seules sur leur ligne",
     "source": _DOC + " §Docstrings", "fn": _r_docstring_own_line},
    {"id": "twisted.no_wildcard_import", "family": "ast",
     "desc": "pas de from x import *",
     "source": _DOC + " §Modules", "fn": _r_no_wildcard_import},
    {"id": "twisted.no_relative_import", "family": "ast",
     "desc": "pas d'import relatif",
     "source": _DOC + " §Modules", "fn": _r_no_relative_import},
    {"id": "twisted.no_print", "family": "ast",
     "desc": "pas de print()",
     "source": _DOC + " (twisted.logger; print hors style repo)", "fn": _r_no_print},
    {"id": "twisted.none_identity", "family": "ast",
     "desc": "comparaisons à None avec is / is not",
     "source": _DOC + " §Fallback (PEP 8 E711)", "fn": _r_none_identity},
    {"id": "twisted.no_tabs", "family": "struct",
     "desc": "pas de tabulation d'indentation",
     "source": _DOC + " §Whitespace (Black)", "fn": _r_no_tabs},
    {"id": "twisted.line_le_88", "family": "struct",
     "desc": "lignes ≤ 88 colonnes (Black), URLs exemptées",
     "source": _DOC + " §Whitespace (Black)", "fn": _r_line_length},
    {"id": "twisted.double_quotes", "family": "struct",
     "desc": "chaînes en double quotes (Black)",
     "source": _DOC + " §Whitespace (Black)", "fn": _r_double_quotes},
]
