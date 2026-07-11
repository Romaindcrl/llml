#!/usr/bin/env python3
"""v2 — règles d'adhérence TigerBeetle (Zig) — UNIQUEMENT famille « struct »
(texte/regex : pas d'AST Python sur du Zig ; le moteur passe tree=None).

Source unique : tigerbeetle/docs/TIGER_STYLE.md @97c7a8ef (clone épinglé).
Entrée : fonction Zig RECONSTRUITE (header + corps) au niveau colonne 0 ;
marche aussi sur un fichier entier.

Lexique via _code() : commentaires //, chaînes "…"/'…' et lignes littérales
multilignes \\ remplacés par des blancs (mêmes offsets). L'extraction de
fonctions se fait par équilibrage d'accolades sur le code nettoyé.

Règles écartées à la calibration (chiffres dans CALIBRATION.md) : interdiction
de usize (908 occurrences dans le train — le style dit « avoid », l'interop
l'impose), catch {} vide (43 usages sanctionnés), littéraux > 1024 nommés
(données de test/checksums légitimes), ≥2 asserts par fonction en absolu
(46 % des fonctions ≥10 lignes du train seulement).
"""
from __future__ import annotations

import re

_DOC = "tigerbeetle/docs/TIGER_STYLE.md @97c7a8ef"


def _strip_zig(src: str) -> str:
    """Vide commentaires //, chaînes et lignes \\ (offsets préservés)."""
    out = []
    for line in src.split("\n"):
        if line.lstrip().startswith("\\\\"):
            out.append(" " * len(line))
            continue
        res, i, n = [], 0, len(line)
        while i < n:
            c = line[i]
            if c == "/" and i + 1 < n and line[i + 1] == "/":
                res.append(" " * (n - i)); break
            if c in "\"'":
                q, j = c, i + 1
                while j < n and line[j] != q:
                    j += 2 if line[j] == "\\" else 1
                j = min(j + 1, n)
                res.append(q + " " * max(0, j - i - 2) + (q if j - i >= 2 else "")); i = j
                continue
            res.append(c); i += 1
        out.append("".join(res))
    return "\n".join(out)


_zig_cache: dict[int, str] = {}


def _code(src: str) -> str:
    key = hash(src)
    if key not in _zig_cache:
        if len(_zig_cache) > 256:
            _zig_cache.clear()
        _zig_cache[key] = _strip_zig(src)
    return _zig_cache[key]


_FN = re.compile(r"(?m)^(?P<ind>[ \t]*)(?:pub\s+|export\s+|extern\s+|inline\s+|noinline\s+)*"
                 r"fn\s+(?P<name>[A-Za-z_]\w*)\s*\(")


def _functions(code: str):
    """[(nom, ligne_début, ligne_fin, corps_texte)] par équilibre d'accolades."""
    out = []
    n = len(code)
    for m in _FN.finditer(code):
        i, depth = m.end() - 1, 0
        while i < n:  # fin de la liste de paramètres
            if code[i] == "(":
                depth += 1
            elif code[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        j = i
        while j < n and code[j] not in "{;":
            j += 1
        if j >= n or code[j] == ";":
            continue  # prototype extern
        depth, k = 0, j
        while k < n:
            if code[k] == "{":
                depth += 1
            elif code[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        start = code[:m.start()].count("\n")
        end = code[:min(k, n - 1)].count("\n")
        out.append((m.group("name"), start, end, code[j:min(k + 1, n)]))
    return out


def _top_level_asserts(text: str):
    """Occurrences assert(…) avec leur argument (parenthèses équilibrées)."""
    out = []
    for m in re.finditer(r"\bassert\s*\(", text):
        i, depth = m.end() - 1, 0
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append(text[m.end():i])
    return out


# --- limites & contrôle de flux -------------------------------------------------------

def _r_fn_max_70(tree, source, path):
    """« hard limit of 70 lines per function » — fonctions feuilles (les
    constructeurs de type `fn XType(…) type { return struct {…} }` contiennent
    leurs méthodes : seules les fonctions SANS fn imbriquée sont jugées)."""
    code = _code(source)
    fns = _functions(code)
    leaves = [(n, s, e, b) for (n, s, e, b) in fns if not re.search(r"\bfn\s+\w+\s*\(", b)]
    if not leaves:
        return False, False, "construction absente"
    bad = [(n, e - s + 1) for (n, s, e, b) in leaves if e - s + 1 > 71]
    return True, not bad, f"fonction(s) > 70 lignes: {bad[:3]}" if bad else \
        f"{len(leaves)} fonction(s) ≤ 70 lignes"


def _r_no_recursion(tree, source, path):
    """« Do not use recursion » : une fonction ne s'appelle pas elle-même."""
    code = _code(source)
    fns = _functions(code)
    if not fns:
        return False, False, "construction absente"
    bad = []
    for name, _s, _e, body in fns:
        inner = body[body.find("{"):]
        if re.search(r"(?<![.\w@])" + re.escape(name) + r"\s*\(", inner):
            bad.append(name)
    return True, not bad, f"récursion directe: {bad[:3]}" if bad else f"{len(fns)} fonction(s) sans récursion"


def _r_no_compound_assert(tree, source, path):
    """« prefer assert(a); assert(b); over assert(a and b); »."""
    code = _code(source)
    args = _top_level_asserts(code)
    if not args:
        return False, False, "construction absente"
    bad = []
    for a in args:
        # masque les sous-parenthèses pour ne juger que le niveau supérieur
        masked = a
        while re.search(r"\([^()]*\)", masked):
            masked = re.sub(r"\([^()]*\)", " ", masked)
        if re.search(r"\band\b", masked):
            bad.append(a.strip()[:40])
    return True, not bad, f"assert composé(s): {bad[:2]}" if bad else f"{len(args)} assert simples ok"


def _r_if_multiline_braces(tree, source, path):
    """« Add braces to the if statement unless it fits on a single line »."""
    code = _code(source)
    lines = code.split("\n")
    tot, bad = 0, []
    for i, line in enumerate(lines[:-1]):
        st = line.strip()
        m = re.match(r"^(?:\}\s*else\s+)?if\s*\(", st)
        if not m:
            continue
        if st.count("(") != st.count(")"):
            continue  # condition multi-lignes : la fermeture porte l'accolade
        tot += 1
        after = st[st.rindex(")") + 1:].strip() if ")" in st else ""
        if after:  # tient sur une ligne (stmt ou {) : conforme
            continue
        nxt = lines[i + 1].strip()
        if not nxt.startswith("{"):
            bad.append(st[:40])
    if tot == 0:
        return False, False, "construction absente"
    return True, not bad, f"if multi-ligne sans accolades: {bad[:2]}" if bad else f"{tot} if conformes"


# --- nommage ---------------------------------------------------------------------------

_STD_ALIAS = re.compile(r"=\s*(?:std|builtin|@import)\b")


def _r_fn_snake_case(tree, source, path):
    """« Use snake_case for function … names » (TitreCase réservé aux
    constructeurs de type, style Zig)."""
    code = _code(source)
    names = [m.group("name") for m in _FN.finditer(code)]
    if not names:
        return False, False, "construction absente"
    bad = [n for n in names if re.match(r"^[a-z]+[A-Z]", n)]
    return True, not bad, f"fonction(s) camelCase: {bad[:3]}" if bad else f"{len(names)} fn snake_case ok"


def _r_var_snake_case(tree, source, path):
    """« Use snake_case for … variable names » : pas de camelCase (alias de
    std exemptés : const expectEqual = std.testing.expectEqual)."""
    code = _code(source)
    decls = list(re.finditer(r"\b(?:const|var)\s+([A-Za-z_]\w*)\b([^\n]*)", code))
    if not decls:
        return False, False, "construction absente"
    bad = [m.group(1) for m in decls
           if re.match(r"^[a-z]+[A-Z]", m.group(1)) and not _STD_ALIAS.search(m.group(2))]
    return True, not bad, f"variable(s) camelCase: {sorted(set(bad))[:4]}" if bad else \
        f"{len(decls)} décl. snake_case ok"


def _r_no_screaming_case(tree, source, path):
    """snake_case aussi pour les constantes : pas de SCREAMING_SNAKE (les
    constantes d'OS recopiées de l'ABI C sont exemptées via alias std)."""
    code = _code(source)
    decls = list(re.finditer(r"\b(?:const|var)\s+([A-Za-z_]\w*)\b([^\n]*)", code))
    if not decls:
        return False, False, "construction absente"
    bad = [m.group(1) for m in decls
           if re.fullmatch(r"[A-Z][A-Z0-9_]*", m.group(1)) and "_" in m.group(1)
           and not _STD_ALIAS.search(m.group(2))]
    return True, not bad, f"SCREAMING_SNAKE: {sorted(set(bad))[:4]}" if bad else "0 SCREAMING_SNAKE"


def _r_units_order(tree, source, path):
    """« put the units or qualifiers last, sorted by descending significance » :
    latency_ms_max, jamais max_latency_ms (qualificateur en tête + unité en fin)."""
    code = _code(source)
    idents = set(re.findall(r"\b[A-Za-z_]\w*\b", code))
    units = r"(?:ms|us|ns|bytes|ticks)"
    concerned = [i for i in idents if re.search(r"_(?:max|min)\b|^(?:max|min)_|_" + units + r"(?:_|$)", i)]
    if not concerned:
        return False, False, "construction absente"
    bad = [i for i in idents
           if re.fullmatch(r"(?:max|min)_[a-z0-9_]+_" + units, i)
           and i not in ("max_path_bytes",)]  # nom hérité de std.fs
    return True, not bad, f"qualificateur en tête: {bad[:3]}" if bad else \
        f"{len(concerned)} identifiant(s) unité/qualif. ok"


# --- zig fmt / style par les nombres -----------------------------------------------------

def _r_line_le_100(tree, source, path):
    """« Hard limit all line lengths … to at most 100 columns » (littéraux
    multilignes \\ exemptés — zig fmt ne les coupe pas)."""
    lines = source.split("\n")
    if not any(l.strip() for l in lines):
        return False, False, "source vide"
    bad = [i + 1 for i, l in enumerate(lines)
           if len(l) > 100 and not l.lstrip().startswith("\\\\")]
    return True, not bad, f"{len(bad)} ligne(s) > 100c (l.{bad[:4]})" if bad else "≤ 100 colonnes ok"


def _r_indent_4(tree, source, path):
    """« Use 4 spaces of indentation » (lignes \\ exemptées)."""
    code = _code(source)
    offenders, tot = [], 0
    for i, line in enumerate(code.split("\n")):
        if not line.strip():
            continue
        tot += 1
        ind = len(line) - len(line.lstrip(" "))
        if ind % 4 != 0:
            offenders.append(i + 1)
    if tot == 0:
        return False, False, "source vide"
    return True, not offenders, f"{len(offenders)}/{tot} ligne(s) hors grille 4 (l.{offenders[:4]})" \
        if offenders else f"{tot} ligne(s) sur grille 4"


def _r_no_tabs(tree, source, path):
    """zig fmt : jamais de tabulation."""
    if not source.strip():
        return False, False, "source vide"
    bad = [i + 1 for i, l in enumerate(source.splitlines()) if "\t" in l]
    return True, not bad, f"tabs lignes {bad[:4]}" if bad else "0 tab"


def _r_trailing_comma_multiline(tree, source, path):
    """« add a trailing comma » pour les appels/structures multi-lignes :
    une ligne fermante `)` ou `},` seule implique une virgule finale avant."""
    code = _code(source)
    lines = code.split("\n")
    tot, bad = 0, []
    for i in range(1, len(lines)):
        st = lines[i].strip()
        if not re.fullmatch(r"\)[;,]?|\}\)?[;,]?", st):
            continue
        prev = ""
        for j in range(i - 1, -1, -1):
            if lines[j].strip():
                prev = lines[j].strip()
                break
        if not prev or prev.endswith(("{", "(")):
            continue
        if st.startswith(")"):
            tot += 1
            if not prev.endswith(","):
                bad.append(i + 1)
    if tot == 0:
        return False, False, "construction absente"
    return True, not bad, f"{len(bad)}/{tot} fermeture(s) sans virgule finale (l.{bad[:4]})" \
        if bad else f"{tot} virgule(s) finale(s) ok"


def _comment_start(line: str) -> int:
    """Index du // ouvrant un commentaire (hors chaînes), -1 sinon."""
    if line.lstrip().startswith("\\\\"):
        return -1
    i, n = 0, len(line)
    while i < n - 1:
        c = line[i]
        if c in "\"'":
            q, i = c, i + 1
            while i < n and line[i] != q:
                i += 2 if line[i] == "\\" else 1
            i += 1
            continue
        if c == "/" and line[i + 1] == "/":
            return i
        i += 1
    return -1


def _r_comment_space(tree, source, path):
    """« Comments are sentences, with a space after the slash »."""
    comments = []
    for line in source.split("\n"):
        idx = _comment_start(line)
        if idx != -1:
            comments.append(line[idx:].rstrip())
    if not comments:
        return False, False, "construction absente"
    bad = [c[:30] for c in comments if not re.match(r"^//[/!]?(?: |$)", c)]
    return True, not bad, f"slash collé: {bad[:3]}" if bad else f"{len(comments)} commentaire(s) espacé(s)"


def _r_comment_sentences(tree, source, path):
    """« Comments are sentences, … with a capital letter and a full stop » —
    blocs de commentaires pleine ligne : majuscule initiale et ponctuation finale."""
    lines = source.split("\n")
    blocks, cur = [], []
    for line in lines:
        st = line.strip()
        if st.startswith("//") and not st.startswith("//!"):
            cur.append(st.lstrip("/").strip())
        else:
            if cur:
                blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    blocks = [b for b in blocks
              if b and b[0] and "http" not in " ".join(b)
              and not any(t in " ".join(b) for t in ("TODO", "FIXME", "zig fmt", "nosemgrep"))]
    if not blocks:
        return False, False, "construction absente"
    bad = 0
    for b in blocks:
        first, last = b[0], b[-1]
        if not (first[0].isupper() or first[0].isdigit() or first[0] in "\"'(`@"):
            bad += 1
        elif not last.rstrip().endswith((".", ":", "!", "?", '"', "`", ")")):
            bad += 1
    ratio = 1 - bad / len(blocks)
    return True, ratio >= 0.8, f"{bad}/{len(blocks)} bloc(s) non-phrase" if bad else \
        f"{len(blocks)} bloc(s) de commentaire en phrases"


def _r_explicit_int_types(tree, source, path):
    """« Use explicitly-sized types like u32 » : pas de c_int/c_long/c_uint
    hors interop extern (les u32/u64/i64… sont la norme du repo)."""
    code = _code(source)
    ints = re.findall(r"\b(?:u|i)(?:8|16|32|64|128)\b|\bc_(?:int|uint|long|ulong|short|ushort)\b", code)
    if not ints:
        return False, False, "construction absente"
    if re.search(r"\bextern\b|\bcallconv\b|@cImport|\bstd\.c\b", code):
        return False, False, "contexte interop C : types c_* sanctionnés"
    bad = [t for t in ints if t.startswith("c_")]
    return True, not bad, f"type(s) C non dimensionné(s): {sorted(set(bad))}" if bad else \
        f"{len(ints)} type(s) entier(s) explicites"


RULES = [
    {"id": "tigerbeetle.fn_max_70_lines", "family": "struct",
     "desc": "fonction feuille ≤ 70 lignes (limite dure)",
     "source": _DOC + " §Safety (hard limit of 70 lines)", "fn": _r_fn_max_70},
    {"id": "tigerbeetle.no_recursion", "family": "struct",
     "desc": "pas de récursion directe",
     "source": _DOC + " §Safety (Do not use recursion)", "fn": _r_no_recursion},
    {"id": "tigerbeetle.no_compound_assert", "family": "struct",
     "desc": "assert(a); assert(b); plutôt que assert(a and b)",
     "source": _DOC + " §Safety (Split compound assertions)", "fn": _r_no_compound_assert},
    {"id": "tigerbeetle.if_multiline_braces", "family": "struct",
     "desc": "accolades sur if multi-ligne (défense goto fail)",
     "source": _DOC + " §Style By The Numbers (braces to the if)", "fn": _r_if_multiline_braces},
    {"id": "tigerbeetle.fn_snake_case", "family": "struct",
     "desc": "fonctions en snake_case (pas de camelCase)",
     "source": _DOC + " §Naming Things (snake_case)", "fn": _r_fn_snake_case},
    {"id": "tigerbeetle.var_snake_case", "family": "struct",
     "desc": "const/var en snake_case (pas de camelCase)",
     "source": _DOC + " §Naming Things (snake_case)", "fn": _r_var_snake_case},
    {"id": "tigerbeetle.no_screaming_case", "family": "struct",
     "desc": "pas de SCREAMING_SNAKE_CASE",
     "source": _DOC + " §Naming Things (snake_case)", "fn": _r_no_screaming_case},
    {"id": "tigerbeetle.units_order", "family": "struct",
     "desc": "unités/qualificateurs par signification décroissante (_ms_max)",
     "source": _DOC + " §Naming Things (units or qualifiers last)", "fn": _r_units_order},
    {"id": "tigerbeetle.line_le_100", "family": "struct",
     "desc": "lignes ≤ 100 colonnes (limite dure)",
     "source": _DOC + " §Style By The Numbers (100 columns)", "fn": _r_line_le_100},
    {"id": "tigerbeetle.indent_4", "family": "struct",
     "desc": "indentation 4 espaces (grille de 4)",
     "source": _DOC + " §Style By The Numbers (4 spaces)", "fn": _r_indent_4},
    {"id": "tigerbeetle.no_tabs", "family": "struct",
     "desc": "jamais de tabulation (zig fmt)",
     "source": _DOC + " §Style By The Numbers (Run zig fmt)", "fn": _r_no_tabs},
    {"id": "tigerbeetle.trailing_comma", "family": "struct",
     "desc": "virgule finale avant fermeture multi-ligne (zig fmt)",
     "source": _DOC + " §Style By The Numbers (add a trailing comma)", "fn": _r_trailing_comma_multiline},
    {"id": "tigerbeetle.comment_space", "family": "struct",
     "desc": "espace après // (commentaires = prose)",
     "source": _DOC + " §Naming Things (Comments are sentences)", "fn": _r_comment_space},
    {"id": "tigerbeetle.comment_sentences", "family": "struct",
     "desc": "blocs de commentaires en phrases (majuscule + point) ≥80 %",
     "source": _DOC + " §Naming Things (Comments are sentences)", "fn": _r_comment_sentences},
    {"id": "tigerbeetle.explicit_int_types", "family": "struct",
     "desc": "types entiers explicites (pas de c_* hors interop)",
     "source": _DOC + " §Safety (explicitly-sized types)", "fn": _r_explicit_int_types},
]
