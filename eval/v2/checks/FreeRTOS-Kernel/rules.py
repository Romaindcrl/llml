#!/usr/bin/env python3
"""v2 — règles d'adhérence FreeRTOS-Kernel (C) — UNIQUEMENT famille « struct »
(texte/regex : pas d'AST Python sur du C ; le moteur passe tree=None après
l'échec attendu d'ast.parse).

Sources : FreeRTOS-Kernel/.github/uncrustify.cfg @9db704cd (formatage, exécuté
en CI) + guide officiel « FreeRTOS coding standard and style guide »
https://www.freertos.org/FreeRTOS-Coding-Standard-and-Style-Guide.html
(notation hongroise, préfixes prv/config/pd, interdits stdlib) + MISRA.md du
clone. Entrée : fonction C RECONSTRUITE (header + corps) au niveau colonne 0 ;
marche aussi sur un fichier entier.

Toutes les analyses lexicales passent par _code() : commentaires, chaînes et
lignes préprocesseur remplacés par des blancs (mêmes offsets de lignes), pour
ne jamais matcher `#if ( configUSE_… )` ou un `if (` dans un commentaire.
Identifiants imposés de l'extérieur exclus (linker `__*`, vecteurs `*Handler`,
`main`) — cf. CALIBRATION.md.
"""
from __future__ import annotations

import re

_CFG = "FreeRTOS-Kernel/.github/uncrustify.cfg @9db704cd"
_GUIDE = "https://www.freertos.org/FreeRTOS-Coding-Standard-and-Style-Guide.html"
_MISRA = "FreeRTOS-Kernel/MISRA.md @9db704cd"


def _strip_comments_strings(src: str) -> str:
    """Commentaires /* */ et //, littéraux "…" et '…' remplacés par des blancs."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j])); i = j
        elif c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i)); i = j
        elif c in "\"'":
            q, j = c, i + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(q + " " * max(0, j - i - 2) + (q if j - i >= 2 else "")); i = j
        else:
            out.append(c); i += 1
    return "".join(out)


def _mask_preproc(code: str) -> str:
    """Vide les lignes préprocesseur (et leurs continuations par backslash)."""
    out, cont = [], False
    for line in code.split("\n"):
        if cont or line.lstrip().startswith("#"):
            cont = line.rstrip().endswith("\\")
            out.append("")
        else:
            cont = False
            out.append(line)
    return "\n".join(out)


_code_cache: dict[int, tuple[str, str]] = {}


def _code(src: str) -> tuple[str, str]:
    """(code sans commentaires/chaînes, idem sans préprocesseur)."""
    key = hash(src)
    if key not in _code_cache:
        stripped = _strip_comments_strings(src)
        _code_cache.clear() if len(_code_cache) > 256 else None
        _code_cache[key] = (stripped, _mask_preproc(stripped))
    return _code_cache[key]


# --- notation hongroise : déclarations --------------------------------------------------

_QUALS = r"(?:static\s+|const\s+|volatile\s+|extern\s+|register\s+|PRIVILEGED_DATA\s+|portDONT_DISCARD\s+)*"


def _decls(code: str, types: str):
    """[(type, star, nom)] des déclarations « TYPE [*] nom [;=,)[] » hors
    typedef/#/return/extern et hors identifiants réservés __*."""
    pat = re.compile(r"(?m)^(?P<head>\s*" + _QUALS + r")(?P<t>" + types +
                     r")\s+(?P<star>\*\s*)?(?:const\s+)?(?P<n>[A-Za-z_]\w*)\s*(?=[;=,\[)])")
    lines = code.split("\n")
    out = []
    for m in pat.finditer(code):
        ln = code[:m.start()].count("\n")
        head_line = lines[ln].lstrip() if ln < len(lines) else ""
        if head_line.startswith(("typedef", "return")) or "extern" in m.group("head"):
            continue
        name = m.group("n")
        if name.startswith("__") or name in ("void", "const"):
            continue
        out.append((m.group("t"), bool(m.group("star")), name))
    # paramètres sur la ligne de signature : TYPE [*] nom suivis de , ou )
    for m in re.finditer(r"[(,]\s*(?:const\s+|volatile\s+)*(" + types +
                         r")\s+(\*\s*)?(?:const\s+)?([A-Za-z_]\w*)\s*(?=[,)\[])", code):
        name = m.group(3)
        if not name.startswith("__"):
            out.append((m.group(1), bool(m.group(2)), name))
    return out


def _mk_hungarian(slug, types, prefixes, label, appl_hint, exact_ok=()):
    types_re = "|".join(types)

    def fn(tree, source, path):
        _, code = _code(source)
        seen = [(t, s, n) for (t, s, n) in _decls(code, types_re) if not s]
        # dédoublonne (params matchés 2x si en tête de ligne)
        seen = sorted(set(seen))
        if not seen:
            return False, False, "construction absente"
        bad = [n for (t, s, n) in seen if not n.startswith(prefixes) and n not in exact_ok]
        return True, not bad, (f"{label}: {sorted(set(bad))[:5]}" if bad
                               else f"{len(seen)} décl. {appl_hint} ok")

    return fn


def _r_ptr_prefix(tree, source, path):
    """Pointeurs déclarés : nom préfixé p (px/pul/puc/pc/pv…)."""
    _, code = _code(source)
    types_re = (r"uint32_t|uint16_t|uint8_t|int32_t|char|void|BaseType_t|UBaseType_t|"
                r"TickType_t|StackType_t|HeapRegion_t|[A-Z]\w*_t")
    ptrs = sorted(set((t, n) for (t, s, n) in _decls(code, types_re) if s))
    if not ptrs:
        return False, False, "construction absente"
    bad = [n for (t, n) in ptrs if not n.lstrip("_").startswith("p")]
    return True, not bad, f"pointeurs sans p: {sorted(set(bad))[:5]}" if bad else f"{len(ptrs)} ptr ok"


def _r_pv_pc_prefix(tree, source, path):
    """void * → pv…, char * → pc… (précision du préfixe pointeur)."""
    _, code = _code(source)
    ptrs = sorted(set((t, n) for (t, s, n) in _decls(code, r"void|char") if s))
    if not ptrs:
        return False, False, "construction absente"
    want = {"void": "pv", "char": "pc"}
    bad = [n for (t, n) in ptrs if not n.lstrip("_").startswith(want[t])]
    return True, not bad, f"pv/pc attendus: {sorted(set(bad))[:5]}" if bad else f"{len(ptrs)} ptr void/char ok"


# --- fonctions ---------------------------------------------------------------------------

_FN_DEF = re.compile(
    r"(?m)^(?P<static>static\s+)?(?:inline\s+|__inline\s+|portFORCE_INLINE\s+)*"
    r"(?P<t>void|BaseType_t|UBaseType_t|TickType_t|uint32_t|uint16_t|uint8_t|size_t|char|"
    r"eSleepModeStatus|eTaskState|eNotifyValue|TaskHandle_t|QueueHandle_t|TimerHandle_t|"
    r"EventGroupHandle_t|StackType_t|HeapStats_t)\s+(?P<star>\*\s*)?(?P<n>[A-Za-z_]\w*)\s*\(")


def _fn_defs(code: str):
    out = []
    for m in _FN_DEF.finditer(code):
        name = m.group("n")
        if name.startswith("__") or "Handler" in name or name in ("main",):
            continue
        if re.match(r"Secure[A-Z]\w*_", name):  # module secure_* : préfixe modulaire sanctionné
            continue
        out.append((bool(m.group("static")), m.group("t"), bool(m.group("star")), name))
    return out


def _r_static_prv(tree, source, path):
    """Fonctions static (privées fichier) préfixées prv."""
    _, code = _code(source)
    statics = [f for f in _fn_defs(code) if f[0]]
    if not statics:
        return False, False, "construction absente"
    bad = [n for (_s, _t, _star, n) in statics if not n.startswith("prv")]
    return True, not bad, f"static sans prv: {bad[:5]}" if bad else f"{len(statics)} static prv ok"


_API_PREF = {
    ("void", False): ("v",), ("void", True): ("pv",),
    ("BaseType_t", False): ("x",), ("UBaseType_t", False): ("ux",),
    ("TickType_t", False): ("x",), ("uint32_t", False): ("ul",),
    ("uint32_t", True): ("pul",), ("uint16_t", False): ("us",),
    ("uint8_t", False): ("uc",), ("uint8_t", True): ("puc",),
    ("size_t", False): ("x", "ux"), ("char", True): ("pc",),
    ("eSleepModeStatus", False): ("e",), ("eTaskState", False): ("e",),
    ("TaskHandle_t", False): ("x",), ("QueueHandle_t", False): ("x",),
    ("TimerHandle_t", False): ("x",), ("EventGroupHandle_t", False): ("x",),
    ("StackType_t", True): ("px",),
}


def _r_api_return_prefix(tree, source, path):
    """Fonctions non static : préfixe = type de retour (v/x/ux/ul/pc/e/pv…)."""
    _, code = _code(source)
    apis = [(t, star, n) for (s, t, star, n) in _fn_defs(code) if not s]
    apis = [(t, star, n) for (t, star, n) in apis if (t, star) in _API_PREF]
    if not apis:
        return False, False, "construction absente"
    bad = [n for (t, star, n) in apis
           if not n.startswith(_API_PREF[(t, star)]) and not n.startswith(("prv", "MPU_", "vApplication"))]
    return True, not bad, f"préfixe/type incohérent: {bad[:5]}" if bad else f"{len(apis)} fonction(s) ok"


# --- formatage uncrustify ------------------------------------------------------------------

def _r_kw_no_space_before_paren(tree, source, path):
    """`if(` collé : sp_before_sparen = remove."""
    _, code = _code(source)
    if not re.search(r"\b(?:if|for|while|switch)\s*\(", code):
        return False, False, "construction absente"
    bad = re.findall(r"\b(?:if|for|while|switch)\s+\(", code)
    return True, not bad, f"{len(bad)} mot-clé(s) avec espace avant (" if bad else "if( collé ok"


def _r_space_inside_parens(tree, source, path):
    """Espaces intérieurs `if( x )` : sp_inside_sparen = force."""
    _, code = _code(source)
    kws = re.findall(r"\b(?:if|for|while|switch)\s*\(\S?", code)
    if not kws:
        return False, False, "construction absente"
    bad_open = re.findall(r"\b(?:if|for|while|switch)\(\S", code)
    return True, not bad_open, (f"{len(bad_open)} parenthèse(s) sans espace intérieur"
                                if bad_open else f"{len(kws)} parenthèse(s) espacée(s) ok")


def _r_allman_braces(tree, source, path):
    """Accolade ouvrante seule sur sa ligne (Allman) : nl_if_brace/nl_fdef_brace = add."""
    _, code = _code(source)
    if "{" not in code:
        return False, False, "construction absente"
    bad = re.findall(r"(?m)(?:\)|\belse|\bdo)[ \t]*\{[ \t]*$", code)
    return True, not bad, f"{len(bad)} accolade(s) K&R en fin de ligne" if bad else "Allman ok"


def _r_else_own_line(tree, source, path):
    """`else` sur sa propre ligne : nl_brace_else/nl_else_brace = add."""
    _, code = _code(source)
    if not re.search(r"\belse\b", code):
        return False, False, "construction absente"
    bad = re.findall(r"\}[ \t]*else\b|\belse[ \t]*\{", code)
    return True, not bad, f"{len(bad)} else mal placé(s)" if bad else "else seuls sur leur ligne"


def _r_no_tabs(tree, source, path):
    """Indentation 4 espaces, jamais de tab (indent_with_tabs = 0)."""
    if not source.strip():
        return False, False, "source vide"
    bad = [i + 1 for i, l in enumerate(source.splitlines()) if l.startswith("\t") or "\t" in l[:24]]
    return True, not bad, f"tabs lignes {bad[:5]}" if bad else "0 tab"


def _r_indent_multiple_4(tree, source, path):
    """Indentation par pas de 4 espaces (indent_columns = 4) — jugée sur les
    DÉBUTS d'instruction (la ligne précédente finit par ; { }), pas sur les
    continuations alignées sur parenthèse (indent_align_paren = true)."""
    code, _ = _code(source)
    offenders, tot = [], 0
    prev_end = ";"
    for i, line in enumerate(code.split("\n")):
        st = line.strip()
        if not st:
            continue
        starts_stmt = prev_end in ";{}"
        prev_end = st[-1]
        if st.startswith(("#", "*")) or not starts_stmt:
            continue
        ind = len(line) - len(line.lstrip(" "))
        tot += 1
        if ind % 4 != 0:
            offenders.append(i + 1)
    if tot == 0:
        return False, False, "source vide"
    return True, not offenders, f"{len(offenders)}/{tot} ligne(s) hors grille 4 (l.{offenders[:4]})" \
        if offenders else f"{tot} début(s) d'instruction sur grille 4"


def _r_braces_mandatory(tree, source, path):
    """Accolades obligatoires après if/else/for/while (mod_full_brace_* = add)."""
    _, code = _code(source)
    lines = code.split("\n")
    tot, bad = 0, []
    for i, line in enumerate(lines):
        st = line.strip()
        m = re.match(r"^(?:\}\s*)?(?:else\s+)?(if|for|while)\s*\(", st) or \
            re.match(r"^\}?\s*(else)\s*$", st)
        if not m:
            continue
        kw = m.group(1)
        if kw == "while" and re.search(r"\)\s*;\s*$", st):
            continue  # fin de do { } while( … );
        # fin de la condition (parenthèses potentiellement multi-lignes)
        j = i
        if kw != "else":
            depth = 0
            while j < len(lines):
                depth += lines[j].count("(") - lines[j].count(")")
                if depth <= 0 and "(" in "".join(lines[i:j + 1]):
                    break
                j += 1
            if j >= len(lines):
                continue  # condition tronquée par la fenêtre : injugeable
        tail = lines[j].strip()
        tot += 1
        after = tail[tail.rindex(")") + 1:].strip() if (kw != "else" and ")" in tail) else ""
        if after and not after.startswith("{"):
            bad.append(i + 1)  # instruction accolée après la parenthèse fermante
            continue
        k = j + 1
        while k < len(lines) and not lines[k].strip():
            k += 1
        nxt = lines[k].strip() if k < len(lines) else ""
        if tail.endswith("{") or nxt.startswith("{"):
            continue
        bad.append(i + 1)
    if tot == 0:
        return False, False, "construction absente"
    return True, not bad, f"{len(bad)} bloc(s) sans accolades (l.{bad[:4]})" if bad else f"{tot} bloc(s) accoladé(s)"


# --- lexique -----------------------------------------------------------------------------

def _r_c_comments_only(tree, source, path):
    """Commentaires /* */ uniquement (cmt_cpp_to_c = true), jamais //."""
    if "/*" not in source and "//" not in source:
        return False, False, "construction absente"
    # scan lexical brut : compter les // qui OUVRENT un commentaire (hors chaînes)
    bad = 0
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if c == "/" and i + 1 < n and source[i + 1] == "*":
            j = source.find("*/", i + 2); i = n if j == -1 else j + 2
        elif c == "/" and i + 1 < n and source[i + 1] == "/":
            bad += 1
            j = source.find("\n", i); i = n if j == -1 else j
        elif c in "\"'":
            q, j = c, i + 1
            while j < n and source[j] != q:
                j += 2 if source[j] == "\\" else 1
            i = min(j + 1, n)
        else:
            i += 1
    return True, bad == 0, f"{bad} commentaire(s) //" if bad else "commentaires C ok"


def _r_no_true_false(tree, source, path):
    """pdTRUE/pdFALSE/pdPASS (préfixe pd) — jamais TRUE/FALSE nus."""
    _, code = _code(source)
    if not re.search(r"\b(?:pdTRUE|pdFALSE|pdPASS|pdFAIL|TRUE|FALSE)\b", code):
        return False, False, "construction absente"
    bad = re.findall(r"\b(?:TRUE|FALSE)\b", code)
    return True, not bad, f"{len(bad)} TRUE/FALSE nu(s)" if bad else "constantes pd* ok"


def _r_if_config_parens(tree, source, path):
    """#if sur macro config* : forme parenthésée `#if ( configX == 1 )`."""
    code = _strip_comments_strings(source)
    tests = re.findall(r"(?m)^\s*#\s*if\s+(.*config\w+.*)$", code)
    tests = [t for t in tests if "defined" not in t]
    if not tests:
        return False, False, "construction absente"
    bad = [t.strip()[:40] for t in tests if "(" not in t]
    return True, not bad, f"#if config sans parenthèses: {bad[:2]}" if bad else f"{len(tests)} #if config ok"


# RÈGLE SUPPRIMÉE À LA CALIBRATION — unsigned_suffix_u (suffixe U sur littéraux
# castés non signés, dérivée MISRA) : le noyau lui-même écrit `( size_t ) 0`
# (secure_heap.c ×3, GCC/RL78/port.c) — 1/5 pass sur le train, 0 déclenchement
# sur les 30 tâches. Pas une pratique du repo. Chiffres dans CALIBRATION.md.


def _r_no_banned_functions(tree, source, path):
    """Pas de sprintf/strcpy/strcat/gets (MISRA 21.6, stdlib bannie du noyau)."""
    _, code = _code(source)
    if not re.search(r"\b[a-z_]+\s*\(", code):
        return False, False, "construction absente"
    bad = re.findall(r"\b(sprintf|strcpy|strcat|gets)\s*\(", code)
    return True, not bad, f"fonctions bannies: {sorted(set(bad))}" if bad else "0 fonction bannie"


# RÈGLE SUPPRIMÉE À LA CALIBRATION — ptr_star_spacing (sp_before/after_ptr_star) :
# les ports historiques écrivent `void *pvParameters` (tâche #07, 47/185 fichiers
# du train) — construct hétérogène dans le repo, non mesurable honnêtement.
# Chiffres dans CALIBRATION.md.

RULES = [
    {"id": "FreeRTOS-Kernel.hungarian_uint32_ul", "family": "struct",
     "desc": "uint32_t locaux/params préfixés ul",
     "source": _GUIDE + " §Naming (variables uint32_t → ul)",
     "fn": _mk_hungarian("ul", ["uint32_t"], ("ul",), "uint32_t sans ul", "uint32_t")},
    {"id": "FreeRTOS-Kernel.hungarian_uint16_us", "family": "struct",
     "desc": "uint16_t préfixés us",
     "source": _GUIDE + " §Naming (uint16_t → us)",
     "fn": _mk_hungarian("us", ["uint16_t"], ("us",), "uint16_t sans us", "uint16_t")},
    {"id": "FreeRTOS-Kernel.hungarian_uint8_uc", "family": "struct",
     "desc": "uint8_t préfixés uc",
     "source": _GUIDE + " §Naming (uint8_t → uc)",
     "fn": _mk_hungarian("uc", ["uint8_t"], ("uc",), "uint8_t sans uc", "uint8_t")},
    {"id": "FreeRTOS-Kernel.hungarian_ubase_ux", "family": "struct",
     "desc": "UBaseType_t préfixés ux (ull sur ports 64 bits, x nu toléré)",
     "source": _GUIDE + " §Naming (UBaseType_t → ux)",
     "fn": _mk_hungarian("ux", ["UBaseType_t"], ("ux", "ull"), "UBaseType_t sans ux",
                          "UBaseType_t", exact_ok=("x",))},
    {"id": "FreeRTOS-Kernel.hungarian_base_x", "family": "struct",
     "desc": "BaseType_t/TickType_t/size_t préfixés x (ux toléré pour size_t)",
     "source": _GUIDE + " §Naming (types non standard → x)",
     "fn": _mk_hungarian("x", ["BaseType_t", "TickType_t", "size_t"], ("x", "ux"),
                          "BaseType/Tick/size_t sans x", "x-typés")},
    {"id": "FreeRTOS-Kernel.ptr_prefix_p", "family": "struct",
     "desc": "pointeurs déclarés préfixés p",
     "source": _GUIDE + " §Naming (pointeurs → p)", "fn": _r_ptr_prefix},
    {"id": "FreeRTOS-Kernel.ptr_pv_pc", "family": "struct",
     "desc": "void* → pv, char* → pc",
     "source": _GUIDE + " §Naming", "fn": _r_pv_pc_prefix},
    {"id": "FreeRTOS-Kernel.static_prv", "family": "struct",
     "desc": "fonctions static préfixées prv",
     "source": _GUIDE + " §Naming (fonctions privées → prv)", "fn": _r_static_prv},
    {"id": "FreeRTOS-Kernel.api_return_prefix", "family": "struct",
     "desc": "préfixe de fonction = type de retour (vTask…, xQueue…)",
     "source": _GUIDE + " §Naming (fonctions)", "fn": _r_api_return_prefix},
    {"id": "FreeRTOS-Kernel.kw_paren_glued", "family": "struct",
     "desc": "if( collé — pas d'espace avant la parenthèse",
     "source": _CFG + " sp_before_sparen=remove", "fn": _r_kw_no_space_before_paren},
    {"id": "FreeRTOS-Kernel.paren_inner_spaces", "family": "struct",
     "desc": "espaces intérieurs if( x )",
     "source": _CFG + " sp_inside_sparen=force", "fn": _r_space_inside_parens},
    {"id": "FreeRTOS-Kernel.allman_braces", "family": "struct",
     "desc": "accolade ouvrante seule sur sa ligne (Allman)",
     "source": _CFG + " nl_if_brace/nl_fdef_brace=add", "fn": _r_allman_braces},
    {"id": "FreeRTOS-Kernel.else_own_line", "family": "struct",
     "desc": "else sur sa propre ligne",
     "source": _CFG + " nl_brace_else/nl_else_brace=add", "fn": _r_else_own_line},
    {"id": "FreeRTOS-Kernel.no_tabs", "family": "struct",
     "desc": "jamais de tabulation",
     "source": _CFG + " indent_with_tabs=0", "fn": _r_no_tabs},
    {"id": "FreeRTOS-Kernel.indent_4", "family": "struct",
     "desc": "indentation par pas de 4 espaces (≥90 % des lignes)",
     "source": _CFG + " indent_columns=4", "fn": _r_indent_multiple_4},
    {"id": "FreeRTOS-Kernel.braces_mandatory", "family": "struct",
     "desc": "accolades obligatoires après if/else/for/while",
     "source": _CFG + " mod_full_brace_*=add", "fn": _r_braces_mandatory},
    {"id": "FreeRTOS-Kernel.c_comments_only", "family": "struct",
     "desc": "commentaires /* */ — jamais //",
     "source": _CFG + " cmt_cpp_to_c=true", "fn": _r_c_comments_only},
    {"id": "FreeRTOS-Kernel.pd_macros", "family": "struct",
     "desc": "pdTRUE/pdFALSE — jamais TRUE/FALSE nus",
     "source": _GUIDE + " §Naming (macros pd*)", "fn": _r_no_true_false},
    {"id": "FreeRTOS-Kernel.if_config_parens", "family": "struct",
     "desc": "#if ( configX == 1 ) parenthésé",
     "source": _CFG + " (pp) + usage config* du guide", "fn": _r_if_config_parens},
    {"id": "FreeRTOS-Kernel.no_banned_functions", "family": "struct",
     "desc": "pas de sprintf/strcpy/strcat/gets",
     "source": _MISRA + " (Dir 4.12/21.6 stdlib)", "fn": _r_no_banned_functions},
]
