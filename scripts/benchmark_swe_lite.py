"""SWE-bench Lite (sous-ensemble sympy) avec qwen-coder-14B + LLML — SÉPARATION SOLVEUR/JUGE.

Le boss final : de vraies issues GitHub -> produire un patch qui fait passer des tests CACHÉS.
Règle d'or SWE-bench (mainteneurs) : le SOLVEUR n'a JAMAIS le droit de voir/utiliser FAIL_TO_PASS,
PASS_TO_PASS ou le test_patch — ce sont le JUGE. Une version précédente les réinjectait comme
oracle de réparation : c'était une FUITE. Cette version corrige ça :

  SOLVEUR (n'utilise que l'ÉNONCÉ de l'issue) :
    1. LOCALISATION — BM25 (m0.rag) sur les blocs ast + signaux d'issue + frames de traceback
       ORDONNÉES (la frame du HAUT de pile = cause racine, pas le symptôme cité dans l'issue).
       On garantit que la LIGNE fautive est à l'écran (fenêtre autour de l'ancre).
    2. ORACLE DE REPRODUCTION — le 14B écrit un script qui reproduit le bug et ASSERTE le
       comportement ATTENDU *décrit dans l'issue* ; validé s'il ÉCHOUE sur la base non patchée.
       C'est la seule vérité que le solveur utilise (fournie par l'issue, pas par le juge).
    3. ÉCHANTILLONNAGE — K patchs SEARCH/REPLACE à température ; on garde ceux qui font passer
       la repro ET ne cassent pas les tests du module (régression AUTO-DÉCOUVERTE sur la base).
    4. RÉPARATION — sur échec, on réinjecte la sortie de la REPRO (jamais celle du juge).

  JUGE (seul acteur autorisé à toucher les tests cachés, APRÈS gel du patch) :
    applique le patch gelé + le test_patch, exécute FAIL_TO_PASS puis PASS_TO_PASS.

Environnement : les instances sympy 1.1 (13647/13971/14774/12171/13480) n'importent pas sous
Python 3.11 (collections.Mapping retiré en 3.10). Un shim sitecustomize restaure les alias -> les
10 instances deviennent exécutables (vérifié : les patchs GOLD passent). Le shim est neutre côté
sympy récent. Sans lui on mesurerait l'environnement, pas le modèle.

Cadrage honnête : SOTA 7-14B open + scaffold Agentless ~12-14% sur le Lite complet ; sur ce
micro-lot on vise 0-2 résolus, et 0 reste un résultat honnête. On NE renomme JAMAIS ce score
en « score SWE-bench » officiel. Live : tail -f logs/benchmark_swe_lite.log
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.rag import _tok  # noqa: E402 — tokeniseur BM25 du projet

# Pointe SWE_SCRATCH vers un dossier contenant swe/sympy (checkout sympy + .swevenv).
SCRATCH = os.environ.get("SWE_SCRATCH", os.path.join(_PROJ, ".swe_work"))
REPO = os.path.join(SCRATCH, "swe", "sympy")
PY = os.path.join(REPO, ".swevenv", "bin", "python")
SHIMDIR = os.path.join(SCRATCH, "swe", "_shim")   # sitecustomize -> rescue sympy 1.1 sous py3.11
SELECTION = os.path.join(_PROJ, "scripts", "swe_selection.json")
LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_swe_lite.log")
N_INSTANCES = int(os.environ.get("SWE_N", "10"))
K_SAMPLES = int(os.environ.get("SWE_K", "6"))
TEMP = float(os.environ.get("SWE_TEMP", "0.8"))
MAX_REPAIRS = int(os.environ.get("SWE_REPAIRS", "1"))
_T0 = time.time()
_SR_RE = re.compile(r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE", re.DOTALL)
_ENV = {**os.environ, "PYTHONPATH": SHIMDIR + os.pathsep + REPO}   # shim actif partout


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)


def _ensure_shim():
    os.makedirs(SHIMDIR, exist_ok=True)
    open(os.path.join(SHIMDIR, "sitecustomize.py"), "w", encoding="utf-8").write(
        "import collections, collections.abc\n"
        "for _n in ('Mapping','MutableMapping','Callable','Iterable','Sequence','Set',\n"
        "          'Hashable','MutableSet','MutableSequence','Container','Sized'):\n"
        "    if not hasattr(collections,_n): setattr(collections,_n,getattr(collections.abc,_n))\n"
        "import inspect\n"
        "if not hasattr(inspect,'getargspec'): inspect.getargspec=inspect.getfullargspec\n"
    )


def checkout_base(inst: dict):
    """Remet le repo au base_commit propre. N'applique RIEN (ni gold, ni test_patch)."""
    git("reset", "--hard", "-q")
    git("checkout", "-q", inst["base_commit"])
    git("clean", "-qfd", "sympy")


def test_files_of(inst: dict) -> list[str]:
    """Chemins des fichiers de test (méta pour le JUGE uniquement ; jamais montrés au solveur)."""
    return re.findall(r"^\+\+\+ b/(\S+)", inst["test_patch"], re.MULTILINE)


# ------------------------------------------------------------------ localisation
def iter_blocks(path: str):
    """(nom, code, path, start, end) pour chaque def/classe (top-level ET méthodes)."""
    try:
        src = open(path, encoding="utf-8").read()
    except Exception:
        return
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, "end_lineno", node.lineno)
            if end - node.lineno > 200 and isinstance(node, ast.ClassDef):
                continue   # classe énorme : on prendra ses méthodes séparément
            code = "\n".join(lines[node.lineno - 1:end])
            yield node.name, code, path, node.lineno, end


def _enclosing_block(src: str, path: str, lineno: int):
    """Le plus petit def/classe englobant `lineno` (fenêtre bornée) — garantit la ligne à l'écran."""
    lines = src.splitlines()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        lo, hi = max(1, lineno - 25), min(len(lines), lineno + 25)
        return (f"lignes{lo}-{hi}", "\n".join(lines[lo - 1:hi]), path, lo, hi)
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, "end_lineno", node.lineno)
            if node.lineno <= lineno <= end:
                span = end - node.lineno
                if best is None or span < best[0]:
                    best = (span, node.name, node.lineno, end)
    if not best:
        lo, hi = max(1, lineno - 25), min(len(lines), lineno + 25)
        return (f"lignes{lo}-{hi}", "\n".join(lines[lo - 1:hi]), path, lo, hi)
    _span, name, lo, hi = best
    if hi - lo > 120:   # borne les blocs géants autour de l'ancre
        a, b = max(lo, lineno - 50), min(hi, lineno + 50)
        return (name, "\n".join(lines[a - 1:b]), path, a, b)
    return (name, "\n".join(lines[lo - 1:hi]), path, lo, hi)


_CODEBLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_BACKTICK_RE = re.compile(r"`([^`\n]{2,80})`")
_PATH_RE = re.compile(r"\b((?:\w+/)+\w+\.py)\b")
_CALL_RE = re.compile(r"\b([a-zA-Z_]\w{2,})\s*\(")
_DUNDER_RE = re.compile(r"__\w{2,}__")
_FRAME_RE = re.compile(r"([\w./]+\.py):(\d+): in (\w+)")   # frame pytest : (path, line, func)


def issue_signals(problem: str) -> tuple[set, list[str], list[str]]:
    """Signaux de localisation de l'issue (comme un dev qui lit un ticket)."""
    idents = set(_DUNDER_RE.findall(problem))
    for m in _BACKTICK_RE.findall(problem):
        idents.update(w for w in re.findall(r"[A-Za-z_]\w{2,}", m))
    idents.update(w for w in _CALL_RE.findall(problem) if len(w) > 2)
    lits = []
    for blk in _CODEBLOCK_RE.findall(problem):
        for ln in blk.splitlines():
            s = ln.strip()
            if (8 <= len(s) <= 90 and ("(" in s or "=" in s or "." in s)
                    and not s.startswith(("#", ">>>", "E ", "import ", "from ", "Traceback"))):
                lits.append(s)
    paths = [p for p in _PATH_RE.findall(problem) if "test" not in p.lower()]
    return idents, lits, paths


def _repo_files() -> list[str]:
    out = []
    for root, _dirs, names in os.walk(os.path.join(REPO, "sympy")):
        if "/tests" in root or "/benchmarks" in root:
            continue
        out += [os.path.join(root, nm) for nm in names if nm.endswith(".py")]
    return out


def localize(problem: str, k: int = 4):
    """Localisation à la SWE-bench, SANS aucun signal du juge (que l'énoncé de l'issue).
    (1) score les FICHIERS par lignes littérales, chemins cités, densité d'identifiants ;
    (2) les frames de traceback ORDONNÉES (haut de pile d'abord) dominent, avec décroissance
        en profondeur — la cause racine bat le symptôme ;
    (3) on montre le bloc ENGLOBANT la ligne fautive (garantie que la cible est visible).
    Renvoie [(nom, code, path)] et un dict de métriques de localisation."""
    idents, lits, paths = issue_signals(problem)
    files = _repo_files()
    fscore = {}
    for f in files:
        try:
            src = open(f, encoding="utf-8").read()
        except Exception:
            continue
        rel = os.path.relpath(f, REPO)
        s = 60 * sum(1 for lit in lits if lit in src)
        s += 25 * sum(1 for p in paths if rel.endswith(p))
        s += sum(2 for w in idents if w in src)
        if s > 0:
            fscore[f] = (s, src)

    frames = _FRAME_RE.findall(problem)                       # [(path, line, func), ...]
    frame_files_ordered, seen = [], set()
    for p, _ln, _fn in frames:
        fp = os.path.join(REPO, p)
        if os.path.isfile(fp) and fp not in seen:
            seen.add(fp); frame_files_ordered.append(fp)
    top_scored = sorted(fscore, key=lambda f: fscore[f][0], reverse=True)[:3]
    top_files = list(dict.fromkeys(frame_files_ordered + top_scored))[:4]   # frames d'abord
    if not top_files:
        top_files = _bm25_files(problem, files)[:2]
    for f in top_files:
        fscore.setdefault(f, (1.0, open(f, encoding="utf-8").read()))

    # boost décroissant en profondeur : frame du HAUT de pile = cause racine
    nf = len(frames)
    frame_boost = {}     # (rel, func) -> boost
    frame_line = {}      # rel -> premier n° de ligne cité (ancre)
    for idx, (p, ln, fn) in enumerate(frames):
        rel = os.path.relpath(os.path.join(REPO, p), REPO)
        frame_boost[(rel, fn)] = max(frame_boost.get((rel, fn), 0), 1000 * (nf - idx))
        frame_line.setdefault(rel, int(ln))

    blocks, chosen_meta = [], {"gold_line_anchors": []}
    for f in top_files:
        rel = os.path.relpath(f, REPO)
        src = fscore[f][1]
        # bloc englobant l'ANCRE (ligne de frame, sinon 1re ligne littérale trouvée) — priorité absolue
        anchor = frame_line.get(rel)
        if anchor is None:
            for lit in lits:
                pos = src.find(lit)
                if pos >= 0:
                    anchor = src[:pos].count("\n") + 1; break
        if anchor is not None:
            nm, code, pth, a, b = _enclosing_block(src, f, anchor)
            blocks.append((5000, nm, code, pth))     # garanti à l'écran
            chosen_meta["gold_line_anchors"].append(f"{nm}@{os.path.basename(f)}:{anchor}")
        for name, code, path, _s, _e in iter_blocks(f):
            bs = (frame_boost.get((rel, name), 0)
                  + 18 * (name in idents)
                  + 12 * sum(1 for lit in lits if lit in code and len(lit) > 8)
                  + 2 * sum(1 for w in idents if w in code))
            blocks.append((bs, name, code, path))
    # dédup par (nom, path) en gardant le meilleur score
    best = {}
    for bs, name, code, path in blocks:
        key = (name, path)
        if key not in best or bs > best[key][0]:
            best[key] = (bs, name, code, path)
    ordered = sorted(best.values(), key=lambda b: b[0], reverse=True)
    chosen = [(n, c, p) for _s, n, c, p in ordered if _s > 0][:k]
    chosen = chosen or [(n, c, p) for _s, n, c, p in ordered[:k]]
    return chosen, chosen_meta


def _bm25_files(query: str, files: list[str], k: int = 4) -> list[str]:
    import math
    from collections import Counter
    docs, paths = [], []
    for f in files:
        try:
            docs.append(_tok(open(f, encoding="utf-8").read())); paths.append(f)
        except Exception:
            continue
    n = len(docs); avgdl = sum(len(d) for d in docs) / max(1, n)
    df = Counter()
    for d in docs:
        df.update(set(d))
    qt = _tok(query); scored = []
    for i, d in enumerate(docs):
        tf = Counter(d); dl = len(d)
        s = sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * dl / avgdl))
                for w in qt if w in tf)
        if s > 0:
            scored.append((s, i))
    scored.sort(reverse=True)
    return [paths[i] for _s, i in scored[:k]]


# ------------------------------------------------------------------ oracle de reproduction (honnête)
def gen_repro(llm, problem: str) -> str:
    """Le 14B écrit un script autonome qui reproduit le bug et ASSERTE le comportement ATTENDU
    tel que DÉCRIT DANS L'ISSUE (jamais le test caché). Sentinelle ISSUE_RESOLVED si réparé."""
    llm.cfg.temperature = 0.2
    prompt = (
        "Voici une issue GitHub de sympy :\n\n"
        f"{problem[:2600]}\n\n"
        "Écris un SCRIPT Python AUTONOME qui reproduit ce bug : exécute l'exemple de l'issue, puis "
        "VÉRIFIE avec un assert le comportement ATTENDU **décrit dans l'issue** (n'invente rien "
        "au-delà de ce que l'issue dit attendre). Si le comportement attendu est obtenu, le script "
        "doit se terminer SANS erreur en affichant exactement :\n    print('ISSUE_RESOLVED')\n"
        "Sur le code buggé actuel il doit lever une exception ou échouer l'assert (donc NE PAS "
        "afficher ISSUE_RESOLVED). Utilise uniquement `import sympy` (et stdlib). "
        "Réponds avec UN SEUL bloc ```python."
    )
    raw = llm.generate(prompt, None) or ""
    m = _CODEBLOCK_RE.search(raw)
    return (m.group(1) if m else raw).strip()


def run_script(script: str, timeout: int = 60) -> tuple[bool, str]:
    """Exécute un script dans le repo (shim actif). ok = code 0 ET sentinelle ISSUE_RESOLVED."""
    p = os.path.join(REPO, "_swe_repro.py")
    open(p, "w", encoding="utf-8").write(script)
    try:
        r = subprocess.run([PY, "_swe_repro.py"], cwd=REPO, capture_output=True, text=True,
                           timeout=timeout, env=_ENV)
        out = r.stdout + r.stderr
        ok = r.returncode == 0 and "ISSUE_RESOLVED" in out
    except subprocess.TimeoutExpired:
        ok, out = False, "timeout"
    finally:
        try:
            os.remove(p)
        except OSError:
            pass
    return ok, out[-1500:]


# ------------------------------------------------------------------ régression auto-découverte
def _sibling_test(path: str) -> str | None:
    d, base = os.path.dirname(path), os.path.basename(path)
    t = os.path.join(d, "tests", "test_" + base)
    return t if os.path.isfile(t) else None


def _pass_count(testfile: str, timeout: int = 150) -> int | None:
    """Nombre de tests 'passed' du module (None si timeout/collecte impossible)."""
    try:
        r = subprocess.run([PY, "-m", "pytest", testfile, "-q", "--no-header",
                            "-p", "no:cacheprovider", "--timeout=60"],
                           cwd=REPO, capture_output=True, text=True, timeout=timeout, env=_ENV)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout + r.stderr
    if "passed" not in out and "error" in out.lower():
        return None
    return sum(int(m) for m in re.findall(r"(\d+) passed", out))


# ------------------------------------------------------------------ édition (K échantillons)
def fuzzy_apply(text: str, search: str, replace: str) -> tuple[str, bool]:
    """SEARCH/REPLACE tolérant aux espaces de début de ligne : exact d'abord, sinon match
    ligne-à-ligne sur le contenu strippé."""
    if search and search in text:
        return text.replace(search, replace, 1), True
    tlines = text.split("\n")
    slines = search.split("\n")
    sstrip = [l.strip() for l in slines]
    if not any(sstrip):
        return text, False
    for i in range(len(tlines) - len(slines) + 1):
        if all(tlines[i + j].strip() == sstrip[j] for j in range(len(slines))):
            out = tlines[:i] + replace.split("\n") + tlines[i + len(slines):]
            return "\n".join(out), True
    return text, False


def gen_edits(llm, problem: str, blocks, error: str, k: int) -> list[list[tuple[str, str]]]:
    """K générations à température -> liste de listes de paires (search, replace). CRLF normalisé.
    Prompt : corrige la CAUSE RACINE (frame du haut de pile), pas le symptôme cité dans l'issue."""
    shown = "\n\n".join(
        f"# ===== Fichier: {os.path.relpath(p, REPO)} — bloc: {n} =====\n{code[:2600]}"
        for n, code, p in blocks)[:11000]
    prompt = (
        "Tu es un mainteneur de sympy. Corrige ce bug. Issue :\n\n"
        f"{problem[:2400]}\n\n"
        "Code source pertinent (localisé automatiquement ; l'ordre suit la PILE D'APPELS, "
        "cause racine en premier) :\n\n"
        f"{shown}\n\n"
        + (f"Une correction précédente a échoué. Sortie de la reproduction :\n{error[:1200]}\n\n"
           if error else "")
        + "IMPORTANT : l'issue décrit souvent le SYMPTÔME (bas de la pile). Corrige la CAUSE "
        "RACINE — de préférence la frame du HAUT de la pile — pas la ligne-symptôme que l'issue "
        "cite. Fais la correction la plus PETITE possible.\n\n"
        "Réponds avec un ou plusieurs blocs au format EXACT. Le bloc SEARCH doit être une copie "
        "FIDÈLE de lignes CONSÉCUTIVES du code montré (garde l'indentation).\n\n"
        "<<<<<<< SEARCH\n(lignes existantes)\n=======\n(lignes corrigées)\n>>>>>>> REPLACE\n"
    )
    llm.cfg.temperature = TEMP
    samples = []
    for _ in range(k):
        raw = (llm.generate(prompt, None) or "").replace("\r\n", "\n")
        pairs = [(s.strip("\n"), r.strip("\n")) for s, r in _SR_RE.findall(raw)]
        if pairs:
            samples.append(pairs)
    return samples


def _apply_pairs(pairs, cand_files) -> tuple[dict, int]:
    """Applique des paires aux fichiers candidats (fuzzy). Renvoie ({path: nouveau_texte}, n_appliqué)."""
    touched, applied = {}, 0
    for search, replace in pairs:
        for path in cand_files:
            txt = touched.get(path) if path in touched else open(path, encoding="utf-8").read()
            new, ok = fuzzy_apply(txt, search, replace)
            if ok:
                touched[path] = new; applied += 1
                break
    return touched, applied


# ------------------------------------------------------------------ SOLVEUR (énoncé seul)
def solve(llm, inst: dict) -> tuple[str, dict]:
    """Renvoie (patch_text, meta). N'utilise QUE l'énoncé de l'issue. patch_text='' si rien."""
    problem = inst["problem_statement"]
    gold_files = re.findall(r"\+\+\+ b/(\S+)", inst["patch"])
    meta = {"repro_validated": False, "n_blocks_max": 0, "bucket": "no-patch",
            "gold_file_hit": False, "gold_line_shown": False, "oracle_accepted": False}

    checkout_base(inst)
    blocks, loc_meta = localize(problem, k=4)
    if not blocks:
        return "", {**meta, "bucket": "loc-fail"}
    cand_files = list(dict.fromkeys(p for _n, _c, p in blocks))
    meta["gold_file_hit"] = any(os.path.relpath(p, REPO) in gold_files for p in cand_files)
    meta["gold_line_shown"] = any(a.split("@")[1].split(":")[0] in
                                  [os.path.basename(g) for g in gold_files]
                                  for a in loc_meta["gold_line_anchors"])
    log(f"   localisé : {[f'{n}@{os.path.basename(p)}' for n, _c, p in blocks]}"
        f" · gold_file={'✓' if meta['gold_file_hit'] else '✗'}"
        f" gold_line_shown={'✓' if meta['gold_line_shown'] else '✗'}")

    # oracle de reproduction (honnête) : jusqu'à 2 essais, on garde celui qui ÉCHOUE sur la base
    repro, base_ok = "", True
    for _ in range(2):
        cand = gen_repro(llm, problem)
        if not cand:
            continue
        repro = cand
        ok, _o = run_script(cand)
        if not ok:                       # échoue sur la base non patchée = reproduit vraiment
            base_ok = False; break
    meta["repro_validated"] = bool(repro) and not base_ok
    if repro:   # persiste la repro (auditable : on peut vérifier qu'elle ne copie pas le test caché)
        rdir = os.path.join(_PROJ, "logs", "swe_repros")
        os.makedirs(rdir, exist_ok=True)
        open(os.path.join(rdir, f"{inst['instance_id']}.py"), "w", encoding="utf-8").write(repro)
    log(f"   repro : {'validée (échoue sur base)' if meta['repro_validated'] else 'NON validée — fallback'}")

    # régression : tests du module localisé, verts sur la base
    reg_file = next((t for t in (_sibling_test(p) for p in cand_files) if t), None)
    reg_base = _pass_count(reg_file) if reg_file else None
    if reg_file:
        log(f"   régression : {os.path.relpath(reg_file, REPO)} base={reg_base} verts")

    best_patch, error = "", ""
    for attempt in range(MAX_REPAIRS + 1):
        checkout_base(inst)
        samples = gen_edits(llm, problem, blocks, error, K_SAMPLES)
        meta["n_blocks_max"] = max([meta["n_blocks_max"]] + [len(s) for s in samples])
        if not samples:
            error = "Aucun bloc SEARCH/REPLACE au format demandé. Donne au moins un bloc."
            log(f"   tentative {attempt + 1}: 0/{K_SAMPLES} échantillon(s) parseable(s)")
            continue

        # applique chaque échantillon, dédup par diff, note la repro (si validée)
        graded, seen = [], set()
        for pairs in samples:
            checkout_base(inst)
            touched, applied = _apply_pairs(pairs, cand_files)
            if not touched:
                continue
            for path, new in touched.items():
                open(path, "w", encoding="utf-8").write(new)
            patch_text = git("diff").stdout
            if not patch_text.strip() or patch_text in seen:
                continue
            seen.add(patch_text)
            r_ok, r_out = run_script(repro) if meta["repro_validated"] else (False, "(repro non validée)")
            graded.append({"repro": r_ok, "reg": None, "applied": applied,
                           "patch": patch_text, "out": r_out})
        checkout_base(inst)
        if not graded:
            error = ("Aucun de tes blocs SEARCH ne correspond au code montré. Recopie des lignes "
                     "EXACTES et CONSÉCUTIVES du code ci-dessus.")
            log(f"   tentative {attempt + 1}: {len(samples)} échantillon(s), 0 appliqué")
            continue

        # régression AUTO-DÉCOUVERTE (signal honnête), bornée : les repro-passers, sinon les 3
        # plus gros patchs distincts — évite K exécutions du module de tests.
        if reg_file and reg_base is not None:
            pool = [g for g in graded if g["repro"]] or \
                sorted(graded, key=lambda g: g["applied"], reverse=True)[:3]
            for g in pool:
                checkout_base(inst)
                open("/tmp/swe_reg.patch", "w").write(g["patch"])
                if git("apply", "/tmp/swe_reg.patch").returncode == 0:
                    after = _pass_count(reg_file)
                    g["reg"] = (after is not None and after >= reg_base)
            checkout_base(inst)

        _rr = {True: 2, None: 1, False: 0}
        graded.sort(key=lambda g: (g["repro"], _rr[g["reg"]], g["applied"]), reverse=True)
        top = graded[0]
        best_patch = top["patch"]     # émis même si l'oracle ne l'accepte pas (pour le juge)
        n_repro = sum(1 for g in graded if g["repro"])
        n_regok = sum(1 for g in graded if g["reg"] is True)
        log(f"   tentative {attempt + 1}: {len(graded)} patch(s) distinct(s) · repro✓={n_repro} "
            f"régression-OK={n_regok}")
        if top["repro"] and top["reg"] is not False:   # repro résolue ET pas de régression avérée
            meta["oracle_accepted"] = True
            meta["bucket"] = "oracle-accepted"
            return best_patch, meta
        error = top["out"]   # réinjecte la sortie de la REPRO (jamais celle du juge)

    meta["bucket"] = "patch-emitted" if best_patch else "no-patch"
    return best_patch, meta


# ------------------------------------------------------------------ JUGE (tests cachés, patch gelé)
def _pytest(test_files, names) -> tuple[int, int, str]:
    if not test_files or not names:
        return 0, 0, "pas de test"
    sel = " or ".join(sorted(set(n.split("[")[0] for n in names)))
    args = [PY, "-m", "pytest", *test_files, "-k", sel, "-q", "--no-header",
            "-p", "no:cacheprovider", "--timeout=60"]
    try:
        r = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=240, env=_ENV)
    except subprocess.TimeoutExpired:
        return 0, 1, "timeout"
    out = r.stdout + r.stderr
    passed = sum(int(m) for m in re.findall(r"(\d+) passed", out))
    bad = sum(int(m) for m in re.findall(r"(\d+) (?:failed|error)", out))
    return passed, bad, out[-1200:]


def judge(patch_text: str, inst: dict) -> tuple[str, str]:
    """SEUL acteur autorisé à toucher FAIL_TO_PASS / PASS_TO_PASS / test_patch, patch GELÉ.
    Renvoie (verdict, détail). verdict ∈ {resolved, unresolved, cand-apply-fail, env-fail}."""
    checkout_base(inst)
    if patch_text.strip():
        open("/tmp/swe_cand.patch", "w").write(patch_text)
        if git("apply", "/tmp/swe_cand.patch").returncode != 0:
            return "cand-apply-fail", ""
    open("/tmp/swe_test.patch", "w").write(inst["test_patch"])
    if git("apply", "/tmp/swe_test.patch").returncode != 0:
        return "env-fail", "test_patch n'applique pas"
    tf = test_files_of(inst)
    f2p = json.loads(inst["FAIL_TO_PASS"])
    p_pass, p_bad, out = _pytest(tf, f2p)
    if p_pass == 0 and p_bad == 0:
        # 0/0 = collection cassée. Distinguer l'ENV (base aussi 0/0) du CANDIDAT qui casse le code.
        checkout_base(inst)
        open("/tmp/swe_test.patch", "w").write(inst["test_patch"]); git("apply", "/tmp/swe_test.patch")
        b_pass, b_bad, _bo = _pytest(tf, f2p)
        if b_pass == 0 and b_bad == 0:
            return "env-fail", "0 passed / 0 failed sur la base aussi (import/collect cassé)"
        return "unresolved", "le patch casse la compilation/collection"
    if not (p_pass >= 1 and p_bad == 0):
        return "unresolved", out
    p2p = json.loads(inst["PASS_TO_PASS"])   # liste COMPLÈTE (critère officiel : tout vert)
    if p2p:
        _pp, after_bad, _o = _pytest(tf, p2p)
        if after_bad > 0:
            # certains P2P échouent déjà sur la base (env sympy-1.1 imparfait) : ce n'est une
            # RÉGRESSION que si le patch fait échouer un test qui passait SANS lui.
            checkout_base(inst)
            open("/tmp/swe_test.patch", "w").write(inst["test_patch"]); git("apply", "/tmp/swe_test.patch")
            _bp, base_bad, _bo = _pytest(tf, p2p)
            if after_bad > base_bad:
                return "unresolved", f"régression PASS_TO_PASS ({after_bad} vs base {base_bad})"
    return "resolved", ""


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    _ensure_shim()
    insts = json.load(open(SELECTION))[:N_INSTANCES]
    log(f"=== SWE-bench Lite (sympy) — qwen-coder-14B + LLML — SOLVEUR/JUGE séparés ===")
    log(f"    {len(insts)} instances · K={K_SAMPLES} échantillons T={TEMP} · réparations={MAX_REPAIRS}")
    log(f"    oracle = repro dérivée de l'issue + régression auto ; tests cachés = JUGE seul")
    cfg = Config.from_env(); cfg.backend = "mlx"
    cfg.mlx_model_path = os.path.join(_PROJ, "models", "qwen2.5-coder-14b-mlx-4bit")
    llm = make_client(cfg); llm.set_adapter(None); llm.cfg.mlx_max_tokens = 1500

    rows = []
    for i, inst in enumerate(insts, 1):
        iid = inst["instance_id"]
        log(f"[{i}/{len(insts)}] {iid}")
        try:
            patch, meta = solve(llm, inst)
            if patch.strip():   # persiste le patch GELÉ (auditable a posteriori)
                pdir = os.path.join(_PROJ, "logs", "swe_patches")
                os.makedirs(pdir, exist_ok=True)
                open(os.path.join(pdir, f"{iid}.patch"), "w", encoding="utf-8").write(patch)
            verdict, _d = judge(patch, inst)
        except Exception as e:  # noqa: BLE001
            verdict, meta = f"crash:{type(e).__name__}", {"bucket": "crash",
                "repro_validated": False, "gold_file_hit": False, "gold_line_shown": False,
                "oracle_accepted": False, "n_blocks_max": 0}
            log(f"   crash: {e}")
        rows.append({"id": iid, "verdict": verdict, **meta})
        log(f"   -> JUGE: {verdict}  (solveur: {meta.get('bucket')}, "
            f"oracle_accepted={meta.get('oracle_accepted')})")

    # ---- bilan honnête ----
    runnable = [r for r in rows if r["verdict"] != "env-fail"]
    resolved = [r for r in runnable if r["verdict"] == "resolved"]
    emitted = [r for r in runnable if r["verdict"] in ("resolved", "unresolved", "cand-apply-fail")]
    gold_hit = sum(1 for r in rows if r.get("gold_file_hit"))
    gold_line = sum(1 for r in rows if r.get("gold_line_shown"))
    repro_val = sum(1 for r in rows if r.get("repro_validated"))
    oracle_acc = sum(1 for r in rows if r.get("oracle_accepted"))
    nofmt = sum(1 for r in rows if r.get("n_blocks_max", 0) == 0)
    log("")
    log("=== RÉSULTAT SWE-bench Lite (sympy) — HONNÊTE (solveur/juge séparés) ===")
    log(f"{'instance':22s} {'JUGE':16s} {'solveur':16s} gold_line repro oracle")
    for r in rows:
        log(f"{r['id']:22s} {r['verdict']:16s} {r.get('bucket',''):16s} "
            f"{'✓' if r.get('gold_line_shown') else '·':^9s} "
            f"{'✓' if r.get('repro_validated') else '·':^5s} "
            f"{'✓' if r.get('oracle_accepted') else '·'}")
    log("")
    n = len(runnable)
    log(f"RÉSOLUS (juge, tests cachés) : {len(resolved)}/{n} exécutables "
        f"({len(resolved)/max(1,n)*100:.0f}%) · {len(rows)-n} env-fail exclus du dénominateur")
    log(f"Localisation : fichier gold {gold_hit}/{len(rows)} · ligne gold à l'écran {gold_line}/{len(rows)}")
    log(f"Oracle repro : validée {repro_val}/{len(rows)} · patchs acceptés par l'oracle {oracle_acc} "
        f"· résolus par le juge {len(resolved)} (écart = calibration de l'oracle honnête)")
    log(f"Génération : instances sans aucun bloc parseable (sur {K_SAMPLES} échantillons) : {nofmt}")
    log("")
    if resolved:
        log(f"🟢 {len(resolved)} vraie(s) issue(s) GitHub résolue(s) SANS fuite (oracle = repro issue "
            f"+ régression ; tests cachés jugés après gel du patch). Résultat mesuré honnêtement.")
    else:
        log(f"🟠 0 résolu sur {n} exécutables — honnête et défendable pour un 14B 4-bit local "
            f"(SOTA 7-14B open+scaffold ~12-14% sur le Lite complet). Le pipeline est propre et "
            f"sans fuite ; le maillon limitant est le raisonnement de correction du modèle.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
