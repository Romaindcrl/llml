"""Apprentissage continu en code — LA courbe honnête (jeu d'éval FIXE + cycles réels).

Corrige la faille du ledger : la courbe pass@1 sur des exercices régénérés à chaque cycle est
truquable (les exercices peuvent devenir plus faciles). Ici : un JEU D'ÉVALUATION FIXE de 8
exercices (validés par solutions de référence), tenu HORS des stores (jamais dans les skills,
jamais de leçon tirée pendant l'éval) — mesuré au cycle 0 (baseline) puis après chaque cycle
d'étude RÉEL (recherche web réelle → exercices auto-générés exécutés → réflexion → compétences).

Sujet choisi : les nouveautés Python 3.11/3.12 (itertools.batched, math.sumprod, tomllib,
hashlib.file_digest, datetime.UTC, ExceptionGroup, asyncio.TaskGroup, génériques PEP 695) —
savoir POSTÉRIEUR à une bonne partie de l'entraînement du 7B => faiblesse réelle, documentée
en ligne, testable en stdlib pure (python3 local = 3.14).

Si la courbe éval-fixe monte cycle après cycle, l'objectif « apprend en continu et devient
plus fort en code » est atteint par le mécanisme honnête (savoir+leçons+expérience en contexte).
Live : tail -f logs/benchmark_continuous_code.log
"""

from __future__ import annotations

import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0 import web  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.learner import ContinuousLearner, Ledger  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.ltm import LTM  # noqa: E402
from m0.rag import RAG  # noqa: E402
from m0.skills import SkillLibrary  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_continuous_code.log")
TOPIC = ("nouveautés de la bibliothèque standard Python 3.11 et 3.12 : itertools.batched, "
         "math.sumprod, tomllib, hashlib.file_digest, datetime.UTC, ExceptionGroup et except*, "
         "asyncio.TaskGroup, syntaxe générique PEP 695")
DOC_URLS = [  # filet de sécurité si la recherche DDG échoue : la doc officielle, direct
    "https://docs.python.org/3/whatsnew/3.12.html",
    "https://docs.python.org/3/whatsnew/3.11.html",
]
N_CYCLES = 3
_T0 = time.time()

# ---- JEU D'ÉVALUATION FIXE (held-out : jamais écrit dans les stores) ----
EVAL = [
    ("Écris une fonction chunk3(seq) qui utilise itertools.batched pour découper seq en tuples "
     "de taille 3 (le dernier peut être plus court) et renvoie la liste de ces tuples.",
     "assert chunk3(range(7)) == [(0,1,2),(3,4,5),(6,)]\nassert chunk3([]) == []"),
    ("Écris une fonction dot(a, b) qui calcule le produit scalaire de deux séquences en "
     "utilisant math.sumprod.",
     "assert dot([1,2],[3,4]) == 11\nassert dot([],[]) == 0"),
    ("Écris une fonction parse_cfg(s) qui parse une chaîne TOML avec le module standard tomllib "
     "et renvoie le dictionnaire.",
     "assert parse_cfg('x = 1') == {'x': 1}\nassert parse_cfg('[a]\\nb = \"c\"') == {'a': {'b': 'c'}}"),
    ("Écris une fonction utc_of(ts) qui convertit un timestamp Unix en datetime AWARE en "
     "utilisant l'alias datetime.UTC (Python 3.11+).",
     "assert utc_of(0).isoformat() == '1970-01-01T00:00:00+00:00'\nassert utc_of(0).tzinfo is not None"),
    ("Écris une fonction sha_file(path) qui renvoie le SHA-256 hexadécimal d'un fichier en "
     "utilisant hashlib.file_digest.",
     "import tempfile, os\n"
     "p = tempfile.mktemp()\n"
     "open(p,'wb').write(b'abc')\n"
     "assert sha_file(p) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'\n"
     "os.unlink(p)"),
    ("Écris une fonction générique first(items) avec la NOUVELLE syntaxe générique PEP 695 de "
     "Python 3.12 (def first[T](items: list[T]) -> T) qui renvoie le premier élément.",
     "assert first([3,4]) == 3\n"
     "assert first.__type_params__ and first.__type_params__[0].__name__ == 'T'"),
    ("Écris une fonction run_all(fns) qui appelle chaque fonction de la liste ; si une ou "
     "plusieurs lèvent une exception, elle lève un ExceptionGroup('errs', [...]) contenant "
     "TOUTES les exceptions ; sinon elle renvoie la liste des résultats.",
     "assert run_all([lambda: 1, lambda: 2]) == [1, 2]\n"
     "try:\n"
     "    run_all([lambda: 1//0, lambda: 2])\n"
     "    assert False, 'aurait dû lever'\n"
     "except ExceptionGroup as eg:\n"
     "    assert len(eg.exceptions) == 1 and isinstance(eg.exceptions[0], ZeroDivisionError)"),
    ("Écris une coroutine squares(ns) qui calcule n*n pour chaque n en tâches concurrentes via "
     "asyncio.TaskGroup (Python 3.11+), et une fonction squares_sync(ns) qui l'exécute avec "
     "asyncio.run et renvoie la liste ordonnée.",
     "assert squares_sync([1,2,3]) == [1,4,9]\nassert squares_sync([]) == []"),
]

REFS = [  # solutions de référence — valident le harnais avant toute mesure
    "from itertools import batched\ndef chunk3(seq):\n    return [tuple(b) for b in batched(seq, 3)]",
    "from math import sumprod\ndef dot(a, b):\n    return sumprod(a, b)",
    "import tomllib\ndef parse_cfg(s):\n    return tomllib.loads(s)",
    "from datetime import datetime, UTC\ndef utc_of(ts):\n    return datetime.fromtimestamp(ts, UTC)",
    "import hashlib\ndef sha_file(path):\n    with open(path, 'rb') as f:\n        return hashlib.file_digest(f, 'sha256').hexdigest()",
    "def first[T](items: list[T]) -> T:\n    return items[0]",
    ("def run_all(fns):\n    out, errs = [], []\n    for fn in fns:\n"
     "        try:\n            out.append(fn())\n        except Exception as e:\n            errs.append(e)\n"
     "    if errs:\n        raise ExceptionGroup('errs', errs)\n    return out"),
    ("import asyncio\nasync def _sq(n):\n    return n*n\n"
     "async def squares(ns):\n    async with asyncio.TaskGroup() as tg:\n"
     "        ts = [tg.create_task(_sq(n)) for n in ns]\n    return [t.result() for t in ts]\n"
     "def squares_sync(ns):\n    return asyncio.run(squares(ns))"),
]


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


_RESEARCHED = {"done": False}


def research_fn(topic):
    """Recherche réelle (DDG + doc officielle en filet) — UNE seule fois : le RAG est
    déjà rempli aux cycles suivants, re-fetcher ne sert qu'à se faire rate-limiter."""
    if _RESEARCHED["done"]:
        return []
    pages = web.research(f"{topic} documentation", k_pages=3)
    if len(pages) < 2:
        for url in DOC_URLS:
            text = web.fetch(url)
            if len(text) > 200:
                pages.append({"title": url.rsplit('/', 1)[-1], "url": url, "text": text})
    _RESEARCHED["done"] = bool(pages)
    return pages[:4]


# curriculum tournant : 3 APIs ciblées par cycle -> les 8 couvertes, puis révision
FOCUS = [
    ["itertools.batched", "math.sumprod", "tomllib.loads"],
    ["datetime.UTC", "hashlib.file_digest", "la syntaxe générique PEP 695 (def f[T](...))"],
    ["ExceptionGroup", "asyncio.TaskGroup", "itertools.batched"],
]


def run_eval(learner, tag):
    """Deux métriques sur le jeu FIXE, LECTURE SEULE (aucun store modifié) :
    pass@1 = le brouillon nu (savoir internalisé — ce que l'étude doit faire monter) ;
    pass@final = avec réparation ≤2 (ce que le SYSTÈME livre — pilier vérification)."""
    p1 = pf = 0
    for i, (statement, tests) in enumerate(EVAL):
        sol = learner._generate_solution(statement, TOPIC)
        hit, out = learner._execute(sol, tests) if sol else (False, "aucun code")
        first = hit
        tries = 0
        while not hit and tries < 2:                      # réparation, SANS stocker
            fixed, _lesson = learner.reflect(statement, sol or "(vide)", out)
            if not fixed:
                break
            sol = fixed
            hit, out = learner._execute(sol, tests)
            tries += 1
        p1 += first; pf += hit
        log(f"   [éval {tag}] {i + 1}/8 draft {'✅' if first else '❌'} système {'✅' if hit else '❌'}")
    log(f"   ÉVAL {tag} : draft {p1}/8 ({p1/8*100:.0f}%) · système {pf}/8 ({pf/8*100:.0f}%)")
    return p1, pf


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    log(f"=== APPRENTISSAGE CONTINU (code) — éval FIXE 8 exercices, {N_CYCLES} cycles réels ===")

    # 0) le harnais est-il valide ? les références doivent faire 8/8
    from m0.tools import Tools
    tools = Tools(os.path.join(_PROJ, "logs", "cc_workdir"))
    ref_ok = 0
    for (statement, tests), ref in zip(EVAL, REFS):
        tools.write_file("attempt.py", f"{ref}\n\n{tests}\nprint('TESTS_OK')\n")
        r = tools.bash("python3 attempt.py")
        ref_ok += r.ok and "TESTS_OK" in r.output
    log(f"harnais : références {ref_ok}/8 {'✓' if ref_ok == 8 else '⚠️ INVALIDE — stop'}")
    if ref_ok < 8:
        return

    cfg = Config.from_env(); cfg.backend = "mlx"
    llm = make_client(cfg); llm.set_adapter(None); llm.cfg.mlx_max_tokens = 650

    # stores DÉDIÉS (n'écrit pas dans le corpus principal du serveur)
    learner = ContinuousLearner(
        llm=llm,
        rag=RAG(os.path.join(_PROJ, "logs", "cc_rag.txt")),
        ltm=LTM(os.path.join(_PROJ, "logs", "cc_ltm.jsonl")),
        skills=SkillLibrary(os.path.join(_PROJ, "logs", "cc_skills.jsonl")),
        ledger=Ledger(os.path.join(_PROJ, "logs", "cc_ledger.jsonl")),
        workdir=os.path.join(_PROJ, "logs", "cc_workdir"),
        research_fn=research_fn,
        max_retries=2,
        log=log,
    )
    learner.rag.clear()

    curve = []
    log("[cycle 0] BASELINE — le modèle n'a encore rien étudié")
    curve.append(run_eval(learner, "0 (baseline)"))
    for c in range(1, N_CYCLES + 1):
        log(f"[cycle {c}] étude réelle")
        summary = learner.study_cycle(TOPIC, n_exercises=3, k_pages=3, focus=FOCUS[c - 1])
        log(f"   cycle {c} : pratique pass@1 {summary['pass1']}/{summary['total']}, "
            f"final {summary['final']}/{summary['total']}, pages {summary['pages']}")
        curve.append(run_eval(learner, str(c)))

    log("")
    log("=== COURBES — jeu d'évaluation FIXE (held-out) ===")
    for i, (p1, pf) in enumerate(curve):
        b1 = "█" * p1 + "░" * (8 - p1)
        bf = "█" * pf + "░" * (8 - pf)
        log(f"   cycle {i} : draft {b1} {p1}/8   ·   système {bf} {pf}/8")
    log(f"   compétences vérifiées : {learner.skills.count()} · flashcards LTM : {learner.ltm.count()} "
        f"· chunks RAG : ~{len(learner.rag.topk('python', 999))}")
    log("")
    d0, dN = curve[0][0], curve[-1][0]
    s0, sN = curve[0][1], curve[-1][1]
    if dN > d0 + 1:
        log(f"🟢 OBJECTIF ATTEINT : le savoir internalisé MONTE ({d0}→{dN}/8 en draft sur étalon fixe) "
            f"et le système livre {sN}/8 — apprentissage continu réel, zéro poids touché.")
    elif sN >= 7 and dN > d0:
        log(f"🟢 SYSTÈME FORT ({s0}→{sN}/8 avec réparation) + progrès du draft ({d0}→{dN}/8).")
    elif sN >= 7:
        log(f"🟡 le SYSTÈME livre ({sN}/8 via réparation) mais le draft n'apprend pas ({d0}→{dN}/8) — "
            "les leçons de la pratique ne migrent pas vers le premier essai.")
    else:
        log(f"🔴 ni progrès du draft ({d0}→{dN}/8) ni système fort ({sN}/8) — à diagnostiquer.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
