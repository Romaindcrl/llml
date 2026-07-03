"""Apprentissage « on-the-job » — le loop HUMAIN : apprendre en travaillant sur de VRAIES tâches.

Insight utilisateur : avant les LLM, on apprenait en cherchant la doc/les forums puis en
TESTANT — sur le vrai problème, jamais sur un examen auto-inventé (c'est le maillon qui a
échoué 4× dans benchmark_continuous_code : l'auto-notation). Ici, la vérité vient du travail
lui-même : chaque tâche arrive avec UN exemple montré (ce qu'un utilisateur donne toujours),
les tests complets restent un juge caché.

Flux de 16 tâches réelles (2 par API moderne : la seconde ronde mesure si l'expérience de la
première a servi). Deux bras :
  CONTRÔLE      : modèle amnésique — chaque tâche traitée isolément (draft nu).
  APPRENTISSAGE : le loop humain — doc indexée une fois, puis pour chaque tâche :
                  draft (avec leçons+compétences accumulées) → vérifié sur l'EXEMPLE montré →
                  réparation ≤2 → si l'exemple passe, la solution devient une compétence et
                  la leçon est validée. Les tests cachés ne servent QU'À noter.
Métrique décisive : pass@1 des DRAFTS en ronde 2 (tâches 9-16) — apprentissage vs contrôle.
Live : tail -f logs/benchmark_onthejob.log
"""

from __future__ import annotations

import os
import re
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from scripts.benchmark_continuous_code import EVAL, REFS, TOPIC, research_fn  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.learner import ContinuousLearner, Ledger, _DEF_RE, extract_code  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.ltm import LTM  # noqa: E402
from m0.rag import RAG  # noqa: E402
from m0.skills import SkillLibrary  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_onthejob.log")
_T0 = time.time()

# ---- ronde 2 : une 2e tâche par API (l'expérience de la ronde 1 doit servir) ----
ROUND2 = [
    ("Écris une fonction batch_sums(seq, n) qui utilise itertools.batched pour sommer seq "
     "par paquets de n et renvoie la liste des sommes.",
     "assert batch_sums([1,2,3,4,5], 2) == [3, 7, 5]",
     "assert batch_sums([1,2,3,4,5], 2) == [3, 7, 5]\nassert batch_sums([], 3) == []\n"
     "assert batch_sums([10], 4) == [10]"),
    ("Écris une fonction wmean(vals, weights) qui calcule la moyenne pondérée en utilisant "
     "math.sumprod, en float.",
     "assert wmean([2, 4], [3, 1]) == 2.5",
     "assert wmean([2, 4], [3, 1]) == 2.5\nassert wmean([1, 3], [1, 1]) == 2.0"),
    ("Écris une fonction toml_get(s, path) qui parse une chaîne TOML avec tomllib puis "
     "navigue avec un chemin pointé ('a.b') et renvoie la valeur.",
     "assert toml_get('x = 1', 'x') == 1",
     "assert toml_get('x = 1', 'x') == 1\nassert toml_get('[a]\\nb = 2', 'a.b') == 2"),
    ("Écris une fonction utc_weekday(ts) qui renvoie le numéro du jour de semaine (lundi=0) "
     "d'un timestamp Unix interprété en UTC via datetime.UTC.",
     "assert utc_weekday(0) == 3",
     "assert utc_weekday(0) == 3\nassert utc_weekday(86400 * 3) == 6"),
    ("Écris une fonction md5_file(path) qui renvoie le MD5 hexadécimal d'un fichier en "
     "utilisant hashlib.file_digest.",
     "import tempfile\np = tempfile.mktemp()\nopen(p,'wb').write(b'abc')\n"
     "assert md5_file(p) == '900150983cd24fb0d6963f7d28e17f72'",
     "import tempfile, os\np = tempfile.mktemp()\nopen(p,'wb').write(b'abc')\n"
     "assert md5_file(p) == '900150983cd24fb0d6963f7d28e17f72'\nos.unlink(p)"),
    ("Écris une classe générique Box avec la syntaxe PEP 695 de Python 3.12 "
     "(class Box[T]) : constructeur qui stocke une valeur, méthode get() qui la renvoie.",
     "b = Box(5)\nassert b.get() == 5",
     "b = Box(5)\nassert b.get() == 5\n"
     "assert Box.__type_params__ and Box.__type_params__[0].__name__ == 'T'"),
    ("Écris une fonction count_zde(fns) qui appelle chaque fonction en collectant les "
     "exceptions ; s'il y en a, elle les regroupe dans un ExceptionGroup qu'elle lève puis "
     "rattrape avec la syntaxe except* pour renvoyer le NOMBRE de ZeroDivisionError. "
     "Attention : return est interdit à l'intérieur d'un bloc except*.",
     "assert count_zde([lambda: 1//0, lambda: 3]) == 1",
     "assert count_zde([lambda: 1//0, lambda: 3]) == 1\n"
     "assert count_zde([lambda: 1//0, lambda: 2//0, lambda: 3]) == 2\n"
     "assert count_zde([lambda: 1]) == 0"),
    ("Écris une coroutine doubles(ns) qui calcule n*2 pour chaque n en tâches concurrentes "
     "via asyncio.TaskGroup, et doubles_sync(ns) qui l'exécute avec asyncio.run (ordre préservé).",
     "assert doubles_sync([1, 2, 3]) == [2, 4, 6]",
     "assert doubles_sync([1, 2, 3]) == [2, 4, 6]\nassert doubles_sync([]) == []"),
]
ROUND2_REFS = [
    "from itertools import batched\ndef batch_sums(seq, n):\n    return [sum(b) for b in batched(seq, n)]",
    "from math import sumprod\ndef wmean(vals, weights):\n    return sumprod(vals, weights) / sum(weights)",
    ("import tomllib\ndef toml_get(s, path):\n    d = tomllib.loads(s)\n"
     "    for k in path.split('.'):\n        d = d[k]\n    return d"),
    "from datetime import datetime, UTC\ndef utc_weekday(ts):\n    return datetime.fromtimestamp(ts, UTC).weekday()",
    ("import hashlib\ndef md5_file(path):\n    with open(path, 'rb') as f:\n"
     "        return hashlib.file_digest(f, 'md5').hexdigest()"),
    "class Box[T]:\n    def __init__(self, v: T):\n        self._v = v\n    def get(self) -> T:\n        return self._v",
    ("def count_zde(fns):\n    errs = []\n    for fn in fns:\n        try:\n            fn()\n"
     "        except Exception as e:\n            errs.append(e)\n    if not errs:\n        return 0\n"
     "    n = 0\n    try:\n        raise ExceptionGroup('errs', errs)\n"
     "    except* ZeroDivisionError as eg:\n        n = len(eg.exceptions)\n"
     "    except* Exception:\n        pass\n    return n"),
    ("import asyncio\nasync def _d(n):\n    return n * 2\n"
     "async def doubles(ns):\n    async with asyncio.TaskGroup() as tg:\n"
     "        ts = [tg.create_task(_d(n)) for n in ns]\n    return [t.result() for t in ts]\n"
     "def doubles_sync(ns):\n    return asyncio.run(doubles(ns))"),
]

# flux complet : ronde 1 = les 8 de l'éval fixe (exemple = 1er assert), ronde 2 = les 8 ci-dessus
def build_stream():
    stream = []
    for (st, tests) in EVAL:
        example = tests.splitlines()[0] if tests.splitlines()[0].startswith("assert") else tests
        stream.append({"statement": st, "example": example, "tests": tests})
    for (st, example, tests) in ROUND2:
        stream.append({"statement": st, "example": example, "tests": tests})
    return stream


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    stream = build_stream()
    log(f"=== ON-THE-JOB — flux de {len(stream)} vraies tâches (2 par API), contrôle vs apprentissage ===")

    # harnais : les références ronde 2 doivent passer leurs tests
    from m0.tools import Tools
    tools = Tools(os.path.join(_PROJ, "logs", "otj_workdir"))
    ok = 0
    for (st, ex, tests), ref in zip(ROUND2, ROUND2_REFS):
        tools.write_file("attempt.py", f"{ref}\n\n{tests}\nprint('TESTS_OK')\n")
        r = tools.bash("python3 attempt.py")
        ok += r.ok and "TESTS_OK" in r.output
    log(f"harnais ronde 2 : références {ok}/8 {'✓' if ok == 8 else '⚠️ INVALIDE — stop'}")
    if ok < 8:
        return

    cfg = Config.from_env(); cfg.backend = "mlx"
    llm = make_client(cfg); llm.set_adapter(None); llm.cfg.mlx_max_tokens = 650

    # ---------- bras CONTRÔLE : amnésique, draft nu, tests cachés comme juge
    log("[1/2] CONTRÔLE (modèle amnésique)")
    ctrl = []
    for i, t in enumerate(stream):
        sol = extract_code(llm.generate(
            f"Exercice : {t['statement']}\nEcris UNIQUEMENT le code Python demande "
            "(fonction complete, imports inclus), dans un bloc ```python.", None))
        tools.write_file("attempt.py", f"{sol}\n\n{t['tests']}\nprint('TESTS_OK')\n")
        r = tools.bash("python3 attempt.py")
        hit = bool(sol) and r.ok and "TESTS_OK" in r.output
        ctrl.append(hit)
        log(f"   [ctrl {i + 1}/16] {'✅' if hit else '❌'}")

    # ---------- bras APPRENTISSAGE : le loop humain
    log("[2/2] APPRENTISSAGE (doc indexée 1×, puis expérience accumulée tâche après tâche)")
    for p in ("otj_rag.txt", "otj_ltm.jsonl", "otj_skills.jsonl", "otj_ledger.jsonl"):
        fp = os.path.join(_PROJ, "logs", p)
        if os.path.exists(fp):
            os.remove(fp)
    learner = ContinuousLearner(
        llm=llm,
        rag=RAG(os.path.join(_PROJ, "logs", "otj_rag.txt")),
        ltm=LTM(os.path.join(_PROJ, "logs", "otj_ltm.jsonl")),
        skills=SkillLibrary(os.path.join(_PROJ, "logs", "otj_skills.jsonl")),
        ledger=Ledger(os.path.join(_PROJ, "logs", "otj_ledger.jsonl")),
        workdir=os.path.join(_PROJ, "logs", "otj_workdir"),
        research_fn=research_fn, max_retries=2, log=log,
    )
    import scripts.benchmark_continuous_code as CC
    CC._RESEARCHED["done"] = False       # autorise UNE recherche pour ce bras
    n_pages = learner.research(TOPIC, k_pages=3)
    log(f"   doc indexée : {n_pages} pages (comme un humain qui ouvre la doc)")

    learn_draft, learn_final = [], []
    for i, t in enumerate(stream):
        # draft avec l'expérience accumulée (leçons + compétences des tâches précédentes)
        sol = learner._generate_solution(t["statement"], TOPIC)
        tools.write_file("attempt.py", f"{sol}\n\n{t['tests']}\nprint('TESTS_OK')\n")
        r = tools.bash("python3 attempt.py")
        draft_hit = bool(sol) and r.ok and "TESTS_OK" in r.output

        # le loop humain : vérifier sur l'EXEMPLE montré, réparer, retenir si validé
        ex_ok, ex_out = learner._execute(sol, t["example"]) if sol else (False, "aucun code")
        tries, lesson = 0, ""
        while not ex_ok and tries < 2:
            fixed, lesson_i = learner.reflect(t["statement"], sol or "(vide)", ex_out)
            lesson = lesson_i or lesson
            if not fixed:
                break
            sol = fixed
            ex_ok, ex_out = learner._execute(sol, t["example"])
            tries += 1
        if ex_ok:                                   # validé par la vérité DISPONIBLE
            fn = _DEF_RE.search(sol or "")
            added = learner.skills.add(fn.group(1) if fn else "solution",
                                       t["statement"], sol, topic=TOPIC, tests=t["example"])
            if lesson:
                learner.rag.add_document(f"LECON ({TOPIC}) : {lesson}")
            if added:
                log(f"      ✓ compétence : {added.name}" + (f" · leçon : {lesson[:60]}" if lesson else ""))
        # note finale sur les tests CACHÉS (le juge)
        tools.write_file("attempt.py", f"{sol}\n\n{t['tests']}\nprint('TESTS_OK')\n")
        r = tools.bash("python3 attempt.py")
        final_hit = bool(sol) and r.ok and "TESTS_OK" in r.output
        learn_draft.append(draft_hit); learn_final.append(final_hit)
        log(f"   [appr {i + 1}/16] draft {'✅' if draft_hit else '❌'} final {'✅' if final_hit else '❌'}")

    # ---------- bilan
    def score(v, lo, hi):
        s = sum(v[lo:hi]); return f"{s}/{hi - lo} ({s / (hi - lo) * 100:.0f}%)"
    log("")
    log("=== RÉSULTAT ON-THE-JOB (drafts = premier essai ; ronde 2 = l'expérience a-t-elle servi ?) ===")
    log(f"{'':22s} | ronde 1 (tâches 1-8) | ronde 2 (tâches 9-16)")
    log(f"{'CONTRÔLE (amnésique)':22s} | {score(ctrl, 0, 8):20s} | {score(ctrl, 8, 16)}")
    log(f"{'APPRENTISSAGE draft':22s} | {score(learn_draft, 0, 8):20s} | {score(learn_draft, 8, 16)}")
    log(f"{'APPRENTISSAGE final':22s} | {score(learn_final, 0, 8):20s} | {score(learn_final, 8, 16)}")
    log(f"   compétences : {learner.skills.count()} · leçons+doc RAG : ~{len(learner.rag.topk('python', 999))} chunks")
    c2, l2 = sum(ctrl[8:]), sum(learn_draft[8:])
    log("")
    if l2 > c2 + 1:
        log(f"🟢 LE LOOP HUMAIN FONCTIONNE : en ronde 2, les drafts avec expérience font {l2}/8 vs "
            f"{c2}/8 amnésique — le système apprend EN TRAVAILLANT, vérité fournie par les tâches réelles.")
    elif l2 > c2:
        log(f"🟡 avantage réel mais mince en ronde 2 ({l2} vs {c2}/8) — plus de flux pour confirmer.")
    else:
        log(f"🔴 pas d'avantage d'expérience en ronde 2 ({l2} vs {c2}/8) — l'accumulation n'aide pas ici.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
