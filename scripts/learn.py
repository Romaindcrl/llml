"""Apprentissage continu — CLI. Le systeme etudie, pratique, et progresse seul.

Usage :
  # etudier un sujet (recherche web -> exercices executes -> lecons -> competences)
  M0_BACKEND=mlx M0_MLX_MODEL_PATH=models/qwen2.5-7b-it-mlx-8bit \
      python scripts/learn.py --topic "python asyncio" --cycles 3

  # mode continu : boucle sur les faiblesses signalees (ledger), a l'infini
  python scripts/learn.py --daemon --interval 300

  # afficher les courbes de progres (pass@1 par sujet, cycle apres cycle)
  python scripts/learn.py --curve

Les connaissances partent dans les MEMES stores que le serveur (logs/rag_corpus.txt,
logs/ltm_qa.jsonl) : ce qui est appris ici est immediatement utilisable en chat, et
consolidable dans les poids par le /sleep existant (gate held-out). Les competences
verifiees vivent dans logs/skills.jsonl, le journal dans logs/learn_ledger.jsonl.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0.config import Config  # noqa: E402
from m0.learner import ContinuousLearner, Ledger  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.ltm import LTM  # noqa: E402
from m0.rag import RAG  # noqa: E402
from m0.skills import SkillLibrary  # noqa: E402

LEDGER_PATH = os.path.join(_PROJ, "logs", "learn_ledger.jsonl")
SKILLS_PATH = os.path.join(_PROJ, "logs", "skills.jsonl")
WORKDIR = os.path.join(_PROJ, "logs", "learn_workdir")


def build_learner(use_web: bool) -> ContinuousLearner:
    cfg = Config.from_env()
    llm = make_client(cfg)
    return ContinuousLearner(
        llm=llm,
        rag=RAG(os.path.join(_PROJ, "logs", "rag_corpus.txt")),
        ltm=LTM(os.path.join(_PROJ, "logs", "ltm_qa.jsonl")),
        skills=SkillLibrary(SKILLS_PATH),
        ledger=Ledger(LEDGER_PATH),
        workdir=WORKDIR,
        use_web=use_web,
        max_retries=int(os.environ.get("M0_LEARN_RETRIES", "2")),
    )


def show_curves() -> None:
    ledger = Ledger(LEDGER_PATH)
    topics = ledger.topics()
    if not topics:
        print("journal vide — lance d'abord un cycle : learn.py --topic \"…\"")
        return
    print("Courbes de progres (pass@1 = reussite au PREMIER essai, sans retry) :\n")
    for topic in topics:
        curve = ledger.curve(topic)
        if not curve:
            continue
        pts = " ".join(f"c{i}:{rate * 100:3.0f}%" for i, rate in curve)
        bar = "".join("▁▂▃▄▅▆▇█"[min(7, int(rate * 8))] for _, rate in curve)
        print(f"  {topic[:44]:44s} {bar}  {pts}")
    print("\nSi la courbe monte, l'apprentissage est reel : le premier essai profite\n"
          "du savoir (RAG/LTM), des lecons et des competences accumulees avant lui.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Boucle d'apprentissage continu LLML")
    ap.add_argument("--topic", action="append", default=[],
                    help="sujet a etudier (repetable) ; sans sujet, suit le ledger")
    ap.add_argument("--cycles", type=int, default=1, help="nb de cycles d'etude")
    ap.add_argument("--exercises", type=int, default=3, help="exercices par cycle")
    ap.add_argument("--no-web", action="store_true",
                    help="pas de recherche internet (RAG existant seulement)")
    ap.add_argument("--daemon", action="store_true",
                    help="boucle sans fin sur les faiblesses du ledger")
    ap.add_argument("--interval", type=int, default=300,
                    help="pause (s) entre cycles en mode --daemon")
    ap.add_argument("--curve", action="store_true", help="afficher les courbes et sortir")
    args = ap.parse_args()

    if args.curve:
        show_curves()
        return

    learner = build_learner(use_web=not args.no_web)
    if args.daemon:
        print(f"[daemon] etude continue (pause {args.interval}s entre cycles, Ctrl+C pour stop)")
        while True:
            done = learner.run(topics=args.topic or None, cycles=1,
                               n_exercises=args.exercises)
            if not done:  # rien a etudier : on attend que l'usage signale une faiblesse
                print(f"[daemon] rien a etudier — nouvelle verification dans {args.interval}s")
            time.sleep(args.interval)
    else:
        learner.run(topics=args.topic or None, cycles=args.cycles,
                    n_exercises=args.exercises)
        show_curves()


if __name__ == "__main__":
    main()
