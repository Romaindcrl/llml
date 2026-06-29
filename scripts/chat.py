"""REPL interactif contre le backend configure (mock par defaut, ollama si
M0_BACKEND=ollama).

Usage :
    python scripts/chat.py
    M0_BACKEND=ollama M0_MODEL=llama3.1:8b python scripts/chat.py

Le REPL cable toute la pile M0 (store, memoire, detecteur, compaction, outils) et
fait tourner un tour d'agent par message utilisateur. Commandes :
    /mem        affiche la memoire active
    /ctx        affiche le model_context courant (events non compactes)
    /pruned     affiche les events prunes (preuve non-destructive)
    /quit       quitte

Avec le backend mock, le modele est deterministe et ne "comprend" pas le texte :
il sert a demontrer la mecanique (compaction, memoire, outils) sans modele. Pour
un vrai dialogue, lancer avec M0_BACKEND=ollama (necessite ollama + un pull).
"""

from __future__ import annotations

import os
import sys
import tempfile

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from m0.agent import Agent  # noqa: E402
from m0.compaction import Compactor  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.detector import TwoShotDetector  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.memory import TextMemory  # noqa: E402
from m0.store import EventStore  # noqa: E402
from m0.tools import Tools  # noqa: E402


def _build(cfg: Config) -> Agent:
    llm = make_client(cfg)
    store = EventStore(cfg.db_path)
    memory = TextMemory(cfg.memory_path, cfg.repo_dir, cfg.memory_inject_cap_tokens)
    detector = TwoShotDetector(memory)
    compactor = Compactor(store, llm, cfg)
    tools = Tools(cfg.workdir)
    agent = Agent(llm, store, memory, detector, compactor, tools, cfg)
    return agent


def main() -> int:
    cfg = Config.from_env()
    # Workdir + memoire isoles pour la session REPL (sauf si deja fixes ailleurs).
    tmp = tempfile.mkdtemp(prefix="m0_chat_")
    cfg.workdir = os.path.join(tmp, "work")
    cfg.repo_dir = tmp
    cfg.memory_path = os.path.join(tmp, "MEMORY.md")
    os.makedirs(cfg.workdir, exist_ok=True)

    agent = _build(cfg)
    print(f"M0 chat REPL — backend={cfg.backend} model={cfg.model}")
    print("Commandes: /mem /ctx /pruned /quit\n")
    if cfg.backend == "mock":
        print("(backend mock: reponses deterministes scriptees, pas de vrai modele)\n")

    while True:
        try:
            line = input("vous> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/mem":
            for e in agent.memory.active_entries():
                print(f"  [{e.id}] ({e.priority}) {e.text}")
            continue
        if line == "/ctx":
            for ev in agent.store.model_context():
                print(f"  #{ev.id} t{ev.turn} {ev.role}/{ev.kind}: {ev.content[:80]}")
            continue
        if line == "/pruned":
            for ev in agent.store.pruned_events():
                print(f"  #{ev.id} t{ev.turn} {ev.role}/{ev.kind}: {ev.content[:80]}")
            continue

        # Un tour d'agent. En mock, un script minimal "echo + done" est injecte.
        task = {
            "id": "repl",
            "prompt": line,
            "sut": "m0",
            "workdir": cfg.workdir,
        }
        if cfg.backend == "mock":
            task["mock_script"] = [{"done": True, "say": f"(mock) recu: {line}"}]
        metrics = agent.run_task(task)
        # Affiche la derniere reponse assistant.
        last_assistant = None
        for ev in agent.store.all_events():
            if ev.role == "assistant" and ev.kind == "message":
                last_assistant = ev
        if last_assistant is not None:
            print(f"agent> {last_assistant.content}")
        print(
            f"  [turns={metrics.turns} ctx_tokens={metrics.context_tokens} "
            f"summary_tokens={metrics.summary_tokens} reerrors={metrics.reerrors}]"
        )

    print("Au revoir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
