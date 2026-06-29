"""Smoke test end-to-end SANS modele (backend mock).

Verifie les 4 invariants load-bearing du jalon TEXTE :
  (a) une COMPACTION se declenche (gros tool_outputs) -> summary cree, events prunes ;
  (b) une entree memoire ecrite puis EXPIREE n'est jamais reinjectee ;
  (c) la MEME erreur provoquee deux fois -> reerror = 1 au 2e tir ;
  (d) NON-DESTRUCTIVITE : les pruned_events restent rechargeables (content intact).

Affiche des assertions PASS/FAIL. Sort 0 si tout passe, 1 sinon.
"""

from __future__ import annotations

import os
import sys
import tempfile

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from m0.compaction import Compactor  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.detector import TwoShotDetector  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.memory import TextMemory  # noqa: E402
from m0.store import EventStore  # noqa: E402

_results: list[tuple[bool, str]] = []


def check(cond: bool, label: str) -> None:
    _results.append((bool(cond), label))
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m0_smoke_")
    memory_path = os.path.join(tmp, "MEMORY.md")
    cfg = Config(backend="mock", db_path=":memory:", memory_path=memory_path,
                 repo_dir=tmp, workdir=os.path.join(tmp, "work"))

    llm = make_client(cfg)
    store = EventStore(cfg.db_path)
    memory = TextMemory(cfg.memory_path, cfg.repo_dir, cfg.memory_inject_cap_tokens)
    detector = TwoShotDetector(memory)
    compactor = Compactor(store, llm, cfg)

    # --------------------------------------------------------------- (a) compaction
    # On fabrique un contexte avec de gros tool_outputs sur d'anciens tours pour
    # depasser compaction_trigger_tokens (4000) tout en gardant assez de
    # recuperable (>800) hors des 2 derniers tours.
    big = "x" * 5000  # ~1250 tokens chacun
    store.append(1, "user", "message", "Objectif: traiter un gros volume.")
    store.append(1, "tool", "tool_output", "OUTPUT-1 " + big)
    store.append(2, "tool", "tool_output", "OUTPUT-2 " + big)
    store.append(3, "tool", "tool_output", "OUTPUT-3 " + big)
    store.append(4, "tool", "tool_output", "OUTPUT-4 " + big)  # tour recent (protege)
    store.append(5, "user", "message", "Continue.")  # tour recent (protege)

    ctx = store.model_context()
    should = compactor.should_compact(ctx)
    check(should, "(a) should_compact = True sur gros contexte")

    n_before_pruned = len(store.pruned_events())
    new_ctx, summary_tokens = compactor.compact(ctx, turn=5)
    pruned = store.pruned_events()
    has_summary = any(e.kind == "summary" for e in new_ctx)
    check(len(pruned) > n_before_pruned, "(a) des events ont ete prunes")
    check(has_summary, "(a) un event summary est present dans le nouveau contexte")
    check(summary_tokens > 0, "(a) summary_tokens > 0")
    # Les tours recents (4 et 5) ne doivent PAS etre prunes.
    pruned_turns = {e.turn for e in pruned}
    check(4 not in pruned_turns and 5 not in pruned_turns,
          "(a) les 2 derniers tours ne sont pas prunes")

    # --------------------------------------------------------------- (b) expiration
    entry = memory.add("Toujours valider les entrees avant lecture.", tags=["lesson"])
    check(entry is not None, "(b) entree memoire ajoutee")
    rendered_before = memory.render_for_injection()
    check(entry.text in rendered_before, "(b) entree active reinjectee avant expiration")

    expired = memory.expire(entry.id)
    check(expired, "(b) expire() = True")
    rendered_after = memory.render_for_injection()
    check(entry.text not in rendered_after,
          "(b) entree expiree JAMAIS reinjectee")
    check(memory.find_by_fingerprint("zzz") is None,
          "(b) find_by_fingerprint inconnu = None")

    # --------------------------------------------------------------- (c) reerror
    err = "Traceback: FileNotFoundError: /abs/path/data/input.csv at 12:00:00"
    v1 = detector.observe_error(err, turn=6)
    # Meme erreur, details variables (chemin/heure differents) -> meme fingerprint.
    err2 = "Traceback: FileNotFoundError: /autre/chemin/data/input.csv at 09:33:21"
    v2 = detector.observe_error(err2, turn=7)
    check(v1 == "new", "(c) 1re occurrence d'erreur = 'new'")
    check(v2 == "reerror", "(c) 2e occurrence (meme empreinte) = 'reerror'")

    # --------------------------------------------------------------- (d) non-destructif
    # Le content prune doit rester intact et rechargeable.
    reloaded = store.pruned_events()
    intact = all(e.content for e in reloaded) and any("OUTPUT-1" in e.content for e in reloaded)
    check(intact, "(d) pruned_events rechargeables avec content intact")
    # all_events contient tout (prunes + actifs) : preuve append-only.
    total = len(store.all_events())
    check(total == len(store.model_context()) + len(reloaded),
          "(d) all_events = model_context + pruned (rien perdu)")

    # Sanity sur le compteur de tokens (pas de dependance lourde).
    check(count_tokens("") == 0 and count_tokens("abcd") == 1,
          "(e) count_tokens stdlib coherent")

    store.close()

    n_pass = sum(1 for ok, _ in _results if ok)
    n_total = len(_results)
    print(f"\n{n_pass}/{n_total} assertions PASS")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
