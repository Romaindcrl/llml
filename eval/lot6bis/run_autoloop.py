#!/usr/bin/env python3
"""Lot 6bis — boucle d'apprentissage autonome (bench interne #12), stack VIVANTE.

Compose les primitives RÉELLES du serveur en une boucle continue, sans aucune
brique nouvelle ni scoring maison :

  cycle k :
    1. l'agent LIT le bloc-corpus k (suite de tours `Agent.chat_turn`) → le
       contexte SATURE → `on_compact` auto-promeut le contenu libéré vers
       LTM (faits) + RAG (texte)               [agent.py:259-273 / serve.py:_on_compact]
    2. `/sleep` gé CUMULATIF entraîne un LoRA sur toute la LTM accumulée
                                                [sleep_train = réplique serve._do_sleep]
    3. le routeur `classify()` décide PAR REQUÊTE : rappel→LoRA-mémoire seul,
       génération→base+RAG                       [rag.classify + serve routing]
    4. éval CUMULATIVE (toutes les QA vues) sous 3 configs : C0 base nue,
       C1 = LLML intégré (routeur), RAG-oracle (force base+RAG).

Mesure H-D1 (apprend : C1 vs C0), H-D3 (oubli : rappel des QA du cycle 1 au fil
des cycles), H-D4 (intégration : C1 vs RAG-oracle), + non-régression génération
(H-D2) via petites tâches codegen exécutées.

Usage (pod) :
  /workspace/venv/bin/python eval/lot6bis/run_autoloop.py \
    --model Qwen/Qwen2.5-7B-Instruct --quant 8bit \
    --corpus eval/lot6bis/frozen_corpus.jsonl --outdir /workspace/results/lot6bis
"""
import argparse
import json
import os
import re
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "eval", "lot4"))

from m0.config import Config          # noqa: E402
from m0.llm import make_client        # noqa: E402
from m0.ltm import LTM                # noqa: E402
from m0.rag import RAG, classify      # noqa: E402
from m0.agent import Agent            # noqa: E402
from m0.compaction import Compactor   # noqa: E402
from m0.detector import TwoShotDetector  # noqa: E402
from m0.memory import TextMemory      # noqa: E402
from m0.store import EventStore       # noqa: E402
from m0.tools import Tools            # noqa: E402
from run_memory import sleep_train    # noqa: E402  (réplique fidèle de serve._do_sleep)


# --- scoring court (identique à lot4b) --------------------------------------------
def sa_prompt(q, ctx=None):
    head = f"Context:\n{ctx}\n\n" if ctx is not None else ""
    return (f"{head}Answer with ONLY the specific fact (a date, version number, code, "
            f"or short name) — no explanation.\n\nQuestion: {q}\nAnswer:")


def sa_match(out, exp):
    o = re.sub(r"\s+", " ", (out or "").lower())
    e = (exp or "").lower().strip()
    if e in o:
        return True
    toks = [t for t in re.split(r"[ ,]+", e) if t and t not in ("and", "the", "a", "of", "to")]
    return len(toks) > 1 and all(t in o for t in toks)


# --- non-régression génération (H-D2) : petites tâches codegen exécutées ----------
GEN_TASKS = [
    {"id": "g1", "prompt": "Write a Python function `is_even(n)` that returns True if n is even, else False. Return only the function.",
     "call": "is_even(4) is True and is_even(7) is False", "fn": "is_even"},
    {"id": "g2", "prompt": "Write a Python function `rev(s)` that returns the reverse of string s. Return only the function.",
     "call": "rev('abc') == 'cba'", "fn": "rev"},
    {"id": "g3", "prompt": "Write a Python function `fac(n)` that returns n factorial (n>=0). Return only the function.",
     "call": "fac(5) == 120 and fac(0) == 1", "fn": "fac"},
]


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*(.+?)```", text or "", re.S)
    code = m.group(1) if m else (text or "")
    return code


def run_gen_task(llm, rag, task):
    """Route = génération → base+RAG (adapter None). Exécute et vérifie."""
    llm.set_adapter(None)
    llm.cfg.mlx_max_tokens = 256
    ctx = "\n".join(rag.topk(task["prompt"], 4))
    out = llm.generate(f"Context:\n{ctx}\n\n{task['prompt']}" if ctx else task["prompt"], None)
    code = _extract_code(out)
    ns = {}
    try:
        exec(code, ns)  # noqa: S102 — sandbox pod, tâches triviales déterministes
        ok = bool(eval(task["call"], ns))  # noqa: S307
    except Exception:  # noqa: BLE001
        ok = False
    return int(ok)


def eval_questions(llm, mem_adapter, rag, questions):
    """Calcule C0 / RAG-oracle / C1(routeur) pour une liste de QA, en minimisant
    les swaps d'adapter (base d'abord, mémoire ensuite)."""
    llm.cfg.mlx_max_tokens = 24
    routes, c0, oracle, c1 = {}, {}, {}, {}
    recall_qs = []

    # --- passe BASE (adapter None) : C0, RAG-oracle, routage, et branche génération de C1
    llm.set_adapter(None)
    for it in questions:
        qid, q, exp = it["qid"], it["q"], it["answer"]
        c0[qid] = int(sa_match(llm.generate(sa_prompt(q), None), exp))
        ctx = "\n".join(rag.topk(q, 4))
        oracle[qid] = int(sa_match(llm.generate(sa_prompt(q, ctx=ctx), None), exp))
        route = classify(q, llm.generate)          # routeur LLM zero-shot
        routes[qid] = route
        if route == "generation":
            c1[qid] = oracle[qid]                  # génération → base+RAG (= même calcul)
        else:
            recall_qs.append(it)                   # rappel → mémoire-poids (passe suivante)

    # --- passe MÉMOIRE (LoRA) : branche rappel de C1 (rappel→poids seuls, fidèle à serve.py)
    if recall_qs and mem_adapter:
        llm.set_adapter(mem_adapter)
        for it in recall_qs:
            c1[it["qid"]] = int(sa_match(llm.generate(sa_prompt(it["q"]), None), it["answer"]))
    else:
        # pas d'adapter commité → la branche rappel retombe sur la base nue (C0)
        for it in recall_qs:
            c1[it["qid"]] = c0[it["qid"]]

    def acc(d):
        return round(sum(d.values()) / max(1, len(d)), 4)
    return {"routes": routes, "c0": c0, "oracle": oracle, "c1": c1,
            "acc": {"C0": acc(c0), "RAG_oracle": acc(oracle), "C1": acc(c1)}}


def build_agent(cfg):
    workdir = os.path.join(cfg.workdir, "chat_workdir")
    os.makedirs(workdir, exist_ok=True)
    cfg.workdir = workdir
    llm = make_client(cfg)
    store = EventStore(":memory:")
    memory = TextMemory(cfg.memory_path, cfg.repo_dir, cfg.memory_inject_cap_tokens)
    detector = TwoShotDetector(memory)
    compactor = Compactor(store, llm, cfg)
    tools = Tools(workdir)
    return Agent(llm, store, memory, detector, compactor, tools, cfg), llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quant", default="8bit")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--compact-trigger", type=int, default=1200)
    ap.add_argument("--paraphrases", type=int, default=6)
    ap.add_argument("--max-flush", type=int, default=6)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    work = os.path.join(a.outdir, "work")
    os.makedirs(work, exist_ok=True)

    cfg = Config.from_env()
    cfg.backend = "hf"
    cfg.hf_model_path = a.model
    cfg.hf_quant = a.quant
    cfg.temperature = 0.0
    cfg.d2l_paraphrases = a.paraphrases
    cfg.compaction_trigger_tokens = a.compact_trigger    # paramètre de faisabilité (pré-enreg.)
    cfg.workdir = work
    cfg.memory_path = os.path.join(work, "MEMORY.md")

    agent, llm = build_agent(cfg)

    # mémoire long-terme partagée (LTM->poids, RAG->contexte), comme serve.py
    ltm = LTM(os.path.join(work, "ltm_qa.jsonl"))
    rag = RAG(os.path.join(work, "rag_corpus.txt"))
    ltm.clear()
    rag.clear()

    promoted = {"facts": 0, "compactions": 0}

    def on_compact(text):
        llm.set_adapter(None)                 # extraction LTM sur le modèle de BASE
        llm.cfg.mlx_max_tokens = 512
        added, _ = ltm.add_document(text, llm.generate)
        rag.add_document(text)
        promoted["facts"] += added
        promoted["compactions"] += 1
    agent.on_compact = on_compact

    corpus = [json.loads(l) for l in open(a.corpus, encoding="utf-8") if l.strip()]
    out_path = os.path.join(a.outdir, "autoloop_results.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["cycle"] for l in open(out_path, encoding="utf-8") if l.strip()}
    fout = open(out_path, "a", encoding="utf-8")

    seen_questions = []          # cumulatif (pour l'oubli + H-D1)
    mem_adapter = None

    for cyc in corpus:
        k = cyc["cycle"]
        if k in done:
            print(f"[cycle {k}] déjà fait — skip", flush=True)
            seen_questions += cyc["questions"]
            continue
        t0 = time.time()
        print(f"[cycle {k}] {cyc['date_range']} — {len(cyc['blocks'])} blocs, "
              f"{len(cyc['questions'])} QA — lecture (agent travaille)…", flush=True)

        # 1) l'agent LIT le corpus → saturation → auto-promotion
        llm.set_adapter(None)
        llm.cfg.mlx_max_tokens = 64
        before_facts = promoted["facts"]
        for bi, block in enumerate(cyc["blocks"]):
            agent.chat_turn(f"Please study this changelog excerpt and note the key "
                            f"facts (versions, dates, rule codes):\n\n{block}", mode="m0")
        # flush : quelques tours neutres pour pousser le matériel hors du contexte récent
        flush = 0
        while flush < a.max_flush:
            agent.chat_turn("Continue reviewing; summarize what you have learned so far.", mode="m0")
            flush += 1
            if promoted["facts"] > before_facts and flush >= 2:
                break
        print(f"    saturation: {promoted['compactions']} compactions cumulées, "
              f"LTM={ltm.count()} faits (+{promoted['facts'] - before_facts})", flush=True)

        # 2) /sleep gé cumulatif
        sl = sleep_train(llm, cfg, ltm, os.path.join(work, f"sleep_c{k}"),
                         os.path.join(work, f"train_c{k}.log"))
        if sl.get("committed"):
            mem_adapter = sl.get("adapter_dir")
        print(f"    /sleep c{k}: {json.dumps({x: sl.get(x) for x in ('ok','acquired','committed')})}"
              f" adapter={'oui' if mem_adapter else 'non'}", flush=True)

        # 3+4) éval cumulative sous C0 / RAG-oracle / C1(routeur)
        seen_questions += cyc["questions"]
        ev = eval_questions(llm, mem_adapter, rag, seen_questions)
        # non-régression génération (H-D2)
        gen = {t["id"]: run_gen_task(llm, rag, t) for t in GEN_TASKS}

        rec = {"cycle": k, "date_range": cyc["date_range"], "versions": cyc["versions"],
               "n_seen": len(seen_questions), "n_new": len(cyc["questions"]),
               "ltm_facts": ltm.count(), "compactions": promoted["compactions"],
               "sleep": {x: sl.get(x) for x in ("ok", "acquired", "committed", "n_facts",
                                                "iters", "n_heldout", "train_loss", "val_loss")},
               "acc": ev["acc"], "routes": ev["routes"],
               "per_q": {"c0": ev["c0"], "oracle": ev["oracle"], "c1": ev["c1"]},
               "gen_nonreg": {"pass": sum(gen.values()), "n": len(gen), "detail": gen},
               "elapsed_s": round(time.time() - t0)}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"    => C0={ev['acc']['C0']} C1={ev['acc']['C1']} "
              f"RAG-oracle={ev['acc']['RAG_oracle']} | gen {sum(gen.values())}/{len(gen)} "
              f"[{round(time.time()-t0)}s]", flush=True)

    print("AUTOLOOP_DONE", flush=True)


if __name__ == "__main__":
    main()
