#!/usr/bin/env python3
"""Lot intégré — LLML COMPLET : routeur + mémoire-poids (replay) + RAG + base.

Teste le SYSTÈME entier, pas les briques isolées. Un flux de requêtes MIXTE
(rappel factuel sur docs internalisés + génération de code) passe par le routeur
`classify()` de LLML (m0/rag.py), qui décide PAR REQUÊTE :
  rappel   -> LoRA mémoire-poids (replay sur tout le LTM)
  génération -> base + RAG

Configs comparées :
  C0     base nue partout (aucun LLML)
  C1     LLML complet : le routeur décide par requête               <- LE système
  C2     mémoire toujours active (routeur désactivé) — mode d'échec  <- Claim B2
  oracle routage parfait (type réel connu) — borne haute du routeur

Métriques : précision de routage (classify vs type réel), rappel closed-book
(choix multiple, déterministe), génération pass@1 (EvalPlus officiel). Le routeur
"gagne sa place" si C1 ≈ oracle et > C0 en rappel SANS dégrader la génération —
là où C2 (toujours-mémoire) dégrade.

Usage (pod) :
  /workspace/venv/bin/python eval/lot_integrated/run_system.py \
    --model Qwen/Qwen2.5-7B-Instruct --quant 8bit \
    --docs eval/lot4/quality_docs.jsonl --recall-per-doc 10 --n-codegen 12 \
    --outdir /workspace/results/integrated
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
sys.path.insert(0, os.path.join(_PROJ, "eval", "lot2"))

from m0.config import Config          # noqa: E402
from m0.llm import make_client        # noqa: E402
from m0.ltm import LTM                # noqa: E402
from m0.rag import RAG, classify      # noqa: E402
from run_memory import chunk_words, mc_prompt, parse_letter, sleep_train  # noqa: E402
import live_score                     # noqa: E402  (EvalPlus scorer, Lot 2)

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
DRAFT_PROMPT = ("Complete the following Python function. Reply with the COMPLETE function "
                "(including the signature) in a single ```python block.\n\n```python\n{prompt}```")


def extract_code(out, prompt, entry_point):
    m = _CODE_RE.findall(out or "")
    code = m[-1].strip() if m else (out or "").strip()
    if f"def {entry_point}" not in code:
        code = prompt + "\n" + code
    return code


def answer_recall(llm, item, adapter):
    """MC closed-book sous l'adapter donné (None=base). Renvoie hit (0/1)."""
    llm.set_adapter(adapter)
    llm.cfg.mlx_max_tokens = 12
    out = llm.generate(mc_prompt(item["q"], item["options"]), None)
    return int(parse_letter(out, item["options"]) == item["gold_idx"])


def gen_code(llm, prob, adapter):
    """Génère une complétion pour un problème HumanEval sous l'adapter donné."""
    llm.set_adapter(adapter)
    llm.cfg.mlx_max_tokens = 640
    draft = llm.generate(DRAFT_PROMPT.format(prompt=prob["prompt"]), None)
    return extract_code(draft, prob["prompt"], prob["entry_point"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quant", default="8bit")
    ap.add_argument("--docs", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--recall-per-doc", type=int, default=10)
    ap.add_argument("--n-codegen", type=int, default=12)
    ap.add_argument("--gate-acq", type=float, default=0.45)
    ap.add_argument("--paraphrases", type=int, default=6)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    cfg = Config.from_env()
    cfg.backend = "hf"
    cfg.hf_model_path = a.model
    cfg.hf_quant = a.quant
    cfg.temperature = 0.0
    cfg.gate_acq = a.gate_acq
    cfg.d2l_paraphrases = a.paraphrases
    llm = make_client(cfg)

    # ---- 1) corpus : ingest TOUS les docs dans UN LTM (replay), un seul /sleep
    docs = [json.loads(l) for l in open(a.docs, encoding="utf-8") if l.strip()]
    work = os.path.join(a.outdir, "work")
    os.makedirs(work, exist_ok=True)
    ltm = LTM(os.path.join(work, "ltm.jsonl"))
    rag = RAG(os.path.join(work, "rag.txt"))
    ltm.clear()
    rag.clear()
    llm.set_adapter(None)
    llm.cfg.mlx_max_tokens = 1024
    tot_facts = 0
    for d in docs:
        for ch in chunk_words(d["text"], words=550):
            n_add, _ = ltm.add_document(ch, llm.generate, n=12)
            tot_facts += n_add
        rag.add_document(d["text"])
    print(f"[corpus] {len(docs)} docs, {tot_facts} faits en LTM — /sleep replay…", flush=True)

    sl = sleep_train(llm, cfg, ltm, work, os.path.join(work, "train.log"))
    adapter = sl.get("adapter_dir") if sl.get("committed") else None
    print(f"[/sleep] {json.dumps({k: sl.get(k) for k in ('ok','acquired','committed','iters')})}",
          flush=True)

    # ---- 2) flux de requêtes MIXTE
    recall_q, gen_q = [], []
    for d in docs:
        for it in d["questions"][: a.recall_per_doc]:
            recall_q.append({"type": "recall", "q": it["q"], "options": it["options"],
                             "gold_idx": it["gold_idx"], "qid": f"{d['doc_id']}:{it['qid']}"})
    from evalplus.data import get_human_eval_plus
    he = get_human_eval_plus()
    he_ids = sorted(he)[: a.n_codegen]
    for tid in he_ids:
        gen_q.append({"type": "generation", "task_id": tid, "prob": he[tid]})
    print(f"[flux] {len(recall_q)} rappels + {len(gen_q)} générations", flush=True)

    # ---- 3) routage (classify) sur CHAQUE requête, une fois
    llm.set_adapter(None)
    llm.cfg.mlx_max_tokens = 8
    routes = {}
    for q in recall_q:
        routes[q["qid"]] = classify(q["q"], llm.generate)
    for q in gen_q:
        # on route la VRAIE requête envoyée au système (une demande de complétion
        # de fonction), pas la seule signature — c'est ce que le routeur voit.
        req = "Complete the following Python function:\n" + q["prob"]["prompt"]
        routes[q["task_id"]] = classify(req, llm.generate)
    route_ok_recall = sum(1 for q in recall_q if routes[q["qid"]] == "recall")
    route_ok_gen = sum(1 for q in gen_q if routes[q["task_id"]] == "generation")
    print(f"[routage] rappel correct {route_ok_recall}/{len(recall_q)} · "
          f"génération correct {route_ok_gen}/{len(gen_q)}", flush=True)

    # ---- 4) chemin par config -> adapter à utiliser pour chaque requête
    def recall_adapter(cfg_name, q):
        if cfg_name == "C0":
            return None
        if cfg_name == "C2":
            return adapter
        if cfg_name == "oracle":
            return adapter          # type réel = recall -> mémoire
        return adapter if routes[q["qid"]] == "recall" else None  # C1

    def gen_adapter(cfg_name, q):
        if cfg_name == "C0":
            return None
        if cfg_name == "C2":
            return adapter
        if cfg_name == "oracle":
            return None             # type réel = generation -> base
        return adapter if routes[q["task_id"]] == "recall" else None  # C1 (mal routé -> mémoire)

    configs = ["C0", "C1", "C2", "oracle"]
    out = {"n_docs": len(docs), "n_facts": tot_facts, "sleep": {k: sl.get(k) for k in
           ("ok", "acquired", "committed", "iters", "n_facts")},
           "routing": {"recall_correct": route_ok_recall, "recall_total": len(recall_q),
                       "gen_correct": route_ok_gen, "gen_total": len(gen_q)},
           "configs": {}}

    for cn in configs:
        # rappel
        rhits = sum(answer_recall(llm, q, recall_adapter(cn, q)) for q in recall_q)
        # génération : produire les complétions puis scorer via EvalPlus officiel
        sols = {}
        for q in gen_q:
            sols[q["task_id"]] = gen_code(llm, q["prob"], gen_adapter(cn, q))
        scored = live_score.score_arm(sorted(he), sols, "humaneval", parallel=4)
        gpass = sum(1 for tid in he_ids if scored.get(tid, {}).get("plus"))
        out["configs"][cn] = {
            "recall_acc": round(rhits / max(1, len(recall_q)), 4), "recall_hits": rhits,
            "recall_n": len(recall_q),
            "gen_pass": gpass, "gen_n": len(gen_q),
            "gen_pass_at_1": round(gpass / max(1, len(gen_q)), 4)}
        print(f"[{cn}] rappel {rhits}/{len(recall_q)} · génération {gpass}/{len(gen_q)}",
              flush=True)
        json.dump(out, open(os.path.join(a.outdir, "system_results.json"), "w"), indent=2)

    print("INTEGRATED_DONE", flush=True)


if __name__ == "__main__":
    main()
