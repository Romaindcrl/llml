#!/usr/bin/env python3
"""Lot 4b — Claim A, volet ÉQUITABLE : mémoire-poids sur docs techniques factuels.

Contrairement à QuALITY (compréhension d'une nouvelle → désaligné avec l'extraction
de faits), ici les questions sont des LOOKUPS FACTUELS (date, version, code de règle,
identifiant) sur des changelogs post-2024 que le modèle nu ne connaît pas — le régime
qui COLLE au design de la mémoire-poids (/sleep extrait des faits courts → LoRA).

Même pipeline /sleep que le volet QuALITY (recette identique), scoring court (le span
attendu apparaît-il dans la réponse). 4 configs closed-book :
  C0 base nue · C1 mémoire-poids · C3 RAG · C4 plein contexte.
Contrôle C0-doit-échouer : sous-ensemble « dur » = questions que le nu rate (post-2024
→ attendu proche de 0). C'est là que la mémoire doit prouver sa valeur.

Usage (pod) :
  /workspace/venv/bin/python eval/lot4b/run_techqa.py --model Qwen/Qwen2.5-7B-Instruct \
      --quant 8bit --docs eval/lot4b/tech_docs.jsonl --outdir /workspace/results/lot4b
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
from m0.rag import RAG                # noqa: E402
from run_memory import chunk_words, sleep_train  # noqa: E402


def sa_prompt(q, doc=None, ctx=None):
    head = ""
    if doc is not None:
        head = f"Document:\n{doc}\n\n"
    elif ctx is not None:
        head = f"Context:\n{ctx}\n\n"
    return (f"{head}Answer with ONLY the specific fact (a date, version number, code, "
            f"or short name) — no explanation.\n\nQuestion: {q}\nAnswer:")


def sa_match(out, exp):
    """Le span attendu (normalisé) apparaît-il dans la sortie ? Multi-tokens
    (ex. 'TY and RUFF', 'over 20') : tous les tokens saillants présents."""
    o = re.sub(r"\s+", " ", (out or "").lower())
    e = (exp or "").lower().strip()
    if e in o:
        return True
    toks = [t for t in re.split(r"[ ,]+", e) if t and t not in ("and", "the", "a", "of", "to")]
    return len(toks) > 1 and all(t in o for t in toks)


def eval_cfg(llm, questions, adapter, doc_text, rag, tag):
    import torch
    llm.set_adapter(adapter)
    llm.cfg.mlx_max_tokens = 24
    hits, lats, plens, per_q = 0, [], [], []
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass
    for it in questions:
        if tag == "C4":
            p = sa_prompt(it["q"], doc=doc_text)
        elif tag == "C3":
            p = sa_prompt(it["q"], ctx="\n".join(rag.topk(it["q"], 4)))
        else:
            p = sa_prompt(it["q"])
        t0 = time.time()
        out = llm.generate(p, None)
        lats.append((time.time() - t0) * 1000)
        plens.append(len(p))
        hit = int(sa_match(out, it["answer"]))
        hits += hit
        per_q.append({"qid": it["qid"], "hit": hit, "answer": it["answer"],
                      "got": (out or "").strip()[:80]})
    try:
        vram = torch.cuda.max_memory_allocated() / 1e9
    except Exception:  # noqa: BLE001
        vram = None
    n = len(questions)
    return {"acc": round(hits / max(1, n), 4), "hits": hits, "n": n, "per_q": per_q,
            "lat_ms_med": round(sorted(lats)[len(lats) // 2], 1) if lats else None,
            "prompt_chars_med": sorted(plens)[len(plens) // 2] if plens else None,
            "vram_gb": round(vram, 2) if vram else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quant", default="8bit")
    ap.add_argument("--docs", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--paraphrases", type=int, default=6)
    ap.add_argument("--gate-acq", type=float, default=0.45)
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

    docs = [json.loads(l) for l in open(a.docs, encoding="utf-8") if l.strip()]
    out_path = os.path.join(a.outdir, "techqa_results.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["doc_id"] for l in open(out_path, encoding="utf-8") if l.strip()}
    fout = open(out_path, "a", encoding="utf-8")

    for di, doc in enumerate(docs, 1):
        if doc["doc_id"] in done:
            print(f"[{di}/{len(docs)}] {doc['doc_id']} déjà fait — skip", flush=True)
            continue
        t0 = time.time()
        qs = doc["questions"]
        print(f"[{di}/{len(docs)}] doc {doc['doc_id']} — {len(qs)} QA, "
              f"{len(doc['text'].split())} mots — ingest…", flush=True)
        work = os.path.join(a.outdir, f"work_{doc['doc_id']}")
        os.makedirs(work, exist_ok=True)
        ltm = LTM(os.path.join(work, "ltm.jsonl"))
        rag = RAG(os.path.join(work, "rag.txt"))
        ltm.clear()
        rag.clear()
        llm.set_adapter(None)
        llm.cfg.mlx_max_tokens = 1024
        added = 0
        for ch in chunk_words(doc["text"], words=550):
            n_add, _ = ltm.add_document(ch, llm.generate, n=14)
            added += n_add
        rag.add_document(doc["text"])
        print(f"    LTM {added} faits — /sleep…", flush=True)

        sl = sleep_train(llm, cfg, ltm, work, os.path.join(work, "train.log"))
        adapter = sl.get("adapter_dir") if sl.get("committed") else None
        print(f"    /sleep: {json.dumps({k: sl.get(k) for k in ('ok','acquired','committed')})}",
              flush=True)

        res = {"doc_id": doc["doc_id"], "title": doc["title"], "n_q": len(qs),
               "sleep": {k: v for k, v in sl.items() if k not in ("res",)}, "configs": {}}
        res["configs"]["C0"] = eval_cfg(llm, qs, None, None, rag, "C0")
        res["configs"]["C4"] = eval_cfg(llm, qs, None, doc["text"], rag, "C4")
        res["configs"]["C3"] = eval_cfg(llm, qs, None, None, rag, "C3")
        if adapter and os.path.exists(os.path.join(adapter, "adapter_model.safetensors")):
            res["configs"]["C1"] = eval_cfg(llm, qs, adapter, None, rag, "C1")
        else:
            res["configs"]["C1"] = {"acc": None, "reason": "pas d'adapter"}
        c = res["configs"]
        print(f"    => C0={c['C0']['acc']} C1={c['C1'].get('acc')} C3={c['C3']['acc']} "
              f"C4={c['C4']['acc']} [{round(time.time()-t0)}s]", flush=True)
        fout.write(json.dumps(res, ensure_ascii=False) + "\n")
        fout.flush()

    print("TECHQA_DONE", flush=True)


if __name__ == "__main__":
    main()
