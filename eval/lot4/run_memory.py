#!/usr/bin/env python3
"""Lot 4 — Claim A : la mémoire-poids de LLML (CDC §4.2).

Pour chaque document externe : on l'internalise via le pipeline /sleep STANDARD
de LLML (recette serve.py:_do_sleep, backend hf), puis on teste le rappel
closed-book (document RETIRÉ du contexte) sur ses questions à choix multiples,
dans 4 configs :

  C0  base nue, sans document           -> contrôle « doit échouer » (≈ hasard)
  C1  adapter /sleep chargé, closed-book -> LA mémoire-poids (le claim)
  C3  RAG (top-k passages en contexte), sans adapter
  C4  document ENTIER dans le prompt      -> borne haute honnête

Hypothèses pré-enregistrées (§4.2) : C1 > C0 largement ; C1 ≥ 80% de C4 ;
C1 bat C4 sur le coût (tokens/VRAM/latence). Kill : C1 < 50% de C4.

Le scoring MC est déterministe (greedy, on parse la lettre). Aucun harness
maison de mémoire : on réutilise le /sleep de LLML tel quel. Résultats écrits en
JSONL résumable (une ligne par doc).

Usage (pod) :
  /workspace/venv/bin/python eval/lot4/run_memory.py \
      --model Qwen/Qwen2.5-7B-Instruct --quant 8bit \
      --docs eval/lot4/quality_docs.jsonl \
      --outdir /workspace/results/lot4 --max-questions 10
"""

import argparse
import json
import os
import string
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ)

from m0 import d2l, d2l_hf  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.ltm import LTM  # noqa: E402
from m0.rag import RAG  # noqa: E402

_NEUTRAL_PROBES = ["Salut, ca va ?", "Combien font 2 + 3 ?",
                   "Quelle est la capitale de l'Allemagne ?"]
_LETTERS = string.ascii_uppercase


def chunk_words(text, words=550):
    """Découpe le document en passages d'environ `words` mots (sur frontières de
    paragraphe quand possible) pour une extraction DENSE : une seule passe sur un
    doc de ~2000 mots ne couvre que le début (budget de génération limité)."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if n + w > words and cur:
            chunks.append("\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += w
    if cur:
        chunks.append("\n".join(cur))
    # doc sans sauts de ligne exploitables : repli sur découpe brute par mots
    if len(chunks) <= 1:
        toks = text.split()
        chunks = [" ".join(toks[i:i + words]) for i in range(0, len(toks), words)] or [text]
    return chunks


def mc_prompt(q, options, doc=None, ctx=None):
    head = ""
    if doc is not None:
        head = f"Document:\n{doc}\n\n"
    elif ctx is not None:
        head = f"Context:\n{ctx}\n\n"
    opts = "\n".join(f"{_LETTERS[i]}) {o}" for i, o in enumerate(options))
    return (f"{head}Read the question and choose the single best answer.\n"
            f"Reply with ONLY the letter.\n\nQuestion: {q}\n{opts}\nAnswer:")


def parse_letter(out, options):
    """Extrait la réponse MC. 1) lettre isolée (A, "A)", "answer is B") ;
    2) sinon texte d'option cité tel quel ; 3) sinon -1 (abstention = faux).
    On évite le repli « 1re lettre quelconque » qui capturait une lettre au
    milieu d'un mot (ex. 'blah' -> B)."""
    n = len(options)
    up = (out or "").strip().upper()
    valid = set(_LETTERS[:n])
    for i, ch in enumerate(up):
        if ch in valid:
            prv = up[i - 1] if i > 0 else " "
            nxt = up[i + 1] if i + 1 < len(up) else " "
            if not prv.isalpha() and not nxt.isalpha():
                return _LETTERS.index(ch)
    for i, o in enumerate(options):  # le modèle a recopié le texte de l'option
        if o and o.strip().upper() in up:
            return i
    return -1


def sleep_train(llm, cfg, ltm, workdir, log_file):
    """Réplique fidèle de serve.py:_do_sleep pour un doc (backend hf)."""
    qa = ltm.all_qa()
    clean = d2l.clean_and_balance(qa, max_per_answer=3)
    if len(clean) < 2:
        return {"ok": False, "reason": "moins de 2 faits extraits"}
    aug = d2l.clean_and_balance(
        d2l.augment_pairs(clean, llm.generate, n_paraphrases=cfg.d2l_paraphrases),
        max_per_answer=12) or clean
    train_pairs, eval_pairs = d2l.split_train_eval(aug, heldout_per_answer=1)
    data_dir = os.path.join(workdir, "data")
    adapter_dir = os.path.join(workdir, "adapter")
    n_rows = d2l.build_chat_dataset(train_pairs, data_dir, repeat=cfg.d2l_repeat,
                                    anchors=d2l.ANCHOR_PAIRS,
                                    anchor_repeat=cfg.d2l_anchor_repeat)
    it = min(400, max(cfg.d2l_iters, 25 * len(clean)))
    # libère la VRAM du modèle d'inférence : le sous-process d'entraînement charge
    # sa PROPRE copie 8-bit, deux modèles ne tiennent pas sur 24 Go (OOM au backward).
    llm.unload()
    res = d2l_hf.train_lora(cfg.hf_model_path, data_dir, adapter_dir,
                            iters=it, num_layers=cfg.d2l_num_layers,
                            learning_rate=cfg.d2l_learning_rate, rank=16,
                            quant=cfg.hf_quant, log_file=log_file)
    if not res.get("ok"):
        return {"ok": False, "reason": "train échoué", "res": res,
                "adapter_dir": adapter_dir}
    # gate d'acquisition (serve.py:_gate_adapter)
    llm.cfg.mlx_max_tokens = 64
    llm.set_adapter(adapter_dir)
    probe = eval_pairs or clean
    ok = sum(1 for q, a in probe if d2l.answer_recalled(llm.generate(q, None), a))
    acquired = ok / max(1, len(probe))
    intact = not any(d2l.looks_degenerate(llm.generate(p, None))
                     for p in _NEUTRAL_PROBES)
    committed = bool(intact and acquired >= cfg.gate_acq)
    if not committed:
        llm.set_adapter(None)  # rollback
    return {"ok": True, "adapter_dir": adapter_dir, "n_facts": len(clean),
            "n_train_rows": n_rows, "iters": it, "n_heldout": len(probe),
            "acquired": round(acquired, 3), "intact": intact,
            "committed": committed, "train_loss": res.get("train_loss"),
            "val_loss": res.get("val_loss")}


def eval_config(llm, questions, adapter, doc_text, rag, tag):
    """Score MC d'une config ; renvoie (accuracy, latence médiane ms, prompt_chars médian, détails)."""
    import torch
    llm.set_adapter(adapter)
    llm.cfg.mlx_max_tokens = 12
    hits, lats, plens, per_q = 0, [], [], []
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass
    for item in questions:
        if tag == "C4":
            p = mc_prompt(item["q"], item["options"], doc=doc_text)
        elif tag == "C3":
            ctx = "\n".join(rag.topk(item["q"], 4))
            p = mc_prompt(item["q"], item["options"], ctx=ctx)
        else:  # C0, C1 : closed-book
            p = mc_prompt(item["q"], item["options"])
        t0 = time.time()
        out = llm.generate(p, None)
        lats.append((time.time() - t0) * 1000)
        plens.append(len(p))
        pred = parse_letter(out, item["options"])
        hit = int(pred == item["gold_idx"])
        hits += hit
        per_q.append({"qid": item["qid"], "pred": pred, "gold": item["gold_idx"],
                      "hit": hit, "difficult": item["difficult"]})
    try:
        vram = torch.cuda.max_memory_allocated() / 1e9
    except Exception:  # noqa: BLE001
        vram = None
    n = len(questions)
    return {"acc": round(hits / max(1, n), 4), "hits": hits, "n": n,
            "lat_ms_med": round(sorted(lats)[len(lats) // 2], 1) if lats else None,
            "prompt_chars_med": sorted(plens)[len(plens) // 2] if plens else None,
            "vram_gb": round(vram, 2) if vram else None, "per_q": per_q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quant", default="8bit")
    ap.add_argument("--docs", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-questions", type=int, default=10)
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
    cfg.d2l_paraphrases = a.paraphrases  # attribut ad hoc lu par sleep_train
    llm = make_client(cfg)

    docs = [json.loads(l) for l in open(a.docs, encoding="utf-8") if l.strip()]
    out_path = os.path.join(a.outdir, "memory_results.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["doc_id"] for l in open(out_path, encoding="utf-8") if l.strip()}
    fout = open(out_path, "a", encoding="utf-8")

    for di, doc in enumerate(docs, 1):
        if doc["doc_id"] in done:
            print(f"[{di}/{len(docs)}] doc {doc['doc_id']} déjà fait — skip", flush=True)
            continue
        t0 = time.time()
        qs = doc["questions"][: a.max_questions]
        print(f"[{di}/{len(docs)}] doc {doc['doc_id']} — {len(qs)} Q, "
              f"{len(doc['text'].split())} mots — ingest…", flush=True)
        workdir = os.path.join(a.outdir, f"work_{doc['doc_id']}")
        os.makedirs(workdir, exist_ok=True)
        ltm = LTM(os.path.join(workdir, "ltm.jsonl"))
        rag = RAG(os.path.join(workdir, "rag.txt"))
        ltm.clear()
        rag.clear()

        llm.set_adapter(None)
        llm.cfg.mlx_max_tokens = 1024  # ~12 paires/chunk sans troncature
        added = extracted = 0
        chunks = chunk_words(doc["text"], words=550)
        for ch in chunks:
            a, e = ltm.add_document(ch, llm.generate, n=12)
            added += a
            extracted += e
        rag.add_document(doc["text"])
        print(f"    LTM {added} faits (extraits {extracted}, {len(chunks)} chunks) — /sleep…",
              flush=True)

        sl = sleep_train(llm, cfg, ltm, workdir,
                         os.path.join(workdir, "train.log"))
        print(f"    /sleep: {json.dumps({k: sl.get(k) for k in ('ok','acquired','committed','iters','val_loss')}, ensure_ascii=False)}",
              flush=True)

        adapter_dir = sl.get("adapter_dir")
        res = {"doc_id": doc["doc_id"], "title": doc["title"],
               "n_words": len(doc["text"].split()), "n_questions": len(qs),
               "extracted": extracted, "sleep": {k: v for k, v in sl.items()
                                                 if k not in ("res",)},
               "configs": {}}
        # C0 base, C4 plein contexte, C3 RAG, C1 adapter closed-book
        res["configs"]["C0"] = eval_config(llm, qs, None, None, rag, "C0")
        res["configs"]["C4"] = eval_config(llm, qs, None, doc["text"], rag, "C4")
        res["configs"]["C3"] = eval_config(llm, qs, None, None, rag, "C3")
        if sl.get("ok") and adapter_dir and os.path.exists(
                os.path.join(adapter_dir, "adapter_model.safetensors")):
            res["configs"]["C1"] = eval_config(llm, qs, adapter_dir, None, rag, "C1")
        else:
            res["configs"]["C1"] = {"acc": None, "reason": "pas d'adapter (train/gate KO)"}

        c = res["configs"]
        c0, c1, c4 = c["C0"]["acc"], c["C1"].get("acc"), c["C4"]["acc"]
        res["summary"] = {
            "C0": c0, "C1": c1, "C3": c["C3"]["acc"], "C4": c4,
            "C1_vs_C4_ratio": round(c1 / c4, 3) if (c1 and c4) else None,
            "C0_must_fail": (c0 is not None and c0 <= 0.35),  # ≈ hasard 4-choix
            "elapsed_s": round(time.time() - t0, 1),
        }
        fout.write(json.dumps(res, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"    => C0={c0} C1={c1} C3={c['C3']['acc']} C4={c4} "
              f"(C1/C4={res['summary']['C1_vs_C4_ratio']}) "
              f"[{res['summary']['elapsed_s']}s]", flush=True)

    print("LOT4_MEMORY_DONE", flush=True)


if __name__ == "__main__":
    main()
