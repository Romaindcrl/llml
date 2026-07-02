"""[STATUT : ⛔ RÉFUTATION #1 (réel). STRUCT 20% ≈ SUMMARY 30% ≈ LLMLingua 20% ≪ FULL 70%.
 La compaction structure-préservante ne bat PAS un résumé générique sur du vrai multi-hop.]

(2) Validation du prototype sur DONNÉES RÉELLES : HotpotQA (multi-hop humain, vrai Wikipédia).

Chaque item : une question multi-hop réelle + 10 paragraphes Wikipédia (2 utiles « gold » + 8
distracteurs). On compare, comme pour le synthétique :
  FULL / SUMMARY (résumé générique) / LLMLingua-2 (token-pruning) / STRUCT (m0.structcompact).
Score SQuAD-style (réponse normalisée présente). On filtre les réponses yes/no (scoring bruité).
Live : tail -f logs/benchmark_structcompact_real.log
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0 import structcompact  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_structcompact_real.log")
SAMPLE = os.path.join(_PROJ, "logs", "hotpot_sample.json")
N = 10
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def _norm(s):
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def hit(pred, gold):
    return bool(_norm(gold)) and _norm(gold) in _norm(pred)


def load_items():
    rows = json.load(open(SAMPLE, encoding="utf-8"))
    items = []
    for r in rows:
        ans = r["answer"]
        if ans.lower() in ("yes", "no") or len(ans) < 3:   # scoring yes/no trop bruité
            continue
        ctx = r["context"]
        paras = [f"{t}: {' '.join(s)}" for t, s in zip(ctx["title"], ctx["sentences"])]
        items.append({"q": r["question"], "a": ans, "ctx": "\n\n".join(paras)})
        if len(items) >= N:
            break
    return items


def _clear():
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def run(items, llm, pc, debug=False):
    R = {k: 0 for k in ("FULL", "SUMMARY", "LLMLingua", "STRUCT")}
    tok = {k: 0 for k in ("FULL", "SUMMARY", "LLMLingua", "STRUCT")}
    for idx, it in enumerate(items):
        ctx = it["ctx"]
        tail = f"\n\nQuestion: {it['q']}\nAnswer concisely (just the answer):"
        summary = llm.generate("Summarize these passages, keeping all names, dates and facts:\n\n" + ctx, None)
        art = structcompact.compact_structured(ctx, llm.generate)
        budget = max(80, count_tokens(art))
        comp = pc.compress_prompt(ctx, question=it["q"], target_token=budget,
                                  force_tokens=['\n', '?', '.', ':', '—', '>'])["compressed_prompt"]
        outs = {
            "FULL": llm.generate("Passages:\n" + ctx + tail, None),
            "SUMMARY": llm.generate("Summary:\n" + summary + tail, None),
            "LLMLingua": llm.generate("Compressed:\n" + comp + tail, None),
            "STRUCT": llm.generate("Structured context:\n" + art + tail, None),
        }
        for k, o in outs.items():
            R[k] += hit(o, it["a"])
        tok["FULL"] += count_tokens(ctx); tok["SUMMARY"] += count_tokens(summary)
        tok["LLMLingua"] += count_tokens(comp); tok["STRUCT"] += budget
        _clear()
        if debug:
            log(f"  [{idx}] a='{it['a']}' | " + " ".join(f"{k}={int(hit(outs[k], it['a']))}" for k in R))
            if idx == 0:
                log(f"     STRUCT artefact (200c): {art[:200]!r}")
        else:
            log(f"   …{idx + 1}/{len(items)} (" + " ".join(f"{k}={R[k]}" for k in R) + ")")
    return R, tok


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    debug = "--debug" in sys.argv
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== (2) HotpotQA réel — prototype structcompact (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg); llm.set_adapter(None)
    from llmlingua import PromptCompressor
    log("init LLMLingua-2…")
    pc = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                          use_llmlingua2=True, device_map="cpu")
    log("init OK")
    items = load_items()
    if debug:
        items = items[:2]
    log(f"{len(items)} questions multi-hop réelles (yes/no filtrées) ; ctx ~{count_tokens(items[0]['ctx'])}tok")
    R, tok = run(items, llm, pc, debug=debug)
    n = len(items)
    log("")
    log("=== RÉSULTAT — HotpotQA réel (multi-hop humain) ===")
    log(f"{'méthode':30s} | exactitude | ctx tokens/item")
    for k in ("FULL", "SUMMARY", "LLMLingua", "STRUCT"):
        log(f"{k:30s} | {R[k]/n*100:5.0f}%    | {tok[k]//n}")
    log("")
    best_base = max(R["SUMMARY"], R["LLMLingua"]) / n
    if R["STRUCT"] / n >= best_base + 0.1 and R["STRUCT"] / n >= 0.5 * R["FULL"] / max(1, n) + 0.0:
        log(f"🟩 GÉNÉRALISE AU RÉEL : STRUCT={R['STRUCT']/n*100:.0f}% vs résumé={R['SUMMARY']/n*100:.0f}% / "
            f"token-pruning={R['LLMLingua']/n*100:.0f}% (FULL={R['FULL']/n*100:.0f}%), à ~{tok['FULL']/max(1,tok['STRUCT']):.0f}× compression.")
    else:
        log(f"🟧 MITIGÉ sur le réel : STRUCT={R['STRUCT']/n*100:.0f}% / SUMMARY={R['SUMMARY']/n*100:.0f}% / "
            f"LLMLingua={R['LLMLingua']/n*100:.0f}% / FULL={R['FULL']/n*100:.0f}% — la marge synthétique se réduit.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
