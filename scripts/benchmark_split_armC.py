"""arm C — une compaction AR pas chère (LLMLingua-2) sur le RÉSIDU volatil referme-t-elle le rot ?

Garde-fou décisif avant tout dLLM : si compresser le résidu (distracteurs + fait volatil) avec
LLMLingua-2 récupère les ~12% que les poids ne peuvent pas sauver à 12k, alors il n'y a PAS de
place pour un compresseur diffusion (la compaction AR suffit). Sinon, cet écart résiduel EST la
niche du compresseur.

Bras (table de routage TOUJOURS dans les POIDS = config arm D validée) :
    D = résidu NON compressé en contexte (réf.) ;
    C = résidu compressé par LLMLingua-2 (target_token court, query-aware) ;
    A = réf. « table en contexte, non compressé » (modèle nu).
On mesure aussi le taux de SURVIE du type volatil dans le compressé (diagnostic).
Live : tail -f logs/benchmark_split_armC.log
"""

from __future__ import annotations

import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import scripts.benchmark_split as S  # noqa: E402
from m0 import d2l  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_split_armC.log")
ADAPTER = os.path.join(_PROJ, "models", "lora", "split_table")
PADDINGS = [4000, 12000]      # 0 inutile (pas de rot)
TARGET_TOK = 400              # budget cible de la compaction AR
SUFFIX = "\nRéponds par le seul nom du service :"
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== arm C — LLMLingua-2 sur le résidu (modèle={os.path.basename(cfg.mlx_model_path)}) ===")

    from llmlingua import PromptCompressor
    log("init LLMLingua-2 (multilingue, CPU)…")
    pc = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                          use_llmlingua2=True, device_map="cpu")
    log("init OK")

    llm = make_client(cfg)
    res = {}
    for pad in PADDINGS:
        # 1) pré-compresse le résidu de chaque item
        prepared = []
        kept = 0
        for idx, (vol, q, ans) in enumerate(S.ITEMS):
            block = S.distractors(pad, idx)
            residual = f"Notes du projet :\n{block}\n\n{vol}"
            comp = pc.compress_prompt(residual, question=q, target_token=TARGET_TOK,
                                      force_tokens=['\n', '?', '.', ':'])["compressed_prompt"]
            vtype = vol.split("de type ")[1].rstrip(".")      # ex. "ALPHA"
            kept += vtype in comp
            prepared.append((vol, q, ans, block, comp))
        ctxC = count_tokens(prepared[0][4])
        log(f"[pad={pad}] résidu compressé ~{ctxC}tok | type volatil survivant {kept}/{len(S.ITEMS)}")

        # 2) A (nu, table en contexte, non compressé)
        llm.set_adapter(None)
        a_ok = sum(d2l.answer_recalled(
            llm.generate(f"{S.TABLE_TEXT}\n\nNotes du projet :\n{block}\n\n{vol}\n\nQuestion : {q}{SUFFIX}", None), ans)
            for vol, q, ans, block, comp in prepared)

        # 3) D (poids, résidu non compressé) + C (poids, résidu compressé LLMLingua-2)
        llm.set_adapter(ADAPTER)
        d_ok = c_ok = 0
        for i, (vol, q, ans, block, comp) in enumerate(prepared):
            d_ok += d2l.answer_recalled(
                llm.generate(f"Notes du projet :\n{block}\n\n{vol}\n\nQuestion : {q}{SUFFIX}", None), ans)
            c_ok += d2l.answer_recalled(llm.generate(f"{comp}\n\nQuestion : {q}{SUFFIX}", None), ans)
            if (i + 1) % 8 == 0:
                log(f"   …pad={pad} {i + 1}/{len(S.ITEMS)} (A={a_ok} D={d_ok} C={c_ok})")
        n = len(S.ITEMS)
        res[pad] = (a_ok / n * 100, d_ok / n * 100, c_ok / n * 100, ctxC, kept / n * 100)
        log(f"[pad={pad}] A(table en ctx)={a_ok/n*100:.0f}% | D(poids, résidu brut)={d_ok/n*100:.0f}% | "
            f"C(poids, résidu LLMLingua-2 ~{ctxC}tok)={c_ok/n*100:.0f}%")

    log("")
    log("=== COURBE — la compaction AR du résidu referme-t-elle le rot ? ===")
    log(f"{'pad':>6} | A en ctx | D poids brut | C poids+LLMLingua2 | survie type")
    closes = []
    for pad in PADDINGS:
        a, d, c, ctxC, ks = res[pad]
        closes.append(c >= d)
        log(f"{pad:>6} | {a:6.0f}% | {d:10.0f}% | {c:16.0f}% | {ks:.0f}%")
    log("")
    if all(c >= 99 for (_, _, c, _, _) in res.values()):
        log("🟥 DIFFUSION NON JUSTIFIÉE : LLMLingua-2 (AR pas cher) referme déjà le rot → pas de niche.")
    elif all(closes):
        log("🟧 AMBIGU : LLMLingua-2 aide mais ne sature pas — niche étroite, à creuser.")
    else:
        log("🟩 NICHE CONFIRMÉE : la compaction AR du résidu N'aide PAS (≤ poids bruts) → "
            "un compresseur qui préserve le fait volatil a un espace à gagner.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
