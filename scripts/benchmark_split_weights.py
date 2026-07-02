"""arm D — le VRAI test du #3 : la table stable dans les POIDS (LoRA) vs dans le contexte.

Le kill test précédent (benchmark_split.py) a montré que le SYSTEM PROMPT est un mauvais proxy des
poids (il dégrade). Ici on met réellement la table de routage dans un LoRA, puis on compare, en
COURBE vs longueur de contexte :
    arm A = modèle nu, table de routage DANS le contexte (doit survivre au rot) ;
    arm D = LoRA (table dans les poids), table ABSENTE du contexte (appliquée « gratuitement »).
Dans les deux cas : distracteurs + fait volatil + question dans le contexte. Le modèle doit composer
(type volatil, lu dans le contexte) + (table stable). Si arm D >= A ET l'écart D-A s'élargit avec la
longueur → l'offload-vers-poids combat le rot (validation propre du #3, comme courbe).
Live : tail -f logs/benchmark_split_weights.log
"""

from __future__ import annotations

import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import scripts.benchmark_split as S  # réutilise TABLE/ITEMS/distractors/TYPES/SERVICES  # noqa: E402
from m0 import d2l  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_split_weights.log")
PADDINGS = [0, 4000, 12000]
ADAPTER = os.path.join(_PROJ, "models", "lora", "split_table")
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def table_qa(generate):
    """Q/R qui internalisent la table type->service, avec augmentation de phrasés."""
    base = []
    for t, s in zip(S.TYPES, S.SERVICES):
        base.append((f"À quel service est affectée une demande de type {t} ?", f"Au service {s}."))
        base.append((f"Une demande de type {t} est routée vers quel service ?", f"Le service {s}."))
        base.append((f"Type {t} : quel service ?", f"{s}."))
    aug = d2l.augment_pairs(base, generate, n_paraphrases=6)
    return d2l.clean_and_balance(base + aug, max_per_answer=14)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== arm D — table dans les POIDS vs contexte (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg); llm.set_adapter(None)
    suffix = "\nRéponds par le seul nom du service :"

    # 1) internalise la table dans un LoRA
    log("[1/3] extraction + augmentation des Q/R de la table")
    qa = table_qa(llm.generate)
    data = os.path.join(_PROJ, "logs", "split_table_data")
    n = d2l.build_chat_dataset(qa, data, repeat=4, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=4)
    iters = min(500, max(200, 12 * len(qa)))
    log(f"[2/3] entraînement LoRA ({len(qa)} Q/R, {n} lignes, {iters} iters)")
    res = d2l.train_lora(cfg.mlx_model_path, data, ADAPTER, iters=iters, num_layers=cfg.d2l_num_layers,
                         learning_rate=cfg.d2l_learning_rate, rank=16, python_exe=sys.executable,
                         log_file=LOG_PATH)
    log(f"  LoRA: ok={res['ok']} val_loss={res['val_loss']}")

    # sanity : la table est-elle bien dans les poids ? (0 contexte, sans la table)
    llm.set_adapter(ADAPTER)
    sane = sum(d2l.answer_recalled(
        llm.generate(f"À quel service est affectée une demande de type {t} ?", None), s)
        for t, s in zip(S.TYPES, S.SERVICES))
    log(f"  sanity poids (0 ctx, table non fournie) : {sane}/{len(S.TYPES)} types rappelés")

    # 2) courbe : arm A (nu, table en contexte) vs arm D (LoRA, table absente)
    log("[3/3] courbe vs longueur")
    res_curve = {}
    for pad in PADDINGS:
        a_ok = d_ok = 0
        for idx, (vol, q, ans) in enumerate(S.ITEMS):
            block = S.distractors(pad, idx)
            core = f"Notes du projet :\n{block}\n\n{vol}\n\nQuestion : {q}{suffix}"
            llm.set_adapter(None)
            outA = llm.generate(f"{S.TABLE_TEXT}\n\n{core}", None)     # table dans le contexte
            llm.set_adapter(ADAPTER)
            outD = llm.generate(core, None)                            # table dans les poids, absente du ctx
            a_ok += d2l.answer_recalled(outA, ans)
            d_ok += d2l.answer_recalled(outD, ans)
            if (idx + 1) % 8 == 0:
                log(f"   …pad={pad} {idx + 1}/{len(S.ITEMS)} (A={a_ok} D={d_ok})")
        nq = len(S.ITEMS)
        ctxA = count_tokens(S.TABLE_TEXT) + count_tokens(S.distractors(pad, 0))
        res_curve[pad] = (a_ok / nq * 100, d_ok / nq * 100, ctxA, count_tokens(S.distractors(pad, 0)))
        log(f"[pad={pad:>5}] A(table en ctx, ~{ctxA}tok)={a_ok}/{nq} ({a_ok/nq*100:.0f}%) | "
            f"D(table en POIDS, ctx~{count_tokens(S.distractors(pad,0))}tok)={d_ok}/{nq} ({d_ok/nq*100:.0f}%) | "
            f"écart D-A={(d_ok-a_ok)/nq*100:+.0f} pts")

    log("")
    log("=== COURBE — table en POIDS (D) vs en CONTEXTE (A) ===")
    log(f"{'pad':>6} | A (en ctx) | D (en poids) | écart D-A | ctx économisé")
    widen = []
    for pad in PADDINGS:
        a, d, ctxA, ctxD = res_curve[pad]
        widen.append(d - a)
        log(f"{pad:>6} | {a:8.0f}% | {d:10.0f}% | {d-a:+6.0f} pts | {ctxA-ctxD:>6} tok")
    grew = widen[-1] >= widen[0] and widen[-1] >= 0
    log("")
    log("✅ #3 VALIDÉ (par les poids, pas le prompt) : D >= A et l'avantage tient/grandit avec la longueur"
        if grew else
        "⚠️ #3 NON validé sur ce régime : même en poids, l'offload ne dépasse pas la table en contexte ici")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
