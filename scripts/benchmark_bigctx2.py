"""Banc D2 — version DISCRIMINANTE de D : la fenêtre est pleine de code LEGACY NON-conforme.

D était saturé (tout le monde à 100%) car le code en fenêtre était conforme -> conventions
gratuites par imitation. Ici, cas réel : vieille codebase à migrer — le code ambiant viole les
7 conventions du standard (accès db directs, print, retours None, camelCase, pas de docstring).
L'imitation devient un PIÈGE ; les conventions ne peuvent venir que du cahier — que le RAG
rate structurellement (§5 : conv 29%) et que la troncature ampute au débordement.

Bras : 14B tout-en-ctx · 14B + RAG-cahier (champion de D) · **14B + LLML (LoRA style entraîné
SUR le 14B, première fois)** + substitution faits · 7B + LLML (référence).
Charges 8k / 24k (débordement). Check de discriminativité loggé (14B+RAG conv < 100 ?).
Live : tail -f logs/benchmark_bigctx2.log
"""

from __future__ import annotations

import gc
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from scripts.benchmark_project import (  # noqa: E402
    FACTS, FILLER_ENTS, HELD, NS, WINDOW, build_spec, fit_code, _FOUNDATION,
)
from scripts.benchmark_spec_final import lookup_facts, substitute  # noqa: E402
from scripts.benchmark_spec_xl import CONV, frac  # noqa: E402
from m0 import d2l  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client, MLXClient  # noqa: E402
from m0.rag import RAG  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_bigctx2.log")
B7 = os.path.join(_PROJ, "models", "qwen2.5-7b-it-mlx-8bit")
B14 = os.path.join(_PROJ, "models", "qwen2.5-coder-14b-mlx-4bit")
AD7 = os.path.join(_PROJ, "models", "lora", "project")
AD14 = os.path.join(_PROJ, "models", "lora", "project14b")
DATA = os.path.join(_PROJ, "logs", "project_data")      # dataset du banc D (réutilisé tel quel)
L_LEVELS = [8000, 24000]
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def _purge():
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def legacy_block(entity, j):
    """Code hérité qui viole LES 7 conventions (imitation = piège)."""
    return (
        f"def getLegacy{entity.capitalize()}{j}(payloadDict):\n"
        f"    # module herite (avant standard) — acces direct, pas de validation\n"
        f"    row = db.query('SELECT * FROM {entity} WHERE id=%s', payloadDict['id'])\n"
        f"    if row is None:\n"
        f"        print('warn: {entity} {j} introuvable')\n"
        f"        return None\n"
        f"    return json.dumps(row)"
    )


def legacy_code(target_tokens):
    blocks, tok, j = [_FOUNDATION], count_tokens(_FOUNDATION), 0
    while tok < target_tokens:
        e = FILLER_ENTS[j % len(FILLER_ENTS)]
        blocks.append(legacy_block(e, j))
        tok += count_tokens(blocks[-1]); j += 1
    return "\n\n".join(blocks)


def task_for(ent):
    return (f"Implémente le module {ent}.py : la fonction get_{ent}(payload) qui récupère un "
            f"{ent} par identifiant, en respectant STRICTEMENT le cahier des charges (standard "
            "Nexus), pas le style du code hérité. N'oublie PAS la ligne d'audit obligatoire "
            "définie dans le module fondation du projet.")


def score_out(out, ent, agg, key, ctx_tok, kept):
    rm, ec = FACTS[ent]
    a = agg[key]
    a["c"].append(frac(out, CONV)); a["f"].append(frac(out, [rm, ec]))
    a["x"].append(1.0 if NS in out else 0.0); a["kept"].append(kept); a["ctx"].append(ctx_tok)
    return frac(out, CONV), frac(out, [rm, ec])


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    if not os.path.isdir(DATA):
        log("ERREUR : dataset project_data absent"); return
    spec = build_spec(); spec_tok = count_tokens(spec)
    rag = RAG(os.path.join(_PROJ, "logs", "rag_bigctx2.txt")); rag.clear(); rag.add_document(spec)
    log(f"=== BANC D2 — code ambiant LEGACY non-conforme (cahier {spec_tok} tok, fenêtre {WINDOW}) ===")

    SUT = ("7B + LLML", "14B tout-en-ctx", "14B + RAG-cahier", "14B + LLML")
    agg = {(L, s): {"c": [], "f": [], "x": [], "kept": [], "ctx": []} for L in L_LEVELS for s in SUT}

    # ---------- Phase 1 : 7B + LLML (référence)
    cfg = Config.from_env(); cfg.backend = "mlx"; cfg.mlx_model_path = B7
    llm = make_client(cfg); llm.set_adapter(AD7); llm.cfg.mlx_max_tokens = 380
    log("[1/3] 7B + LLML (référence)")
    for L in L_LEVELS:
        code = legacy_code(L)
        c_fit = fit_code(code, WINDOW - 600)
        for ent in HELD:
            draft = llm.generate(f"Code du projet :\n{c_fit}\n\n{task_for(ent)}"
                                 "\nÉcris uniquement le code Python du module.", None)
            out = substitute(draft, *lookup_facts(ent, rag)[:2])
            c, f = score_out(out, ent, agg, (L, "7B + LLML"), count_tokens(c_fit), 1.0 if NS in c_fit else 0.0)
            log(f"   [L={L} {ent}] conv={c*100:.0f}% faits={f*100:.0f}%")
        _purge()
    del llm; MLXClient._cache.clear(); _purge()

    # ---------- Phase 2 : LoRA style SUR LE 14B (première fois — données du banc D réutilisées)
    log("[2/3] entraînement du LoRA-cahier SUR le 14B (600 iters, lr 5e-5, rang 16)")
    res = d2l.train_lora(B14, DATA, AD14, iters=600, num_layers=8, learning_rate=5e-5,
                         rank=16, python_exe=sys.executable, log_file=LOG_PATH)
    log(f"   LoRA-14B ok={res['ok']} val_loss={res['val_loss']}")
    if not res["ok"]:
        log("   ⚠️ entraînement 14B échoué — le bras 14B+LLML sera sauté")
    _purge()

    # ---------- Phase 3 : les trois bras 14B
    cfg14 = Config.from_env(); cfg14.backend = "mlx"; cfg14.mlx_model_path = B14
    llm14 = make_client(cfg14); llm14.cfg.mlx_max_tokens = 380
    log("[3/3] 14B : tout-en-ctx · RAG-cahier · LLML(poids)")
    for L in L_LEVELS:
        code = legacy_code(L)
        for ent in HELD:
            task = task_for(ent)
            # (a) tout-en-ctx
            llm14.set_adapter(None)
            c_fit = fit_code(code, WINDOW - spec_tok - 700)
            o = llm14.generate(f"Cahier des charges :\n{spec}\n\nCode du projet :\n{c_fit}\n\n{task}"
                               "\nÉcris uniquement le code Python du module.", None)
            c, f = score_out(o, ent, agg, (L, "14B tout-en-ctx"),
                             spec_tok + count_tokens(c_fit), 1.0 if NS in c_fit else 0.0)
            log(f"   [L={L} {ent}] 14B-tout : conv={c*100:.0f}% faits={f*100:.0f}% (fondation {int(NS in c_fit)})")
            _purge()
            # (b) RAG-cahier
            chunks = "\n".join(rag.topk(task, 6))
            c_rag = fit_code(code, WINDOW - count_tokens(chunks) - 700)
            o = llm14.generate(f"Cahier (extraits) :\n{chunks}\n\nCode du projet :\n{c_rag}\n\n{task}"
                               "\nÉcris uniquement le code Python du module.", None)
            c, f = score_out(o, ent, agg, (L, "14B + RAG-cahier"),
                             count_tokens(chunks) + count_tokens(c_rag), 1.0 if NS in c_rag else 0.0)
            log(f"   [L={L} {ent}] 14B-RAG  : conv={c*100:.0f}% faits={f*100:.0f}%")
            _purge()
            # (c) 14B + LLML (poids)
            if res["ok"]:
                llm14.set_adapter(AD14)
                c_full = fit_code(code, WINDOW - 600)
                draft = llm14.generate(f"Code du projet :\n{c_full}\n\n{task}"
                                       "\nÉcris uniquement le code Python du module.", None)
                o = substitute(draft, *lookup_facts(ent, rag)[:2])
                c, f = score_out(o, ent, agg, (L, "14B + LLML"),
                                 count_tokens(c_full), 1.0 if NS in c_full else 0.0)
                log(f"   [L={L} {ent}] 14B-LLML : conv={c*100:.0f}% faits={f*100:.0f}% audit={int(NS in o)}")
                _purge()

    def av(L, s, k):
        v = agg[(L, s)][k]; return sum(v) / len(v) * 100 if v else 0.0
    log("")
    log("=== RÉSULTAT D2 — code ambiant NON-conforme (l'imitation est un piège) ===")
    log(f"{'L':>6} | {'méthode':18s} | conv | faits | audit | fondation | ctx")
    for L in L_LEVELS:
        for s in SUT:
            if agg[(L, s)]["c"]:
                log(f"{L:>6} | {s:18s} | {av(L,s,'c'):3.0f}% | {av(L,s,'f'):4.0f}% | {av(L,s,'x'):4.0f}% "
                    f"| {av(L,s,'kept'):3.0f}%      | {int(sum(agg[(L,s)]['ctx'])/max(len(agg[(L,s)]['ctx']),1))}")
        log("       " + "-" * 62)
    ragc = av(L_LEVELS[1], "14B + RAG-cahier", "c")
    log("")
    log(f"discriminativité : 14B+RAG conv = {ragc:.0f}% au débordement "
        f"({'✓ discriminant' if ragc < 95 else '✗ toujours saturé'})")
    if agg[(L_LEVELS[1], "14B + LLML")]["c"]:
        lc, lf = av(L_LEVELS[1], "14B + LLML", "c"), av(L_LEVELS[1], "14B + LLML", "f")
        if lc > ragc + 5:
            log(f"🟢 14B+LLML conv {lc:.0f}% > 14B+RAG {ragc:.0f}% : le cahier en POIDS résiste au "
                "piège d'imitation que le RAG ne couvre pas — LLML monte sur le gros modèle.")
        else:
            log(f"🟠 14B+LLML {lc:.0f}%/{lf:.0f}% vs RAG {ragc:.0f}% — à analyser.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
