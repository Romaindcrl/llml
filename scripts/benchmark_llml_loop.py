"""Auto-amélioration × SYSTÈME LLML COMPLET, en conditions réelles (fenêtre 32k, vs 14B).

La boucle complète (pas le fine-tune seul) :
  échec → doc trouvée → [t+0s] indexée en RAG (récupération IMMÉDIATE) → [idle] consolidation
  en poids (auto-étude validée) → ensuite : 2-ÉTAPES = poids + vérification RAG (le mode LLML).

Conditions réelles :
  - CHARGE : les mêmes questions quand ~20k tokens de code projet occupent la fenêtre 32k
    (le prompt de génération/vérif LLML n'a PAS besoin du code → side-channel ~250 tok).
  - GROS MODÈLE : baseline qwen2.5-coder-14B-4bit (nu, doc en contexte, avec/sans charge)
    → teste « 7B + LLML ≥ 14B nu ? ».

Phases : 1 base→échec · 2 RAG immédiat · 3 étude+LoRA · 4 poids seuls · 5 LLML 2-étapes ·
6 sous charge 20k (7B doc-en-ctx vs LLML) · 7 14B (sanity/doc-ctx/doc+charge).
Live : tail -f logs/benchmark_llml_loop.log
"""

from __future__ import annotations

import gc
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from scripts.benchmark_selfimprove import DOC, KNOWLEDGE  # noqa: E402
from m0 import d2l  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client, MLXClient  # noqa: E402
from m0.rag import RAG  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_llml_loop.log")
ADAPTER = os.path.join(_PROJ, "models", "lora", "corvex_loop")
B7 = os.path.join(_PROJ, "models", "qwen2.5-7b-it-mlx-8bit")
B14 = os.path.join(_PROJ, "models", "qwen2.5-coder-14b-mlx-4bit")
LOAD_TOK = 20000
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


def make_noise(target_tok):
    """~20k tokens de code projet plausible (la fenêtre est occupée)."""
    lines, j = [], 0
    while count_tokens("\n".join(lines)) < target_tok:
        lines.append(f"def util_{j}(x, y={j % 9}):  # helper {j} du module ingestion\n"
                     f"    acc = x * {j % 7} + y - {j % 13}\n"
                     f"    return acc if acc > 0 else {j % 5}  # borne basse, statut ok")
        j += 1
    return "\n".join(lines)


def score(llm, items, prompt_fn, cap=70, tag=""):
    ok = 0
    llm.cfg.mlx_max_tokens = cap
    for i, (q, a) in enumerate(items):
        out = llm.generate(prompt_fn(q), None)
        ok += d2l.answer_recalled(out, a)
        if tag and (i + 1) % 4 == 0:
            log(f"   …{tag} {i + 1}/{len(items)} (ok={ok})")
    return ok


def two_step(llm, rag, q, a):
    """LLML : réponse par les POIDS puis VÉRIFICATION contre la mémoire externe."""
    llm.cfg.mlx_max_tokens = 60
    draft = llm.generate(q + "\nRéponds brièvement :", None)
    chunks = "\n".join(rag.topk(q, 3))
    corr = llm.generate(
        f"Extraits de la documentation :\n{chunks}\n\nQuestion : {q}\nRéponse proposée : {draft}\n"
        "Corrige la réponse si les extraits la contredisent ou la précisent. "
        "Donne UNIQUEMENT la réponse finale :", None)
    return d2l.answer_recalled(corr, a), count_tokens(chunks) + count_tokens(draft) + 30


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    nk = len(KNOWLEDGE)
    R, ctx_cost = {}, {}

    cfg = Config.from_env(); cfg.backend = "mlx"; cfg.mlx_model_path = B7
    llm = make_client(cfg); llm.set_adapter(None)
    log(f"=== BOUCLE LLML COMPLÈTE (7B={os.path.basename(B7)}, charge {LOAD_TOK} tok, vs 14B) ===")

    log("[1/7] échec initial (base, 0 ctx)")
    R["base"] = score(llm, KNOWLEDGE, lambda q: q + "\nRéponds brièvement :")
    ctx_cost["base"] = 0
    log(f"   base {R['base']}/{nk}")

    log("[2/7] doc trouvée → RAG : récupération IMMÉDIATE (t+0, aucun entraînement)")
    rag = RAG(os.path.join(_PROJ, "logs", "rag_corvex_loop.txt")); rag.clear()
    rag.add_document(DOC)
    R["RAG immédiat"] = score(llm, KNOWLEDGE,
                              lambda q: f"Extraits :\n{chr(10).join(rag.topk(q, 4))}\n\nQuestion : {q}\nRéponds brièvement :")
    ctx_cost["RAG immédiat"] = count_tokens("\n".join(rag.topk(KNOWLEDGE[0][0], 4)))
    log(f"   RAG {R['RAG immédiat']}/{nk} (ctx ~{ctx_cost['RAG immédiat']} tok)")

    log("[3/7] consolidation idle : auto-étude (stratégie validée) + LoRA")
    facts = [ln.strip() for ln in DOC.splitlines() if len(ln.strip()) > 30]
    llm.cfg.mlx_max_tokens = 620
    qa = d2l.extract_qa(DOC, llm.generate, n=24); _purge()
    llm.cfg.mlx_max_tokens = 220
    for fact in facts:
        qa += d2l.extract_qa(fact, llm.generate, n=6); _purge()
    qa = d2l.clean_and_balance(qa, max_per_answer=3)
    train = d2l.clean_and_balance(qa + d2l.augment_pairs(qa, llm.generate, n_paraphrases=6), max_per_answer=12)
    blob = " ".join(q + " " + a for q, a in train).lower()
    cov = sum(1 for _, a in KNOWLEDGE if a.lower() in blob)
    log(f"   {len(train)} Q/R auto-générées, couverture {cov}/{nk}")
    data = os.path.join(_PROJ, "logs", "corvex_loop_data")
    n = d2l.build_chat_dataset(train, data, repeat=5, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=4)
    iters = min(600, max(350, 9 * len(train)))
    res = d2l.train_lora(cfg.mlx_model_path, data, ADAPTER, iters=iters, num_layers=cfg.d2l_num_layers,
                         learning_rate=cfg.d2l_learning_rate, rank=16, python_exe=sys.executable,
                         log_file=LOG_PATH)
    log(f"   LoRA ok={res['ok']} val_loss={res['val_loss']}")
    _purge()

    log("[4/7] poids seuls (0 ctx)")
    llm.set_adapter(ADAPTER)
    R["poids seuls"] = score(llm, KNOWLEDGE, lambda q: q + "\nRéponds brièvement :")
    ctx_cost["poids seuls"] = 0
    log(f"   poids {R['poids seuls']}/{nk}")

    log("[5/7] LLML 2-ÉTAPES : poids + vérification RAG (le vrai mode système)")
    ok = cost = 0
    for i, (q, a) in enumerate(KNOWLEDGE):
        h, c = two_step(llm, rag, q, a)
        ok += h; cost = max(cost, c)
        if (i + 1) % 4 == 0:
            log(f"   …2-étapes {i + 1}/{nk} (ok={ok})")
    R["LLML 2-étapes"] = ok; ctx_cost["LLML 2-étapes"] = cost
    log(f"   LLML {ok}/{nk} (ctx ~{cost} tok)")

    log(f"[6/7] SOUS CHARGE : {LOAD_TOK} tok de code dans la fenêtre 32k")
    noise = make_noise(LOAD_TOK)
    llm.set_adapter(None)
    R["7B doc+code en ctx"] = score(
        llm, KNOWLEDGE,
        lambda q: f"Code du projet :\n{noise}\n\nDocumentation :\n{DOC}\n\nQuestion : {q}\nRéponds brièvement :",
        cap=60, tag="7B chargé")
    ctx_cost["7B doc+code en ctx"] = count_tokens(noise) + count_tokens(DOC)
    log(f"   7B doc+code {R['7B doc+code en ctx']}/{nk} (ctx ~{ctx_cost['7B doc+code en ctx']} tok)")
    log("   (LLML 2-étapes = side-channel : par construction insensible à la charge, ctx ~"
        f"{ctx_cost['LLML 2-étapes']} tok)")
    _purge()

    log("[7/7] BASELINE GROS MODÈLE : qwen2.5-coder-14B-4bit nu")
    del llm
    MLXClient._cache.clear(); _purge()
    cfg14 = Config.from_env(); cfg14.backend = "mlx"; cfg14.mlx_model_path = B14
    llm14 = make_client(cfg14); llm14.set_adapter(None)
    R["14B nu (0 ctx)"] = score(llm14, KNOWLEDGE, lambda q: q + "\nRéponds brièvement :")
    ctx_cost["14B nu (0 ctx)"] = 0
    log(f"   14B 0ctx {R['14B nu (0 ctx)']}/{nk} (sanity : savoir privé inconnu)")
    R["14B + doc en ctx"] = score(llm14, KNOWLEDGE,
                                  lambda q: f"Documentation :\n{DOC}\n\nQuestion : {q}\nRéponds brièvement :")
    ctx_cost["14B + doc en ctx"] = count_tokens(DOC)
    log(f"   14B doc {R['14B + doc en ctx']}/{nk}")
    R["14B doc+code en ctx"] = score(
        llm14, KNOWLEDGE,
        lambda q: f"Code du projet :\n{noise}\n\nDocumentation :\n{DOC}\n\nQuestion : {q}\nRéponds brièvement :",
        cap=60, tag="14B chargé")
    ctx_cost["14B doc+code en ctx"] = count_tokens(noise) + count_tokens(DOC)
    log(f"   14B doc+code {R['14B doc+code en ctx']}/{nk}")

    log("")
    log("=== RÉSULTAT — boucle LLML complète, conditions réelles ===")
    log(f"{'bras':26s} | score | ctx (tok)")
    for k, v in R.items():
        log(f"{k:26s} | {v}/{nk} ({v/nk*100:3.0f}%) | {ctx_cost[k]}")
    log("")
    llml, b14 = R["LLML 2-étapes"], R["14B + doc en ctx"]
    llml_l, b14_l = R["LLML 2-étapes"], R["14B doc+code en ctx"]
    if llml >= b14 and llml_l >= b14_l:
        log(f"🟢 PARI TENU : 7B+LLML ({llml}/{nk}) ≥ 14B nu+doc ({b14}/{nk}) — et sous charge 20k, "
            f"LLML ({llml_l}/{nk}, ~{ctx_cost['LLML 2-étapes']} tok) ≥ 14B chargé ({b14_l}/{nk}, "
            f"~{ctx_cost['14B doc+code en ctx']} tok).")
    else:
        log(f"🟠 PARTIEL : 7B+LLML {llml}/{nk} vs 14B+doc {b14}/{nk} ; sous charge {llml_l} vs {b14_l}.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
