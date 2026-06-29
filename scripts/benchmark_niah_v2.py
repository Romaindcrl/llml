"""NIAH v2 — la RECETTE est-elle le coupable de notre 20% ?

Original (benchmark_niah.py) : extraction Q/R lossy du document -> 20% de rappel-poids.
Hypothèse (corrigée avec l'utilisateur) : de bons LoRA d'encodage existent (D2L le prouve) ;
notre 20% vient de la recette (couverture incomplète + phrasé d'entraînement ≠ phrasé de test).

Ici on FORCE la couverture des 5 aiguilles + augmentation lourde (plein de phrasés par aiguille)
+ entraînement plus poussé, puis on teste le rappel. Si ça saute vers le haut -> c'est la recette.
Live : tail -f logs/benchmark_niah_v2.log
"""

from __future__ import annotations

import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0 import d2l  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.rag import RAG  # noqa: E402
from scripts.benchmark_niah import NEEDLES, build_haystack, fetch_filler  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_niah_v2.log")
L = os.path.join(_PROJ, "models", "lora")
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
    tag = os.path.basename(cfg.mlx_model_path)
    log(f"=== NIAH v2 — recette améliorée (modèle={tag}) ===")
    llm = make_client(cfg)

    haystack = build_haystack(fetch_filler())
    log(f"[1/3] Botte de foin = {count_tokens(haystack)} tokens, {len(NEEDLES)} aiguilles")

    # --- recette améliorée : couverture forcée + augmentation lourde ---
    base = [(q, a) for (_, q, a) in NEEDLES]                       # les 5 Q/R-aiguilles
    facts = [(f"Rappelle l'information confidentielle suivante.", fact) for (fact, _, _) in NEEDLES]
    log("[2/3] Entraînement : couverture forcée des 5 aiguilles + augmentation (phrasés variés)")
    aug = d2l.augment_pairs(base, llm.generate, n_paraphrases=6)   # ~6 phrasés par aiguille
    train_pairs = base + aug + facts
    log(f"  {len(train_pairs)} paires d'entraînement (5 base + {len(aug)} augmentées + {len(facts)} faits)")
    data, adapter = f"{_PROJ}/logs/niah_v2_data", f"{L}/niah_v2"
    n = d2l.build_chat_dataset(train_pairs, data, repeat=4, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=3)
    iters = min(600, max(400, 12 * len(train_pairs)))
    res = d2l.train_lora(cfg.mlx_model_path, data, adapter, iters=iters, num_layers=cfg.d2l_num_layers,
                         learning_rate=5e-5, rank=16, python_exe=sys.executable, log_file=LOG_PATH)
    log(f"  LoRA: ok={res['ok']} val_loss={res['val_loss']} ({n} lignes, {iters} iters)")

    R = {m: 0 for m in ("in-context", "RAG", "poids-v2")}
    rag = RAG(); rag.add_document(haystack)
    nq = len(NEEDLES)
    log("[3/3] Rappel des aiguilles")
    for i, (_, q, a) in enumerate(NEEDLES, 1):
        llm.set_adapter(None)
        ic = d2l.answer_recalled(llm.generate(f"Document :\n{haystack}\n\nQuestion : {q}\nReponds en quelques mots :", None), a)
        ctx = "\n".join(rag.topk(q, 4))
        rr = d2l.answer_recalled(llm.generate(f"Extraits :\n{ctx}\n\nQuestion : {q}\nReponds en quelques mots :", None), a)
        llm.set_adapter(adapter)
        wt = d2l.answer_recalled(llm.generate(f"{q}\nReponds en quelques mots :", None), a)
        R["in-context"] += ic; R["RAG"] += rr; R["poids-v2"] += wt
        log(f"  [aiguille {i}/{nq} '{a}'] in-context={int(ic)} RAG={int(rr)} poids-v2={int(wt)}")

    log("")
    log(f"=== RÉSULTATS NIAH v2 ({tag}) — recette améliorée ===")
    for m in ("in-context", "RAG", "poids-v2"):
        log(f"{m:12s} | {R[m]}/{nq} ({R[m]/nq*100:.0f}%)")
    log("  (rappel : recette originale = poids 20% = 1/5)")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
