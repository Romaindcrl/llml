"""Benchmark NIAH (Needle-in-a-Haystack) pour notre méthode-POIDS — le terrain de D2L (Sakana).

On insère K « aiguilles » (faits uniques) dans un long document de remplissage, puis on teste
le rappel de chaque aiguille sous 3 méthodes :
  - in-context : tout le document dans le prompt + question.
  - RAG        : on découpe, on récupère, on répond.
  - poids      : on internalise le document dans un LoRA (extraction Q/R + entraînement), on vide
                 le contexte, on répond depuis les poids seuls.
But : voir si notre méthode-poids attrape l'aiguille comme D2L, et si la précision (bf16 vs
8-bit, via M0_MLX_MODEL_PATH) y change quelque chose.
Logs par aiguille (cuttable). Live : tail -f logs/benchmark_niah.log
"""

from __future__ import annotations

import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import httpx  # noqa: E402

from m0 import d2l  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.rag import RAG, chunk_text  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_niah.log")
L = os.path.join(_PROJ, "models", "lora")
_T0 = time.time()

# aiguilles : (fait inséré, question, réponse) — réponses = tokens uniques, faciles à scorer
NEEDLES = [
    ("Le code d'acces du laboratoire Zephyr est 7F3A9K.",
     "Quel est le code d'acces du laboratoire Zephyr ?", "7F3A9K"),
    ("La salle attribuee a l'equipe Helios est la B214.",
     "Quelle salle est attribuee a l'equipe Helios ?", "B214"),
    ("Le serveur de production s'appelle nyxprod04.",
     "Comment s'appelle le serveur de production ?", "nyxprod04"),
    ("La cle de signature du service Orion est skorion91x.",
     "Quelle est la cle de signature du service Orion ?", "skorion91x"),
    ("Le jeton de secours du projet Tycho est TYCHO2231.",
     "Quel est le jeton de secours du projet Tycho ?", "TYCHO2231"),
]


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def fetch_filler(max_chars=20000):
    """Texte neutre (Wikipédia) comme botte de foin."""
    r = httpx.get("https://fr.wikipedia.org/w/api.php",
                  params={"format": "json", "action": "query", "prop": "extracts",
                          "explaintext": 1, "redirects": 1, "titles": "Histoire de l'informatique"},
                  timeout=30.0, headers={"User-Agent": "m0-niah/0.1"})
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return (page.get("extract", "") or "")[:max_chars]


def build_haystack(filler):
    """Insère les aiguilles à intervalles réguliers dans le remplissage."""
    sents = [s for s in filler.split(". ") if len(s) > 20]
    n = len(sents); k = len(NEEDLES)
    out = []
    for i, s in enumerate(sents):
        out.append(s.strip() + ".")
        for j, (fact, _, _) in enumerate(NEEDLES):
            if i == int((j + 1) * n / (k + 1)):
                out.append(fact)
    return " ".join(out)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    tag = os.path.basename(cfg.mlx_model_path)
    log(f"=== NIAH (modèle={tag}) ===")
    llm = make_client(cfg)

    haystack = build_haystack(fetch_filler())
    log(f"[1/3] Botte de foin = {count_tokens(haystack)} tokens, {len(NEEDLES)} aiguilles")

    # --- poids : internalisation du document dans un LoRA ---
    log("[2/3] Internalisation du document dans les poids (extraction Q/R + LoRA)")
    qa = d2l.clean_and_balance(d2l.extract_qa(haystack, llm.generate, n=30), max_per_answer=2)
    aug = d2l.augment_pairs(qa, llm.generate, n_paraphrases=1)
    data, adapter = f"{_PROJ}/logs/niah_data", f"{L}/niah"
    n = d2l.build_chat_dataset(aug + qa, data, repeat=4, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=3)
    iters = min(500, max(250, 8 * len(qa)))
    res = d2l.train_lora(cfg.mlx_model_path, data, adapter, iters=iters, num_layers=cfg.d2l_num_layers,
                         learning_rate=cfg.d2l_learning_rate, rank=16, python_exe=sys.executable,
                         log_file=LOG_PATH)
    log(f"  LoRA: ok={res['ok']} val_loss={res['val_loss']} ({n} lignes, {iters} iters ; {len(qa)} Q/R)")

    R = {m: 0 for m in ("in-context", "RAG", "poids")}
    rag = RAG(); rag.add_document(haystack)
    nq = len(NEEDLES)
    log("[3/3] Rappel des aiguilles")
    for i, (_, q, a) in enumerate(NEEDLES, 1):
        # in-context (modèle de base, doc entier)
        llm.set_adapter(None)
        ic = d2l.answer_recalled(llm.generate(f"Document :\n{haystack}\n\nQuestion : {q}\nReponds en quelques mots :", None), a)
        # RAG
        ctx = "\n".join(rag.topk(q, 4))
        rr = d2l.answer_recalled(llm.generate(f"Extraits :\n{ctx}\n\nQuestion : {q}\nReponds en quelques mots :", None), a)
        # poids (aucun contexte)
        llm.set_adapter(adapter)
        wt = d2l.answer_recalled(llm.generate(f"{q}\nReponds en quelques mots :", None), a)
        R["in-context"] += ic; R["RAG"] += rr; R["poids"] += wt
        log(f"  [aiguille {i}/{nq} '{a}'] in-context={int(ic)} RAG={int(rr)} poids={int(wt)}")

    log("")
    log(f"=== RÉSULTATS NIAH ({tag}) — aiguilles rappelées ===")
    for m in ("in-context", "RAG", "poids"):
        log(f"{m:12s} | {R[m]}/{nq} ({R[m]/nq*100:.0f}%)")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
