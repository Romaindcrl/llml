"""RAG ITÉRATIF (agentique) sur le résidu structuré — le dernier garde-fou avant tout dLLM.

Le RAG mono-passe échoue (0%) car la question ne contient que le nœud de DÉPART. Le RAG itératif
re-interroge à CHAQUE hop avec la nouvelle clé : récupère « clé X mène à ? », extrait le maillon
suivant (1 appel LLM), recommence. Ça devrait suivre la chaîne — la vraie question est le COÛT
(K+1 appels séquentiels au lieu d'1). On compare exactitude ET nb d'appels LLM.

Bilan attendu : si l'itératif réussit à ~100% mais coûte K+1 appels, le compresseur diffusion ne
gagne que (a) si le résidu est RÉUTILISÉ sur plusieurs requêtes (il amortit 1 passe), ou (b) si K
appels séquentiels coûtent plus (latence/$) qu'une compression. Si l'itératif échoue aussi → niche
plus forte. Réutilise EXACTEMENT les items de benchmark_residual_rag.
Live : tail -f logs/benchmark_residual_iterative.log
"""

from __future__ import annotations

import os
import re
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import scripts.benchmark_residual_rag as RR  # mêmes items/contextes/constantes  # noqa: E402
from m0 import d2l  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.rag import RAG  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_residual_iterative.log")
_NODE_RE = re.compile(r"N\d+H\d+")
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def find_value(text):
    low = (text or "").lower()
    for v in RR.VALUES:
        if v.lower() in low:
            return v
    return None


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== RAG ITÉRATIF — chaîne {RR.K}-hop (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg); llm.set_adapter(None)

    items = RR.build_items()
    rag = RAG(os.path.join(_PROJ, "logs", "rag_residual_iter.txt"))
    it_ok = 0
    total_calls = 0
    for idx, it in enumerate(items):
        chunks = RR.make_context(it, idx)
        rag.clear()
        for c in chunks:
            rag.add_document(c)

        current = it["start"]
        final = None
        calls = 0
        seen = set()
        for _hop in range(RR.K + 2):          # borne de sécurité
            if current in seen:
                break
            seen.add(current)
            top = rag.topk(f"La clé {current} mène à", RR.RAG_K)
            prompt = ("Voici des extraits de notes :\n" + "\n".join(top) +
                      f"\n\nDans ces extraits, la clé {current} « mène à » quoi ? "
                      "Réponds en UN seul mot : la clé ou la valeur qui suit.")
            ans = llm.generate(prompt, None)
            calls += 1
            v = find_value(ans)
            if v:                              # on a atteint une valeur terminale
                final = v
                break
            m = _NODE_RE.search(ans)
            if not m:
                break
            current = m.group(0)
        total_calls += calls
        hit = bool(final) and d2l.answer_recalled(final, it["value"])
        it_ok += hit
        if (idx + 1) % 4 == 0:
            log(f"   …{idx + 1}/{RR.N} (itératif OK={it_ok}, appels cumulés={total_calls})")

    n = RR.N
    log("")
    log(f"=== RÉSULTAT — résidu structuré ({RR.K}-hop), même contexte que le test RAG-proof ===")
    log(f"{'bras':32s} | exactitude | appels LLM / item")
    log(f"{'RAG mono-passe (réf.)':32s} |     0%     | 1")
    log(f"{'LLMLingua-2 (réf.)':32s} |     6%     | 1 (+compress)")
    log(f"{'FULL contexte (réf.)':32s} |   100%     | 1 (gros ctx)")
    log(f"{'RAG ITÉRATIF (agentique)':32s} | {it_ok/n*100:6.0f}%    | {total_calls/n:.1f}")
    log("")
    if it_ok / n >= 0.8:
        log(f"🟧 LE RAG ITÉRATIF RÉSOUT LA CHAÎNE ({it_ok}/{n}) au prix de ~{total_calls/n:.1f} appels/item. "
            "→ le compresseur diffusion ne gagne que si le résidu est RÉUTILISÉ (amortit 1 passe) "
            "ou si la latence/coût de K appels séquentiels dépasse une compression. Niche = réutilisation.")
    else:
        log(f"🟩 MÊME L'ITÉRATIF PEINE ({it_ok}/{n}) → niche plus forte : suivre la structure par "
            "retrieval reste fragile ; un compresseur qui préserve la structure a un espace net.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
