"""Test RAG-proof : un résidu NON-récupérable (chaîne multi-hop) défait-il le RAG ?

Le résidu n'est PAS un fait discret mais une CHAÎNE de pointeurs : N0 → N1 → N2 → VALEUR,
éparpillée dans du bruit (beaucoup d'autres « La clé X mène à Y »). La réponse exige de SUIVRE
toute la chaîne (composer), pas d'en RETROUVER un bout. La question ne contient que le nœud de
DÉPART → le RAG (BM25, 1 passe) ne peut récupérer que le 1er lien, pas les suivants.

Bras : FULL (tout en contexte = plafond) · RAG (BM25 top-k) · LLMLingua-2 (compaction AR).
Si FULL réussit MAIS RAG échoue → le résidu est RAG-proof → niche réelle pour un compresseur
qui préserve la structure (justifie d'explorer la diffusion). Si RAG s'en sort → pas de niche,
stable→poids + volatil→RAG suffit (= déjà LLML). Diagnostic : couverture de la chaîne par le RAG.
Live : tail -f logs/benchmark_residual_rag.log
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

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_residual_rag.log")
K = 2            # nombre de hops (N0->N1->VALEUR = 2 liens)
N = 16
PAD = 1200       # tokens de pointeurs distracteurs + filler
RAG_K = 4        # top-k généreux pour le RAG
SUFFIX = "\nDonne UNIQUEMENT le mot de la valeur finale atteinte :"
_T0 = time.time()

VALUES = ("Azurite Basalte Cinabre Dolomie Emeraude Feldspath Grenat Hematite "
          "Iolite Jade Kunzite Lazulite Malachite Nacre Opale Pyrite").split()
_FILL = ("Orion Cygnus Perseus Auriga Dorado Tucana Grus Indus Norma Pictor Volans Mensa "
         "Reticulum Caelum Fornax Sculptor Antlia Pyxis Crater Hydra Corvus Lupus Ara Vela").split()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def build_items():
    items = []
    for i in range(N):
        nodes = [f"N{i}H{x}" for x in range(K)]
        value = VALUES[i % len(VALUES)]
        chain = nodes + [value]
        links = [f"La clé {a} mène à {b}." for a, b in zip(chain, chain[1:])]
        q = (f"En partant de la clé {nodes[0]} et en suivant les liens « mène à » de proche "
             f"en proche, quelle est la valeur finale atteinte ?")
        items.append({"start": nodes[0], "links": links, "q": q, "value": value})
    return items


def filler_sentences(start_j, n):
    out = []
    for j in range(start_j, start_j + n):
        out.append(f"L'entité {_FILL[j % len(_FILL)]}{j} a le statut {'actif' if j % 2 else 'archivé'} "
                   f"et la priorité {j % 7}.")
    return out


def distractor_pointers(start_j, n):
    # pointeurs LEURRES, espace de noms disjoint (préfixe D), ne se relient à aucune chaîne
    return [f"La clé D{start_j + j}A mène à D{start_j + j}B." for j in range(n)]


def make_context(item, salt):
    """contexte ~PAD tokens : liens de la chaîne ÉPARPILLÉS parmi pointeurs leurres + filler."""
    noise = distractor_pointers(salt * 50, 40) + filler_sentences(salt * 50 + 200, 30)
    # remplit jusqu'à ~PAD
    j = salt * 50 + 400
    while count_tokens(" ".join(noise)) < PAD:
        noise += distractor_pointers(j, 10); j += 10
    # insère les K liens de la chaîne à des positions espacées et déterministes
    chunks = list(noise)
    step = max(1, len(chunks) // (len(item["links"]) + 1))
    for h, link in enumerate(item["links"]):
        pos = min(len(chunks), (h + 1) * step + (salt % 3))
        chunks.insert(pos, link)
    return chunks  # liste de phrases (= unités RAG)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== Résidu RAG-proof — chaîne {K}-hop (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg); llm.set_adapter(None)

    from llmlingua import PromptCompressor
    log("init LLMLingua-2…")
    pc = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                          use_llmlingua2=True, device_map="cpu")
    log("init OK")

    items = build_items()
    rag = RAG(os.path.join(_PROJ, "logs", "rag_residual.txt"))
    full_ok = rag_ok = ll_ok = 0
    cover_sum = 0
    for idx, it in enumerate(items):
        chunks = make_context(it, idx)
        ctx_full = "Notes du projet :\n" + " ".join(chunks)
        tail = f"\n\nQuestion : {it['q']}{SUFFIX}"

        # FULL
        outF = llm.generate(ctx_full + tail, None)
        full_ok += d2l.answer_recalled(outF, it["value"])

        # RAG : chaque phrase = un chunk ; top-k sur la question
        rag.clear()
        for c in chunks:
            rag.add_document(c)
        top = rag.topk(it["q"], RAG_K)
        cover = sum(1 for lk in it["links"] if any(lk in t or t in lk for t in top))
        cover_sum += cover
        outR = llm.generate("Extraits pertinents :\n" + "\n".join(top) + tail, None)
        rag_ok += d2l.answer_recalled(outR, it["value"])

        # LLMLingua-2 sur le contexte complet
        comp = pc.compress_prompt(" ".join(chunks), question=it["q"], target_token=400,
                                  force_tokens=['\n', '?', '.', ':'])["compressed_prompt"]
        outL = llm.generate("Notes compressées :\n" + comp + tail, None)
        ll_ok += d2l.answer_recalled(outL, it["value"])

        if (idx + 1) % 4 == 0:
            log(f"   …{idx + 1}/{N} (FULL={full_ok} RAG={rag_ok} LL={ll_ok} | couv.chaîne RAG={cover_sum}/{(idx+1)*K})")

    ctxF = count_tokens("Notes du projet :\n" + " ".join(make_context(items[0], 0)))
    log("")
    log(f"=== RÉSULTAT (chaîne {K}-hop, contexte FULL ~{ctxF}tok, RAG top-{RAG_K}) ===")
    log(f"{'bras':28s} | exactitude")
    log(f"{'FULL (tout en contexte)':28s} | {full_ok/N*100:4.0f}%  ({full_ok}/{N})")
    log(f"{'RAG (BM25 top-k)':28s} | {rag_ok/N*100:4.0f}%  ({rag_ok}/{N})")
    log(f"{'LLMLingua-2 (compaction AR)':28s} | {ll_ok/N*100:4.0f}%  ({ll_ok}/{N})")
    log(f"couverture moyenne de la chaîne par le RAG : {cover_sum}/{N*K} liens "
        f"({cover_sum/(N*K)*100:.0f}%)")
    log("")
    if full_ok / N >= 0.7 and rag_ok / N <= full_ok / N - 0.25 and ll_ok / N <= full_ok / N - 0.25:
        log("🟩 NICHE RAG-PROOF CONFIRMÉE : FULL réussit, RAG ET LLMLingua-2 échouent sur le "
            "résidu structuré → un compresseur préservant la structure (diffusion) a un espace réel.")
    elif full_ok / N < 0.7:
        log("⚠️ CONFONDU : même en contexte complet le modèle ne suit pas la chaîne (plafond trop bas) "
            "→ réduire K ou prendre un modèle plus fort.")
    else:
        log("🟥 PAS DE NICHE : le RAG (ou LLMLingua-2) tient le résidu → stable→poids + volatil→RAG suffit.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
