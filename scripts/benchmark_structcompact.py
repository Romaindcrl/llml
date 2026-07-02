"""[STATUT : ⚠️ RÉSULTAT SYNTHÉTIQUE TROMPEUR (STRUCT 100% vs résumé/LLMLingua 0%) = ARTEFACT de
 design (la réponse se réduisait à suivre des flèches explicites). Réfuté ensuite sur données
 réelles (HotpotQA, VIS). Conservé pour mémoire.]

Prototype : compaction STRUCTURE-PRÉSERVANTE sur du contexte agentique RÉALISTE (non-regexable).

Le résidu = des « sorties d'outils » prose-lourdes où une chaîne de dépendances multi-hop est
exprimée en LANGAGE NATUREL VARIÉ (« délègue à », « s'appuie sur », « lit sa config depuis »…) →
un filtre regex ne peut PAS l'extraire ; il faut que le compacteur COMPRENNE. Question multi-hop.

4 méthodes :
  FULL      = tout en contexte (plafond) ;
  SUMMARY   = résumé générique (ce que fait la compaction type Claude Code) ;
  LLMLingua = token-pruning (budget = celui de STRUCT) ;
  STRUCT    = m0.structcompact (extrait le graphe de dépendances + faits) = LE PROTOTYPE.
Si STRUCT ≈ FULL et >> SUMMARY/LLMLingua à budget réduit → le prototype remplit la niche.
Live : tail -f logs/benchmark_structcompact.log
"""

from __future__ import annotations

import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0 import d2l, structcompact  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_structcompact.log")
N = 10
K = 2                 # hops (2-hop : plafond FULL atteignable par le 7B)
PAD = 1500
_T0 = time.time()

# espaces de noms DISJOINTS : chaîne vs distracteurs (sinon le modèle les confond)
CHAIN_BASES = "authgw crypto vault cfgstore edge tokenmint policy gateway ledger scheduler".split()
DISTR_BASES = ("metrics logpipe tracer healthz probe sysmon collectd statsd beacon heartbeat relay "
               "proxycache dnscache cdnedge blobstore coldstore").split()
# relations VARIÉES (toutes = une arête de dépendance, NON-regexables en un motif)
REL = ["délègue la validation des jetons à", "s'appuie sur", "lit sa configuration depuis",
       "est protégé par", "transmet ses requêtes à", "récupère ses secrets via",
       "dépend du cache fourni par"]


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def _node(i, x):
    return f"{CHAIN_BASES[(i * 3 + x) % len(CHAIN_BASES)]}{i}"


def build_items():
    items = []
    for i in range(N):
        nodes = [_node(i, x) for x in range(K + 1)]
        links = [f"{nodes[h]} {REL[(i + h) % len(REL)]} {nodes[h + 1]}." for h in range(K)]
        q = (f"En partant de {nodes[0]} et en suivant les dépendances de proche en proche "
             f"(chaque composant dépend du suivant), quel est le DERNIER composant de la chaîne ?")
        items.append({"start": nodes[0], "links": links, "q": q, "value": nodes[-1]})
    return items


def _prose(j):
    c = f"{DISTR_BASES[j % len(DISTR_BASES)]}{j}"
    return (f"[t+{j}s] {c} a démarré en 0.{j % 9}s, statut nominal ; {1000 + j} requêtes traitées, "
            f"latence p95 {30 + j % 40}ms, aucune alerte sur la fenêtre courante.")


def _distractor_dep(j):
    a = f"{DISTR_BASES[(j * 3) % len(DISTR_BASES)]}{j}"
    b = f"{DISTR_BASES[(j * 3 + 1) % len(DISTR_BASES)]}{j}"
    return f"{a} {REL[j % len(REL)]} {b}."


def make_context(item, salt):
    chunks, j = [], salt * 50 + 1
    while count_tokens(" ".join(chunks)) < PAD:          # prose de fond
        chunks.append(_prose(j)); j += 1
    for d in range(5):                                    # dépendances LEURRES éparpillées
        chunks.insert(min(len(chunks), (d + 1) * len(chunks) // 10), _distractor_dep(salt * 7 + d))
    step = max(1, len(chunks) // (len(item["links"]) + 1))
    for h, link in enumerate(item["links"]):             # la vraie chaîne, éparpillée
        chunks.insert(min(len(chunks), (h + 1) * step + (salt % 3)), link)
    return " ".join(chunks)


def _clear_cache():
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== Prototype compaction structure-préservante ({K}-hop, modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg); llm.set_adapter(None)

    from llmlingua import PromptCompressor
    log("init LLMLingua-2…")
    pc = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                          use_llmlingua2=True, device_map="cpu")
    log("init OK")

    items = build_items()
    R = {k: 0 for k in ("FULL", "SUMMARY", "LLMLingua", "STRUCT")}
    tok = {k: 0 for k in ("FULL", "SUMMARY", "LLMLingua", "STRUCT")}
    for idx, it in enumerate(items):
        ctx = make_context(it, idx)
        tail = f"\n\nQuestion : {it['q']}\nRéponds par le SEUL nom du composant :"

        summary = llm.generate("Résume ces notes en 4-5 phrases en gardant l'essentiel :\n\n" + ctx, None)
        art = structcompact.compact_structured(ctx, llm.generate)
        budget = max(80, count_tokens(art))
        comp = pc.compress_prompt(ctx, question=it["q"], target_token=budget,
                                  force_tokens=['\n', '?', '.', ':', '>'])["compressed_prompt"]

        outs = {
            "FULL": llm.generate("Notes :\n" + ctx + tail, None),
            "SUMMARY": llm.generate("Résumé du contexte :\n" + summary + tail, None),
            "LLMLingua": llm.generate("Notes compressées :\n" + comp + tail, None),
            "STRUCT": llm.generate("Contexte compact (graphe) :\n" + art + tail, None),
        }
        for k, o in outs.items():
            R[k] += d2l.answer_recalled(o, it["value"])
        tok["FULL"] += count_tokens(ctx); tok["SUMMARY"] += count_tokens(summary)
        tok["LLMLingua"] += count_tokens(comp); tok["STRUCT"] += budget
        _clear_cache()
        log(f"   …{idx + 1}/{N} (FULL={R['FULL']} SUMMARY={R['SUMMARY']} LLMLingua={R['LLMLingua']} STRUCT={R['STRUCT']})")

    n = len(items)
    log("")
    log(f"=== RÉSULTAT — contexte agentique réaliste, structure NON-regexable ({K}-hop) ===")
    log(f"{'méthode':30s} | exactitude | ctx tokens/item")
    for k in ("FULL", "SUMMARY", "LLMLingua", "STRUCT"):
        log(f"{k:30s} | {R[k]/n*100:5.0f}%    | {tok[k]//n}")
    comp_ratio = tok['FULL'] / max(1, tok['STRUCT'])
    log("")
    if R["STRUCT"] / n >= 0.7 and R["STRUCT"] / n - max(R["SUMMARY"], R["LLMLingua"]) / n >= 0.25:
        log(f"🟩 PROTOTYPE VALIDÉ : la compaction structure-préservante tient {R['STRUCT']/n*100:.0f}% "
            f"à ~{comp_ratio:.0f}× compression, là où résumé={R['SUMMARY']/n*100:.0f}% et "
            f"token-pruning={R['LLMLingua']/n*100:.0f}%. Sur structure NON-regexable → il faut bien "
            "COMPRENDRE (LLM/dLLM), pas filtrer. En agentique : compacté 1×, relu N tours.")
    else:
        log(f"🟧 MITIGÉ : STRUCT={R['STRUCT']/n*100:.0f}% vs SUMMARY={R['SUMMARY']/n*100:.0f}% "
            f"vs LLMLingua={R['LLMLingua']/n*100:.0f}% — à analyser.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
