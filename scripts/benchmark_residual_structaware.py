"""La niche est-elle REMPLISSABLE ? Compaction structure-préservante vs token-pruning (budget égal).

Recadrage (objection utilisateur, juste) : en agentique le résidu structuré EST le contexte, re-lu
à chaque tour → régime RÉUTILISATION = défaut. Le concurrent n'est pas le RAG itératif mais la
COMPACTION (que l'agent fait au débordement), qui détruit la structure (LLMLingua-2 → 6%).

Test : résidu PROSE-LOURD (≈ sorties d'outils agentiques) avec une chaîne multi-hop noyée. À budget
de tokens compressé ÉGAL, on compare :
  FULL      = tout en contexte (plafond, gros) ;
  LLMLingua = token-pruning AR (ce que fait une compaction générique) ;
  STRUCT    = compaction STRUCTURE-PRÉSERVANTE (garde les relations « mène à », jette la prose) —
              upper-bound déterministe de ce qu'un compresseur answer-preserving (diffusion) vise.
Si STRUCT ≈ FULL >> LLMLingua au MÊME petit budget → la niche est réelle ET remplissable, et en
régime réutilisation 1 compression s'amortit sur N tours. Live : tail -f logs/...structaware.log
"""

from __future__ import annotations

import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import scripts.benchmark_residual_rag as RR  # mêmes chaînes/valeurs  # noqa: E402
from m0 import d2l  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_residual_structaware.log")
PAD = 3500
SUFFIX = "\nDonne UNIQUEMENT le mot de la valeur finale atteinte :"
_T0 = time.time()

_MODS = ("Atlas Borealis Cobalt Delta Echo Foxtrot Gamma Helix Indigo Juno Kilo Lima Meridian "
         "Nova Onyx Quartz Raven Sierra Tango Umbra Vortex Willow Xenon Yuki Zephyr").split()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def prose(j):
    n = _MODS[j % len(_MODS)] + str(j)
    return (f"Le module {n} a été inspecté durant la passe de revue ; aucune anomalie n'a été "
            f"relevée et le statut reste nominal pour la fenêtre de maintenance courante.")


def make_context_prose(item, salt):
    """contexte prose-lourd (~3:1 prose:pointeur) avec les liens de la chaîne éparpillés."""
    chunks, j = [], salt * 60 + 1
    while count_tokens(" ".join(chunks)) < PAD:
        chunks.append(prose(j)); chunks.append(prose(j + 1)); chunks.append(prose(j + 2))
        chunks.append(f"La clé D{j}A mène à D{j}B.")   # pointeur leurre
        j += 3
    step = max(1, len(chunks) // (len(item["links"]) + 1))
    for h, link in enumerate(item["links"]):
        chunks.insert(min(len(chunks), (h + 1) * step + (salt % 3)), link)
    return chunks


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== Compaction structure-préservante vs token-pruning ({RR.K}-hop, modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg); llm.set_adapter(None)

    from llmlingua import PromptCompressor
    log("init LLMLingua-2…")
    pc = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                          use_llmlingua2=True, device_map="cpu")
    log("init OK")

    items = RR.build_items()
    f_ok = l_ok = s_ok = 0
    full_tok = struct_tok = ll_tok = 0
    for idx, it in enumerate(items):
        chunks = make_context_prose(it, idx)
        full_ctx = " ".join(chunks)
        # STRUCT : garde les phrases relationnelles, jette la prose
        skeleton = " ".join(c for c in chunks if "mène à" in c)
        budget = count_tokens(skeleton)                      # budget cible commun
        # LLMLingua-2 au MÊME budget
        comp = pc.compress_prompt(full_ctx, question=it["q"], target_token=budget,
                                  force_tokens=['\n', '?', '.', ':'])["compressed_prompt"]
        tail = f"\n\nQuestion : {it['q']}{SUFFIX}"
        f_ok += d2l.answer_recalled(llm.generate("Notes :\n" + full_ctx + tail, None), it["value"])
        s_ok += d2l.answer_recalled(llm.generate("Notes (relations) :\n" + skeleton + tail, None), it["value"])
        l_ok += d2l.answer_recalled(llm.generate("Notes compressées :\n" + comp + tail, None), it["value"])
        full_tok += count_tokens(full_ctx); struct_tok += budget; ll_tok += count_tokens(comp)
        if (idx + 1) % 4 == 0:
            log(f"   …{idx + 1}/{RR.N} (FULL={f_ok} STRUCT={s_ok} LLMLingua={l_ok})")

    n = RR.N
    log("")
    log(f"=== RÉSULTAT — budget compressé ÉGAL (~{struct_tok//n} tok), résidu prose-lourd ~{full_tok//n} tok ===")
    log(f"{'bras':38s} | exactitude | ctx tokens/item")
    log(f"{'FULL (tout en contexte)':38s} | {f_ok/n*100:5.0f}%    | {full_tok//n}")
    log(f"{'LLMLingua-2 (token-pruning)':38s} | {l_ok/n*100:5.0f}%    | {ll_tok//n}")
    log(f"{'STRUCT (structure-préservante)':38s} | {s_ok/n*100:5.0f}%    | {struct_tok//n}")
    log("")
    if s_ok / n >= 0.8 and s_ok / n - l_ok / n >= 0.3:
        log(f"🟩 NICHE REMPLISSABLE : à budget égal (~{struct_tok//n}tok, ~{full_tok//struct_tok:.0f}× compression), "
            f"préserver la structure tient {s_ok/n*100:.0f}% là où le token-pruning fait {l_ok/n*100:.0f}%. "
            "En régime réutilisation (agentique) 1 telle compression s'amortit sur N tours → la cible du dLLM.")
    else:
        log(f"🟥 NON CONCLUANT : STRUCT={s_ok/n*100:.0f}% / LLMLingua={l_ok/n*100:.0f}% — l'écart attendu n'apparaît pas.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
