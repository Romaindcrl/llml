"""Benchmark PROJET MULTI-FICHIERS — notre système COMPLET vs compaction(s), en 32k.

Scénario réaliste : le cahier des charges tient dans 32k, MAIS le code déjà écrit s'accumule
dans le contexte et sature la fenêtre. Variable clé : L = tokens de code projet déjà présents.
Plus L monte, moins il reste de place -> un compacteur doit rogner le cahier des charges
(perd les faits, puis les conventions). NOUS : le cahier des charges est dans les POIDS
(style-LoRA) -> 0 contexte -> la fenêtre reste libre pour le code, adhérence stable.

SUT (toutes sous budget 32k, on génère un module pour une entité held-out) :
  - compaction-seule : résumé(cahier) tronqué au budget restant + code existant.
  - RAG + compaction : extraits RAG (faits de l'entité) + résumé tronqué + code existant.
  - NOTRE SYSTÈME    : conventions dans les POIDS (style-LoRA, 0 ctx) + RAG (faits) + vérif.
Score par module : conventions (pervasif) + faits spécifiques. Tracé contre L.
Logs par point (cuttable). Live : tail -f logs/benchmark_project.log
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
from scripts.benchmark_spec_final import lookup_facts, substitute  # noqa: E402
from scripts.benchmark_spec_xl import (  # noqa: E402
    ACTIONS, CODES, CONV, CONVENTIONS_TEXT, VERBS, conforming_code, entity_api, frac,
)

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_project.log")
L_LEVELS = [22000, 31000]          # tokens de code projet en contexte (31k -> DÉBORDEMENT)
HELD = ["region", "shipment"]      # modules à générer (entités jamais entraînées)
WINDOW = 32000                     # fenêtre de contexte DURE
NS = "nx_7f3a"                     # signal inter-fichiers (défini SEULEMENT dans le module fondation)
_T0 = time.time()

# ~160 entités : cahier des charges ~13k tokens (tient dans 32k), faits idiosyncratiques.
_PREF = "core edge meta sub super micro macro multi uni intra inter trans".split()
_NOUN = ("order user invoice account product payment ledger ticket session vendor contract "
         "asset audit policy tenant device webhook quota coupon refund dispute payout balance "
         "transfer merchant terminal receipt catalog bundle discount tax address carrier").split()
_COMBO = [f"{p}{n}" for p in _PREF for n in _NOUN]
TRAIN_ENTS = _COMBO[:50]
FILLER_ENTS = _COMBO[50:225]
ALL = TRAIN_ENTS + FILLER_ENTS + HELD          # held à la FIN (hors du début résumé)
FACTS = {e: (f"{VERBS[i % len(VERBS)]}_{e}", f"E_{e.upper()}_{CODES[i % len(CODES)]}")
         for i, e in enumerate(ALL)}


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def build_spec():
    parts = [CONVENTIONS_TEXT]
    for e in ALL:
        rm, ec = FACTS[e]
        parts.append(entity_api(e, rm, ec))
    return "\n".join(parts)


_FOUNDATION = (
    "# === MODULE FONDATION (foundation.py) ===\n"
    "# CONVENTION PROJET : toute fonction get_* doit inclure, juste après la docstring,\n"
    f"# la ligne d'audit obligatoire :  audit_tag = \"{NS}\"\n"
    "# (cette valeur n'existe QUE dans ce module fondation, en début de projet).\n")


def filler_code(target_tokens):
    """Code projet déjà écrit. Commence par le MODULE FONDATION (qui définit audit_tag) :
    c'est le premier à tomber quand on tronque par l'avant."""
    blocks, tok = [_FOUNDATION], count_tokens(_FOUNDATION)
    if target_tokens <= 0:
        return _FOUNDATION
    for e in FILLER_ENTS:
        rm, ec = FACTS[e]
        for act in ACTIONS:
            blocks.append(conforming_code(act, e, rm, ec))
            tok += count_tokens(blocks[-1])
            if tok >= target_tokens:
                return "\n\n".join(blocks)
    return "\n\n".join(blocks)


def fit_code(full_code, avail_tok):
    """Tronque le code à la fenêtre : garde le PLUS RÉCENT (fin), lâche le plus ancien (début,
    = le module fondation en premier)."""
    if avail_tok <= 0:
        return ""
    chars = avail_tok * 4
    return full_code[-chars:] if len(full_code) > chars else full_code


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== BENCHMARK PROJET MULTI-FICHIERS (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg)

    spec = build_spec()
    rag = RAG(os.path.join(_PROJ, "logs", "rag_project.txt")); rag.clear()
    rag.add_document(spec)
    log(f"[1/3] Cahier des charges = {count_tokens(spec)} tokens ({len(ALL)} entités) — "
        f"tient dans {WINDOW}, mais pas avec le code")

    # style-LoRA : conventions dans les POIDS (no-contexte, entités d'entraînement)
    pairs = []
    for e in TRAIN_ENTS:
        rm, ec = FACTS[e]
        for act, fr in ACTIONS.items():
            pairs.append((f"Implémente la fonction {act}_{e} qui {fr} un {e} à partir de payload.",
                          conforming_code(act, e, rm, ec)))
    log(f"[2/3] Entraînement style-LoRA : {len(pairs)} ex (conventions -> poids), rang 16")
    data, adapter = f"{_PROJ}/logs/project_data", f"{_PROJ}/models/lora/project"
    d2l.build_chat_dataset(pairs, data, repeat=2, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=3)
    res = d2l.train_lora(cfg.mlx_model_path, data, adapter, iters=600, num_layers=cfg.d2l_num_layers,
                         learning_rate=5e-5, rank=16, python_exe=sys.executable, log_file=LOG_PATH)
    log(f"  style-LoRA : ok={res['ok']} val_loss={res['val_loss']}")

    # résumé de compaction, CONVENTIONS d'abord puis faits (la troncature garde les conventions)
    summary = llm.generate(
        "Résume ce cahier des charges. D'ABORD toutes les conventions de code, PUIS, pour chaque "
        "entité, son repo.<méthode> et son code d'erreur exacts.\n" + spec, None)
    log(f"  résumé de compaction = {count_tokens(summary)} tokens")

    def gen(prompt):
        return llm.generate(prompt + "\nÉcris uniquement le code Python du module.", None)

    SUT = ("compaction-seule", "RAG+compaction", "NOTRE-SYSTÈME")
    agg = {(L, s): {"c": [], "f": [], "x": [], "kept": []} for L in L_LEVELS for s in SUT}

    log("[3/3] Génération sous fenêtre DURE 32k — au débordement, les baselines doivent "
        "lâcher du code pour garder le cahier ; nous gardons tout le code")
    for L in L_LEVELS:
        code_full = filler_code(L)
        Lt = count_tokens(code_full)
        spec_fits = count_tokens(spec) <= (WINDOW - Lt - 600)
        comp = spec if spec_fits else summary
        log(f"-- L={L} (code {Lt} tok ; cahier {'COMPLET' if spec_fits else 'RÉSUMÉ'})")
        for ent in HELD:
            rm, ec = FACTS[ent]; ck = [rm, ec]
            task = (f"Implémente le module {ent}.py : la fonction get_{ent}(payload) qui récupère "
                    f"un {ent} par identifiant, en respectant le cahier des charges. N'oublie PAS "
                    f"la ligne d'audit obligatoire définie dans le module fondation du projet.")
            chunks = "\n".join(rag.topk(task, 6))
            comp_tok, chunk_tok = count_tokens(comp), count_tokens(chunks)
            # budgets de code restants (fenêtre dure) : le cahier mange la place chez les baselines
            avail = {"compaction-seule": WINDOW - comp_tok - 600,
                     "RAG+compaction": WINDOW - comp_tok - chunk_tok - 600,
                     "NOTRE-SYSTÈME": WINDOW - 600}          # nous : 0 cahier en contexte
            c_comp = fit_code(code_full, avail["compaction-seule"])
            c_rag = fit_code(code_full, avail["RAG+compaction"])
            c_nous = fit_code(code_full, avail["NOTRE-SYSTÈME"])

            llm.set_adapter(None)
            o1 = gen(f"Cahier (résumé) :\n{comp}\n\nCode du projet :\n{c_comp}\n\n{task}")
            o2 = gen(f"Cahier (extraits) :\n{chunks}\n\nCahier :\n{comp}\n\n"
                     f"Code du projet :\n{c_rag}\n\n{task}")
            llm.set_adapter(adapter)
            draft = gen(f"Code du projet :\n{c_nous}\n\n{task}")
            o3 = substitute(draft, *lookup_facts(ent, rag)[:2])

            for s, out, kept in (("compaction-seule", o1, c_comp), ("RAG+compaction", o2, c_rag),
                                 ("NOTRE-SYSTÈME", o3, c_nous)):
                agg[(L, s)]["c"].append(frac(out, CONV))
                agg[(L, s)]["f"].append(frac(out, ck))
                agg[(L, s)]["x"].append(1.0 if NS in out else 0.0)          # signal inter-fichiers
                agg[(L, s)]["kept"].append(1.0 if NS in kept else 0.0)      # fondation en contexte ?
            log(f"  [L={L} {ent}] conv/faits/audit — "
                f"comp={frac(o1,CONV)*100:.0f}/{frac(o1,ck)*100:.0f}/{int(NS in o1)} "
                f"rag={frac(o2,CONV)*100:.0f}/{frac(o2,ck)*100:.0f}/{int(NS in o2)} "
                f"nous={frac(o3,CONV)*100:.0f}/{frac(o3,ck)*100:.0f}/{int(NS in o3)}")

    def av(L, s, k):
        v = agg[(L, s)][k]; return sum(v) / len(v) * 100 if v else 0.0
    log("")
    log("=== RÉSULTATS — fenêtre dure 32k (conv% / faits% / audit inter-fichiers%) ===")
    log(f"{'charge L':>9s} | {'méthode':16s} | conv | faits | audit | (fondation gardée)")
    for L in L_LEVELS:
        for s in SUT:
            log(f"{L:>9d} | {s:16s} | {av(L,s,'c'):3.0f}% | {av(L,s,'f'):4.0f}% | "
                f"{av(L,s,'x'):4.0f}% | {av(L,s,'kept'):3.0f}%")
        log("          " + "-" * 52)
    log("=== FIN ===")


if __name__ == "__main__":
    main()
