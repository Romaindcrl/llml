"""Benchmark SPEC FINAL — les deux voies pour réaliser la complémentarité robuste.

(a) 2-ÉTAPES : style-LoRA (no-contexte, conventions robustes 0-ctx) GÉNÈRE le code, puis une
    PASSE DE VÉRIFICATION déterministe substitue les faits spécifiques (repo.<méthode>, code
    d'erreur) en les LOOKUPant dans la mémoire externe. Pas de fusion dans le forward pass.
(b) FUSION-HI : LoRA contexte-aware à plus de CAPACITÉ (rang 32, lr 5e-5) — voir si plus de
    capacité rend la fusion en un seul pass robuste (le rang 16 échouait à l'échelle).

Réf : base · RAG · compaction · style-seul. Spec XL (8745 tok, idiosyncratique, 5 held-out).
Logs par tâche (cuttable). Live : tail -f logs/benchmark_spec_final.log
"""

from __future__ import annotations

import os
import re
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0 import d2l  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.rag import RAG  # noqa: E402
from scripts.benchmark_spec_xl import (  # noqa: E402
    ALL_ENT, ACTIONS, CONV, HELD, TRAIN_NOUNS, build_spec, conforming_code, frac,
)

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_spec_final.log")
L = os.path.join(_PROJ, "models", "lora")
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def lookup_facts(entity, rag):
    """Mémoire externe : retrouve la méthode repo et le code d'erreur exacts de l'entité.
    Récupère assez de chunks pour couvrir toute la section API de l'entité (repo + NexusError
    sont dans des phrases/chunks distincts)."""
    chunk = " ".join(rag.topk(f"{entity} repo charger NexusError absent enregistrement", 6))
    m = re.search(rf"repo\.(\w*{re.escape(entity)}\w*)", chunk) or re.search(r"repo\.(\w+)", chunk)
    e = re.search(rf'NexusError\("(E_{re.escape(entity).upper()}[^"]*)"\)', chunk) \
        or re.search(r'NexusError\("([^"]+)"\)', chunk) \
        or re.search(rf"(E_{re.escape(entity).upper()}_[A-Z]+)", chunk)
    return (m.group(1) if m else None), (e.group(1) if e else None), count_tokens(chunk)


def substitute(code, method, err):
    """Passe de vérification : remplace les faits hallucinés par les faits vérifiés."""
    if method:
        code = re.sub(r"repo\.\w+", f"repo.{method}", code, count=1)
    if err:
        code = re.sub(r'NexusError\("[^"]*"\)', f'NexusError("{err}")', code)
    return code


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== BENCHMARK SPEC FINAL (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg)

    spec = build_spec()
    rag = RAG(os.path.join(_PROJ, "logs", "rag_spec_f.txt")); rag.clear()
    rag.add_document(spec)
    log(f"[1/4] Spec = {count_tokens(spec)} tokens, {len(ALL_ENT)} entités, {len(HELD)} held-out")

    # (a) style-LoRA : no-contexte, conventions robustes (rang 16)
    style_pairs = []
    for ent in TRAIN_NOUNS:
        rm, ec = ALL_ENT[ent]
        for act, fr in ACTIONS.items():
            instr = f"Implémente la fonction {act}_{ent} qui {fr} un {ent} à partir de payload."
            style_pairs.append((instr, conforming_code(act, ent, rm, ec)))
    log(f"[2/4] (a) style-LoRA : {len(style_pairs)} ex no-contexte, rang 16")
    d_s, a_style = f"{_PROJ}/logs/specf_style_data", f"{L}/specf_style"
    d2l.build_chat_dataset(style_pairs, d_s, repeat=2, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=3)
    rs = d2l.train_lora(cfg.mlx_model_path, d_s, a_style, iters=500, num_layers=cfg.d2l_num_layers,
                        learning_rate=1e-4, rank=16, python_exe=sys.executable, log_file=LOG_PATH)
    log(f"  style : ok={rs['ok']} val_loss={rs['val_loss']}")

    # (b) fusion-hi : contexte-aware, rang 32, lr 5e-5 (plus de capacité)
    fus_pairs = []
    for ent in TRAIN_NOUNS:
        rm, ec = ALL_ENT[ent]
        eq = f"Implémente la fonction get_{ent} qui récupère un {ent} par identifiant."
        ectx = "\n".join(rag.topk(eq, 4))
        for act, fr in ACTIONS.items():
            instr = f"Implémente la fonction {act}_{ent} qui {fr} un {ent} à partir de payload."
            fus_pairs.append((f"Spec (extraits pertinents) :\n{ectx}\n\n{instr}",
                              conforming_code(act, ent, rm, ec)))
    log(f"[3/4] (b) fusion-hi : {len(fus_pairs)} ex contexte-aware, rang 32, lr 5e-5")
    d_f, a_fus = f"{_PROJ}/logs/specf_fus_data", f"{L}/specf_fus"
    d2l.build_chat_dataset(fus_pairs, d_f, repeat=2, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=3)
    rf = d2l.train_lora(cfg.mlx_model_path, d_f, a_fus, iters=600, num_layers=cfg.d2l_num_layers,
                        learning_rate=5e-5, rank=32, max_seq_length=1024, mask_prompt=True,
                        python_exe=sys.executable, log_file=LOG_PATH)
    log(f"  fusion-hi : ok={rf['ok']} val_loss={rf['val_loss']}")

    tasks = [(ent, f"Implémente la fonction get_{ent} qui récupère un {ent} par identifiant.",
              CONV, list(ALL_ENT[ent])) for ent in HELD]
    nt = len(tasks)
    summary = llm.generate("Résume ce standard en gardant conventions ET noms d'API exacts :\n"
                           + spec, None)
    sum_tok = count_tokens(summary)

    def gen(prompt):
        return llm.generate(prompt + "\nÉcris uniquement le code Python de la fonction.", None)

    SUT = ("base", "RAG", "compaction", "style-seul", "2-étapes(a)", "fusion-hi(b)")
    C = {s: [] for s in SUT}; F = {s: [] for s in SUT}; CTX = {s: [] for s in SUT}

    log("[4/4] Évaluation (6 conditions)")
    for i, (ent, instr, conv_ck, fact_ck) in enumerate(tasks, 1):
        ctx = "\n".join(rag.topk(instr, 4))
        rag_prompt = f"Spec (extraits pertinents) :\n{ctx}\n\n{instr}"
        # base / RAG / compaction
        llm.set_adapter(None)
        ob = gen(instr)
        orr = gen(rag_prompt)
        oc = gen(f"Spec (résumé) :\n{summary}\n\n{instr}")
        # (a) style-seul -> puis 2-étapes (substitution déterministe)
        llm.set_adapter(a_style)
        os_ = gen(instr)
        meth, err, lk_tok = lookup_facts(ent, rag)
        o2 = substitute(os_, meth, err)
        # (b) fusion-hi
        llm.set_adapter(a_fus)
        ofz = gen(rag_prompt)
        for s, out, ct in (("base", ob, 0), ("RAG", orr, count_tokens(ctx)),
                           ("compaction", oc, sum_tok), ("style-seul", os_, 0),
                           ("2-étapes(a)", o2, lk_tok), ("fusion-hi(b)", ofz, count_tokens(ctx))):
            C[s].append(frac(out, conv_ck)); F[s].append(frac(out, fact_ck)); CTX[s].append(ct)
        log(f"[tâche {i}/{nt} {ent:9s}] conv: style={C['style-seul'][-1]*100:.0f} "
            f"2ét={C['2-étapes(a)'][-1]*100:.0f} fus={C['fusion-hi(b)'][-1]*100:.0f} | "
            f"faits: rag={F['RAG'][-1]*100:.0f} 2ét={F['2-étapes(a)'][-1]*100:.0f} "
            f"fus={F['fusion-hi(b)'][-1]*100:.0f}")

    def avg(d, s):
        return sum(d[s]) / len(d[s]) * 100 if d[s] else 0.0
    log("")
    log(f"=== RÉSULTATS FINAL — {nt} held-out · spec {count_tokens(spec)} tok ===")
    log(f"{'méthode':14s} | conventions | faits spéc. | global | ctx tok/req")
    for s in SUT:
        g = (avg(C, s) + avg(F, s)) / 2
        log(f"{s:14s} | {avg(C, s):9.0f}%  | {avg(F, s):9.0f}%  | {g:5.0f}% | "
            f"{sum(CTX[s])/len(CTX[s]):5.0f}")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
