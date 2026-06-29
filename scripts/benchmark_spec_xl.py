"""Benchmark CAHIER DES CHARGES XL — robustesse + régime "trop gros pour le contexte".

Combine les deux directions :
  - ROBUSTESSE : 50 entités d'entraînement diverses (vs 16) -> tue la dégénérescence.
  - TROP GROS : spec ~100 entités (≈10k+ tokens) -> la compaction ne peut PAS tout résumer
    utilement (perd les faits par entité), là où l'avantage 0-contexte des poids compte.

Faits IDIOSYNCRATIQUES par entité (verbe varié : scan_/probe_/locate_…) -> impossibles à
DEVINER -> le modèle est FORCÉ de lire la mémoire externe pour les faits (corrige le biais
"le LoRA applique un motif mémorisé et ignore le contexte").

Entraînement MIXTE (sans-ctx = grave les conventions ; avec-ctx = apprend à lire les faits)
+ mask-prompt. 5 conditions, entités HELD-OUT jamais vues.
Logs par tâche (cuttable). Live : tail -f logs/benchmark_spec_xl.log
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

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_spec_xl.log")
L = os.path.join(_PROJ, "models", "lora")
_T0 = time.time()

VERBS = ["scan", "locate", "probe", "fetch", "load", "read", "pull", "resolve",
         "lookup", "select", "gather", "retrieve"]
CODES = ["LOCKED", "MISSING", "INVALID", "DENIED", "EXPIRED", "FROZEN", "FULL",
         "STALE", "BANNED", "CLOSED"]
HELD = ["region", "shipment", "warehouse", "inventory", "supplier"]
POOL = (
    "account user order invoice product payment ledger ticket session vendor contract asset "
    "audit policy tenant device webhook quota subscription coupon refund dispute payout balance "
    "transfer merchant terminal receipt catalog bundle discount tax address carrier manifest "
    "customs pallet dock route driver vehicle fuel permit incident requisition batch lot serial "
    "warranty claim voucher loyalty reward tier campaign segment lead opportunity pipeline "
    "forecast budget expense reimbursement timesheet shift roster schedule booking reservation "
    "venue event sponsor speaker agenda track room badge survey rating review thread message "
    "notification alert reminder milestone sprint backlog epic release deploy artifact registry "
    "cluster node service secret config volume snapshot backup"
).split()
TRAIN_NOUNS = POOL[:50]
FILLER_NOUNS = POOL[50:]
ALL_ENT = {}
for _i, _e in enumerate(TRAIN_NOUNS + HELD + FILLER_NOUNS):
    ALL_ENT[_e] = (f"{VERBS[_i % len(VERBS)]}_{_e}", f"E_{_e.upper()}_{CODES[_i % len(CODES)]}")

ACTIONS = {"get": "récupère", "update": "met à jour", "delete": "supprime", "archive": "archive"}
CONV = ["validate_input", "log.", "repo.", "result", "nexuserror", '"""', "serialize_envelope"]

CONVENTIONS_TEXT = (
    "Standard de code Nexus — regles obligatoires pour tout handler du service.\n"
    "Tout point d'entree est nomme en snake_case sous la forme verbe_objet.\n"
    "Tout point d'entree debute par une docstring triple-guillemets decrivant l'effet.\n"
    "La premiere instruction valide l'entree via validate_input(payload).\n"
    "Tout traitement journalise son demarrage via log.info(...).\n"
    "Tout acces aux donnees passe par la couche repo (repo.methode), jamais d'acces direct.\n"
    "Les anomalies sont signalees via raise NexusError(\"CODE\"), jamais d'exception nue.\n"
    "En cas de succes, renvoyer Result.ok(serialize_envelope(record)).\n"
    "Ne jamais renvoyer un dict brut ni un None implicite.\n"
)


def conforming_code(action, entity, rm, ec):
    return (
        f'def {action}_{entity}(payload):\n'
        f'    """{action.capitalize()} the {entity} identified by payload. Returns a Result."""\n'
        f'    validate_input(payload)\n'
        f'    log.info(f"{action}_{entity}: start")\n'
        f'    record = repo.{rm}(payload["id"])\n'
        f'    if record is None:\n'
        f'        raise NexusError("{ec}")\n'
        f'    return Result.ok(serialize_envelope(record))'
    )


def entity_api(ent, rm, ec):
    return (f"Module {ent}. Pour charger un {ent}, appeler repo.{rm}(id) qui retourne "
            f"l'enregistrement {ent} ou None. Si le {ent} est absent, lever NexusError(\"{ec}\"). "
            f"Le {ent} possede les champs id, statut, date_creation et metadonnees. "
            f"Les operations sur {ent} sont soumises aux quotas du tenant courant.")


def build_spec():
    parts = [CONVENTIONS_TEXT]
    for ent, (rm, ec) in ALL_ENT.items():
        parts.append(entity_api(ent, rm, ec))
    return "\n".join(parts)


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def frac(text, checklist):
    t = (text or "").lower()
    return sum(1 for tok in checklist if tok.lower() in t) / len(checklist)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== BENCHMARK SPEC XL (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg)

    spec = build_spec()
    rag = RAG(os.path.join(_PROJ, "logs", "rag_spec_xl.txt")); rag.clear()
    rag.add_document(spec)
    log(f"[1/3] Spec = {count_tokens(spec)} tokens, {rag.count()} chunks, "
        f"{len(ALL_ENT)} entités ({len(TRAIN_NOUNS)} train / {len(HELD)} held-out / "
        f"{len(FILLER_NOUNS)} filler) — trop gros pour le contexte")

    # entraînement PUR AVEC-CONTEXTE sur 50 entités (held-out exclues), faits idiosyncratiques.
    # (le mélange sans-ctx+avec-ctx déstabilise l'entraînement ; on garde la recette pure qui
    #  a donné region 100/100, scalée à 50 entités pour la robustesse.) Conventions apprises
    #  via les CIBLES de code ; faits variables lus depuis le contexte. mask-prompt.
    train_pairs = []
    for ent in TRAIN_NOUNS:
        rm, ec = ALL_ENT[ent]
        eq = f"Implémente la fonction get_{ent} qui récupère un {ent} par identifiant."
        ectx = "\n".join(rag.topk(eq, 4))
        for act, fr in ACTIONS.items():
            instr = f"Implémente la fonction {act}_{ent} qui {fr} un {ent} à partir de payload."
            code = conforming_code(act, ent, rm, ec)
            train_pairs.append((f"Spec (extraits pertinents) :\n{ectx}\n\n{instr}", code))
    log(f"[2/3] Entraînement POIDS PUR-CONTEXTE : {len(train_pairs)} ex / {len(TRAIN_NOUNS)} "
        f"entités (style depuis poids + LIRE les faits du contexte), mask-prompt")
    data, adapter = f"{_PROJ}/logs/spec_xl_data", f"{L}/spec_xl"
    n = d2l.build_chat_dataset(train_pairs, data, repeat=2, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=3)
    iters = min(800, max(600, 3 * len(train_pairs)))
    res = d2l.train_lora(cfg.mlx_model_path, data, adapter, iters=iters, num_layers=cfg.d2l_num_layers,
                         learning_rate=cfg.d2l_learning_rate, rank=16, max_seq_length=1024,
                         mask_prompt=True, python_exe=sys.executable, log_file=LOG_PATH)
    log(f"  POIDS : ok={res['ok']} val_loss={res['val_loss']} ({n} lignes, {iters} iters)")

    tasks = [(ent, f"Implémente la fonction get_{ent} qui récupère un {ent} par identifiant.",
              CONV, list(ALL_ENT[ent])) for ent in HELD]
    nt = len(tasks)

    log("  résumé de compaction (la spec entière → un résumé borné)…")
    summary = llm.generate("Résume ce standard en gardant conventions ET noms d'API exacts :\n"
                           + spec, None)
    sum_tok = count_tokens(summary)

    def gen(prompt):
        return llm.generate(prompt + "\nÉcris uniquement le code Python de la fonction.", None)

    SUT = ("base", "RAG", "compaction", "poids", "complémentaire")
    C = {s: [] for s in SUT}; F = {s: [] for s in SUT}; CTX = {s: [] for s in SUT}

    log("[3/3] Génération + scoring (5 conditions)")
    for i, (ent, instr, conv_ck, fact_ck) in enumerate(tasks, 1):
        ctx = "\n".join(rag.topk(instr, 4))
        rag_prompt = f"Spec (extraits pertinents) :\n{ctx}\n\n{instr}"
        sum_prompt = f"Spec (résumé) :\n{summary}\n\n{instr}"
        llm.set_adapter(None)
        ob, orr, oc = gen(instr), gen(rag_prompt), gen(sum_prompt)
        llm.set_adapter(adapter)
        op, ox = gen(instr), gen(rag_prompt)
        for s, out, ct in (("base", ob, 0), ("RAG", orr, count_tokens(ctx)),
                           ("compaction", oc, sum_tok), ("poids", op, 0),
                           ("complémentaire", ox, count_tokens(ctx))):
            C[s].append(frac(out, conv_ck)); F[s].append(frac(out, fact_ck)); CTX[s].append(ct)
        log(f"[tâche {i}/{nt} {ent:9s}] conv: rag={C['RAG'][-1]*100:.0f} comp={C['compaction'][-1]*100:.0f} "
            f"poids={C['poids'][-1]*100:.0f} compl={C['complémentaire'][-1]*100:.0f} | "
            f"faits: rag={F['RAG'][-1]*100:.0f} comp={F['compaction'][-1]*100:.0f} "
            f"compl={F['complémentaire'][-1]*100:.0f}")

    def avg(d, s):
        return sum(d[s]) / len(d[s]) * 100 if d[s] else 0.0
    log("")
    log(f"=== RÉSULTATS XL — {nt} entités held-out · spec {count_tokens(spec)} tok ===")
    log(f"{'méthode':16s} | conventions | faits spéc. | global | ctx tok/req")
    for s in SUT:
        g = (avg(C, s) + avg(F, s)) / 2
        log(f"{s:16s} | {avg(C, s):9.0f}%  | {avg(F, s):9.0f}%  | {g:5.0f}% | "
            f"{sum(CTX[s])/len(CTX[s]):5.0f}")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
