"""Benchmark CAHIER DES CHARGES — teste la COMPLÉMENTARITÉ poids + mémoire externe.

Régime que les benchmarks précédents ratent : une spec TROP GROSSE pour le contexte, avec
  - des CONVENTIONS pervasives (s'appliquent à chaque génération) -> candidat POIDS (style),
  - des FAITS spécifiques par entité (à vérifier ponctuellement)   -> candidat MÉMOIRE EXTERNE.
Les conventions ne sont PAS lexicalement pertinentes à une requête de feature -> le RAG naïf
les RATE. Les poids les portent toujours, sans polluer le contexte.

Entités d'entraînement (poids voient leur style + leurs faits) vs entités HELD-OUT (jamais
vues : conventions doivent transférer ; faits spécifiques uniquement dans la spec/RAG).

5 conditions : base · RAG · compaction · poids · complémentaire(poids+RAG ciblé).
Score séparé : adhérence aux CONVENTIONS (pervasif) vs FAITS spécifiques (lookup).
Logs par tâche (cuttable). Live : tail -f logs/benchmark_spec.log
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

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_spec.log")
L = os.path.join(_PROJ, "models", "lora")
_T0 = time.time()

# entités VUES à l'entraînement (style + faits appris)
TRAIN_ENT = {
    "user": ("find_by_id", "E_USER_MISSING"),
    "order": ("fetch_order", "E_ORDER_INVALID"),
    "invoice": ("load_invoice", "E_INVOICE_LOCKED"),
    "account": ("read_account", "E_ACCOUNT_FROZEN"),
}
# entités HELD-OUT (jamais entraînées : conventions doivent transférer ; faits seulement spec)
HELD_ENT = {
    "region": ("scan_by_region", "E_REGION_LOCKED"),
    "shipment": ("locate_shipment", "E_SHIPMENT_LOST"),
    "warehouse": ("probe_warehouse", "E_WAREHOUSE_FULL"),
}
# entités de remplissage (grossissent la spec -> trop grosse pour le contexte de travail)
FILLER_ENT = {
    e: (f"query_{e}", f"E_{e.upper()}_FAIL") for e in
    ("product", "payment", "ledger", "ticket", "session", "vendor", "contract",
     "asset", "audit", "policy", "tenant", "device", "webhook", "quota",
     "subscription", "coupon", "refund", "dispute", "payout", "balance", "transfer",
     "merchant", "terminal", "receipt", "catalog", "bundle", "discount", "tax",
     "address", "carrier", "manifest", "customs", "pallet", "dock", "route",
     "driver", "vehicle", "fuel", "permit", "incident")
}
ACTIONS = {"get": "récupère", "update": "met à jour", "delete": "supprime", "archive": "archive"}

CONV = ["validate_input", "log.", "repo.", "result", "nexuserror", '"""', "serialize_envelope"]


def conforming_code(action, entity, repo_method, error_code):
    return (
        f'def {action}_{entity}(payload):\n'
        f'    """{action.capitalize()} the {entity} identified by payload. Returns a Result."""\n'
        f'    validate_input(payload)\n'
        f'    log.info(f"{action}_{entity}: start")\n'
        f'    record = repo.{repo_method}(payload["id"])\n'
        f'    if record is None:\n'
        f'        raise NexusError("{error_code}")\n'
        f'    return Result.ok(serialize_envelope(record))'
    )


# Décrit les règles dans SON propre vocabulaire (pas celui des requêtes de feature),
# comme un vrai guide de style -> non récupérable par BM25 sur une requête "get_<entité>".
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


def entity_api(ent, rm, ec):
    return (f"Module {ent}. Pour charger un {ent}, appeler repo.{rm}(id) qui retourne "
            f"l'enregistrement {ent} ou None. Si le {ent} est absent, lever NexusError(\"{ec}\"). "
            f"Le {ent} possede les champs id, statut, date_creation et metadonnees. "
            f"Les operations sur {ent} sont soumises aux quotas du tenant courant.")


def build_spec():
    parts = [CONVENTIONS_TEXT]
    for ent, (rm, ec) in {**TRAIN_ENT, **HELD_ENT, **FILLER_ENT}.items():
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
    log(f"=== BENCHMARK CAHIER DES CHARGES (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg)

    spec = build_spec()
    rag = RAG(os.path.join(_PROJ, "logs", "rag_spec.txt")); rag.clear()
    rag.add_document(spec)
    log(f"[1/3] Spec = {count_tokens(spec)} tokens, {rag.count()} chunks "
        f"(trop gros pour le contexte de travail)")

    # entraînement POIDS MIXTE sur 24 entités diverses (held-out exclues) :
    #  - exemples SANS contexte  -> grave les CONVENTIONS dans les poids (gratuit à l'inférence)
    #  - exemples AVEC contexte   -> apprend à LIRE les faits variables depuis la mémoire externe
    # mask-prompt : la loss ne porte que sur le code (le contexte n'infle pas la loss).
    train_ents = list({**TRAIN_ENT, **FILLER_ENT}.items())[:24]
    train_pairs = []
    for ent, (rm, ec) in train_ents:
        eq = f"Implémente la fonction get_{ent} qui récupère un {ent} par identifiant."
        ectx = "\n".join(rag.topk(eq, 4))
        for act, fr in ACTIONS.items():
            instr = f"Implémente la fonction {act}_{ent} qui {fr} un {ent} à partir de payload."
            code = conforming_code(act, ent, rm, ec)
            train_pairs.append((instr, code))                                     # sans contexte
            train_pairs.append((f"Spec (extraits pertinents) :\n{ectx}\n\n{instr}", code))  # avec
    pairs = train_pairs
    log(f"[2/3] Entraînement POIDS MIXTE : {len(pairs)} ex / {len(train_ents)} entités "
        f"(sans-ctx=conventions + avec-ctx=lire les faits)")
    data, adapter = f"{_PROJ}/logs/spec_data", f"{L}/spec"
    n = d2l.build_chat_dataset(pairs, data, repeat=2, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=3)
    iters = min(800, max(500, 3 * len(pairs)))
    res = d2l.train_lora(cfg.mlx_model_path, data, adapter, iters=iters, num_layers=cfg.d2l_num_layers,
                         learning_rate=cfg.d2l_learning_rate, rank=16, max_seq_length=1024,
                         mask_prompt=True, python_exe=sys.executable, log_file=LOG_PATH)
    log(f"  POIDS : ok={res['ok']} val_loss={res['val_loss']} ({n} lignes, {iters} iters)")

    # tâches HELD-OUT : entités jamais entraînées
    tasks = []
    for ent, (rm, ec) in HELD_ENT.items():
        instr = f"Implémente la fonction get_{ent} qui récupère un {ent} par identifiant."
        tasks.append((ent, instr, CONV, [rm, ec]))
    nt = len(tasks)

    summary = llm.generate("Résume ce standard en gardant les conventions ET les noms d'API "
                           "exacts :\n" + spec, None)
    sum_tok = count_tokens(summary)

    def gen(prompt):
        return llm.generate(prompt + "\nÉcris uniquement le code Python de la fonction.", None)

    SUT = ("base", "RAG", "compaction", "poids", "complémentaire")
    C = {s: [] for s in SUT}   # conventions
    F = {s: [] for s in SUT}   # faits spécifiques
    CTX = {s: [] for s in SUT}  # tokens de contexte / requête

    log("[3/3] Génération + scoring (5 conditions)")
    for i, (ent, instr, conv_ck, fact_ck) in enumerate(tasks, 1):
        chunks = rag.topk(instr, 4)  # budget réaliste : la spec entière ne tient pas
        ctx = "\n".join(chunks)
        rag_prompt = f"Spec (extraits pertinents) :\n{ctx}\n\n{instr}"
        sum_prompt = f"Spec (résumé) :\n{summary}\n\n{instr}"

        # base / RAG / compaction -> modèle de base
        llm.set_adapter(None)
        ob = gen(instr)
        orr = gen(rag_prompt)
        oc = gen(sum_prompt)
        # poids / complémentaire -> LoRA style
        llm.set_adapter(adapter)
        op = gen(instr)
        ox = gen(rag_prompt)

        for s, out, ct in (("base", ob, 0), ("RAG", orr, count_tokens(ctx)),
                           ("compaction", oc, sum_tok), ("poids", op, 0),
                           ("complémentaire", ox, count_tokens(ctx))):
            C[s].append(frac(out, conv_ck)); F[s].append(frac(out, fact_ck)); CTX[s].append(ct)
        log(f"[tâche {i}/{nt} {ent:9s}] "
            f"conv: base={C['base'][-1]*100:.0f} rag={C['RAG'][-1]*100:.0f} "
            f"poids={C['poids'][-1]*100:.0f} compl={C['complémentaire'][-1]*100:.0f} | "
            f"faits: rag={F['RAG'][-1]*100:.0f} poids={F['poids'][-1]*100:.0f} "
            f"compl={F['complémentaire'][-1]*100:.0f}")

    def avg(d, s):
        return sum(d[s]) / len(d[s]) * 100 if d[s] else 0.0
    log("")
    log("=== RÉSULTATS — adhérence (held-out, entités jamais entraînées) ===")
    log(f"{'méthode':16s} | conventions | faits spéc. | global | ctx tok/req")
    for s in SUT:
        g = (avg(C, s) + avg(F, s)) / 2
        log(f"{s:16s} | {avg(C, s):9.0f}%  | {avg(F, s):9.0f}%  | {g:5.0f}% | "
            f"{sum(CTX[s])/len(CTX[s]):5.0f}")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
