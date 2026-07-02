"""Auto-amélioration déclenchée par l'ÉCHEC : échoue → lit la doc → s'auto-entraîne → réessaie.

Idée utilisateur : un petit modèle qui galère sur une tâche va chercher la documentation,
l'internalise dans ses poids (LoRA), et réessaie. Ici le modèle S'AUTO-GÉNÈRE ses données
d'entraînement depuis la doc (extract_qa + augmentation = self-edits, style SEAL 2506.10943).

Prédiction (fondée sur nos mesures) : ça marche pour les trous de SAVOIR (API/conventions
inconnues), PAS pour les trous de CAPACITÉ (raisonnement) — d'où 2 sondes de capacité qui ne
doivent PAS bouger. Domaine : SDK fictif « Corvex » (zéro fuite paramétrique).

Arms : AVANT (base, 0 ctx) → boucle d'auto-amélioration → APRÈS (poids, 0 ctx) · RAG (référence).
Live : tail -f logs/benchmark_selfimprove.log
"""

from __future__ import annotations

import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0 import d2l  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_selfimprove.log")
ADAPTER = os.path.join(_PROJ, "models", "lora", "corvex_self")
_T0 = time.time()

DOC = """Documentation du SDK Corvex (version 3).

Connexion : la connexion s'ouvre avec corvex.connect(project_key, region). La région par
défaut est "eu-west". Le jeton d'authentification est lu dans la variable d'environnement
CORVEX_TOKEN.

Envoi : les frames sont envoyées une à une avec session.push_frame(frame, tags=[...]).
Convention stricte : tous les tags doivent être en kebab-case minuscule. Un lot entier est
envoyé avec session.flush(mode="atomic") ; utiliser mode="stream" pour le flux continu.

Stockage : les frames sont conservées dans des "vaults". Le vault par défaut s'appelle "main".
La limite d'ingestion est de 120 frames par minute et par projet.

Erreurs : quand le schéma des données dérive, le SDK lève l'exception CorvexDriftError.
La version du schéma se fige avec corvex.pin(schema=3).

Fermeture : une session se termine par session.seal() — jamais par close().
"""

# held-out : phrasés différents de la doc
KNOWLEDGE = [
    ("Avec quelle fonction ouvre-t-on la connexion au SDK Corvex ?", "connect"),
    ("Quelle variable d'environnement contient le jeton d'authentification Corvex ?", "CORVEX_TOKEN"),
    ("Quelle exception Corvex est levée quand le schéma dérive ?", "CorvexDriftError"),
    ("Quelle méthode faut-il appeler pour terminer proprement une session Corvex ?", "seal"),
    ("Combien de frames par minute Corvex accepte-t-il au maximum ?", "120"),
    ("Quelle est la région par défaut d'une connexion Corvex ?", "eu-west"),
    ("Dans quoi Corvex conserve-t-il les frames stockées ?", "vault"),
    ("Quelle méthode Corvex envoie un lot de frames en mode atomique ?", "flush"),
]
# sondes de CAPACITÉ : la doc n'aide pas ; ne doivent PAS bouger (frontière savoir/capacité)
CAPABILITY = [
    ("Calcule 4729 multiplié par 8351. Donne uniquement le nombre.", "39492479"),
    ("Écris le mot 'consolidation' à l'envers, uniquement le résultat.", "noitadilosnoc"),
]


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def _purge():
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def score(llm, items, ctx=None):
    ok = 0
    for q, a in items:
        p = q if ctx is None else f"Documentation :\n{ctx}\n\nQuestion : {q}\nRéponds brièvement :"
        ok += d2l.answer_recalled(llm.generate(p, None), a)
    return ok


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== AUTO-AMÉLIORATION sur échec (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg); llm.set_adapter(None)
    nk, nc = len(KNOWLEDGE), len(CAPABILITY)

    log("[1/5] tentative initiale (0 contexte) — le modèle 'galère' ?")
    k0 = score(llm, KNOWLEDGE); c0 = score(llm, CAPABILITY)
    log(f"   savoir {k0}/{nk} ({k0/nk*100:.0f}%) | capacité {c0}/{nc}")

    log("[2/5] le modèle 'recherche' : lit la doc et S'AUTO-GÉNÈRE ses données (self-edits)")
    # double stratégie d'étude, 100% auto-générée : vue d'ensemble (whole-doc, exige un cap
    # de tokens élevé pour ne pas tronquer les 24 paires) + fiches par paragraphe
    facts = [ln.strip() for ln in DOC.splitlines() if len(ln.strip()) > 30]
    llm.cfg.mlx_max_tokens = 620          # cap élevé SEULEMENT pour la vue d'ensemble (24 paires)
    qa = d2l.extract_qa(DOC, llm.generate, n=24)
    _purge()
    log(f"   …vue d'ensemble : {len(qa)} Q/R")
    llm.cfg.mlx_max_tokens = 220          # fiches par-fait : 6 paires max, cap court
    for i, fact in enumerate(facts):
        qa += d2l.extract_qa(fact, llm.generate, n=6)
        _purge()
    qa = d2l.clean_and_balance(qa, max_per_answer=3)
    train = qa + d2l.augment_pairs(qa, llm.generate, n_paraphrases=6)
    train = d2l.clean_and_balance(train, max_per_answer=12)
    llm.cfg.mlx_max_tokens = 90           # éval : réponses courtes
    _purge()
    log(f"   {len(qa)} Q/R auto-extraites (doc + par-fait) -> {len(train)} après augmentation")
    # couverture des 8 faits testés par les données d'auto-étude (diagnostic décisif)
    blob = " ".join(q + " " + a for q, a in train).lower()
    cov = sum(1 for _, a in KNOWLEDGE if a.lower() in blob)
    log(f"   couverture des faits testés par l'auto-étude : {cov}/{len(KNOWLEDGE)}")
    for q, a in train[:10]:
        log(f"     ex: Q:{q[:55]} | R:{a[:30]}")

    log("[3/5] auto-entraînement LoRA")
    data = os.path.join(_PROJ, "logs", "corvex_self_data")
    n = d2l.build_chat_dataset(train, data, repeat=5, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=4)
    iters = min(600, max(300, 9 * len(train)))
    res = d2l.train_lora(cfg.mlx_model_path, data, ADAPTER, iters=iters, num_layers=cfg.d2l_num_layers,
                         learning_rate=cfg.d2l_learning_rate, rank=16, python_exe=sys.executable,
                         log_file=LOG_PATH)
    log(f"   LoRA ok={res['ok']} val_loss={res['val_loss']} ({n} lignes, {iters} iters)")

    log("[4/5] NOUVELLE tentative (0 contexte, poids auto-améliorés)")
    llm.set_adapter(ADAPTER)
    k1 = score(llm, KNOWLEDGE); c1 = score(llm, CAPABILITY)
    log(f"   savoir {k1}/{nk} ({k1/nk*100:.0f}%) | capacité {c1}/{nc}")

    log("[5/5] référence : RAG (doc en contexte, modèle nu)")
    llm.set_adapter(None)
    kr = score(llm, KNOWLEDGE, ctx=DOC)
    log(f"   RAG savoir {kr}/{nk} ({kr/nk*100:.0f}%)")

    log("")
    log("=== RÉSULTAT — boucle échec → doc → poids → retry ===")
    log(f"{'':22s} | savoir (API Corvex) | capacité (sondes)")
    log(f"{'AVANT (base, 0 ctx)':22s} | {k0}/{nk} ({k0/nk*100:3.0f}%)         | {c0}/{nc}")
    log(f"{'APRÈS (poids, 0 ctx)':22s} | {k1}/{nk} ({k1/nk*100:3.0f}%)         | {c1}/{nc}")
    log(f"{'RAG (référence)':22s} | {kr}/{nk} ({kr/nk*100:3.0f}%)         | —")
    log("")
    if k1 >= k0 + nk * 0.5 and c1 <= c0 + 1:
        log(f"✅ AUTO-AMÉLIORATION VALIDÉE pour les trous de SAVOIR : {k0/nk*100:.0f}% → {k1/nk*100:.0f}% "
            f"à 0 token de contexte, données d'entraînement auto-générées. La CAPACITÉ ne bouge pas "
            f"({c0}→{c1}/{nc}) — les poids ajoutent du savoir, pas de l'intelligence brute (frontière confirmée).")
    else:
        log(f"🟧 MITIGÉ : savoir {k0}→{k1}/{nk}, capacité {c0}→{c1}/{nc} — à analyser.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
