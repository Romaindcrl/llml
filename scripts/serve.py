"""Serveur compatible OpenAI exposant M0 (memoire + compaction + agent + outils)
pour une UI de chat type Open WebUI.

Contrairement a un appel Ollama brut, CHAQUE message passe par la boucle M0 :
injection de la memoire texte, compaction de contexte, usage d'outils, failure-hook
et ecriture de lecons. L'etat (store + MEMORY.md) persiste pendant la vie du process.

Lancer :
  M0_BACKEND=ollama M0_MODEL=qwen2.5:7b ./.venv/bin/python scripts/serve.py
  (defaut : http://127.0.0.1:8000 ; surcharge via M0_SERVE_HOST / M0_SERVE_PORT)

Brancher Open WebUI -> Settings -> Connections -> OpenAI API :
  Base URL = http://localhost:8000/v1   |   API key = n'importe quoi.

Endpoints utiles hors OpenAI :
  GET /health      -> backend/modele/seuils actifs
  GET /m0/state    -> entrees memoire + compteurs d'events (pour inspecter M0)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time

# Rendre le package m0 importable quel que soit le cwd.
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from m0 import d2l, lora_merge  # noqa: E402
from m0.ltm import LTM  # noqa: E402
from m0.rag import RAG, is_generation  # noqa: E402
from m0.agent import Agent  # noqa: E402
from m0.compaction import Compactor  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.detector import TwoShotDetector  # noqa: E402
from m0.llm import MLXClient, make_client  # noqa: E402
from m0.memory import TextMemory  # noqa: E402
from m0.store import EventStore  # noqa: E402
from m0.tools import Tools  # noqa: E402

MODEL_ID = "m0"


def build_agent() -> tuple[Agent, Config]:
    cfg = Config.from_env()
    # Workdir stable pour les outils du chat (confines a ce dossier).
    workdir = os.path.join(_PROJ, "logs", "chat_workdir")
    os.makedirs(workdir, exist_ok=True)
    cfg.workdir = workdir

    llm = make_client(cfg)
    store = EventStore(":memory:")
    memory = TextMemory(cfg.memory_path, cfg.repo_dir, cfg.memory_inject_cap_tokens)
    detector = TwoShotDetector(memory)
    compactor = Compactor(store, llm, cfg)
    tools = Tools(workdir)
    return Agent(llm, store, memory, detector, compactor, tools, cfg), cfg


app = FastAPI(title="M0 — OpenAI-compatible server")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
_AGENT, _CFG = build_agent()

# Mémoire long-terme : LTM (Q/R -> poids par replay) + RAG (docs bruts -> contexte pour
# la génération). Le benchmark montre : poids = RAPPEL de faits, RAG = GÉNÉRATION.
_LTM = LTM(os.path.join(_PROJ, "logs", "ltm_qa.jsonl"))
_RAG = RAG(os.path.join(_PROJ, "logs", "rag_corpus.txt"))


def _on_compact(text):
    """Auto-promotion à la saturation : le contenu libéré nourrit LTM (faits→poids)
    ET RAG (texte brut→référence de génération)."""
    _RAG.add_document(text)
    _LTM.add_document(text, _AGENT.llm.generate)


_AGENT.on_compact = _on_compact


def _mem_adapter():
    """Chemin du LoRA-mémoire courant s'il existe (sinon None = modèle de base)."""
    p = os.path.join(_PROJ, "models", "lora", "memory")
    return p if os.path.exists(os.path.join(p, "adapters.safetensors")) else None


def _ensure_adapter(target):
    """Charge l'adapter cible (None = base) seulement si différent de l'état courant."""
    llm = _AGENT.llm
    if isinstance(llm, MLXClient) and getattr(llm, "adapter_path", None) != target:
        llm.set_adapter(target)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "backend": _CFG.backend,
        "model": _CFG.model if _CFG.backend == "ollama" else _CFG.mlx_model_path,
        "seuils": {
            "memory_inject_cap_tokens": _CFG.memory_inject_cap_tokens,
            "compaction_trigger_tokens": _CFG.compaction_trigger_tokens,
            "min_recoverable_tokens": _CFG.min_recoverable_tokens,
            "max_context_tokens": _CFG.max_context_tokens,
            "keep_recent_turns": _CFG.keep_recent_turns,
        },
    }


@app.get("/v1/models")
def models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "m0"}
        ],
    }


@app.get("/m0/state")
def state() -> dict:
    mem = _AGENT.memory
    return {
        "memory_entries": [
            {
                "id": e.id,
                "status": e.status,
                "priority": e.priority,
                "tags": e.tags,
                "text": e.text[:200],
            }
            for e in mem.all_entries()
        ],
        "events_total": len(_AGENT.store.all_events()),
        "model_context_events": len(_AGENT.store.model_context()),
        "pruned_events": len(_AGENT.store.pruned_events()),
    }


# Open WebUI peut envoyer plusieurs requetes en parallele (chat + titre + tags) :
# on serialise les acces a l'agent partage (store/memoire mono-instance).
_LOCK = threading.Lock()

# Trace du dernier /sleep (alimente la commande /info).
_LAST_SLEEP: dict = {}


def _extract_text(content) -> str:
    """Aplati un `content` OpenAI : str OU liste de parts [{type,text}, ...]."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text") or p.get("content")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _messages_text(messages: list[dict]) -> tuple[str, str]:
    """Retourne (dernier message user, dernier message system), texte aplati."""
    user_text = ""
    system_text = ""
    for m in messages or []:
        role = m.get("role")
        txt = _extract_text(m.get("content"))
        if role == "user":
            user_text = txt  # le dernier gagne
        elif role == "system":
            system_text = txt
    return user_text, system_text


def _is_owui_utility(system_text: str, user_text: str) -> bool:
    """Detecte les requetes utilitaires d'Open WebUI (titre, tags, suggestions)
    qui ne doivent PAS passer par M0 (sinon elles polluent la memoire)."""
    blob = f"{system_text}\n{user_text}"
    markers = ("### Task:", "Generate a concise", "Create a concise",
               "JSON format:", "chat_history", "follow-up")
    return any(mk in blob for mk in markers)


# --------------------------------------------------------------- commandes slash

_HELP = (
    "Commandes M0 (mémoire 2 étages : contexte ↔ poids) :\n"
    "• /remember — pousse le dernier document/texte donné vers la MÉMOIRE LONG-TERME "
    "(corpus texte). (Auto aussi : quand le contexte sature, le contenu libéré y va.)\n"
    "• /sleep — consolidation REPLAY : réentraîne un LoRA depuis la base sur TOUT le "
    "corpus long-terme (le savoir passe du texte aux poids).\n"
    "• /ctxt_clear — vide le contexte (garde le LoRA). Pour tester la mémoire-poids.\n"
    "• /info — corpus long-terme + dernier entraînement.\n"
    "• /reset — remise à zéro : modèle de BASE, LoRA + LTM + dataset effacés.\n"
    "• /state — état courant : adapter chargé, events, LTM.\n"
    "• /help — cette aide."
)


def _conversation_text(store) -> str:
    """Reconstitue la conversation en texte lisible (pour l'extraction de Q/R)."""
    lines = []
    for e in store.all_events():
        if e.kind == "message" and e.role == "user":
            lines.append(f"Utilisateur: {e.content}")
        elif e.kind == "message" and e.role == "assistant":
            lines.append(f"Assistant: {e.content}")
    return "\n".join(lines)


def _buffer_pairs(store) -> list[tuple[str, str]]:
    """Reconstitue les echanges (user, assistant) du buffer pour l'entrainement."""
    evs = [e for e in store.all_events()
           if e.kind == "message" and e.role in ("user", "assistant")]
    pairs: list[tuple[str, str]] = []
    pending_user = None
    for e in evs:
        if e.role == "user":
            pending_user = e.content
        elif e.role == "assistant" and pending_user is not None:
            pairs.append((pending_user, e.content))
            pending_user = None
    return pairs


def _state_text() -> str:
    actives = _AGENT.memory.active_entries()
    adapter = getattr(_AGENT.llm, "adapter_path", None)
    base = _CFG.model if _CFG.backend == "ollama" else os.path.basename(_CFG.mlx_model_path)
    lines = [
        f"backend={_CFG.backend} · base={base}",
        f"LoRA charge : {adapter or 'aucun'}",
        f"events contexte : {len(_AGENT.store.model_context())} · "
        f"memoire : {len(actives)} entrees actives",
    ]
    for e in actives[:6]:
        lines.append(f"  - [{e.priority}] {e.text[:80]}")
    return "\n".join(lines)


def _info_text() -> str:
    """Mémoire long-terme + texte + ce qui a été envoyé au dernier entraînement LoRA."""
    out: list[str] = []

    # 0) Mémoire LONG-TERME (corpus texte -> poids par replay)
    ltm = _LTM.all_qa()
    out.append(f"🧠 MÉMOIRE LONG-TERME (LTM, → poids) — {len(ltm)} faits Q/R")
    for q, a in ltm[:12]:
        out.append(f"  • {q}  →  {a}")
    if len(ltm) > 12:
        out.append(f"  … (+{len(ltm) - 12})")
    out.append("")

    # 1) Memoire texte (MEMORY.md)
    entries = _AGENT.memory.all_entries()
    out.append(f"📒 MÉMOIRE TEXTE (MEMORY.md) — {len(entries)} entrée(s)")
    if not entries:
        out.append("  (vide)")
    for e in entries:
        out.append(f"  • [{e.status}/{e.priority}] {e.text}")

    # 2) Dernier entrainement LoRA
    out.append("")
    if _LAST_SLEEP:
        ls = _LAST_SLEEP
        out.append(f"🏋️ DERNIER ENTRAÎNEMENT LoRA ({ls.get('when', '?')})")
        out.append(f"  adapter   : {ls.get('adapter')}")
        out.append(
            f"  réglages  : iters={ls.get('iters')} · "
            f"train_loss={ls.get('train_loss')} · val_loss={ls.get('val_loss')}"
        )
        ex = ls.get("examples", [])
        out.append(f"  {ls.get('qa_count', '?')} faits extraits → {len(ex)} exemples Q/R distincts :")
        for q, a in ex:
            out.append(f"    Q: {q}\n    R: {a}")
    else:
        # repli : relire le dernier dataset ecrit sur disque
        path = os.path.join(_PROJ, "logs", "d2l_data", "train.jsonl")
        if os.path.exists(path):
            out.append("🏋️ DERNIER DATASET D'ENTRAÎNEMENT (logs/d2l_data/train.jsonl)")
            seen = set()
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msgs = row.get("messages", [])
                    u = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                    a = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
                    key = (u, a)
                    if key not in seen:
                        seen.add(key)
                        out.append(f"    Q: {u}\n    R: {a}")
        else:
            out.append("🏋️ DERNIER ENTRAÎNEMENT LoRA : aucun /sleep effectué.")
    return "\n".join(out)


def _do_reset() -> str:
    """Remise a zero complete de l'etat APPRIS : modele de base (LoRA detache +
    supprime), MEMORY.md vide, dataset d'entrainement efface. Ne touche pas au
    contexte de conversation (cf. /ctxt_clear)."""
    global _LAST_SLEEP
    llm = _AGENT.llm
    before = getattr(llm, "adapter_path", None)
    if hasattr(llm, "set_adapter"):
        llm.set_adapter(None)  # revient au modele de base
    # supprime l'adapter et le dernier dataset sur disque
    shutil.rmtree(os.path.join(_PROJ, "models", "lora", "current"), ignore_errors=True)
    shutil.rmtree(os.path.join(_PROJ, "models", "lora", "memory"), ignore_errors=True)
    shutil.rmtree(os.path.join(_PROJ, "logs", "d2l_data"), ignore_errors=True)
    _LTM.clear()  # vide la memoire long-terme (corpus texte)
    _RAG.clear()  # vide l'index RAG (docs bruts)
    # vide la memoire texte (fichier + objet en memoire) et le detecteur
    try:
        os.remove(_CFG.memory_path)
    except OSError:
        pass
    _AGENT.memory = TextMemory(_CFG.memory_path, _CFG.repo_dir, _CFG.memory_inject_cap_tokens)
    _AGENT.detector = TwoShotDetector(_AGENT.memory)
    _LAST_SLEEP = {}
    return (
        "♻️ RESET complet :\n"
        f"• modèle de BASE restauré (LoRA détaché : {before or 'aucun'} + supprimé)\n"
        "• mémoire long-terme (LTM) vidée + MEMORY.md vidé\n"
        "• dataset d'entraînement effacé (logs/d2l_data)\n"
        "(Le contexte de conversation n'est pas touché — fais /ctxt_clear pour le vider.)"
    )


# Sondes neutres pour le controle d'integrite. DISJOINTES du jeu d'ancrage
# (d2l.ANCHOR_PAIRS) : on teste la preservation des capacites generales sur du
# generaliste NON entraine (pas de fuite), pas sur les ancres elles-memes.
_NEUTRAL_PROBES = [
    "Salut, ca va ?",
    "Combien font 2 + 3 ?",
    "Quelle est la capitale de l'Allemagne ?",
    "Quel est le contraire de 'chaud' ?",
    "Cite une couleur.",
]


def _gate_adapter(adapter_dir: str, eval_qa: list) -> tuple[float, bool]:
    """Charge le LoRA candidat et le teste : (taux d'acquisition, integrite_ok).

    - acquisition : fraction des faits reellement rappeles (reponse correcte) ;
    - integrite   : aucune sortie degeneree sur des questions neutres.
    """
    llm = _AGENT.llm
    llm.set_adapter(adapter_dir)
    ok = 0
    for q, a in eval_qa:
        if d2l.answer_recalled(llm.generate(q, None), a):
            ok += 1
    acquired = ok / max(1, len(eval_qa))
    intact = not any(d2l.looks_degenerate(llm.generate(p, None)) for p in _NEUTRAL_PROBES)
    return acquired, intact


def _do_sleep() -> str:
    """Consolidation REPLAY : promeut la conversation courante en LTM, puis réentraîne
    un LoRA FRAIS depuis la base sur TOUT le corpus LTM (texte = source de vérité).
    Approche qui scale (M3) ; aucune fusion d'adapters."""
    global _LAST_SLEEP
    llm = _AGENT.llm
    if not isinstance(llm, MLXClient):
        return ("/sleep necessite le backend MLX. Relance :\n"
                "M0_BACKEND=mlx M0_MLX_MODEL_PATH=models/qwen2.5-7b-it-mlx-8bit "
                "./.venv/bin/python scripts/serve.py")
    # 1) corpus LTM = source de vérité. Alimenté par /remember (explicite) + auto-promotion
    #    à la compaction. On NE re-extrait PAS la conversation ici (évite les redites).
    qa = _LTM.all_qa()
    if len(qa) < 2:
        return ("💤 Mémoire long-terme quasi vide. Donne un document puis /remember "
                "(ou discute des faits), puis /sleep.")
    clean = d2l.clean_and_balance(qa, max_per_answer=3)
    aug = d2l.clean_and_balance(
        d2l.augment_pairs(clean, llm.generate, n_paraphrases=3), max_per_answer=8) or clean
    train_pairs, eval_pairs = d2l.split_train_eval(aug, heldout_per_answer=1)
    if not eval_pairs:
        eval_pairs = clean
    if not train_pairs:
        train_pairs = aug

    data_dir = os.path.join(_PROJ, "logs", "d2l_data")
    adapter_dir = os.path.join(_PROJ, "models", "lora", "memory")  # le LoRA replay EST la mémoire
    prev_adapter = getattr(llm, "adapter_path", None)
    n = d2l.build_chat_dataset(
        train_pairs, data_dir, repeat=_CFG.d2l_repeat,
        anchors=d2l.ANCHOR_PAIRS, anchor_repeat=_CFG.d2l_anchor_repeat,  # rehearsal anti-oubli
    )
    if n == 0:
        return "Corpus vide apres nettoyage — rien a entrainer."

    # 7) ENTRAINEMENT REPLAY sous GATE HELD-OUT, retry adaptatif. iters proportionnels
    #    au corpus (replay : plus de donnees -> plus d'iters pour tout binder).
    layers = _CFG.d2l_num_layers
    it = min(400, max(_CFG.d2l_iters, 25 * len(clean)))
    lr, rank = _CFG.d2l_learning_rate, 16
    attempts: list[str] = []
    committed = None
    res = None
    for _ in range(3):
        res = d2l.train_lora(
            _CFG.mlx_model_path, data_dir, adapter_dir,
            iters=it, num_layers=layers, learning_rate=lr, rank=rank,
            python_exe=sys.executable,
        )
        if not res["ok"]:
            attempts.append(f"it={it} lr={lr:g} L={layers} r={rank}: echec train (rc={res['returncode']})")
            it, lr = max(30, it // 2), lr / 2.0
            continue
        acq, intact = _gate_adapter(adapter_dir, eval_pairs)
        attempts.append(
            f"it={it} lr={lr:g} L={layers} r={rank}: acquisition(held-out)={acq:.0%} "
            f"integrite={'OK' if intact else 'CASSE'} val_loss={res['val_loss']}"
        )
        if intact and acq >= _CFG.gate_acq:
            committed = {"it": it, "lr": lr, "layers": layers, "rank": rank, "acq": acq, "res": res}
            break
        llm.set_adapter(prev_adapter)  # rollback avant le prochain essai
        # num_layers reste CONSTANT (sinon les LoRA de sessions differentes couvrent
        # des couches differentes -> TIES/merge impossible). On ajuste iters/lr/rang.
        if not intact:  # degenere -> plus doux
            it, lr, rank = max(40, it // 2), lr / 2.0, max(8, rank // 2)
        else:           # sous-entraine -> plus fort (iters ; rang 16 fixe, stable en 8-bit)
            it = int(it * 1.5)

    # 8) DECISION de la gate
    if not committed:
        llm.set_adapter(prev_adapter)  # on NE committe PAS un LoRA rate
        return (
            "💤 Sleep : LoRA REJETÉ par la gate (held-out) — etat precedent conserve.\n"
            + "\n".join("  - " + a for a in attempts)
            + "\n➡️ Astuce : enseigne plus de faits DISTINCTS (plus de donnees = liaison plus stable)."
        )

    llm.set_adapter(adapter_dir)  # le LoRA replay = la mémoire-poids complète (corpus LTM)
    res = committed["res"]
    _LAST_SLEEP = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adapter": adapter_dir,
        "iters": committed["it"],
        "train_loss": res["train_loss"],
        "val_loss": res["val_loss"],
        "qa_count": len(clean),
        "examples": train_pairs[:40],
    }
    return (
        "💤 Sleep terminé — mémoire-poids RÉENTRAÎNÉE par replay ✅\n"
        f"• corpus LTM = {len(clean)} faits · {len(train_pairs)} ex train · {len(eval_pairs)} held-out\n"
        f"• acquisition(held-out)={committed['acq']:.0%} · intégrité=OK · "
        f"it={committed['it']} · val_loss={res['val_loss']}\n"
        f"• adapter chargé : {adapter_dir}\n"
        "➡️ /ctxt_clear puis re-teste : savoir = POIDS (corpus LTM entier)."
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    user_text, system_text = _messages_text(body.get("messages", []))
    stream = bool(body.get("stream"))

    def _work() -> str:
        # --- detection ROBUSTE des commandes slash ---
        # Open WebUI peut prefixer du contexte (RAG d'un document joint) avant le
        # message : on cherche donc une commande connue en tete de N'IMPORTE quelle
        # ligne, et on matche le 1er token exactement (evite les faux positifs).
        cmd = ""
        for line in (user_text or "").splitlines():
            s = line.strip().lower()
            if s.startswith("/"):
                cmd = s
                break
        first = cmd.split()[0] if cmd else ""

        if first in ("/help", "/aide", "/?"):
            return _HELP
        if first == "/info":
            return _info_text()
        if first in ("/state", "/etat"):
            return _state_text()
        if first in ("/remember", "/learn"):
            # promeut le dernier document/texte de l'utilisateur vers la LTM
            docs = [e.content for e in reversed(_AGENT.store.all_events())
                    if e.role == "user" and e.kind == "message"]
            if not docs:
                return ("Rien à mémoriser — donne d'abord le contenu d'un document/texte "
                        "(message normal), puis /remember.")
            with _LOCK:
                added, extracted = _LTM.add_document(docs[0], _AGENT.llm.generate)
                _RAG.add_document(docs[0])  # aussi indexé pour la génération (RAG)
            return (f"🧠 Mémoire long-terme : +{added} faits (sur {extracted} extraits) → "
                    f"LTM = {_LTM.count()} faits · RAG = {_RAG.count()} passages.\n"
                    "Fais /sleep pour graver les faits dans les poids.")
        if first in ("/ctxt_clear", "/ctx_clear", "/clear"):
            n = _AGENT.clear_context()
            adapter = getattr(_AGENT.llm, "adapter_path", None)
            return (f"🧹 Contexte vide ({n} events effaces). LoRA : {adapter or 'aucun'} · "
                    "MEMORY.md conserve.\nLe modele ne 'voit' plus la conversation — "
                    "teste sa memoire-poids.")
        if first == "/reset":
            with _LOCK:
                return _do_reset()
        if first in ("/sleep", "/dodo"):
            with _LOCK:
                return _do_sleep()
        # --- requete utilitaire Open WebUI -> modele brut, hors M0 ---
        if _is_owui_utility(system_text, user_text):
            return _AGENT.llm.generate(user_text or system_text, system_text or None)
        # --- ROUTAGE (conclusions des benchmarks : poids = rappel, RAG = génération) ---
        if is_generation(user_text):
            # GÉNÉRATION -> modèle de BASE + RAG, puis PASSE DE VÉRIFICATION (2-étapes).
            # Le LoRA-mémoire est entraîné sur des faits -> dégrade la génération : on le retire.
            # La vérification corrige les noms d'API/identifiants hallucinés vs la doc — le
            # bench "spec" montre que générer-puis-vérifier bat la fusion LoRA+contexte.
            _ensure_adapter(None)
            ctx = "\n".join(_RAG.topk(user_text, 4))
            with _LOCK:
                if not ctx:
                    return _AGENT.llm.generate(user_text, None)
                draft = _AGENT.llm.generate(f"Documentation pertinente :\n{ctx}\n\n{user_text}", None)
                return _AGENT.llm.generate(
                    f"Documentation de référence :\n{ctx}\n\nRéponse générée :\n{draft}\n\n"
                    "Corrige UNIQUEMENT les noms d'API, identifiants ou valeurs qui ne "
                    "correspondent pas à la documentation ci-dessus ; sinon renvoie la réponse "
                    "inchangée. Renvoie la version finale uniquement.", None)
        # RAPPEL / chat -> POIDS (LoRA-mémoire) via la boucle M0
        _ensure_adapter(_mem_adapter())
        with _LOCK:
            return _AGENT.chat_turn(user_text, "m0")

    try:
        # Bloquant (appel LLM) -> threadpool pour ne pas geler l'event loop.
        answer = await run_in_threadpool(_work)
    except Exception as exc:  # jamais de 500 text/plain vers Open WebUI
        answer = f"⚠️ Erreur M0 ({type(exc).__name__}): {exc}"

    created = int(time.time())
    cid = f"chatcmpl-m0-{created}"

    if not stream:
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    def gen():
        first = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(first)}\n\n"
        last = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(last)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("M0_SERVE_HOST", "127.0.0.1")
    port = int(os.environ.get("M0_SERVE_PORT", "8000"))
    print(f"M0 server -> http://{host}:{port}/v1  (backend={_CFG.backend})")
    uvicorn.run(app, host=host, port=port)
