"""Agent : boucle agentique mono-utilisateur, 3 modes (baseline/bctrl/m0).

La boucle, a chaque tour :
  1) construit messages = [system, memoire injectee, events du model_context] ;
  2) si compaction necessaire (mode != baseline) -> compacte (cumule summary_tokens) ;
  3) appelle llm.chat(messages, tool_schemas()) ;
  4) si tool_calls -> dispatch via tools, append tool_output, detector.observe_error
     sur les sorties en erreur ;
  5) detector.observe_message sur le content (TIR 1) ; en mode m0, les lecons
     candidates sont ecrites en memoire ;
  6) detecte la fin ([[DONE]] dans le content OU oracle satisfait) ; stop a max_turns.

Modes (task["sut"]) :
  - baseline : compaction OFF, memoire OFF.
  - bctrl    : compaction ON, memoire OFF.
  - m0       : compaction ON, memoire ON (injection + ecriture lecons + failure-hook).

Metrics :
  turns, context_tokens (pic du model_context), summary_tokens (cumule),
  net_tokens = (somme tokens des events prunes) - summary_tokens,
  reerrors (nb de "reerror"), injection_total/relevant, success via oracle.
"""

from __future__ import annotations

import os

from .config import Config, count_tokens
from .detector import TwoShotDetector
from .llm import DONE_MARKER, LLMClient
from .store import EventStore
from .tools import Tools, dispatch, tool_schemas
from .types import Metrics, ToolResult

_SYSTEM_PROMPT = (
    "Tu es un assistant agentique local. Tu disposes des outils bash, read_file, "
    "write_file, edit, confines a un repertoire de travail. Resous la tache de "
    "l'utilisateur puis termine en ecrivant le marqueur " + DONE_MARKER + "."
)


class Agent:
    """Orchestre la boucle de resolution d'une tache."""

    def __init__(self, llm, store, memory, detector, compactor, tools, cfg: Config) -> None:
        self.llm = llm
        self.store = store
        self.memory = memory
        self.detector: TwoShotDetector = detector
        self.compactor = compactor
        self.tools: Tools = tools
        self.cfg = cfg
        self._turn = 0  # compteur persistant pour le mode conversationnel (chat_turn)
        # callback(text) appele a la compaction avec le contenu LIBERE du contexte
        # (auto-promotion vers la memoire long-terme). Pose par le serveur.
        self.on_compact = None

    # ------------------------------------------------------------- assemblage
    def _build_messages(self, mode: str) -> list[dict]:
        """Assemble les messages a envoyer au LLM pour le tour courant."""
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Memoire injectee (mode m0 uniquement).
        if mode == "m0":
            injected = self.memory.render_for_injection()
            if injected:
                messages.append({"role": "system", "content": injected})

        # Events du model_context (non compactes), ordre chronologique.
        for ev in self.store.model_context():
            role = ev.role
            # On ramene summary/reinjection/tool a des roles valides pour l'API chat.
            if role in ("summary", "tool"):
                role = "system" if role == "summary" else "tool"
            if role not in ("user", "assistant", "system", "tool"):
                role = "system"
            messages.append({"role": role, "content": ev.content})
        return messages

    # ------------------------------------------------------------- oracle
    @staticmethod
    def _oracle_satisfied(oracle: dict | None, tools: Tools) -> bool:
        """Evalue l'oracle de la tache. Formes supportees :
          {"file_exists": "relpath"}
          {"file_contains": {"path": "relpath", "text": "..."}}
          {"file_equals": {"path": "relpath", "text": "..."}}
        Plusieurs cles -> toutes doivent etre vraies (AND). None -> True.
        """
        if not oracle:
            return True
        workdir = tools.workdir

        def _path(rel: str) -> str:
            return os.path.join(workdir, rel)

        if "file_exists" in oracle:
            if not os.path.isfile(_path(oracle["file_exists"])):
                return False
        if "file_contains" in oracle:
            spec = oracle["file_contains"]
            p = _path(spec["path"])
            if not os.path.isfile(p):
                return False
            with open(p, "r", encoding="utf-8") as fh:
                if spec["text"] not in fh.read():
                    return False
        if "file_equals" in oracle:
            spec = oracle["file_equals"]
            p = _path(spec["path"])
            if not os.path.isfile(p):
                return False
            with open(p, "r", encoding="utf-8") as fh:
                if fh.read() != spec["text"]:
                    return False
        return True

    # ------------------------------------------------------------- run
    def run_task(self, task: dict) -> Metrics:
        """Execute une tache et renvoie ses Metrics."""
        mode = task.get("sut", "m0")
        compaction_on = mode in ("bctrl", "m0")
        memory_on = mode == "m0"

        # Script du mock (si backend mock). On (re)charge a chaque tache.
        script = task.get("mock_script")
        if script is not None and hasattr(self.llm, "set_script"):
            self.llm.set_script(script)

        # Workdir de la tache : confine les outils.
        workdir = task.get("workdir")
        if workdir:
            self.tools = Tools(workdir)

        oracle = task.get("oracle")
        # Annotations de pertinence d'injection (optionnel) : dict id_entry->bool
        # ou simplement un booleen global ; sinon relevant = total.
        relevance = task.get("injection_relevance")

        # Goal initial = prompt utilisateur.
        prompt = task.get("prompt", "")
        self.store.append(0, "user", "message", prompt)

        peak_context = 0
        summary_tokens_total = 0
        pruned_tokens_total = 0
        reerrors = 0
        injection_total = 0
        injection_relevant = 0
        success = False
        turn = 0

        # Injection mesuree (mode m0) : combien d'entrees actives injectees.
        if memory_on:
            active = self.memory.active_entries()
            rendered = self.memory.render_for_injection()
            for e in active:
                if e.text and e.text in rendered:
                    injection_total += 1
                    if self._is_relevant(e, relevance):
                        injection_relevant += 1

        for turn in range(1, self.cfg.max_turns + 1):
            # Compaction eventuelle AVANT l'appel.
            if compaction_on:
                ctx = self.store.model_context()
                if self.compactor.should_compact(ctx):
                    before_ids = {e.id for e in ctx}
                    _, s_tokens = self.compactor.compact(ctx, turn)
                    summary_tokens_total += s_tokens
                    # Tokens reellement prunes lors de cette compaction.
                    for pe in self.store.pruned_events():
                        if pe.id in before_ids:
                            pruned_tokens_total += pe.tokens
                    # On evite de re-compter les memes events au prochain tour :
                    # before_ids ne contient que les ids de CE contexte, et
                    # mark_compacted ne re-marque pas (compacted_at deja pose).

            messages = self._build_messages(mode)
            ctx_tokens = sum(count_tokens(m["content"]) for m in messages)
            peak_context = max(peak_context, ctx_tokens)

            result = self.llm.chat(messages, tool_schemas())
            content = result.get("content", "") or ""
            tool_calls = result.get("tool_calls", []) or []

            # Trace l'assistant.
            if content:
                self.store.append(turn, "assistant", "message", content)

            # TIR 1 : lecons candidates dans le message assistant.
            if memory_on and content:
                for lesson in self.detector.observe_message(content):
                    self.memory.add(lesson, tags=["lesson"], priority="normal")

            # Outils.
            for call in tool_calls:
                self.store.append(
                    turn, "assistant", "tool_call", f"{call.name}({call.args})"
                )
                res: ToolResult = dispatch(self.tools, call)
                self.store.append(turn, "tool", "tool_output", res.output)
                if not res.ok and memory_on:
                    # TIR 2 : failure-hook sur sortie en erreur.
                    verdict = self.detector.observe_error(res.output, turn)
                    if verdict == "reerror":
                        reerrors += 1

            # Fin de tache ?
            if DONE_MARKER in content:
                success = self._oracle_satisfied(oracle, self.tools)
                break
            if self._oracle_satisfied(oracle, self.tools) and oracle:
                success = True
                break
        else:
            # Boucle epuisee sans break : on evalue l'oracle final.
            success = self._oracle_satisfied(oracle, self.tools)

        # context_tokens : pic du model_context observe (ou final si jamais mesure).
        final_ctx = sum(e.tokens for e in self.store.model_context())
        context_tokens = peak_context if peak_context else final_ctx

        net_tokens = pruned_tokens_total - summary_tokens_total

        return Metrics(
            success=success,
            turns=turn,
            context_tokens=context_tokens,
            summary_tokens=summary_tokens_total,
            net_tokens=net_tokens,
            reerrors=reerrors,
            injection_total=injection_total,
            injection_relevant=injection_relevant,
        )

    # ------------------------------------------------------------- chat (UI)
    def chat_turn(self, user_text: str, mode: str = "m0", max_steps: int = 6) -> str:
        """Un tour de conversation pour une UI de chat (sans tache/oracle).

        Maintient l'etat (store + memoire) ENTRE les appels : c'est ce qui
        distingue M0 d'un appel LLM brut (injection memoire, compaction,
        failure-hook, lecons). Petite boucle interne pour laisser le modele
        utiliser des outils puis repondre.
        """
        compaction_on = mode in ("bctrl", "m0")
        memory_on = mode == "m0"
        # Outils desactives par defaut en conversation (cf. Config.chat_use_tools) :
        # un petit modele a qui on expose des outils emet du JSON de tool-call au
        # lieu de repondre. Les outils restent disponibles en mode tache (run_task).
        use_tools = bool(getattr(self.cfg, "chat_use_tools", False))
        schemas = tool_schemas() if use_tools else None

        self._turn += 1
        self.store.append(self._turn, "user", "message", user_text)

        answer_parts: list[str] = []
        last_content = ""
        for _ in range(max_steps):
            if compaction_on:
                ctx = self.store.model_context()
                if self.compactor.should_compact(ctx):
                    before_ids = {e.id for e in ctx}
                    self.compactor.compact(ctx, self._turn)
                    # auto-promotion : le contenu LIBERE du contexte -> memoire long-terme
                    if self.on_compact is not None:
                        freed = [e.content for e in self.store.pruned_events()
                                 if e.id in before_ids and e.kind in ("message", "tool_output")]
                        text = "\n".join(t for t in freed if t.strip())
                        if text.strip():
                            try:
                                self.on_compact(text)
                            except Exception:  # noqa: BLE001 — best-effort
                                pass

            messages = self._build_messages(mode)
            result = self.llm.chat(messages, schemas)
            content = result.get("content", "") or ""
            tool_calls = (result.get("tool_calls", []) or []) if use_tools else []

            if content:
                last_content = content
                self.store.append(self._turn, "assistant", "message", content)
                if memory_on:
                    for lesson in self.detector.observe_message(content):
                        self.memory.add(lesson, tags=["lesson"], priority="normal")

            if not tool_calls:
                visible = content.replace(DONE_MARKER, "").strip()
                if visible:
                    answer_parts.append(visible)
                break

            for call in tool_calls:
                self.store.append(
                    self._turn, "assistant", "tool_call", f"{call.name}({call.args})"
                )
                res: ToolResult = dispatch(self.tools, call)
                self.store.append(self._turn, "tool", "tool_output", res.output)
                if not res.ok and memory_on:
                    self.detector.observe_error(res.output, self._turn)

            if DONE_MARKER in content:
                break

        answer = "\n".join(answer_parts).strip()
        if not answer:  # fallback : dernier contenu produit, meme sans break propre
            answer = last_content.replace(DONE_MARKER, "").strip()
        return answer or "(M0 n'a pas produit de texte)"

    def clear_context(self) -> int:
        """Vide le contexte conversationnel (nouveau store vierge), en CONSERVANT
        la memoire texte et le LoRA charge. Sert a tester la memoire-poids : apres
        /sleep puis /ctxt_clear, le savoir ne peut venir que des poids + MEMORY.md.

        Retourne le nombre d'events effaces.
        """
        n = len(self.store.all_events())
        self.store = EventStore(":memory:")
        self.compactor.store = self.store  # le compactor doit suivre le nouveau store
        self._turn = 0
        return n

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _is_relevant(entry, relevance) -> bool:
        """Determine si une entree injectee est 'pertinente'.

        - relevance None         -> tout pertinent (relevant = total).
        - relevance bool         -> applique a toutes les entrees.
        - relevance dict id->bool-> par entree (defaut True si absent).
        """
        if relevance is None:
            return True
        if isinstance(relevance, bool):
            return relevance
        if isinstance(relevance, dict):
            return bool(relevance.get(entry.id, True))
        return True
