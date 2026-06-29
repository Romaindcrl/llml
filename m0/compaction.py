"""Compaction du contexte : prune non-destructif + re-injection verbatim + summary.

Origine opencode (transposee a une fenetre ~8K) :
  - on garde les outputs recents ; on prune (= sort du model_context) les
    tool_output verbeux SAUF ceux des 2 derniers tours et ceux marques "skill" ;
  - on ne prune que s'il y a assez de tokens recuperables (min_recoverable_tokens) ;
  - on re-injecte VERBATIM l'etat vivant (dernier goal user, derniers diffs /
    fichiers, derniers 2 tours) pour que rien d'essentiel ne soit perdu ;
  - on resume en 1 appel llm.generate (gist 5 champs) stocke comme event summary.

NON-DESTRUCTIF : le prune passe par store.mark_compacted (UPDATE compacted_at
seulement). Le contenu reste recuperable via store.pruned_events().
"""

from __future__ import annotations

from .config import Config, count_tokens
from .llm import LLMClient
from .types import Event

# Marqueur permettant de proteger les outputs de "skill" (transpose d'opencode
# qui protege les outputs de skill). Un tool_output dont le content contient ce
# marqueur ne sera jamais prune.
SKILL_MARKER = "[[SKILL]]"


class Compactor:
    """Decide et execute la compaction du model_context."""

    def __init__(self, store, llm: LLMClient, cfg: Config) -> None:
        self.store = store
        self.llm = llm
        self.cfg = cfg

    # ------------------------------------------------------------------ utils

    def _recent_turns(self, ctx: list[Event]) -> set[int]:
        """Les `keep_recent_turns` tours les plus recents presents dans ctx."""
        turns = sorted({e.turn for e in ctx})
        if not turns:
            return set()
        return set(turns[-self.cfg.keep_recent_turns:])

    @staticmethod
    def _is_skill(event: Event) -> bool:
        return SKILL_MARKER in (event.content or "")

    def _prunable(self, event: Event, recent_turns: set[int]) -> bool:
        """Prunable = contenu FROID (hors `keep_recent_turns` derniers tours, hors skill,
        pas deja compacte) : tool_output verbeux ET messages (docs/historique). On ne
        prune jamais les events de compaction eux-memes (summary / reinjection)."""
        if event.compacted_at is not None:
            return False
        if event.turn in recent_turns:
            return False
        if self._is_skill(event):
            return False
        return event.kind in ("tool_output", "message")

    def _recoverable_tokens(self, ctx: list[Event]) -> int:
        recent = self._recent_turns(ctx)
        return sum(e.tokens for e in ctx if self._prunable(e, recent))

    # ----------------------------------------------------------------- decide

    def should_compact(self, ctx: list[Event]) -> bool:
        """Vrai si le contexte depasse le trigger ET qu'assez est recuperable."""
        total = sum(e.tokens for e in ctx)
        if total <= self.cfg.compaction_trigger_tokens:
            return False
        return self._recoverable_tokens(ctx) > self.cfg.min_recoverable_tokens

    # --------------------------------------------------------- re-injection

    def _live_state(self, ctx: list[Event]) -> str:
        """Extrait VERBATIM l'etat vivant a re-injecter.

        - dernier goal user (dernier message role=user),
        - derniers diffs / fichiers (derniers tool_output de write_file/edit ou
          contenant un diff), heuristique simple sur le content,
        - integralite des `keep_recent_turns` derniers tours.
        """
        recent = self._recent_turns(ctx)
        parts: list[str] = []

        # Dernier goal user.
        last_user = None
        for e in ctx:
            if e.role == "user" and e.kind == "message":
                last_user = e
        if last_user is not None:
            parts.append(f"### Goal (dernier message user)\n{last_user.content}")

        # Derniers diffs / fichiers : tool_output hors tours recents (les recents
        # sont deja inclus integralement plus bas) qui ressemblent a des
        # modifications de fichiers.
        file_hits = [
            e
            for e in ctx
            if e.kind == "tool_output"
            and e.turn not in recent
            and (
                "diff" in (e.content or "").lower()
                or "--- " in (e.content or "")
                or "+++ " in (e.content or "")
                or "write_file" in (e.content or "")
                or "edit" in (e.content or "").lower()
            )
        ]
        if file_hits:
            last_file = file_hits[-1]
            parts.append(
                f"### Derniers diffs / fichiers (verbatim)\n{last_file.content}"
            )

        # Integralite des derniers tours.
        recent_events = [e for e in ctx if e.turn in recent]
        if recent_events:
            lines = [
                f"[t{e.turn}] {e.role}/{e.kind}: {e.content}" for e in recent_events
            ]
            parts.append("### Derniers tours (verbatim)\n" + "\n".join(lines))

        return "\n\n".join(parts)

    # ----------------------------------------------------------------- summary

    def _summary_prompt(self, ctx: list[Event]) -> str:
        """Construit le prompt de resume a partir du contexte a compacter."""
        transcript = "\n".join(
            f"[t{e.turn}] {e.role}/{e.kind}: {e.content}" for e in ctx
        )
        return (
            "Resume la session ci-dessous en 5 champs (gist) : Goal, "
            "Instructions, Decouvertes, Travail, Fichiers. Sois factuel et "
            "concis.\n\n--- TRANSCRIPT ---\n" + transcript
        )

    # ------------------------------------------------------------------ compact

    def compact(self, ctx: list[Event], turn: int) -> tuple[list[Event], int]:
        """Execute la compaction et retourne (nouveau model_context, summary_tokens).

        Etapes :
          1) PRUNE non-destructif des tool_output verbeux (hors 2 derniers tours
             et hors skill) via store.mark_compacted.
          2) RE-INJECTION verbatim de l'etat vivant (event role=system
             kind=reinjection).
          3) SUMMARY : 1 appel llm.generate -> event role=summary kind=summary.
        """
        recent = self._recent_turns(ctx)
        prunable = [e for e in ctx if self._prunable(e, recent)]

        # 2) RE-INJECTION (calculee AVANT le prune, sur le contexte complet).
        live = self._live_state(ctx)
        if live:
            self.store.append(turn, "system", "reinjection", live)

        # 3) SUMMARY (1 seul appel LLM).
        summary_text = self.llm.generate(
            self._summary_prompt(ctx),
            system="Tu es un compacteur de contexte. Produis un gist en 5 champs.",
        )
        summary_event = self.store.append(turn, "summary", "summary", summary_text)
        summary_tokens = count_tokens(summary_text)

        # 1) PRUNE non-destructif : on marque compactes les outputs verbeux en
        #    rattachant le summary comme reference.
        if prunable:
            self.store.mark_compacted(
                [e.id for e in prunable], summary_event.id
            )

        return self.store.model_context(), summary_tokens
