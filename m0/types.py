"""Dataclasses du domaine M0. AUCUNE logique (sauf la property derivee Metrics).

Conventions :
  Event.role  in {user, assistant, tool, system, summary}
  Event.kind  in {message, tool_call, tool_output, summary, reinjection}
  MemoryEntry.priority in {normal, high}
  MemoryEntry.status   in {active, expired}
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Event:
    """Un evenement append-only du contexte de conversation.

    compacted_at != None signifie que l'event a ete prune lors d'une compaction :
    la donnee reste presente (non-destructif) mais sort du model_context.
    """

    id: int
    turn: int
    role: str  # user | assistant | tool | system | summary
    kind: str  # message | tool_call | tool_output | summary | reinjection
    content: str
    tokens: int
    created_at: str
    compacted_at: str | None = None


@dataclass
class ToolCall:
    """Un appel d'outil demande par le modele."""

    name: str
    args: dict


@dataclass
class ToolResult:
    """Le resultat d'execution d'un outil."""

    name: str
    ok: bool
    output: str


@dataclass
class MemoryEntry:
    """Une entree de la memoire textuelle persistante (MEMORY.md)."""

    id: str
    date: str
    tags: list[str]
    priority: str  # normal | high
    status: str  # active | expired
    fingerprint: str | None
    text: str


@dataclass
class Metrics:
    """Mesures d'une execution de tache (par SUT)."""

    success: bool
    turns: int
    context_tokens: int
    summary_tokens: int
    net_tokens: int
    reerrors: int
    injection_total: int
    injection_relevant: int

    @property
    def injection_precision(self) -> float | None:
        """Precision d'injection = relevant / total, ou None si rien injecte."""
        if self.injection_total == 0:
            return None
        return self.injection_relevant / self.injection_total
