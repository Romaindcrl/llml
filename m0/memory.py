"""Memoire textuelle persistante (MEMORY.md) versionnee par git (best-effort).

La memoire est un fichier markdown LISIBLE : une section par entree avec un
front-matter par bloc. Au demarrage on RECHARGE le fichier s'il existe, ce qui
rend la memoire persistante entre executions.

Contrat (voir specification M0) :
  add(text, tags, priority="normal", fingerprint=None) -> MemoryEntry | None
      None si l'ajout ferait depasser cap_tokens le RENDU ACTIF (jamais de
      troncature silencieuse) -> on log "memory full".
  expire(entry_id) -> bool                 # active -> expired, puis commit
  active_entries() / all_entries()
  render_for_injection() -> str            # SEULEMENT les actives, sous cap_tokens
  find_by_fingerprint(fp) -> MemoryEntry | None   # parmi les actives

Chaque mutation (add/expire) tente 1 commit git (best-effort : si git est absent
ou echoue, on ne plante pas).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from datetime import datetime, timezone

from .config import count_tokens
from .types import MemoryEntry

logger = logging.getLogger("m0.memory")

# Delimiteurs du format de fichier. Chaque entree est un bloc :
#   <!-- m0:entry -->
#   ---
#   id: ...
#   date: ...
#   tags: a, b
#   priority: normal
#   status: active
#   fingerprint: ...        (ligne omise si None)
#   ---
#   <texte libre, potentiellement multi-lignes>
#   <!-- m0:end -->
_ENTRY_START = "<!-- m0:entry -->"
_ENTRY_END = "<!-- m0:end -->"
_ENTRY_RE = re.compile(
    re.escape(_ENTRY_START) + r"\s*\n(?P<body>.*?)\n?" + re.escape(_ENTRY_END),
    re.DOTALL,
)
_FRONT_RE = re.compile(r"^---\s*\n(?P<front>.*?)\n---\s*\n?(?P<text>.*)$", re.DOTALL)

_FILE_HEADER = "# MEMORY\n\nMemoire textuelle persistante du systeme M0 (jalon TEXTE).\n"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class TextMemory:
    """Memoire markdown persistante avec plafond de tokens a l'injection."""

    def __init__(self, path: str, repo_dir: str, cap_tokens: int) -> None:
        self.path = path
        self.repo_dir = repo_dir
        self.cap_tokens = cap_tokens
        self._entries: list[MemoryEntry] = []
        self._counter = 0  # pour generer des ids stables et croissants

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._git_init_best_effort()
        self._load()

    # ------------------------------------------------------------------ git
    def _git(self, *args: str) -> bool:
        """Lance une commande git dans repo_dir. Best-effort : renvoie False sans
        lever si git est absent ou echoue."""
        try:
            subprocess.run(
                ["git", *args],
                cwd=self.repo_dir,
                check=True,
                capture_output=True,
                timeout=30,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    def _git_init_best_effort(self) -> None:
        if not os.path.isdir(self.repo_dir):
            return
        if os.path.isdir(os.path.join(self.repo_dir, ".git")):
            return
        self._git("init")

    def _git_commit(self, message: str) -> None:
        """1 commit par mutation (best-effort)."""
        if not self._git("add", self.path):
            return
        self._git("commit", "-m", message)

    # --------------------------------------------------------------- parsing
    def _load(self) -> None:
        """Recharge les entrees depuis le fichier MEMORY.md s'il existe."""
        self._entries = []
        self._counter = 0
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError:
            return

        for m in _ENTRY_RE.finditer(raw):
            entry = self._parse_block(m.group("body"))
            if entry is not None:
                self._entries.append(entry)

        # Le compteur reprend au-dela du plus grand suffixe numerique rencontre.
        for e in self._entries:
            n = _id_suffix(e.id)
            if n > self._counter:
                self._counter = n

    @staticmethod
    def _parse_block(body: str) -> MemoryEntry | None:
        fm = _FRONT_RE.match(body)
        if not fm:
            return None
        front = fm.group("front")
        text = fm.group("text").rstrip("\n")

        fields: dict[str, str] = {}
        for line in front.splitlines():
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            fields[key.strip()] = val.strip()

        entry_id = fields.get("id")
        if not entry_id:
            return None

        tags_raw = fields.get("tags", "")
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

        fp = fields.get("fingerprint")
        if fp in ("", "none", "None"):
            fp = None

        return MemoryEntry(
            id=entry_id,
            date=fields.get("date", _today()),
            tags=tags,
            priority=fields.get("priority", "normal") or "normal",
            status=fields.get("status", "active") or "active",
            fingerprint=fp,
            text=text,
        )

    # ---------------------------------------------------------- serialisation
    @staticmethod
    def _render_block(e: MemoryEntry) -> str:
        lines = [
            _ENTRY_START,
            "---",
            f"id: {e.id}",
            f"date: {e.date}",
            f"tags: {', '.join(e.tags)}",
            f"priority: {e.priority}",
            f"status: {e.status}",
        ]
        if e.fingerprint:
            lines.append(f"fingerprint: {e.fingerprint}")
        lines.append("---")
        lines.append(e.text)
        lines.append(_ENTRY_END)
        return "\n".join(lines)

    def _persist(self) -> None:
        """Reecrit l'integralite du fichier MEMORY.md (lisible)."""
        parts = [_FILE_HEADER]
        for e in self._entries:
            parts.append(self._render_block(e))
        content = "\n\n".join(parts) + "\n"
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(content)

    # ------------------------------------------------------------- mutations
    def _new_id(self) -> str:
        self._counter += 1
        return f"mem-{self._counter:04d}"

    def add(
        self,
        text: str,
        tags: list[str],
        priority: str = "normal",
        fingerprint: str | None = None,
    ) -> MemoryEntry | None:
        """Ajoute une entree active.

        Renvoie None (et log "memory full") si l'entree ferait depasser cap_tokens
        au RENDU ACTIF -> on ne tronque jamais en silence.
        """
        candidate = MemoryEntry(
            id=self._new_id(),
            date=_today(),
            tags=list(tags),
            priority=priority,
            status="active",
            fingerprint=fingerprint,
            text=text,
        )

        # On simule le rendu actif AVEC la candidate pour verifier le plafond.
        # _render_active s'arrete AVANT de depasser cap_tokens : si la candidate
        # n'apparait pas dans le rendu, c'est qu'elle ne tient pas -> on refuse.
        # (Ce test attrape aussi le cas d'une seule entree trop grosse, ou le
        #  rendu retournerait vide.)
        projected = self._render_active([*self._active(), candidate])
        marker = self._format_entry_for_injection(candidate)
        if marker not in projected:
            logger.warning(
                "memory full: ajout refuse (cap_tokens=%d depasse), entree non ecrite: %r",
                self.cap_tokens,
                text[:80],
            )
            # On a consomme un id du compteur : on le rend pour rester stable.
            self._counter -= 1
            return None

        self._entries.append(candidate)
        self._persist()
        self._git_commit(f"memory: add {candidate.id}")
        return candidate

    def expire(self, entry_id: str) -> bool:
        """Passe une entree active -> expired. Renvoie False si introuvable/deja
        expiree. Commit best-effort en cas de changement."""
        for e in self._entries:
            if e.id == entry_id and e.status == "active":
                e.status = "expired"
                self._persist()
                self._git_commit(f"memory: expire {entry_id}")
                return True
        return False

    # --------------------------------------------------------------- lectures
    def _active(self) -> list[MemoryEntry]:
        return [e for e in self._entries if e.status == "active"]

    def active_entries(self) -> list[MemoryEntry]:
        return list(self._active())

    def all_entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def find_by_fingerprint(self, fp: str) -> MemoryEntry | None:
        """Cherche une entree ACTIVE par fingerprint."""
        if not fp:
            return None
        for e in self._active():
            if e.fingerprint == fp:
                return e
        return None

    # --------------------------------------------------------------- injection
    @staticmethod
    def _format_entry_for_injection(e: MemoryEntry) -> str:
        tag_str = f" [tags: {', '.join(e.tags)}]" if e.tags else ""
        prio = " (priorite: high)" if e.priority == "high" else ""
        return f"- {e.text}{tag_str}{prio}"

    def _render_active(self, entries: list[MemoryEntry]) -> str:
        """Rend une liste d'entrees actives en respectant cap_tokens.

        On ajoute les entrees une par une et on s'arrete AVANT de depasser le
        plafond (jamais au-dela). Les entrees high-priority passent d'abord.
        """
        actives = [e for e in entries if e.status == "active"]
        if not actives:
            return ""

        ordered = sorted(actives, key=lambda e: (e.priority != "high",))

        header = "## Memoire persistante\n"
        body_lines: list[str] = []
        for e in ordered:
            line = self._format_entry_for_injection(e)
            tentative = header + "\n".join([*body_lines, line])
            if count_tokens(tentative) > self.cap_tokens:
                break
            body_lines.append(line)

        if not body_lines:
            return ""
        return header + "\n".join(body_lines)

    def render_for_injection(self) -> str:
        """Concatene SEULEMENT les entrees actives, dans la limite cap_tokens."""
        return self._render_active(self._active())


def _id_suffix(entry_id: str) -> int:
    """Extrait le suffixe numerique d'un id 'mem-0007' -> 7, sinon 0."""
    m = re.search(r"(\d+)\s*$", entry_id or "")
    return int(m.group(1)) if m else 0
