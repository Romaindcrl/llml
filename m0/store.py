"""Stockage append-only des evenements de conversation (SQLite, mode WAL).

Principes :
  - APPEND-ONLY : on n'UPDATE JAMAIS le content d'un event. La seule mutation
    autorisee est de positionner compacted_at (marquage de prune lors d'une
    compaction). La donnee reste donc presente -> non-destructif / rewindable.
  - WAL : active uniquement pour une base fichier (inutile et non supporte de
    facon utile pour ":memory:").
  - tokens calcule a l'insertion via config.count_tokens ; created_at = ISO UTC.

Le model_context() ne renvoie que les events NON compactes (compacted_at IS NULL),
en ordre chronologique. pruned_events() renvoie l'inverse (preuve de
non-destructivite, rewind possible).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .config import count_tokens
from .types import Event


def _now_iso() -> str:
    """Horodatage ISO 8601 en UTC (deterministe quant au format)."""
    return datetime.now(timezone.utc).isoformat()


class EventStore:
    """Journal append-only des events, persiste dans SQLite."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # check_same_thread=False : usage mono-thread cote agent, mais on evite
        # une exception si l'objet est passe entre threads (bench/REPL).
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # WAL seulement pour une base fichier (":memory:" n'en beneficie pas).
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                turn         INTEGER NOT NULL,
                role         TEXT    NOT NULL,
                kind         TEXT    NOT NULL,
                content      TEXT    NOT NULL,
                tokens       INTEGER NOT NULL,
                created_at   TEXT    NOT NULL,
                compacted_at TEXT
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"],
            turn=row["turn"],
            role=row["role"],
            kind=row["kind"],
            content=row["content"],
            tokens=row["tokens"],
            created_at=row["created_at"],
            compacted_at=row["compacted_at"],
        )

    # ------------------------------------------------------------------- ecriture

    def append(self, turn: int, role: str, kind: str, content: str) -> Event:
        """Ajoute un event (append-only). Calcule tokens et created_at."""
        tokens = count_tokens(content)
        created_at = _now_iso()
        cur = self._conn.execute(
            """
            INSERT INTO events (turn, role, kind, content, tokens, created_at, compacted_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL)
            """,
            (turn, role, kind, content, tokens, created_at),
        )
        self._conn.commit()
        event_id = int(cur.lastrowid)
        return Event(
            id=event_id,
            turn=turn,
            role=role,
            kind=kind,
            content=content,
            tokens=tokens,
            created_at=created_at,
            compacted_at=None,
        )

    def mark_compacted(self, event_ids: list[int], summary_event_id: int | None) -> None:
        """Marque des events comme compactes (positionne compacted_at).

        NON-DESTRUCTIF : seul compacted_at est mis a jour ; le content reste
        intact et reste interrogeable via pruned_events()/all_events().

        On ne marque jamais le summary lui-meme (summary_event_id) : il doit
        rester dans le model_context comme substitut des events prunes.
        """
        if not event_ids:
            return
        compacted_at = _now_iso()
        ids = [i for i in event_ids if i != summary_event_id]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(
            f"""
            UPDATE events
               SET compacted_at = ?
             WHERE id IN ({placeholders})
               AND compacted_at IS NULL
            """,
            (compacted_at, *ids),
        )
        self._conn.commit()

    # ------------------------------------------------------------------- lecture

    def model_context(self) -> list[Event]:
        """Events NON compactes, en ordre chronologique (id croissant)."""
        rows = self._conn.execute(
            """
            SELECT id, turn, role, kind, content, tokens, created_at, compacted_at
              FROM events
             WHERE compacted_at IS NULL
             ORDER BY id ASC
            """
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def pruned_events(self) -> list[Event]:
        """Events compactes (compacted_at NOT NULL), ordre chronologique.

        Sert de preuve de non-destructivite et permet un rewind : la donnee
        prunee reste recuperable.
        """
        rows = self._conn.execute(
            """
            SELECT id, turn, role, kind, content, tokens, created_at, compacted_at
              FROM events
             WHERE compacted_at IS NOT NULL
             ORDER BY id ASC
            """
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def all_events(self) -> list[Event]:
        """Tous les events (compactes ou non), ordre chronologique."""
        rows = self._conn.execute(
            """
            SELECT id, turn, role, kind, content, tokens, created_at, compacted_at
              FROM events
             ORDER BY id ASC
            """
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    # ------------------------------------------------------------------- cloture

    def close(self) -> None:
        """Ferme la connexion SQLite (best-effort)."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass
