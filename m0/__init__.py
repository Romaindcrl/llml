"""m0 — Memoire par poids persistants, jalon TEXTE (M0).

Assistant agentique LOCAL mono-utilisateur. Zero poids / zero LoRA a ce jalon :
toute la "memoire" est textuelle (MEMORY.md + compaction de contexte).

Sous-modules :
  - config      : count_tokens (approximation stdlib) + Config + from_env
  - types       : dataclasses du domaine (Event, ToolCall, ToolResult, MemoryEntry, Metrics)
  - llm         : LLMClient (ABC), OllamaClient (httpx), MockClient (deterministe), make_client
  - store       : EventStore (SQLite append-only)
  - memory      : TextMemory (MEMORY.md versionne git)
  - detector    : TwoShotDetector (deux tirs : lessons + reerror fingerprint)
  - tools        : Tools confines a un workdir
  - compaction  : Compactor (prune + reinjection + summary)
  - agent       : Agent (boucle de tache, 3 SUT : baseline / bctrl / m0)
"""

__version__ = "0.1.0"

__all__ = [
    "config",
    "types",
    "llm",
]
