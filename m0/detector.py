"""Detecteur de lecons a deux tirs (two-shot).

  TIR 1 (proactif) : scan_for_lessons() repere des phrases candidates dans un
  message (impératif / negation / correction / emphase, en FR et EN). L'agent
  decide ensuite d'ecrire ou non en memoire.

  TIR 2 (reactif sur erreur) : observe_error() calcule une empreinte normalisee
  de l'erreur. Si une entree memoire active porte deja cette empreinte, c'est une
  RE-ERREUR : on remonte sa priorite a "high" et on retourne "reerror". Sinon on
  enregistre un candidat et on retourne "new".

Le detecteur s'appuie sur TextMemory (m0.memory) pour la persistance ; il est
importe paresseusement (lazy) pour ne pas creer de cycle d'import.
"""

from __future__ import annotations

import hashlib
import re

from .memory import TextMemory

# --------------------------------------------------------------------------- #
# TIR 1 : detection de lecons dans un message (regex FR + EN).
# --------------------------------------------------------------------------- #

# Flags lexicaux declenchant une candidature de lecon. On couvre :
#   - imperatif / regle absolue : toujours, jamais, never, always, must, do not
#   - negation / correction     : ne pas, en fait non, corrige, actually no, fix
#   - emphase / attention       : attention, important, careful, note
_LESSON_FLAGS = [
    # FR
    r"toujours",
    r"jamais",
    r"attention",
    r"en\s+fait\s+non",
    r"ne\s+(?:pas|jamais|plus)\b",
    r"surtout\s+(?:pas|ne)",
    r"il\s+faut",
    r"corrige[rz]?",
    r"important",
    r"erreur",
    r"évite[rz]?",  # evite / eviter / evitez (avec accent)
    r"evite[rz]?",
    r"rappel",
    # EN
    r"never",
    r"always",
    r"must(?:\s+not)?",
    r"do\s*n'?t",
    r"do\s+not",
    r"actually\s+no",
    r"careful",
    r"warning",
    r"note\s+that",
    r"remember",
    r"fix(?:es|ed|ing)?",
    r"correct(?:ion|s)?",
]

_LESSON_RE = re.compile("|".join(_LESSON_FLAGS), re.IGNORECASE)

# Decoupage en phrases : sur ponctuation forte ou saut de ligne.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def scan_for_lessons(text: str) -> list[str]:
    """Repere les phrases candidates a une lecon dans `text`.

    Retourne la liste des phrases (nettoyees) contenant au moins un flag
    lexical FR/EN. Liste vide si rien. L'ordre d'apparition est preserve et
    les doublons exacts sont elimines.
    """
    if not text or not text.strip():
        return []

    candidates: list[str] = []
    seen: set[str] = set()
    for raw in _SENTENCE_SPLIT_RE.split(text):
        sentence = raw.strip()
        if not sentence:
            continue
        if _LESSON_RE.search(sentence):
            if sentence not in seen:
                seen.add(sentence)
                candidates.append(sentence)
    return candidates


# --------------------------------------------------------------------------- #
# Empreinte d'erreur : normalisation puis sha1 court.
# --------------------------------------------------------------------------- #

# Ordre des normalisations : du plus specifique au plus generique pour eviter
# qu'un motif large mange un motif precis (ex: hex 0x avant les nombres nus).
_NORM_HEX0X_RE = re.compile(r"0x[0-9a-fA-F]+")
_NORM_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_NORM_TIME_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b")
# Chemins absolus POSIX (/a/b/c) et Windows (C:\a\b). On capture le chemin entier.
_NORM_PATH_POSIX_RE = re.compile(r"(?<![\w./])/(?:[^\s/]+/)*[^\s/]+")
_NORM_PATH_WIN_RE = re.compile(r"[A-Za-z]:\\(?:[^\s\\]+\\?)*")
# Longues sequences hex nues (sha, uuid sans tirets, adresses memoire).
_NORM_HEXLONG_RE = re.compile(r"\b[0-9a-fA-F]{8,}\b")
_NORM_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_NORM_WS_RE = re.compile(r"\s+")


def error_fingerprint(text: str) -> str:
    """Empreinte stable d'une erreur : normalise puis sha1 court (12 hex).

    Normalisation (pour que deux occurrences "de la meme erreur" coincident
    malgre des details variables) :
      - retire timestamps, heures,
      - retire adresses 0x et longues sequences hex (hashes, uuid, @0x...),
      - retire chemins absolus POSIX et Windows,
      - retire les nombres,
      - normalise les espaces et la casse.
    """
    norm = text or ""
    norm = _NORM_TIMESTAMP_RE.sub("<TS>", norm)
    norm = _NORM_TIME_RE.sub("<TIME>", norm)
    norm = _NORM_HEX0X_RE.sub("<HEX>", norm)
    norm = _NORM_PATH_WIN_RE.sub("<PATH>", norm)
    norm = _NORM_PATH_POSIX_RE.sub("<PATH>", norm)
    norm = _NORM_HEXLONG_RE.sub("<HEX>", norm)
    norm = _NORM_NUMBER_RE.sub("<N>", norm)
    norm = _NORM_WS_RE.sub(" ", norm).strip().lower()

    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()
    return digest[:12]


# --------------------------------------------------------------------------- #
# Detecteur two-shot.
# --------------------------------------------------------------------------- #


class TwoShotDetector:
    """Detecteur a deux tirs adosse a une memoire textuelle.

    TIR 1 : observe_message() -> candidats lecon (regex). N'ecrit rien : l'agent
            decide.
    TIR 2 : observe_error() -> "reerror" si l'empreinte est deja active en
            memoire (et alors remontee de priorite a high), sinon "new" apres
            enregistrement d'un candidat porteur de l'empreinte.
    """

    def __init__(self, memory: TextMemory) -> None:
        self.memory = memory

    def observe_message(self, text: str) -> list[str]:
        """TIR 1 : retourne les phrases candidates a une lecon (sans ecrire)."""
        return scan_for_lessons(text)

    def observe_error(self, error_text: str, turn: int) -> str:
        """TIR 2 : classe une erreur en 'reerror' ou 'new'.

        - Calcule l'empreinte normalisee de `error_text`.
        - Si une entree memoire ACTIVE porte cette empreinte : c'est une
          repetition -> on remonte sa priorite a "high" et on retourne
          "reerror".
        - Sinon : on enregistre un candidat (tag 'error', empreinte attachee)
          et on retourne "new". L'enregistrement peut etre refuse par la memoire
          (cap_tokens depasse) ; on retourne quand meme "new" puisque ce n'est
          pas une re-erreur connue.
        """
        fp = error_fingerprint(error_text)
        existing = self.memory.find_by_fingerprint(fp)
        if existing is not None:
            # Re-erreur : on durcit la priorite a high (best-effort).
            self._upgrade_priority_high(existing)
            return "reerror"

        # Premiere occurrence : on memorise un candidat porteur de l'empreinte.
        snippet = (error_text or "").strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + " ..."
        lesson = f"[erreur tour {turn}] {snippet}" if snippet else f"[erreur tour {turn}]"
        self.memory.add(
            lesson,
            tags=["error"],
            priority="normal",
            fingerprint=fp,
        )
        return "new"

    def _upgrade_priority_high(self, entry) -> None:
        """Remonte la priorite d'une entree a "high" (best-effort).

        TextMemory etant append-friendly via MEMORY.md, on essaie d'abord une
        methode dediee si elle existe (set_priority), sinon on mute l'objet en
        memoire et on demande une re-persistance si l'API l'expose. On ne plante
        jamais : l'upgrade est un bonus, pas un invariant dur du detecteur.
        """
        if getattr(entry, "priority", None) == "high":
            return
        # 1) API dediee si la memoire en fournit une.
        setter = getattr(self.memory, "set_priority", None)
        if callable(setter):
            try:
                setter(entry.id, "high")
                return
            except Exception:
                pass
        # 2) Fallback : mutation directe + re-persistance optionnelle.
        try:
            entry.priority = "high"
        except Exception:
            return
        for meth_name in ("persist", "save", "_flush", "_persist"):
            meth = getattr(self.memory, meth_name, None)
            if callable(meth):
                try:
                    meth()
                except Exception:
                    pass
                break
