"""RAG local (BM25) sur le corpus de documents BRUTS.

Le benchmark montre que GÉNÉRER (code/style/spec) veut la référence EN CONTEXTE, pas dans
les poids. Le RAG sert donc la voie 'génération' ; les poids (LoRA) servent le RAPPEL de
faits. Lexical (BM25) : zéro dépendance, fort quand la requête partage des mots avec le doc.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter


# Mots vides FR+EN : retirés avant BM25 pour que la pertinence repose sur les mots de
# contenu (sinon des chunks non pertinents remontent via 'la/par/the/of' = bruit).
_STOP = frozenset((
    "le la les un une des de du au aux et ou ni mais donc or car que qui quoi dont ou "
    "a as à en dans par pour sur sous avec sans vers chez entre est sont ete etre il elle "
    "ils elles on ce cet cette ces son sa ses leur leurs se ne pas plus tout toute tous "
    "the a an of to in for on with by is are be at as it its this that these those and or "
    "qui qu d l n s je tu nous vous"
).split())


def _tok(s: str) -> list[str]:
    return [w for w in re.findall(r"\w+", (s or "").lower()) if w not in _STOP and len(w) > 1]


def chunk_text(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"\n+|(?<=[.!?])\s+", text or "") if len(s.strip()) > 12]


class RAG:
    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.chunks: list[str] = []
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.chunks = [ln.rstrip("\n") for ln in f if ln.strip()]
        self._reindex()

    def _reindex(self) -> None:
        self.toks = [_tok(c) for c in self.chunks]
        self.N = len(self.chunks)
        self.avgdl = sum(len(t) for t in self.toks) / max(1, self.N)
        self.df = Counter()
        for t in self.toks:
            self.df.update(set(t))

    def add_document(self, text: str) -> int:
        new = [c for c in chunk_text(text) if c not in self.chunks]
        if not new:
            return 0
        self.chunks += new
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for c in new:
                    f.write(c.replace("\n", " ") + "\n")
        self._reindex()
        return len(new)

    def _idf(self, w: str) -> float:
        n = self.df.get(w, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def topk(self, query: str, k: int = 4) -> list[str]:
        if not self.chunks:
            return []
        qt = _tok(query)
        scored = []
        for i, t in enumerate(self.toks):
            tf = Counter(t)
            dl = len(t)
            s = sum(self._idf(w) * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * dl / self.avgdl))
                    for w in qt if w in tf)
            scored.append((s, i))
        scored.sort(reverse=True)
        return [self.chunks[i] for _, i in scored[:k]]

    def count(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks = []
        self._reindex()
        if self.path:
            try:
                os.remove(self.path)
            except OSError:
                pass


# --- Routeur : décider si une requête est de la GÉNÉRATION (→ base+RAG) ou du RAPPEL (→ poids)
_GEN_MARKERS = (
    "ecris", "écris", "ecrire", "écrire", "genere", "génère", "generer", "générer",
    "code", "script", "implemente", "implémente", "fonction", "classe", "programme",
    "write", "generate", "implement", "function", "snippet", "redige", "rédige",
    "cree un", "crée un", "create a", "fais un",
    # tournures indirectes (rattrapées via l'éval routeur) :
    "montre comment", "comment ferais", "exemple d", "donne un exemple",
    "comment utiliser", "comment faire pour", "comment on fait", "coder",
)


def is_generation(text: str) -> bool:
    """Heuristique mots-clés : la requête demande-t-elle de PRODUIRE du code/texte ?"""
    t = (text or "").lower()
    return any(m in t for m in _GEN_MARKERS)


def classify(query: str, generate_fn=None) -> str:
    """Routeur AFFINÉ. Retourne 'generation' ou 'recall'.
    - avec generate_fn : classifieur LLM zero-shot (robuste aux formulations indirectes
      type 'montre comment…', 'donne un exemple…') ;
    - sinon (ou si échec) : repli sur l'heuristique mots-clés is_generation().
    """
    if generate_fn is not None:
        try:
            out = generate_fn(
                "Classe la requête utilisateur en UN seul mot.\n"
                "- GENERATION : elle demande de PRODUIRE/ÉCRIRE quelque chose (code, script, "
                "fonction, exemple d'utilisation, texte à rédiger).\n"
                "- RAPPEL : elle demande une INFORMATION ou un FAIT (réponse courte).\n"
                f"Requête : {query}\nClasse (un seul mot, GENERATION ou RAPPEL) :",
                None,
            ).strip().lower()
            if "genera" in out or "génér" in out:
                return "generation"
            if "rappel" in out or "recall" in out or "fact" in out or "info" in out:
                return "recall"
        except Exception:  # noqa: BLE001
            pass
    return "generation" if is_generation(query) else "recall"
