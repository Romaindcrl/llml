"""Bibliotheque de competences VERIFIEES (style Voyager, arXiv:2305.16291).

C'est la troisieme jambe de la memoire — apres le RAG (savoir volatil) et les poids
(savoir stable) : du CODE QUI A DEJA MARCHE. Une competence n'entre ici que si sa
solution a PASSE ses tests en execution reelle (jamais sur la foi du modele). A la
prochaine tache proche, elle est reinjectee en contexte : le systeme repart de ses
reussites au lieu de re-deriver de zero — exactement ce qu'un humain appelle
« l'experience ».

Persistance : JSONL append-only, une competence par ligne. Recherche : BM25-lite
(reutilise la tokenisation du module rag — lexical, zero dependance).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

from .rag import _tok


@dataclass
class Skill:
    name: str          # nom de la fonction/du pattern (ex: "parse_kebab_tags")
    description: str   # l'enonce que ce code a resolu
    code: str          # la solution VERIFIEE (a passe ses tests)
    topic: str = ""    # sujet d'etude d'origine
    tests: str = ""    # les tests qu'elle a passes (re-verifiables)
    created: float = field(default_factory=time.time)
    uses: int = 0      # nb de reinjections en contexte (utilite mesuree)


class SkillLibrary:
    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.skills: list[Skill] = []
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        self.skills.append(Skill(**{k: d[k] for k in d if k in Skill.__dataclass_fields__}))
                    except (json.JSONDecodeError, TypeError):
                        continue

    # ------------------------------------------------------------------ ecriture
    def add(self, name: str, description: str, code: str, *, topic: str = "",
            tests: str = "") -> Skill | None:
        """Ajoute une competence verifiee. Deduplique par code normalise (la meme
        solution retrouvee deux fois n'apporte rien). Retourne None si doublon."""
        norm = re.sub(r"\s+", " ", (code or "")).strip()
        if not norm or not (name or "").strip():
            return None
        for s in self.skills:
            if re.sub(r"\s+", " ", s.code).strip() == norm:
                return None
        skill = Skill(name=name.strip(), description=(description or "").strip(),
                      code=code.strip(), topic=topic, tests=(tests or "").strip())
        self.skills.append(skill)
        self._append(skill)
        return skill

    def _append(self, skill: Skill) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(skill), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ lecture
    def topk(self, query: str, k: int = 3) -> list[Skill]:
        """BM25-lite sur name+description+topic : les competences les plus proches
        de la tache courante."""
        if not self.skills:
            return []
        docs = [_tok(f"{s.name} {s.description} {s.topic}") for s in self.skills]
        n_docs = len(docs)
        avgdl = sum(len(d) for d in docs) / max(1, n_docs)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        qt = _tok(query)
        scored = []
        for i, d in enumerate(docs):
            tf = Counter(d)
            dl = len(d)
            s = sum(
                math.log(1 + (n_docs - df[w] + 0.5) / (df[w] + 0.5))
                * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * dl / avgdl))
                for w in qt if w in tf
            )
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [self.skills[i] for _, i in scored[:k]]

    def render_for_context(self, query: str, k: int = 2, cap_chars: int = 2400) -> str:
        """Rend les competences pertinentes pour injection en contexte de generation.
        Compte les usages (mesure d'utilite reelle de la bibliotheque)."""
        hits = self.topk(query, k=k)
        if not hits:
            return ""
        parts = ["### Solutions deja VERIFIEES sur des taches proches (reutilise-les) :"]
        used = 0
        for s in hits:
            block = f"# {s.name} — {s.description}\n{s.code}"
            if used + len(block) > cap_chars:
                break
            parts.append(block)
            used += len(block)
            s.uses += 1
        return "\n\n".join(parts) if len(parts) > 1 else ""

    def count(self) -> int:
        return len(self.skills)
