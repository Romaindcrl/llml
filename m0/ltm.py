"""Mémoire long-terme (LTM) — le corpus TEXTE qui sera gravé dans les poids.

C'est la SOURCE DE VÉRITÉ : un jeu de Q/R accumulé (jsonl). `/sleep` réentraîne un
LoRA FRAIS depuis la base sur tout ce corpus (replay) — l'approche qui scale (M3).

Alimenté par : (1) /remember (promotion manuelle d'un document), (2) l'auto-promotion
quand le contexte sature (le contenu libéré de la compaction est poussé ici).
"""

from __future__ import annotations

import json
import os

from m0 import d2l


class LTM:
    def __init__(self, path: str) -> None:
        self.path = path

    def all_qa(self) -> list[tuple[str, str]]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("q") and d.get("a"):
                    out.append((d["q"], d["a"]))
        return out

    def add_qa(self, pairs) -> int:
        """Ajoute des paires Q/R (dédupliquées). Retourne le nb réellement ajouté."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        seen = {(q.lower(), a.lower()) for q, a in self.all_qa()}
        added = 0
        with open(self.path, "a", encoding="utf-8") as f:
            for q, a in pairs:
                key = (q.lower(), a.lower())
                if key in seen:
                    continue
                seen.add(key)
                f.write(json.dumps({"q": q, "a": a}, ensure_ascii=False) + "\n")
                added += 1
        return added

    def add_document(self, text: str, generate_fn, n: int = 18) -> tuple[int, int]:
        """Extrait des Q/R groundées d'un document/texte et les ajoute. Retourne
        (ajoutés, extraits)."""
        qa = d2l.clean_and_balance(d2l.extract_qa(text, generate_fn, n=n), max_per_answer=2)
        return self.add_qa(qa), len(qa)

    def count(self) -> int:
        return len(self.all_qa())

    def clear(self) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass
