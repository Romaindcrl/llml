"""Éval du routeur : heuristique mots-clés vs classifieur LLM, sur un set étiqueté
incluant des formulations AMBIGUËS (où les mots-clés se trompent)."""

from __future__ import annotations

import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0 import rag  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402

# (requête, label attendu) — inclut des cas durs pour les mots-clés
LABELED = [
    ("Quel est le port par defaut de Helios ?", "recall"),
    ("Combien vaut MAX_LAYERS ?", "recall"),
    ("Qui a cree le langage Rust ?", "recall"),
    ("Quelle est la licence du projet ?", "recall"),
    ("A quelle altitude orbite la station ?", "recall"),
    ("Ecris un script qui charge une image et l'exporte.", "generation"),
    ("Genere une fonction de fusion de canvas.", "generation"),
    ("Implemente un pipeline de rendu complet.", "generation"),
    ("Redige un court resume de la doc.", "generation"),
    # --- cas AMBIGUS (les mots-clés se trompent souvent) ---
    ("Montre comment fusionner deux canvas.", "generation"),       # pas de 'écris/code'
    ("Donne un exemple d'utilisation de glyph.load.", "generation"),  # 'exemple d'utilisation'
    ("Comment ferais-tu pour exporter en haute resolution ?", "generation"),
    ("Rappelle-moi le dpi par defaut.", "recall"),                 # 'rappelle' != generation
    ("Peux-tu me coder la rotation ?", "generation"),
]


def main():
    cfg = Config.from_env(); cfg.backend = "mlx"
    llm = make_client(cfg)
    kw_ok = llm_ok = 0
    print(f"{'requête':52s} | attendu     | mots-clés | LLM")
    for q, gold in LABELED:
        kw = rag.classify(q, None)                 # mots-clés seuls
        ll = rag.classify(q, llm.generate)         # classifieur LLM
        kw_ok += (kw == gold); llm_ok += (ll == gold)
        mark = lambda x: "✓" if x == gold else "✗"
        print(f"{q[:52]:52s} | {gold:10s} | {kw:9s}{mark(kw)} | {ll}{mark(ll)}")
    n = len(LABELED)
    print(f"\nPrécision routeur : mots-clés = {kw_ok}/{n} ({kw_ok/n:.0%}) | "
          f"LLM = {llm_ok}/{n} ({llm_ok/n:.0%})")


if __name__ == "__main__":
    main()
