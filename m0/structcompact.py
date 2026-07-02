"""[STATUT : ⛔ PROTOTYPE RÉFUTÉ sur données réelles — conservé comme expérience négative documentée.
 Synthétique : 100% vs 0% (artefact). HotpotQA : 20% ≈ baselines. Vrai code VIS : 38% < résumé
 générique 75%. Un bon résumé bat l'extraction de graphe. NE PAS recâbler dans serve.py.]

Compaction de contexte STRUCTURE-PRÉSERVANTE (prototype).

Contrairement à un résumé générique (qui perd les relations) ou au token-pruning (qui jette les
tokens rares), ce compacteur extrait explicitement le GRAPHE DE DÉPENDANCES + les faits/contraintes
essentiels, sous une forme compacte et traçable. Cible : le résidu structuré d'une tâche agentique
(transcript, sorties d'outils) re-lu à chaque tour → on le compacte UNE fois, on le relit N fois.

Prototype « extraction LLM » : une passe autorégressive. À remplacer ensuite par un dLLM (compression
en une passe bidirectionnelle = même résultat, plus rapide — cf. verdict « diffusion = efficacité »).
"""

from __future__ import annotations

from typing import Callable

_PROMPT = """Tu es un COMPACTEUR DE CONTEXTE qui PRÉSERVE LA STRUCTURE.

À partir des notes ci-dessous, produis un résumé TÉLÉGRAPHIQUE qui conserve tout ce qui permet de
RAISONNER en CHAÎNANT plusieurs informations (multi-hop) :

1) RELATIONS — pour chaque lien entre deux entités (dépendance, appartenance, localisation, rôle,
   causalité, « X est/fait/contient/dépend de/se situe dans Y »), écris UNE ligne :
       X — relation → Y
   en utilisant les VRAIS NOMS EXACTS des entités tels qu'ils apparaissent. N'écris JAMAIS de
   lettres génériques (« A », « B »). N'oublie AUCUNE relation utile au raisonnement.
2) FAITS — valeurs, dates, attributs essentiels, une par ligne, préfixés « Fait: ».

Ignore la prose de remplissage, le bavardage, les détails non pertinents. Sois le plus court
possible SANS perdre une seule relation ou un fait clé nécessaire pour répondre.

NOTES :
{context}

SORTIE (lignes « X — relation → Y » avec les vrais noms, puis « Fait: … ») :"""


def compact_structured(context: str, generate: Callable[[str, str | None], str]) -> str:
    """Compacte `context` en préservant le graphe de dépendances. Retourne l'artefact compact."""
    return generate(_PROMPT.format(context=context), None).strip()
