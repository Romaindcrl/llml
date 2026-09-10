# Lot 4 — Claim A : mémoire-poids (résultats)

Réplication publique du pilier **mémoire-poids** de LLML : un document externe est
internalisé via le pipeline `/sleep` (extraction de faits → LoRA replay, avec gate
d'acquisition held-out + rollback), puis on teste le **rappel closed-book**
(document RETIRÉ du contexte) sur ses questions à choix multiples.

- Modèle : `Qwen/Qwen2.5-7B-Instruct` (M1), 8-bit, greedy.
- Corpus : **QuALITY** (dev), 3 documents tirés seed=42 (CDC §4.2), QA natives
  (rédigées par des humains — indépendance forte, aucune génération).
- 4 configs sur les MÊMES questions : **C0** modèle nu · **C1** mémoire-poids
  (LoRA `/sleep`, closed-book) · **C3** RAG (passages récupérés) · **C4** plein
  contexte (document entier dans le prompt, borne haute honnête).

## Résultat poolé — 3 docs, 30 questions

| Config | Rappel |
|--------|--------|
| Modèle nu (C0)          | 19/30 = **63,3 %** |
| Mémoire-poids (C1)      | 19/30 = **63,3 %** |
| RAG (C3)                | 15/30 = **50,0 %** |
| Plein contexte (C4)     | 25/30 = **83,3 %** |

- **Gate d'acquisition** : 0,84 en moyenne, **committed 3/3** — le LoRA rappelle
  bien ~84 % de SES PROPRES faits extraits en held-out. La mécanique fonctionne.
- **Coût** (l'atout réel) : C1 = **309 caractères** de prompt vs C4 = **12 170**
  (**÷39**), latence 687 ms vs 886 ms. La mémoire-poids est très bon marché.

## Sous-ensemble DUR — questions que le modèle nu rate (le vrai test)

C0 sur QuALITY est déjà haut (beaucoup de MC devinables). Le test décisif : sur les
**11 questions où C0 échoue** (là où la mémoire DOIT aider), que récupère chaque config ?

| Config | Récupéré sur les 11 « dures » |
|--------|-------------------------------|
| Modèle nu (C0)      | 0/11 (par définition) |
| **Mémoire-poids (C1)** | **0/11 = 0 %** |
| RAG (C3)            | 3/11 = 27 % |
| Plein contexte (C4) | 9/11 = 82 % |

**La mémoire-poids récupère EXACTEMENT ZÉRO** des questions dures. Le plein
contexte en récupère 9, le RAG 3.

## Verdict des hypothèses pré-enregistrées (§4.2)

| Hypothèse | Attendu | Mesuré | Verdict |
|-----------|---------|--------|---------|
| C1 > C0 (largement) | oui | C1 = C0 (63,3 %) | ❌ **réfutée** |
| C1 ≥ 80 % de C4 | ≥ 0,80 | 0,633/0,833 = **0,76** | ❌ sous le seuil |
| Kill : C1 < 50 % de C4 | — | poolé 0,76 (pas de kill) ; **sous-ensemble dur 0 %** | ⚠️ kill sur le sous-ensemble pertinent |
| C1 bat C4 sur le coût | oui | ÷39 contexte, latence < | ✅ **confirmée** |

## Lecture honnête

Sur ce corpus, la mémoire-poids telle qu'implémentée (extraction de faits → LoRA)
**n'internalise pas le document de façon à répondre closed-book à ses questions** :
elle mémorise ses propres faits extraits (gate 84 %) mais ceux-ci ne recouvrent
pas les questions de compréhension de QuALITY — sur les questions dures, elle
n'apporte **rien** (0/11), là où le contexte (C4) et le RAG (C3) apportent.

**Ce que ça établit :** la mémoire-poids est un **stockage bon marché de faits
auto-extraits** (÷39 de contexte), pas un substitut du contexte pour le rappel de
contenu arbitraire d'un document.

**Caveat important (à ne pas surinterpréter) :** QuALITY teste la *compréhension/
inférence* sur des nouvelles littéraires ; le pipeline extrait des *spans
factuels*. Ce **désalignement** (extraire des faits ≠ répondre à des questions de
compréhension) explique probablement une partie du nul. Le volet **documents
techniques** de Claim A (§4.2, questions de *lookup factuel* — non encore exécuté)
serait un test plus favorable au *design* de cette mémoire ; c'est la prochaine
étape pour être complet et équitable. n=30 reste petit.

## Fichiers

- `memory_results.jsonl` — par doc : configs C0/C1/C3/C4 (accuracy, per_q, coût),
  gate `/sleep` (acquisition, committed), extraction.

Repro : `eval/lot4/prep_quality.py` (corpus) + `eval/lot4/run_memory.py` (ingest →
`/sleep` → éval 4 configs). Scoreboard : `eval/lot4/make_mem_scoreboard.py`.
