# Lot 4b — Claim A, volet ÉQUITABLE : mémoire-poids sur du factuel post-cutoff

**Question.** Le volet QuALITY (Lot 4) a conclu que la mémoire-poids ne récupère
**rien** sur le sous-ensemble dur (0/11). Objection légitime : QuALITY teste la
*compréhension* d'une nouvelle, alors que le pipeline `/sleep` extrait des **faits
courts**. Il y avait donc un désalignement tâche↔design. Ce lot corrige ça en
plaçant la mémoire dans **le régime le plus favorable à son design** :

- **2 documents techniques réels post-2024** (changelogs `uv` 0.11.x et `ruff`
  0.15.x, dates/versions de juin 2026) — inconnus du modèle nu (cutoff 2023).
- **17 questions = lookups factuels purs** : une date, un numéro de version, un
  code de règle, un identifiant court. Exactement ce que `/sleep` extrait.
- Scoring court : le span attendu apparaît-il dans la réponse (`sa_match`).

Modèle **Qwen2.5-7B-Instruct** 8-bit. Recette `/sleep` identique au volet QuALITY
(gate d'acquisition 0.45, rollback). 4 configs closed-book :
`C0` base nue · `C1` mémoire-poids · `C3` RAG · `C4` plein contexte.

## Résultats poolés (17 QA)

| Config | Score | % |
|--------|:-----:|:---:|
| **C0** base nue | 0/17 | **0,0 %** |
| **C1** mémoire-poids (`/sleep`+LoRA) | 2/17 | **11,8 %** |
| **C3** RAG (top-4 chunks) | 8/17 | **47,1 %** |
| **C4** plein contexte | 16/17 | **94,1 %** |

Par document :

| Doc | QA | C0 | C1 | C3 | C4 | gate `/sleep` |
|-----|:--:|:--:|:--:|:--:|:--:|:--:|
| uv (changelog 0.11.x) | 10 | 0,00 | 0,10 | 0,50 | 0,90 | 0,923 (commit) |
| ruff (changelog 0.15.x) | 7 | 0,00 | 0,143 | 0,429 | 1,00 | 1,0 (commit) |

**Sous-ensemble dur** = questions que le nu rate. Ici `C0 = 0/17`, donc le
sous-ensemble dur **est la totalité des 17 questions**. C'est le contrôle idéal :
le nu ne sait rien (post-cutoff confirmé), toute réussite est un vrai gain.

## Verdict : la mémoire-poids **ne recolle pas**, même sur le factuel

- Le `/sleep` **s'engage** (gate 0.92 et 1.0, adaptateurs commités) : le pipeline
  fonctionne mécaniquement. Ce n'est pas un échec d'entraînement.
- Mais il ne récupère que **2/17 (11,8 %)**, très loin de RAG (47 %) et du plein
  contexte (94 %) pour un coût mémoire comparable à l'inférence.
- **Les 2 seuls succès sont tous les deux `2026-06-18`** — une date qui apparaît
  dans **les deux** documents (uv 0.11.22 et ruff 0.15.18 sortis ce jour-là). La
  mémoire n'a donc pas *rappelé* un fait précis : elle a **collapsé sur le token
  de date le plus fréquent** du corpus. C'est un artefact statistique, pas de la
  récupération.
- Sur tout le reste, elle **hallucine des quasi-réponses** plausibles mais fausses :
  `0.6.3` → `0.13.0` · `SARIF` → `json` · `TY and RUFF` → `UV_BIN_PATH` ·
  `pre-commit-uv` → `Meshmixer` · `3.15.0b3` → `3.6.0`.

## Comparaison honnête au volet QuALITY

| Volet | Régime | Mémoire (sous-ens. dur) |
|-------|--------|:-----------------------:|
| Lot 4 — QuALITY | compréhension de nouvelle | **0/11** |
| Lot 4b — tech docs | lookups factuels (design-friendly) | **2/17 (dont 2 coïncidences de date)** |

Placée dans son **meilleur régime possible**, la mémoire-poids passe de 0 % à
~12 % apparent — mais ce ~12 % s'explique entièrement par une coïncidence de
fréquence, pas par de la récupération factuelle. **La conclusion du Lot 4 tient :
pour injecter de la connaissance factuelle nouvelle, RAG et le plein contexte
dominent nettement la mémoire-poids**, à coût comparable et sans hallucination.

## Reproduire

```bash
/workspace/venv/bin/python eval/lot4b/run_techqa.py \
  --model Qwen/Qwen2.5-7B-Instruct --quant 8bit \
  --docs eval/lot4b/tech_docs.jsonl --outdir /workspace/results/lot4b
```

`tech_docs.jsonl` : 2 docs + 17 QA rédigées par un générateur non-Qwen (Claude)
et **validées par un sous-agent Claude indépendant** (amendement #2). Les
changelogs bruts sont ceux d'`astral-sh/uv` et `astral-sh/ruff` (juin 2026).
