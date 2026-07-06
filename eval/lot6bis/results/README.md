# Lot 6bis — Résultats : boucle d'apprentissage autonome (bench interne #12)

> **La stack LLML entière, vivante, en continu.** Premier run où rien n'est
> appelé à la main : l'agent lit un flux → son contexte **sature** →
> **auto-promotion** en LTM+RAG → **`/sleep` gé** (LoRA) → **routeur** décide
> par requête. 5 cycles cumulatifs. Modèle Qwen2.5-7B 8-bit, greedy.
> Pré-enregistrement (H-D1..D4, design, critères) : `../PRE_REGISTRATION.md`,
> committé **avant** ce run.

## Résultats bruts (`autoloop_results.jsonl`, `run.log`)

| cycle | LTM | gate `/sleep` | commité | C0 | **C1 (LLML intégré)** | RAG-oracle | gén. non-rég. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 14 | 0.385 | ✗ (rollback) | 0/4 | **0/4** | 1/4 (25%) | 3/3 |
| 2 | 21 | 0.538 | ✓ | 0/8 | **0/8** | 3/8 (37.5%) | 3/3 |
| 3 | 25 | 0.538 | ✓ | 0/11 | **0/11** | 4/11 (36.4%) | 3/3 |
| 4 | 52 | 0.538 | ✓ | 0/16 | **0/16** | 7/16 (43.8%) | 3/3 |
| 5 | 67 | 0.538 | ✓ | 0/17 | **0/17** | 6/17 (35.3%) | 3/3 |

- **Routage : 56/56 questions factuelles → `recall`** (100%). Le routeur classe
  parfaitement, et envoie donc tout vers la mémoire-poids (jamais le RAG).
- **Auto-promotion : opérationnelle** — 6 compactions cumulées, 67 faits promus
  en LTM sans intervention. Le mécanisme « contexte sature → LTM » marche.
- **Oubli (QA du cycle 1 sous C1)** : 0/4 à *tous* les cycles — rien à oublier,
  C1 n'a jamais rien rappelé.

## Le point qui tue : la gate se ment à elle-même

À partir du cycle 2, le `/sleep` **passe sa gate** (acquired 0.538 ≥ 0.45) et
**commit** l'adapter. Et pourtant **C1 reste 0/17**. Pourquoi ?

> La gate mesure `acquired` = rappel d'un **held-out de paraphrases des faits
> qu'elle vient d'entraîner** (`d2l.answer_recalled` sur `eval_pairs`). Elle
> valide donc que le LoRA a mémorisé *ses propres phrases d'entraînement*, pas
> qu'il **généralise** aux vraies questions du benchmark. Résultat : le système
> croit avoir appris (gate verte, adapter commité), mais sur les questions
> réelles, la branche `recall` répond **0**.

C'est un diagnostic mécanique, pas une opinion : `acquired=0.538` et `C1=0` sur
le **même** adapter, au **même** cycle.

## Verdict des 4 hypothèses pré-enregistrées

| Hyp. | Énoncé | Verdict |
|------|--------|---------|
| **H-D1** | Le système apprend (C1 ≥ C0 +20 pts) | **RÉFUTÉE pour C1** (0/17 = C0). Le seul « apprentissage » mesurable passe par **RAG-oracle** (35%), pas par la boucle intégrée. |
| **H-D2** | Pas de régression génération | **CONFIRMÉE** : 3/3 à tous les cycles. Le routeur bascule la génération sur base+RAG et la protège de bout en bout. |
| **H-D3** | Pas d'oubli catastrophique | **NON-APPLICABLE** : C1 = 0 partout, il n'y a jamais eu de rappel à oublier. |
| **H-D4** | Intégration vs RAG-oracle | **CONFIRMÉE (dans le sens prédit)** : RAG-oracle ≫ C1 à chaque cycle. **Router les faits vers les poids plutôt que le RAG est le goulot du système.** |

## Comparaison honnête (mémoire-poids, tous régimes)

| Lot | Régime | Rappel mémoire-poids |
|-----|--------|:---:|
| Lot 4 — QuALITY | compréhension (sous-ens. dur) | 0/11 |
| Lot 4b — tech docs | lookup factuel closed-book | 2/17 (dont 2 coïncidences de date) |
| **Lot 6bis — boucle vivante** | factuel, en boucle end-to-end | **0/17 (C1 intégré)** |

Le résultat converge dans les trois régimes : **la consolidation en poids
n'apporte pas de rappel factuel net.**

## Conclusion — a-t-on testé la stack entière ?

**Oui.** Pour la première fois, tout a tourné **ensemble et en continu**, sans
orchestration manuelle : saturation → auto-promotion → `/sleep` gé → routage.

Ce que la boucle prouve, honnêtement :
- **Le squelette d'intégration fonctionne** : routage 100%, protection de la
  génération (3/3), gating + rollback opérationnels, auto-promotion à la
  saturation. L'ingénierie du système est réelle et robuste.
- **La brique de consolidation-poids ne tient pas** : même gate passée, elle ne
  généralise pas aux vraies questions (C1 = 0/17). Sa valeur affichée
  (`acquired`) est un artefact de la métrique de gate.
- **La valeur réelle du système vient du RAG + du routeur de protection**, pas de
  la mémoire-poids. La conséquence actionnable : **router les rappels factuels
  vers le RAG** (ou fusionner poids+RAG dans la branche `recall`) plutôt que vers
  les poids seuls corrigerait le goulot — c'est exactement ce que H-D4 pointe.

Résultat négatif sur le claim central de la mémoire-poids, **publié tel quel**
(politique CDC : le négatif est publiable). Le reste de la stack — verify
(Lot 2), routeur de non-régression (Lot intégré), protection de la génération
(ici) — tient.

## Reproduire

```bash
python eval/lot6bis/build_corpus.py          # dérive frozen_corpus.jsonl (déterministe)
/workspace/venv/bin/python eval/lot6bis/run_autoloop.py \
  --model Qwen/Qwen2.5-7B-Instruct --quant 8bit \
  --corpus eval/lot6bis/frozen_corpus.jsonl --outdir /workspace/results/lot6bis
```

Paramètre de faisabilité documenté (pré-enreg.) : `M0_COMPACT_TRIGGER` 4000→1200
pour que le corpus modeste sature le contexte. Mécanisme inchangé.
