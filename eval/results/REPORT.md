# LLML — Rapport de synthèse d'évaluation

**Validation des claims de LLML sur benchmarks publics, méthodologie
pré-enregistrée et intégralement reproductible.**

Modèle primaire : `Qwen/Qwen2.5-7B-Instruct`, quantization **8-bit (bitsandbytes)**
partout, décodage **greedy** (`temperature=0`). Backend CUDA `hf` (transformers +
bitsandbytes + peft), portage fidèle du client MLX de LLML. Scoring **uniquement**
via harnesses standards (EvalPlus, extraction MC/short-answer déterministe) —
jamais de harness maison. Pré-enregistrement des plans avant chaque run
([issue #1](https://github.com/Romaindcrl/llml/issues/1), amendements documentés).

---

## Résumé exécutif

LLML combine trois mécanismes. Mesurés proprement, séparément **puis en boucle
vivante de bout en bout** :

| Claim | Mécanisme | Verdict |
|---|---|:---:|
| **C** | Boucle **verify** (draft → run → repair) | ✅ **tient** — gain réel, zéro régression |
| **B** | **Routeur** de non-régression (protège les capacités de base) | ✅ **tient** — sans lui, la génération se dégrade |
| **A** | **Mémoire-poids** (`/sleep` → LoRA replay pour le rappel factuel) | ❌ **ne tient pas** — pas de rappel factuel net |

**La valeur réelle du système vient du routeur (protection) + du RAG (rappel),
pas de la consolidation en poids.** Le squelette d'intégration — routage, gating,
rollback, auto-promotion à la saturation, protection de la génération — est réel
et robuste. La brique qui ne délivre pas est la mémoire-poids.

---

## Claim C — la boucle verify apporte un gain réel

**Lot 2.** C0 (draft seul) vs C0+verify (draft → exécution d'exemples → réparation),
scoring **EvalPlus** (base + plus), sur l'intégralité de HumanEval+ (164) et MBPP+ (378).

| Benchmark | | base pass | plus pass | régressions |
|---|---|:---:|:---:|:---:|
| HumanEval+ (164) | C0 | 136 | 129 | — |
| | **C0+verify** | **137** | **130** | **0** |
| MBPP+ (378) | C0 | 307 | 261 | — |
| | **C0+verify** | **312** | **264** | **0** |

- **4 problèmes** passent de `fail` → `pass` grâce à la vérification
  (HumanEval/19 ; Mbpp/6, /259, /391), **0 régression** sur 542 problèmes.
- Gain modeste mais **réel et strictement non-régressif** — cohérent avec un
  mécanisme qui ne corrige que ce qu'il peut exécuter et vérifier.
- Recadrage honnête : le « 92→98 » du repo interne était mesuré sur 40 problèmes
  avec un harness maison ; sur harness standard complet, l'effet réel est ce
  +4/542. Le mécanisme est bon, l'ampleur annoncée était surévaluée.

→ détail : [`eval/lot2/results/`](../lot2/results/)

---

## Claim B — le routeur préserve les capacités de base

**Lot intégré.** Charge mixte (rappel factuel + génération de code entrelacés),
3 documents → 114 faits, `/sleep` commité (acquisition 0.769). Le routeur
`classify()` décide par requête ; on compare le système avec routeur (C1),
sans routeur / mémoire forcée (C2), et routage parfait (oracle).

| Config | routage | rappel | génération pass@1 |
|---|:---:|:---:|:---:|
| **routage** | — | — | **42/42 = 100%** |
| C0 (base) | — | 0.633 | 0.917 |
| **C1 (LLML, routeur)** | actif | 0.633 | **0.917** |
| C2 (mémoire always-on) | **désactivé** | 0.633 | **0.75** |
| oracle (routage parfait) | parfait | 0.633 | 0.917 |

- **Routage parfait : 42/42.** Le classifieur ne se trompe jamais sur ce mix.
- **B1 confirmée** : C1 = C0 = oracle → le routeur ne dégrade **rien**.
- **B2 confirmée** : forcer l'adapter en always-on (C2) fait chuter la génération
  **92% → 75%**. Le routeur existe précisément pour éviter ce mode de défaillance,
  et il le fait.
- Confirmé une 2ᵉ fois en boucle continue (Lot 6bis) : génération **3/3 à tous
  les cycles** malgré 5 `/sleep` successifs.

→ détail : [`eval/lot_integrated/results/`](../lot_integrated/results/)

---

## Claim A — la mémoire-poids ne restitue pas les faits

Testée dans **trois régimes de plus en plus favorables**. Le résultat converge.

| Lot | Régime | Mémoire-poids | RAG | Contexte plein |
|---|---|:---:|:---:|:---:|
| **4 — QuALITY** | compréhension (sous-ens. dur) | **0/11** | 3/11 | 9/11 |
| **4b — tech docs** | lookup factuel closed-book (design-friendly) | **2/17** | 8/17 | 16/17 |
| **6bis — boucle vivante** | factuel, end-to-end, 5 cycles | **0/17** (C1) | ~35% (RAG-oracle) | — |

- Lot 4b : les **2 seuls succès** sont la **même date** (`2026-06-18`, présente
  dans les deux docs) → collapse sur le token fréquent, pas du rappel. Ailleurs,
  hallucinations *near-miss* (`SARIF`→`json`, `0.6.3`→`0.13.0`).
- Même quand le `/sleep` **passe sa gate** (Lot 6bis cycle 2+, acquisition 0.538,
  adapter commité), le rappel réel reste **0** : la gate mesure le rappel de
  **paraphrases de son propre entraînement**, pas la généralisation aux vraies
  questions. Le système croit avoir appris, valide, commit — sans effet.
- Dans les trois régimes, **RAG et contexte plein dominent nettement** la
  mémoire-poids, à coût comparable et sans hallucination.

→ détail : [`eval/lot4/results/`](../lot4/results/) · [`eval/lot4b/results/`](../lot4b/results/)

---

## Lot 6bis — la stack ENTIÈRE, vivante (bench interne #12)

Le seul test qui exerce **tout le système en continu, sans appel manuel** :
l'agent lit un flux → son contexte **sature** → **auto-promotion** en LTM+RAG →
**`/sleep` gé** (LoRA) → **routeur** décide. 5 cycles cumulatifs, corpus gelé
post-cutoff. Hypothèses **pré-enregistrées** (`eval/lot6bis/PRE_REGISTRATION.md`)
avant le run.

| cycle | LTM | gate | commit | C0 | **C1 intégré** | RAG-oracle | génération |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 14 | 0.385 | ✗ | 0 | **0** | 25% | 3/3 |
| 2 | 21 | 0.538 | ✓ | 0 | **0** | 37.5% | 3/3 |
| 3 | 25 | 0.538 | ✓ | 0 | **0** | 36.4% | 3/3 |
| 4 | 52 | 0.538 | ✓ | 0 | **0** | 43.8% | 3/3 |
| 5 | 67 | 0.538 | ✓ | 0 | **0** | 35.3% | 3/3 |

**Verdict des 4 hypothèses pré-enregistrées :**
- **H-D1 (apprend)** — ❌ RÉFUTÉE pour C1 (0/17). Le gain n'existe que via RAG-oracle.
- **H-D2 (non-régression)** — ✅ CONFIRMÉE (3/3 partout ; le routeur protège la génération).
- **H-D3 (oubli)** — ➖ sans objet (C1 = 0 partout, rien à oublier).
- **H-D4 (intégration vs RAG)** — ✅ CONFIRMÉE : RAG-oracle ≫ C1. **Router les faits
  vers les poids plutôt que vers le RAG est le goulot du système.**

**Mécanismes qui marchent, prouvés vivants** : routage 56/56, auto-promotion
(67 faits promus tout seuls à la saturation), gating + rollback, protection de la
génération. **Mécanisme qui ne marche pas** : la consolidation-poids.

→ détail : [`eval/lot6bis/results/`](../lot6bis/results/) · artefact live :
<https://claude.ai/code/artifact/0194e31f-2b41-4028-b94b-a557f4a3be58>

---

## Recommandation actionnable

Le diagnostic est net et convergent sur 3 régimes + la boucle vivante :

> **Faire router la branche `recall` vers le RAG (ou fusionner poids + RAG dans
> le rappel) au lieu des poids seuls.** Le routeur classe déjà parfaitement les
> requêtes factuelles ; il les envoie juste au mauvais back-end. C'est
> exactement ce que pointe H-D4, et ça transformerait un rappel de 0% en ~35%+
> sans toucher au reste de la stack.

La `/sleep`-gate mérite aussi d'être re-spécifiée : mesurer l'acquisition sur des
**questions held-out réellement distinctes** (pas des paraphrases du train), sinon
elle valide de l'overfit.

---

## Méthodologie & reproductibilité

- **Harnesses standards uniquement** : EvalPlus (code), scoring MC/short-answer
  déterministe (rappel). Aucune réimplémentation de benchmark.
- **8-bit partout**, greedy, révisions HF épinglées (`eval/requirements.lock`).
- **Pré-enregistrement** avant chaque run (issue #1 + `PRE_REGISTRATION.md` du Lot 6bis) ;
  tout écart documenté, jamais silencieux.
- **Rapatriement par benchmark** avant tout `podTerminate` (règle durcie après
  incident Lot 1) : aucun chiffre publié sans artefact brut.
- **Résultats négatifs publiés tels quels** (politique CDC) — le Claim A négatif
  est un livrable, pas un échec caché.
- **Coût GPU total** : sous le plafond de 50 $ fixé au CDC (pods RunPod 4090
  8-bit, arrêtés après chaque lot).

## Index des lots

| Lot | Objet | Résultats |
|---|---|---|
| 1 | Baselines C0 (M1–M4) + contrôle bf16 | `eval/lot1/` |
| 2 | **Claim C** — verify | `eval/lot2/results/` |
| 4 | **Claim A** — mémoire (QuALITY) | `eval/lot4/results/` |
| 4b | **Claim A** — mémoire (tech docs factuels) | `eval/lot4b/results/` |
| intégré | **Claim B** — routeur (charge mixte) | `eval/lot_integrated/results/` |
| 6bis | **Stack entière vivante** (bench #12) | `eval/lot6bis/results/` |
