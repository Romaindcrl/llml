# LLML — Rapport de synthèse d'évaluation

**Synthèse des mesures publiques archivées de juillet 2026.**

Revue de portée du 10 septembre 2026 : ces archives sont partielles ; elles ne
certifient pas toute la matrice préenregistrée. Les résultats ci-dessous n'ont pas
été réexécutés pendant cette revue. Voir [le statut des campagnes](CAMPAIGN_STATUS.md)
pour les écarts, les pièces manquantes et le devenir des issues #1 et #3.

Modèle primaire : `Qwen/Qwen2.5-7B-Instruct`, quantization **8-bit (bitsandbytes)**
partout, décodage **greedy** (`temperature=0`). Backend CUDA `hf` (transformers +
bitsandbytes + peft), portage fidèle du client MLX de LLML. Scoring **uniquement**
via harnesses standards (EvalPlus, extraction MC/short-answer déterministe) —
jamais de harness maison. Pré-enregistrement des plans avant chaque run
([issue #1](https://github.com/Romaindcrl/llml/issues/1), amendements documentés).

---

## Résumé exécutif

LLML combine trois mécanismes, évalués sur les sous-ensembles décrits ci-dessous :

| Claim | Mécanisme | Verdict |
|---|---|:---:|
| **C** | Boucle **verify** (draft → run → repair) | +4 succès / 542, aucune régression observée ; portée limitée à ce run |
| **B** | **Routeur** de non-régression | Préservation observée sur 12 tâches ; matrice B1/B2 complète non établie |
| **A** | **Mémoire-poids** (`/sleep` → LoRA replay pour le rappel factuel) | Résultat défavorable à la recette testée face au retrieval/contexte |

Les observations favorisent la récupération externe et la protection du modèle
de base par rapport à la consolidation factuelle testée. Elles ne démontrent pas
un gain commercial, une garantie de non-régression ou la robustesse de chaque
chemin d'intégration. Les corrections de maintenance de septembre ne changent
pas les scores archivés.

---

## Claim C — gain observé de la boucle verify

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
- Gain descriptif modeste sur cet échantillon. Passer les exemples visibles ne
  garantit pas le passage des tests cachés : 6 modifications ont été adoptées,
  dont 2 sans amélioration du verdict caché. Les statistiques appariées demandées
  par H-C1 ne sont pas certifiées par ces seules synthèses.
- Recadrage honnête : le « 92→98 » du repo interne était mesuré sur 40 problèmes
  avec un harness maison ; sur harness standard complet, l'effet réel est ce
  +4/542. Le mécanisme est bon, l'ampleur annoncée était surévaluée.

→ détail : [`eval/lot2/results/`](../lot2/results/)

---

## Claim B — préservation observée sur un petit lot intégré

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
- C1 = C0 = oracle sur **12 tâches de génération** ; ce résultat ne certifie pas B1 sur toute la matrice.
- Sur ces 12 tâches, l'adapter always-on (C2) passe de **11/12 à 9/12**.
  Cette observation ne remplace pas le test statistique B2 préenregistré.
- Lot 6bis : génération **3/3 à chaque cycle** sur les mêmes trois tâches,
  malgré cinq `/sleep`. Ce n'est pas la courbe B3 prévue sur dix cycles.

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
  mémoire-poids sur les cas rapportés ; aucun avantage général de coût ou
  absence d'hallucination n'est établi ici.

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
- **H-D2 du protocole local 6bis (non-régression)** — 3/3 observé à chaque cycle ; conclusion limitée à ces trois tâches.
- **H-D3 (oubli)** — ➖ sans objet (C1 = 0 partout, rien à oublier).
- **H-D4 (intégration vs RAG)** — ✅ CONFIRMÉE : RAG-oracle ≫ C1. **Router les faits
  vers les poids plutôt que vers le RAG est le goulot du système.**

**Mécanismes exercés dans les archives** : routage 56/56, auto-promotion
(67 faits promus à la saturation), gate et maintien des trois tâches de génération.
Le rappel factuel par consolidation-poids reste nul dans ce run.

Les noms H-D1..D4 de ce protocole local ne correspondent pas aux hypothèses de
l'amendement SDK initial de l'issue #1. Cette boucle factuelle ne valide pas la
progression sur des tâches SDK annoncée dans cet amendement.

→ détail : [`eval/lot6bis/results/`](../lot6bis/results/) · artefact live :
<https://claude.ai/code/artifact/0194e31f-2b41-4028-b94b-a557f4a3be58>

---

## Recommandation actionnable

Le diagnostic est net et convergent sur 3 régimes + la boucle vivante :

> **Faire router la branche `recall` vers le RAG (ou fusionner poids + RAG dans
> le rappel) au lieu des poids seuls.** Le routeur classe déjà parfaitement les
> requêtes factuelles ; il les envoie juste au mauvais back-end. C'est
> une piste suggérée par la comparaison à RAG-oracle. Un changement réel du serveur
> devra être évalué ; les ~35 % de l'oracle ne sont pas un résultat déployé.

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
- Les pièces effectivement présentes sont inventoriées dans [CAMPAIGN_STATUS.md](CAMPAIGN_STATUS.md).
  Certaines sorties détaillées et certains adaptateurs annoncés ont été perdus ;
  la reproductibilité intégrale ne peut pas être affirmée.
- **Résultats négatifs publiés tels quels** (politique CDC) — le Claim A négatif
  est un livrable, pas un échec caché.
- **Coût GPU historique** : annoncé sous le plafond de 50 $ dans le journal de campagne.
  Aucun rapprochement de facturation n'a été effectué lors de la revue de septembre,
  et aucune ressource cloud n'a été relancée.

## Index des lots

| Lot | Objet | Résultats |
|---|---|---|
| 1 | Baselines C0 (M1–M4) + contrôle bf16 | `eval/lot1/` |
| 2 | **Claim C** — verify | `eval/lot2/results/` |
| 4 | **Claim A** — mémoire (QuALITY) | `eval/lot4/results/` |
| 4b | **Claim A** — mémoire (tech docs factuels) | `eval/lot4b/results/` |
| intégré | **Claim B** — routeur (charge mixte) | `eval/lot_integrated/results/` |
| 6bis | **Stack entière vivante** (bench #12) | `eval/lot6bis/results/` |
