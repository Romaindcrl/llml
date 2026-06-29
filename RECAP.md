# Mémoire par poids persistants — Récap complet du projet

> Assistant LLM local (MacBook M5, 24 Go, MLX) explorant comment doter un modèle d'une
> mémoire à long terme : **poids (fine-tuning LoRA)** vs **mémoire externe (RAG / compaction)**.
> Modèle de référence : `qwen2.5-7b-instruct` MLX 8-bit. Date : juin 2026.
> Auteur : **Romain Decrand--Lardière**.

---

## 1. Question de départ
Peut-on internaliser durablement du savoir dans les **poids** d'un petit LLM local (via LoRA),
plutôt que de tout garder en contexte ? Et si oui, **quand** est-ce mieux que le RAG / la
compaction ?

## 2. Architecture construite (`m0/`, `scripts/serve.py`)
Mémoire à 2 étages, inspirée CLS (Complementary Learning Systems) :
- **Contexte** (court terme) → compaction quand saturé.
- **Mémoire texte** (`MEMORY.md`, `m0/ltm.py`) = corpus Q/R, source de vérité.
- **Poids** (`m0/d2l.py`) = LoRA entraîné par **replay** sur le corpus à chaque `/sleep`.
- **RAG** (`m0/rag.py`, BM25 + filtrage stopwords) = docs bruts récupérables.
- **Routeur** (`is_generation` / `classify`) : rappel → poids ; génération → base+RAG.
- Serveur compatible OpenAI (Open WebUI) : `/remember`, `/sleep`, `/ctxt_clear`, `/reset`, `/info`.

## 3. Résultats des benchmarks (les chiffres)

### 3.1 QA factuelle — benchmark MAISON (held-out = reformulations des faits entraînés)
| méthode | classique | prog | global | ctx tok/q |
|---|---|---|---|---|
| base | 0 | 0 | 0 | 0 |
| RAG | 78 | 100 | 85 | 69 |
| compaction | 22 | 0 | 15 | 266 |
| **poids** | 100 | 100 | **100** | 0 |
→ *Trompeur* : held-out = paraphrases → le LoRA n'a qu'à reconnaître. **Biais de protocole.**

### 3.2 QA factuelle — VRAI benchmark SQuAD (questions humaines indépendantes)
| méthode | score |
|---|---|
| base | 59 |
| **RAG** | **88** |
| compaction | 72 |
| **poids** | **34** (sous la base !) |
→ Sur du réel, **les poids échouent** : ne capturent que les Q/R entraînées, pas une
compréhension du doc, **et dégradent le savoir pré-existant** (oubli). Tentative de sauvetage
(extraction exhaustive n=24, passage entier) : 34→38 % — **l'oubli, pas la couverture, est le mur.**

### 3.3 Génération de code (framework fictif)
| méthode | moyenne |
|---|---|
| base | 53 |
| RAG | 67 |
| **compaction** | **82** |
| poids (QA) | 6 |
| poids (code) | 24 |
| hybride LoRA+RAG | 26 |
→ Pour **générer**, le contexte gagne ; le fine-tuning **dégrade la génération** (même sur code).

### 3.4 Cahier des charges XL (spec 8745 tok, conventions pervasives + faits, 5 entités held-out)
| méthode | conventions | faits | global | ctx tok |
|---|---|---|---|---|
| base | 0 | 0 | 0 | 0 |
| RAG | 29 | 100 | 64 | 65 |
| compaction | 91 | **0** | 46 | 456 |
| style-seul (poids) | 100 | 10 | 55 | 0 |
| **2-étapes (poids + vérif. externe)** | **100** | **100** | **100** | 108 |
| fusion-hi (LoRA rang 32 + contexte) | 97 | 70 | 84 | 65 |
→ **Le résultat clé** : pour une grosse réf (codebase/spec) mêlant **style pervasif** + **faits** :
- les **poids** internalisent le **style/conventions** parfaitement et **généralisent** (100 %, 0 ctx) ;
- la **compaction s'effondre sur les faits** (0 %) quand la réf est trop grosse pour un résumé ;
- la **vérification externe** (lookup ciblé) fournit les faits exacts sans polluer le contexte ;
- **l'archi 2-étapes (poids style + vérif. externe) atteint 100 % et bat tout.**
- Plus de **capacité** (rang 32 vs 16) sauve aussi la fusion en un pass (0→84 %) → la fragilité
  était une limite de capacité, pas conceptuelle.

## 4. Conclusion — la dichotomie validée
| Type de savoir | Meilleur outil |
|---|---|
| **Faits** (rappel ponctuel, ouvert) | **mémoire externe (RAG)** — les poids perdent + oublient |
| **Style / conventions pervasives** | **poids (LoRA)** — internalise, généralise, 0 contexte |
| **Grosse réf (spec/codebase) style+faits** | **2-étapes** : poids (style) + vérif. externe (faits) |
| **Génération en général** | base + contexte (ne PAS fine-tuner pour générer) |

## 5. ⚠️ Nouveauté — recherche de prior art (honnête)
**Ces résultats ne sont PAS nouveaux : ils reproduisent des travaux déjà publiés.** Nous les
avons redécouverts indépendamment, ce qui valide la méthodo — mais ne justifie pas un papier.

- **RAG > fine-tuning pour les faits, + oubli, + "exposer à des variations du fait aide"** →
  Ovadia et al., *Fine-Tuning or Retrieval?* (arXiv:2312.05934, 2023). = nos §3.2.
- **Fine-tuning = style/comportement, RAG = faits** → consensus établi (mécanisme : LoRA agit
  surtout sur l'attention, le savoir factuel vit dans les FFN). = nos §3.3-3.4.
- **Entraîner le modèle à LIRE le contexte récupéré (fusion)** → **RAFT** (arXiv:2403.10131,
  2024). = notre "fusion-hi".
- **Capacité d'un LoRA / oubli / le rang compte / géométrie des sous-espaces** → *How Much
  Knowledge Can You Pack into a LoRA* (arXiv:2502.14502), *LoRA Rank Trade-offs* (2512.15634),
  *OPLoRA* (2510.13003). = nos observations rang 16 vs 32, oubli.
- **Générer-puis-vérifier / correction post-hoc des hallucinations d'API contre la doc/AST** →
  RARR, RAC (arXiv:2410.15667), *Correcting Hallucinations in Code via AST* (arXiv:2601.19106,
  100 % détection / 77 % correction). = notre "2-étapes".

**Verdict :** synthèse d'ingénierie solide, pas une contribution de recherche. Un papier
exigerait une vraie lacune + des benchmarks rigoureux (jeux réels, plusieurs modèles, baselines
vs RAFT/Ovadia) — non identifiée ici.

## 6. Recommandations pratiques (ce qui marche, à utiliser)
1. **Faits / connaissances** → RAG (BM25 ou dense). Ne pas fine-tuner.
2. **Style / conventions d'un projet** → un LoRA de style (entraîné sur du code conforme).
3. **Codebase/spec trop grosse** → 2-étapes : style en poids + vérification ciblée des faits.
4. **Génération** → toujours garder le modèle de base (le LoRA-faits dégrade la génération).
5. `serve.py` : génération = base+RAG **+ passe de vérification** ; rappel = poids.

## 7. Limites
- Petit N (toy specs fictives, 3-5 tâches), un seul modèle 7B 8-bit.
- 8-bit + rang 16 fragile ; rang 32 aide mais limite mémoire (24 Go) ; pas de fp16 testé.
- `val_loss` bruité (`--val-batches 1`). `--mask-prompt` + `--max-seq-length≥1024` requis pour
  l'entraînement contexte-aware.

## 8. Fichiers clés
- `m0/d2l.py` (LoRA, extraction Q/R, ancres, `mask_prompt`), `m0/rag.py` (BM25 + routeur),
  `m0/ltm.py`, `m0/compaction.py`, `m0/agent.py`, `scripts/serve.py`.
- Benchmarks : `benchmark.py`, `benchmark_code.py`/`_v2`, `benchmark_routed.py`,
  `benchmark_squad.py`/`_rescue`, `benchmark_spec.py`/`_xl`/`_final`, `router_eval.py`.
