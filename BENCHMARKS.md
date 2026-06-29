# Benchmarks

Every number here was produced locally (MacBook M-series, MLX) by the scripts in `scripts/`.
We report the results that flatter the system **and the ones that don't** — the point of this
repo is to measure honestly *when* weight-memory helps and when it hurts.

Default model: `qwen2.5-7b-instruct` MLX 8-bit, unless noted. Reproduce with the command under
each section.

---

## 1. Factual QA — homemade (⚠️ biased, kept as a cautionary tale)
`scripts/benchmark.py` — internalize 3 Wikipedia pages, answer held-out questions.

| method | score |
|---|---|
| base | 0% |
| RAG | 85% |
| compaction | 15% |
| **weights (ours)** | **100%** |

**Takeaway:** looks like a blowout for weights — but it's an **artifact**: the held-out questions
were *paraphrases of the trained facts*, so the LoRA only had to recognize a reformulation. This
result is why we then ran a *real* benchmark ↓.

## 2. Factual QA — SQuAD (real, the reality check)
`scripts/benchmark_squad.py` — internalize 8 Wikipedia paragraphs, answer **32 real human
SQuAD questions** never seen in training.

| method | score |
|---|---|
| base | 59% |
| **RAG** | **88–94%** |
| compaction | 72% |
| weights — naive recipe | 34% |
| weights — **good recipe** (heavy paraphrase augmentation) | **44%** |

**Takeaway:** on real, *unpredictable* human questions, **weight-internalization loses to RAG**.
A much better recipe lifts the weights from 34% to 44% — but no further, because you can't
pre-cover questions you don't know. (Contrast §9: with *known* queries — Needle-in-a-Haystack —
the same good recipe takes the weights to **100%**.) This matches the literature (Ovadia et al.
2023), but the limit is **query coverage**, not a hard "weights can't store facts".

## 3. Code generation
`scripts/benchmark_code_v2.py` — generate code using a fictional framework.

| method | mean |
|---|---|
| base | 53% |
| RAG | 67% |
| **compaction** | **82%** |
| weights (QA-trained) | 6% |
| weights (code-trained) | 24% |
| hybrid (LoRA+RAG) | 26% |

**Takeaway:** for *generation*, the reference must be **in context**. Fine-tuning — even on code —
**degrades generation below the base model**. Don't fine-tune to generate.

## 4. The router
`scripts/router_eval.py` — classify a query as recall vs generation.

| router | accuracy |
|---|---|
| keyword heuristic | 79% |
| LLM zero-shot | 93% |

On the 32 real SQuAD questions, the keyword router mis-routed **0** (all correctly → recall).

## 5. Style / conventions — where weights actually win
`scripts/benchmark_spec_final.py` — a spec with **pervasive conventions** + **per-entity facts**,
tested on entities never trained. Conventions% / facts% / global / context-tokens.

| method | conv | facts | global | ctx |
|---|---|---|---|---|
| base | 0 | 0 | 0 | 0 |
| RAG | 29 | 100 | 64 | 65 |
| compaction | 91 | 0 | 46 | 456 |
| style-LoRA only | 100 | 10 | 55 | 0 |
| **2-step (weights + verify)** | **100** | **100** | **100** | 108 |
| fusion (LoRA rank-32 + context) | 97 | 70 | 84 | 65 |

**Takeaway:** **weights internalize pervasive style/conventions perfectly and generalize at 0
context cost.** A **2-step** architecture — style in weights + a verification pass that fixes
specific facts against external memory — hits 100/100 and beats RAG, compaction, and fusion.

## 6. Multi-file project under a spec (the realistic test), 32k window
`scripts/benchmark_project.py` — a cahier des charges that fits in 32k, but the **accumulating
project code** fills the window (`L` = tokens of code already in context). conventions% / facts%
/ context-tokens-for-the-spec.

| code load L | method | conv | facts | ctx (spec) |
|---|---|---|---|---|
| 0 | compaction | 79 | 50 | 20 382 |
| | RAG+compaction | 93 | 100 | 20 484 |
| | **OURS** | **100** | **100** | **150** |
| 12k | compaction | 93 | **0** | 13 806 |
| | RAG+compaction | 100 | 100 | 13 908 |
| | **OURS** | **100** | **100** | 12 284 |
| 22k | compaction | 93 | **0** | 23 963 |
| | RAG+compaction | 93 | 100 | 24 064 |
| | **OURS** | **100** | **100** | 22 441 |

**Takeaway:** as code fills context, **compaction collapses on facts** (its summary can't hold the
spec). Ours stays **100/100 at near-zero spec context** because the cahier des charges lives in
the weights — the window stays free for code.

## 7. True overflow — hard 32k window
Same script, pushing `L` to 31k with a hard window. A *foundation module* at the start of the
project defines a cross-file marker that is truncated first.

| L | method | conv | facts | foundation kept |
|---|---|---|---|---|
| 31k | compaction | 100 | **0** | **0%** |
| | RAG+compaction | **93** | 100 | **0%** |
| | **OURS** | **100** | **100** | **100%** |

**Takeaway (mixed, honest):** at overflow, **only ours is 100/100** (compaction's facts die,
RAG+compaction's conventions slip), and the truncation mechanism works exactly as designed —
baselines drop the foundation module to keep the spec, **ours keeps all the code**. **But** a
cross-file metric built on that foundation came out ~0 for *everyone*: the 7B can't recall a
marker from the start of a 22–31k context (lost-in-the-middle). So the "preserve project context"
advantage exists *mechanically* but the local 7B is too weak to exploit it — that part would need
a stronger long-context model.

## 8. End-to-end coding capability (the model is the ceiling)
`scripts/demo_codeproject.py` — generate a real module, run a hidden test suite.

| task | model | mode | score |
|---|---|---|---|
| LRU cache | qwen-7B | one-shot | 6/6 ✅ |
| expression parser | qwen-7B | one-shot | 0/16 |
| expression parser | Coder-7B | one-shot | 2/16 |
| expression parser | Coder-7B | agentic (retry) | 2 → **0** (thrashes) |
| KV-store | Coder-7B | agentic | 5 → **13/15** |
| KV-store | **Coder-14B** | agentic, 32k | 9 → **15/15** ✅ |

**Takeaway:** the memory/orchestration layer is sound; raw reasoning power comes from the model.
Self-repair **amplifies a competent model** (KV-store) but **thrashes when the task exceeds the
model's capability** (the parser) — it fixes near-misses, not capability gaps.

## 9. Weight-recall: the *recipe*, not a wall (Needle-in-a-Haystack)
`scripts/benchmark_niah.py` / `_v2.py` — insert K unique facts ("needles") into a long document,
internalize it into the weights, then recall each needle.

| recipe (same Qwen-7B 8-bit) | needles recalled |
|---|---|
| naive (lossy QA extraction) | 20% (1/5) |
| **good (forced coverage + heavy paraphrase augmentation)** | **100% (5/5)** |

in-context and RAG are 100% throughout. **Precision is *not* the lever:** 8-bit and **bf16**
(full precision) both give 20% with the naive recipe, even with every needle in the training set
~16×. The 20% → 100% jump came purely from a better recipe (covering the query phrasings).

**Takeaway (refines §2):** weights *can* store facts reliably — the earlier "weights lose" was
partly a **recipe** failure (lossy extraction + train-phrasing ≠ test-phrasing), not a hard limit.
The real trade-off is **query coverage**: if you can anticipate/cover the queries (or faithfully
encode the doc, like Sakana's *Doc-to-LoRA* — an efficiency win, not an accuracy-vs-RAG one), the
weights work; for **open, unpredictable** questions (§2, SQuAD), RAG wins because it retrieves the
source instead of betting on coverage.

---

## Overall conclusions
- **Open, unpredictable facts → RAG.** Weights *can* store facts with a good recipe (§9), but for
  questions you can't anticipate, RAG is more robust — it retrieves the source (§2).
- **Generation → base model + context.** Fine-tuning hurts generation (§3).
- **Pervasive style / a big cahier des charges → weights**, with a verification pass for the
  specific facts (§5–7). This is the one regime where weight-memory clearly wins, and it holds
  even when the context window saturates with code.
- **Raw reasoning (hard algorithms, long-context recall) is the model's job**, not the memory
  layer's (§8, §7).

None of the individual techniques are novel (RAG-vs-FT: Ovadia et al. 2312.05934; train-to-read-
context: RAFT 2403.10131; generate-then-verify: RAC 2410.15667; LoRA capacity/forgetting:
2502.14502). The contribution is the **integrated, local, honestly-measured system** — see
[`RECAP.md`](RECAP.md) and [`README.md`](README.md).

---
*Author: **Romain Decrand--Lardière** · MIT License.*
