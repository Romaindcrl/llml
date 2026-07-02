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

# Part II — the systems campaign (July 2026)

Same rules: everything measured locally, failures reported alongside wins.

## 10. Where should stable knowledge live? (the sign-flip)
`benchmark_split.py` / `benchmark_split_weights.py` — a stable lookup table, distractor-loaded
context, the ONLY variable is where the table lives.

| context size | table in context | in system prompt | **in weights (LoRA)** |
|---|---|---|---|
| ~0 | 100% | 100% (+0) | 100% (+0) |
| ~4k | 81% | 62% (**−19**) | **100% (+19)** |
| ~12k | 81% | 75% (−6) | 88% (+6) |

**Takeaway:** the system prompt is NOT a proxy for weights (it *hurts*); weights win exactly
when context rot bites, at ~0 context cost. Weight-knowledge is load-invariant (confirmed again
under 20k load in §13).

## 11. Structured residuals — and an honest self-refutation
`benchmark_residual_*.py`, `benchmark_structcompact*.py`, `m0/structcompact.py` (kept as a
documented negative result). Multi-hop pointer chains: in-context 100% / single-pass RAG **0%**
/ token-pruning 6% / **iterative RAG 94%** (~2 calls). A structure-preserving compactor won
spectacularly on synthetic data (100% vs 0%) — then **failed on real data** (HotpotQA: 20% ≈
baselines; real-codebase import bridges: a *generic summary* 75% beats it at 38%).

**Takeaway:** synthetic wins must be re-tested on real data; a good generic summary is hard to
beat; token-pruning compression is reliably harmful (0-19% everywhere we measured it).

## 12. Failure-triggered self-improvement (`benchmark_selfimprove.py`, `benchmark_llml_loop.py`)
The model fails a task (0/8 on an unseen SDK) → reads the doc → **generates its own study
Q/A** → trains a LoRA → retries: **0% → 62% at 0 context, fully autonomous**. Capability
probes unchanged (0/2 → 0/2): the loop adds knowledge, not intelligence. The recipe is the
bottleneck (naive self-study: 12-38%). The full-system loop: RAG gives **75-88% immediately**
(t+50s), weights consolidate during idle — RAG now, weights forever.

## 13. Multi-tenant serving + a self-improving expert library
`multitenant_serve.py`, `serve_multitenant.py` — one frozen base + N×46MB adapters:
**~2ms hot-swap** (~590× cheaper than a model reload), per-tenant isolation verified, ~300
tenants fit on a 24GB machine. OpenAI-compatible endpoint routed by `X-Tenant`.
The MoE-of-experts system (router → expert weights → verify vs the expert's corpus):
**routing 12/12, system accuracy 92%**, and verify-caught corrections **retrain the failing
experts autonomously** (pure-weight drafts: 8/12 → 10/12, zero human ground truth). Unchanged
under 20k of context load. *(Run against a real private project spec + two synthetic tenants;
the private-spec scripts are withheld.)*

## 14. Code SKILLS via LoRA: refuted (5 convergent measurements)
`benchmark_code_skills.py` / `_skills2.py` — training on *verified, test-passing* code traces
(own = STaR-lite, or a 14B teacher = distillation), mask_prompt, anchors, reduced iters:
**every arm degraded held-out codegen** (−17 to −50 pts), consistent with §3. We also could
not create a 7B-vs-14B gap on self-contained tasks (the 7B matched or beat the 14B-coder-4bit
twice). **Memory ≠ capability; at this scale the LoRA path does not buy code skill.**

## 15. The imitation trap — the discriminating regime (`benchmark_bigctx2*.py`)
20k-token spec, hard 32k window, and the window full of **legacy code that violates all 7
conventions** (the real "migrate this codebase" scenario). With conforming ambient code
(`benchmark_bigctx_14b.py`), everything saturates — conventions come free by imitation. With
legacy ambient code:

| L=24k (overflow) | conv | facts | foundation kept |
|---|---|---|---|
| 14B, everything-in-context | 100% | 100% | **0%** (mechanically dropped) |
| 14B + RAG-spec | **36%** | 100% | 100% |
| **7B + LLML (spec in weights)** | **100%** | **100%** | **100%** |
| **14B + LLML** | **100%** | **100%** | **100%** |

Retrieval can't fetch *pervasive* rules (they're lexically unrelated to any query) and the
model imitates the legacy style; **spec-in-weights resists the trap on both models**. First
LoRA trained ON the 14B-4bit locally (val 0.066, ~9 min, 24GB). Also discovered: **rigidity**
— the baked skeleton ignored a *novel in-context* audit rule (0%) until the deterministic
verify pass was extended to read the value from the project's foundation module at runtime
(never baked into weights) → **100% on all four metrics, both models, both loads**
(`benchmark_bigctx2_auditfix.py`).

## 16. Executable deliverables — the test that counts (`benchmark_realtask*.py`)
A behavioral harness (stubbed runtime; 16 executed asserts per entity: nominal flow, exact
repo method, exact error code, validation, envelope). One-shot module generation: spec-in-ctx
62% (7B) / 78% (14B); the weight arms **collapsed to 0-25%** (rigidity again: a LoRA trained
on single functions can't emit a 4-function module). **Decomposed generation** (one function
per call — the adapter's training format — assembled + fact substitution):

| arm | behavioral asserts | recurring spec cost |
|---|---|---|
| 7B, spec in context | 62% | 20,333 tok/call |
| 14B, spec in context | 78% | 20,333 tok/call |
| **7B + LLML, decomposed** | **81%** | **0 tok** |
| **14B + LLML, decomposed** | **81%** | **0 tok** |

**Takeaway:** on *executed* deliverables, the full system (weights + decomposition + fact
substitution + verification) beats the bigger model with the whole spec in context — at zero
recurring spec cost. The system design (decomposition + verification) is what compensates the
weights' rigidity; remove any piece and the result collapses.

---

## 17. Public certification — HumanEval (`benchmark_humaneval*.py`)
40 problems, official hidden tests executed, pass@1, greedy. Five arms, one story:

| arm | pass@1 | Δ |
|---|---|---|
| 7B bare (one-shot) | 37/40 (92%) | — |
| 7B + spec-LoRA **always on** | **3/40 (8%)** | **−84 pts** |
| 7B + code-LoRA always on | 31/40 (78%) | −15 pts |
| **MoE system** (router w/ GENERAL fallback) | **37/40 (92%)** | **±0** — routed 40/40 out-of-domain to the bare model |
| **LLML verification loop** (draft → run documented examples → repair ≤2×) | **39/40 (98%)** | **+6 pts** |

**Takeaways:** (1) an always-on adapter taxes general ability — spec-LoRA catastrophically
(rigid baked skeleton), even a verified code-LoRA by −15 pts: **weight-memory must be routed,
never ambient**; (2) the router's GENERAL fallback fully restores the base model (100% correct
out-of-domain routing); (3) the *system* improves the public benchmark **92% → 98%** — the gain
comes from the verification pillar (execution-checked repair), not from memory, and we say so.

---

## Overall conclusions
- **Open, unpredictable facts → RAG.** Weights *can* store facts with a good recipe (§9), but for
  questions you can't anticipate, RAG is more robust — it retrieves the source (§2).
- **Generation → base model + context.** Fine-tuning hurts generation (§3) — including on
  verified code traces (§14). Memory ≠ capability.
- **Pervasive style / a big cahier des charges → weights**, with a verification pass for the
  specific facts (§5–7). This regime holds under window saturation and is where retrieval
  *structurally* fails: the imitation trap (§15, 100% vs 36%).
- **Small docs → in-context wins.** A few hundred tokens of reference next to the question
  beats everything (incl. under 20k of clean load); weights pay off in tokens and persistence,
  not accuracy, in that regime.
- **The system is the product, not the LoRA**: weights (stable) + deterministic verification
  (facts + contextual rules) + decomposition (format) + RAG (unpredictable) + router
  (multi-domain). Every failed arm in Part II was a missing piece; every green one had all five
  (§13, §15, §16).
- **Raw reasoning (hard algorithms, long-context recall) is the model's job**, not the memory
  layer's (§8, §7, §14).

None of the individual techniques are novel (RAG-vs-FT: Ovadia et al. 2312.05934; train-to-read-
context: RAFT 2403.10131; generate-then-verify: RAC 2410.15667; LoRA capacity/forgetting:
2502.14502). The contribution is the **integrated, local, honestly-measured system** — see
[`RECAP.md`](RECAP.md) and [`README.md`](README.md).

---
*Author: **Romain Decrand--Lardière** · MIT License.*
