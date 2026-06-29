# LLML — Local Lifelong Memory for LLMs

> A **local-first** (Apple Silicon / MLX) memory system that gives a small on-device LLM a
> two-tier long-term memory — **text memory + model weights** — with a `/sleep` consolidation
> cycle, RAG, task routing, and a generate-then-verify path.
>
> **Honest disclaimer.** None of the individual techniques here are novel — this is an
> *integration* of established ideas (see [Prior art](#prior-art)) into a usable local system,
> shipped with a **reproducible benchmark suite that honestly measures when each memory tier
> helps and when it hurts** (including negative results). That measurement is the point.

---

## What it is

A single 7B model (default `qwen2.5-7b-instruct`, MLX 8-bit) augmented with:

```
            ┌─────────────── CONTEXT (short-term) ───────────────┐
 documents ─▶  chat + tool use ──(saturation)──▶ compaction       │
            └───────────────────────┬─────────────────────────────┘
                                     │ freed content auto-promoted
                                     ▼
         TEXT MEMORY (MEMORY.md / LTM corpus)  ◀── /remember (manual)
                                     │
                          /sleep  (consolidation)
                                     ▼
              WEIGHTS  (LoRA, retrained by replay on the corpus)
                                     │
   query ──▶ ROUTER ──┬─ recall ──▶ weights (0-context fact recall)
                      └─ generation ▶ base model + RAG + verification pass
```

Inspired by **Complementary Learning Systems** (hippocampus = context, neocortex = weights):
new info enters as context, is written to a text corpus, and is periodically **consolidated
into the weights during `/sleep`** (replay = retrain a fresh LoRA on the growing corpus).

## Why it exists — the honest finding

We benchmarked weight-memory (fine-tuning) vs RAG vs compaction across tasks. The takeaway:

| Knowledge type | Best tool |
|---|---|
| **Open factual recall** | **RAG** — weights *lose and degrade prior knowledge* (SQuAD: weights 34% < base 59% < RAG 88%) |
| **Pervasive style / conventions** | **Weights** — internalize and generalize, at **0 context cost** (100% on unseen cases) |
| **Big spec/codebase (style + facts)** | **2-step**: weights (style) + external verification (facts) — beats RAG/compaction/fusion |
| **Generation in general** | base model + context — **do not fine-tune for generation** (it degrades it) |

The repo ships the benchmarks that produce these numbers, including the ones where our own
weight-memory approach **loses**. Most "agent memory" projects never measure this.

## Quickstart

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install mlx-lm httpx fastapi uvicorn numpy
# download an MLX model, e.g. mlx-community/Qwen2.5-7B-Instruct-8bit -> models/qwen2.5-7b-it-mlx-8bit

# OpenAI-compatible server (point Open WebUI at http://localhost:8000/v1)
M0_BACKEND=mlx M0_MLX_MODEL_PATH=models/qwen2.5-7b-it-mlx-8bit ./.venv/bin/python scripts/serve.py
```

Run without any model (deterministic mock, end-to-end smoke): `./.venv/bin/python scripts/smoke.py`

### Slash commands (in chat)
| Command | Effect |
|---|---|
| `/remember` | Add the last document to text memory (LTM) **and** the RAG index |
| `/sleep` | Consolidate: extract Q/R, retrain a LoRA by replay on the full corpus, hot-swap it |
| `/ctxt_clear` | Clear the context (keep weights + text memory) — to test recall from weights |
| `/reset` | Wipe everything (base model + LoRA + MEMORY.md + corpus + RAG) |
| `/info`, `/state`, `/help` | Inspect memory / state |

**Automatic mode (default on).** You don't have to type anything: documents you paste are
auto-indexed into RAG, and digested into long-term memory **in the background during idle time**
(*sleep-time compute*). `/remember` and `/sleep` become optional manual overrides. Weight
consolidation (`/sleep`) stays **opt-in** (`M0_AUTO_SLEEP=1`) — because the benchmarks show RAG
wins for facts, so we don't auto-bake everything into the weights.

## Benchmarks (reproducible)

Full results, tables, and honest takeaways (including where our approach **loses**) are in
**[`BENCHMARKS.md`](BENCHMARKS.md)**.

```bash
M0_BACKEND=mlx M0_MLX_MODEL_PATH=models/qwen2.5-7b-it-mlx-8bit \
  PYTHONPATH="$PWD" ./.venv/bin/python -u scripts/benchmark_squad.py      # facts: RAG wins
#  benchmark_code_v2.py     generation: context wins, fine-tuning hurts
#  benchmark_spec_final.py  big spec: 2-step (weights+verify) wins (100%)
#  benchmark_project.py     multi-file project @32k: ours holds, compaction collapses
#  demo_codeproject.py      end-to-end: generate a real module, run a hidden test suite
#  router_eval.py           keyword vs LLM router accuracy
```

## Limitations
- Small-scale evaluation (toy specs, N=3–5 tasks, single 7B 8-bit model).
- 8-bit + LoRA rank 16 is fragile; rank 32 needs lower LR; no fp16 tested.
- Single-pass weight+context *fusion* is not robust at scale — the **2-step** path is.
- BM25 RAG (lexical); a dense retriever would likely lift factual scores further.

## Prior art

This project re-derives, and stands on, established work:
- Ovadia et al., *Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs* — arXiv:2312.05934
- Zhang et al., *RAFT: Adapting Language Model to Domain Specific RAG* — arXiv:2403.10131
- *RAC: Efficient LLM Factuality Correction with Retrieval Augmentation* — arXiv:2410.15667
- *How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?* — arXiv:2502.14502
- Sleep / memory-consolidation for LLMs (CLS-inspired replay to weights): arXiv:2603.14517,
  2604.20943, 2605.26099; *Sleep-time Compute* — arXiv:2504.13171; Larimar — arXiv:2403.11901

See [`RECAP.md`](RECAP.md) for the full method, every benchmark number, and the analysis.

## Configuration (env vars)

| Variable | Default | Role |
|---|---|---|
| `M0_BACKEND` | `mock` | `mock` \| `mlx` \| `ollama` |
| `M0_MLX_MODEL_PATH` | `models/mlx-3b-4bit` | MLX model dir (mlx backend) |
| `M0_GATE_ACQ` | `0.45` | acquisition gate threshold for `/sleep` |
| `M0_AUTO_LEARN` | `1` | auto-index documents + background long-term memory |
| `M0_AUTO_SLEEP` | `0` | opt-in: auto-consolidate to weights when idle |
| `M0_AUTO_SLEEP_AFTER` / `M0_AUTO_IDLE_SEC` | 12 / 120 | new-facts threshold / idle seconds before auto-`/sleep` |
| `M0_D2L_ITERS` / `_LAYERS` / `_REPEAT` / `_LR` | 120 / 8 / 4 / 1e-4 | LoRA training knobs |
| `M0_MEMORY_CAP` | `1200` | injected text-memory cap |
| `M0_COMPACT_TRIGGER` | `4000` | compaction trigger threshold |
| `M0_SERVE_HOST` / `_PORT` | `127.0.0.1` / `8000` | OpenAI-compatible server |

## Architecture (package `m0`)

| Module | Role |
|---|---|
| `config.py` | `count_tokens`, `Config`, `from_env` |
| `llm.py` | `LLMClient` ABC, `MLXClient`, `OllamaClient`, `MockClient` |
| `d2l.py` | LoRA training, Q/R extraction + grounding, rehearsal anchors, `mask_prompt` |
| `rag.py` | BM25 retrieval (+ stopwords) and the `is_generation` / `classify` router |
| `ltm.py` | long-term text corpus (source of truth for replay) |
| `compaction.py` | non-destructive prune + live-state reinjection + summary |
| `agent.py` | agent loop, context clearing, auto-promotion on compaction |
| `scripts/serve.py` | OpenAI-compatible server + slash commands + routing |

## Author
**Romain Decrand--Lardière** — local LLM memory R&D.

## License
MIT © 2026 Romain Decrand--Lardière — see [`LICENSE`](LICENSE).
