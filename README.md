# LLML — local LLM memory research

LLML explores external retrieval, LoRA memory and verification for local language
models. It is a **research prototype**, with an MLX backend for Apple Silicon and
an experimental Hugging Face/CUDA backend.

**Current status:** the public evaluation archives are available. They do not
establish that LoRA memory beats a well-configured retrieval baseline, or that
LLML generally prevents regressions. The procedural-generation campaign is not
being continued under its old protocol. See the [campaign status](eval/results/CAMPAIGN_STATUS.md).

## What the archived public evaluations show

| Mechanism | Recorded observation | Scope |
|---|---|---|
| Verification | 4 additional successes out of 542 paired HumanEval+/MBPP+ tasks; 0 observed regressions | One Qwen2.5-7B 8-bit run; not a general safety or productivity guarantee |
| Routing | 42/42 routing decisions; generation 11/12 with routing vs 9/12 with an always-on adapter | Small mixed workload, not the full pre-registered non-regression matrix |
| Factual weight memory | 0/11 hard QuALITY cases, 2/17 technical-document cases, 0/17 at the end of the continuous loop | Retrieval/context performed better on these recorded cases |
| Procedural LoRA | No completed comparison available | Historical v2 task/generation artifacts are missing |

These are archived measurements and summaries, not runs repeated during the
September maintenance review. The [evaluation report](eval/results/REPORT.md)
records their limits and the [evaluation directory](eval/README.md) indexes the
available artifacts.

Earlier internal experiments used small synthetic or selected workloads. Their
results remain in [BENCHMARKS.md](BENCHMARKS.md) and the
[historical README](https://github.com/Romaindcrl/llml/blob/21a4e5ce23ed78b2c82ef2e1a5d2e6633fd56d35/README.md).
They should not be presented as current product guarantees or compared directly
to the later public evaluation.

## Components

- `m0/rag.py`, `m0/ltm.py`, `m0/memory.py`: retrieval and external memory.
- `m0/llm.py`: mock, Ollama and MLX clients; `m0/hf.py`: optional CUDA client.
- `m0/d2l.py`, `m0/d2l_hf.py`: experimental LoRA training backends.
- `m0/agent.py`, `scripts/serve.py`: orchestration and an OpenAI-compatible server.
- `eval/`: historical public evaluation tools and artifacts.

The current server still contains the experimental weight-recall path. Proposed
routing and gate improvements in old reports are not evidence that they have
been implemented or validated.

## Local checks without a model

```bash
python3 scripts/smoke.py
python3 -m unittest discover -s tests
```

These checks exercise orchestration and regression cases without downloading a
model or starting training. CUDA/MLX inference and training need their respective
runtimes and model weights; they are not covered by the mock checks.

The [example configuration](.env.example) and [CUDA evaluation notes](eval/README.md)
describe the experimental backends. Historical RunPod provisioning scripts are
explicit tools, not part of local startup.

## Next research question

Can a local specialist learn recurring, project-specific corrections and improve
on the same base model with compact rules and retrieved examples?

The initial feasibility audit found real historical corrections, but no qualified
single-family dataset with reliable no-op cases and an independent final test.
Dataset qualification comes before a detailed design or training. A positive
technical benchmark would still not establish commercial viability or human
time savings.

## License

MIT — see [LICENSE](LICENSE).
