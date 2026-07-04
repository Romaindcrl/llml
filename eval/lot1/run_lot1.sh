#!/usr/bin/env bash
# Lot 1 — baselines C0 pour UN modèle et UNE quantization, résumable.
# Usage (pod) : bash eval/lot1/run_lot1.sh <HF_MODEL> <TAG> <8bit|bf16> [benchs...]
#   ex: bash eval/lot1/run_lot1.sh Qwen/Qwen2.5-7B-Instruct M1 8bit gsm8k humaneval mbpp ifeval mmlu_pro
# Chaque benchmark écrit ses sorties sous /workspace/results/lot1/<TAG>_<QUANT>/
# et pose un marqueur .done_<bench> (reprise sur préemption, CDC §7).
set -uo pipefail

MODEL="$1"; TAG="$2"; QUANT="$3"; shift 3
BENCHS=("${@:-gsm8k humaneval mbpp ifeval mmlu_pro}")
[ $# -eq 0 ] && BENCHS=(gsm8k humaneval mbpp ifeval mmlu_pro)

export HF_HOME=/workspace/hf
PY=/workspace/venv/bin/python
LLML=/workspace/llml
OUT=/workspace/results/lot1/${TAG}_${QUANT}
CSV=/workspace/results/lot1/scores.csv
mkdir -p "$OUT"

if [ "$QUANT" = "8bit" ]; then
  MA="pretrained=${MODEL},load_in_8bit=True"
  BS="${LMEVAL_BS:-16}"
else
  MA="pretrained=${MODEL},dtype=bfloat16"
  BS="${LMEVAL_BS:-8}"
fi

# Choix de template gelés (journalisés AGENTS.md) : chat template + fewshot
# multiturn pour modèles instruct (méthodo Open LLM Leaderboard v2).
# Batch EXPLICITE : `auto` sérialisait la génération (~14 s/item mesuré, Lot 1
# Phase A) ; un entier fixe fait batcher generate_until par lm-eval. Le greedy
# par item est inchangé (padding gauche géré par le harness).
LMEVAL_COMMON=(--model hf --model_args "$MA" --batch_size "$BS" --seed 42
               --apply_chat_template --fewshot_as_multiturn --log_samples)

step_done() { [ -f "$OUT/.done_$1" ]; }
mark_done() { date -u +%FT%TZ > "$OUT/.done_$1"; }
say() { echo "[lot1 ${TAG}_${QUANT}] $(date -u +%T) $*"; }

for B in "${BENCHS[@]}"; do
  if step_done "$B"; then say "$B deja fait — skip"; continue; fi
  say "=== $B START ==="
  T0=$SECONDS
  case "$B" in
    gsm8k)
      $PY -m lm_eval "${LMEVAL_COMMON[@]}" --tasks gsm8k --num_fewshot 8 \
          --output_path "$OUT/gsm8k" 2>&1 | tail -30
      RC=$?
      J=$(ls -t "$OUT"/gsm8k/*/results_*.json 2>/dev/null | head -1)
      [ $RC -eq 0 ] && [ -n "$J" ] && $PY "$LLML/eval/lot1/collect.py" lmeval \
          --json "$J" --model "$TAG" --quant "$QUANT" --config C0 --csv "$CSV" \
          && mark_done "$B"
      ;;
    ifeval)
      $PY -m lm_eval "${LMEVAL_COMMON[@]}" --tasks ifeval --num_fewshot 0 \
          --output_path "$OUT/ifeval" 2>&1 | tail -30
      RC=$?
      J=$(ls -t "$OUT"/ifeval/*/results_*.json 2>/dev/null | head -1)
      [ $RC -eq 0 ] && [ -n "$J" ] && $PY "$LLML/eval/lot1/collect.py" lmeval \
          --json "$J" --model "$TAG" --quant "$QUANT" --config C0 --csv "$CSV" \
          && mark_done "$B"
      ;;
    mmlu_pro)
      # 6 domaines pré-enregistrés (seed=42, issue #1)
      TASKS=mmlu_pro_biology,mmlu_pro_business,mmlu_pro_computer_science,mmlu_pro_economics,mmlu_pro_math,mmlu_pro_other
      $PY -m lm_eval "${LMEVAL_COMMON[@]}" --tasks "$TASKS" \
          --output_path "$OUT/mmlu_pro" 2>&1 | tail -40
      RC=$?
      J=$(ls -t "$OUT"/mmlu_pro/*/results_*.json 2>/dev/null | head -1)
      [ $RC -eq 0 ] && [ -n "$J" ] && $PY "$LLML/eval/lot1/collect.py" lmeval \
          --json "$J" --model "$TAG" --quant "$QUANT" --config C0 --csv "$CSV" \
          && mark_done "$B"
      ;;
    humaneval|mbpp)
      SMP="$OUT/${B}_samples.jsonl"
      $PY "$LLML/eval/lot1/gen_evalplus.py" --model "$MODEL" --dataset "$B" \
          --quant "$QUANT" --out "$SMP"
      RC=$?
      if [ $RC -eq 0 ]; then
        $PY -m evalplus.evaluate --dataset "$B" --samples "$SMP" \
            2>&1 | tee "$OUT/${B}_evalplus.log" | tail -12
        grep -q "pass@1" "$OUT/${B}_evalplus.log" && \
          $PY "$LLML/eval/lot1/collect.py" evalplus --log "$OUT/${B}_evalplus.log" \
              --dataset "$B" --model "$TAG" --quant "$QUANT" --config C0 --csv "$CSV" \
          && mark_done "$B"
      fi
      ;;
    *) say "benchmark inconnu: $B" ;;
  esac
  say "=== $B END ($((SECONDS-T0)) s) — done=$(step_done "$B" && echo oui || echo NON) ==="
done

say "TERMINE. Marqueurs:"
ls "$OUT"/.done_* 2>/dev/null || true
[ -f "$CSV" ] && { say "scores.csv:"; cat "$CSV"; }
