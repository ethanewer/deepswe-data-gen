#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# swebench-multilingual-easy is a fixed 10-task debugging subset used for fast
# SFT iteration. It is intentionally separate from the predictive-30 benchmark.
TRIALS="${TRIALS:-5}"
if ! [[ "$TRIALS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRIALS must be a positive integer; got $TRIALS" >&2
  exit 1
fi
TRIAL_START="${TRIAL_START:-1}"
TRIAL_INDICES="${TRIAL_INDICES:-}"
if ! [[ "$TRIAL_START" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRIAL_START must be a positive integer; got $TRIAL_START" >&2
  exit 1
fi

export INSTANCE_IDS_PATH="${INSTANCE_IDS_PATH:-$SCRIPT_DIR/swebench_multilingual_easy_10_instance_ids.txt}"
export SUBSET_LABEL="${SUBSET_LABEL:-swebench-multilingual-easy}"
export BENCHMARK="${BENCHMARK:-multilingual}"
export BENCHMARK_LABEL="${BENCHMARK_LABEL:-swebench-ml-${SUBSET_LABEL}}"
export DEVICE_LABEL="${DEVICE_LABEL:-h200}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft/checkpoints/qwen3_4b_thinking_swe260617_v73_mixed50_weighted_one_epoch_65k_sft}"
export CHECKPOINT_STEP_DIR="${CHECKPOINT_STEP_DIR:-$CHECKPOINT_DIR/epoch_0_step_49}"
export CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-v73-mixed50-weighted-oneepoch-s49}"
export CONTEXT_LABEL="${CONTEXT_LABEL:-65k}"
export EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-2}"
export MAX_TOKENS="${MAX_TOKENS:-8192}"
export GENERATION_STEP_LIMIT="${GENERATION_STEP_LIMIT:-250}"

RUN_SUFFIX_PREFIX="${RUN_SUFFIX_PREFIX:-easy-n${TRIALS}}"
DEFAULT_RUN_STEM="qwen3-4b-thinking-sft-${CHECKPOINT_LABEL}-${CONTEXT_LABEL}"

if [ -n "$TRIAL_INDICES" ]; then
  read -r -a trial_ids <<< "$TRIAL_INDICES"
else
  mapfile -t trial_ids < <(seq "$TRIAL_START" "$((TRIAL_START + TRIALS - 1))")
fi

for trial in "${trial_ids[@]}"; do
  if ! [[ "$trial" =~ ^[1-9][0-9]*$ ]]; then
    echo "Trial indices must be positive integers; got $trial" >&2
    exit 1
  fi
  export RUN_STEM="${RUN_STEM:-$DEFAULT_RUN_STEM}"
  export RUN_SUFFIX="${RUN_SUFFIX_PREFIX}-t${trial}"
  echo "Starting swebench-multilingual-easy trial ${trial}: $CHECKPOINT_LABEL"
  "$SCRIPT_DIR/run_qwen3_4b_swe260612_step50_l40s_eval.sh"
done
