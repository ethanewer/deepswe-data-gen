#!/usr/bin/env bash
set -euo pipefail

# Run repeated swebench-multilingual-easy trials on one L40S allocation.
#
# This wrapper delegates each trial to the existing Qwen3 4B L40S eval script
# and only varies RUN_SUFFIX and BASE_PORT_OVERRIDE. The underlying script owns
# checkpoint consolidation, vLLM startup/cleanup, Mini-SWE generation, and
# SWE-bench grading for a single trial.

DEFAULT_REPO_ROOT="/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$DEFAULT_REPO_ROOT}}"
if [ ! -d "$REPO_ROOT/eval/benchmarks/swebench_multilingual" ]; then
  REPO_ROOT="$DEFAULT_REPO_ROOT"
fi

REAL_EVAL_SCRIPT="${SINGLE_TRIAL_EVAL_SCRIPT:-$REPO_ROOT/eval/benchmarks/swebench_multilingual/run_qwen3_4b_swe260612_step50_l40s_eval.sh}"
if [ ! -x "$REAL_EVAL_SCRIPT" ]; then
  echo "Missing executable eval script: $REAL_EVAL_SCRIPT" >&2
  exit 1
fi

N_TRIALS="${N_TRIALS:-3}"
if ! [[ "$N_TRIALS" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_TRIALS must be a positive integer; got $N_TRIALS" >&2
  exit 1
fi

export BENCHMARK="${BENCHMARK:-multilingual_easy}"
export EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-8}"
export EVAL_ACCELERATOR_LABEL="${EVAL_ACCELERATOR_LABEL:-l40s}"
export GENERATION_WORKERS="${GENERATION_WORKERS:-$EVAL_GPU_COUNT}"
export EVAL_WORKERS="${EVAL_WORKERS:-$EVAL_GPU_COUNT}"

RUN_SUFFIX_PREFIX="${RUN_SUFFIX_PREFIX:-easy-n${N_TRIALS}}"
BASE_PORT_SEED=$((20000 + (${SLURM_JOB_ID:-0} % 20000)))

for trial in $(seq 1 "$N_TRIALS"); do
  export RUN_SUFFIX="${RUN_SUFFIX_PREFIX}-t${trial}"
  export BASE_PORT_OVERRIDE="$((BASE_PORT_SEED + (trial - 1) * 1000))"
  echo "[$(date -Is)] Starting ${BENCHMARK} trial ${trial}/${N_TRIALS}: RUN_SUFFIX=$RUN_SUFFIX BASE_PORT_OVERRIDE=$BASE_PORT_OVERRIDE"
  "$REAL_EVAL_SCRIPT"
done
