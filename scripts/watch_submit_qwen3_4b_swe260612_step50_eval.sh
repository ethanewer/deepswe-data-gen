#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/sft/qwen3-sft/checkpoints/qwen3_4b_thinking_swe260612_highquality_65k_online_packed_sft_fixed_h200_4gpu}"
CHECKPOINT_STEP_DIR="${CHECKPOINT_STEP_DIR:-$CHECKPOINT_DIR/epoch_0_step_49}"
EVAL_SBATCH="${EVAL_SBATCH:-$REPO_ROOT/eval/benchmarks/swebench_multilingual/slurm_qwen3_4b_swe260612_step50_l40s_4gpu.sbatch}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABILIZE_SECONDS="${STABILIZE_SECONDS:-120}"
LOG_PATH="${LOG_PATH:-$REPO_ROOT/logs/watch-qwen3-4b-swe260612-step50-eval.log}"

mkdir -p "$(dirname "$LOG_PATH")"

{
  echo "[$(date -Is)] Waiting for $CHECKPOINT_STEP_DIR/model/.metadata"
  while [ ! -f "$CHECKPOINT_STEP_DIR/model/.metadata" ]; do
    sleep "$POLL_SECONDS"
  done
  echo "[$(date -Is)] Found checkpoint metadata; waiting ${STABILIZE_SECONDS}s for writes to settle"
  sleep "$STABILIZE_SECONDS"
  echo "[$(date -Is)] Submitting L40S eval job"
  sbatch \
    --export=ALL,CHECKPOINT_STEP_DIR="$CHECKPOINT_STEP_DIR",CHECKPOINT_LABEL=step50 \
    "$EVAL_SBATCH"
} >>"$LOG_PATH" 2>&1
