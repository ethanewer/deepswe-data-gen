#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/sft/qwen3-sft/checkpoints/qwen3_4b_thinking_swe260612_miniswe_aligned_65k_toolcall_only_h200_4gpu_sft}"
CHECKPOINT_STEP_DIR="${CHECKPOINT_STEP_DIR:-$CHECKPOINT_DIR/epoch_0_step_49}"
EVAL_SBATCH_8GPU="${EVAL_SBATCH_8GPU:-$REPO_ROOT/eval/benchmarks/swebench_multilingual/slurm_qwen3_4b_swe260612_step50_l40s_8gpu.sbatch}"
EVAL_SBATCH_4GPU="${EVAL_SBATCH_4GPU:-$REPO_ROOT/eval/benchmarks/swebench_multilingual/slurm_qwen3_4b_swe260612_step50_l40s_4gpu.sbatch}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABILIZE_SECONDS="${STABILIZE_SECONDS:-120}"
LOG_PATH="${LOG_PATH:-$REPO_ROOT/logs/watch-qwen3-4b-swe260612-step50-eval.log}"

mkdir -p "$(dirname "$LOG_PATH")"

choose_eval_sbatch() {
  if [ -n "${EVAL_SBATCH:-}" ]; then
    echo "$EVAL_SBATCH"
    return
  fi

  idle_8gpu_nodes=""
  if command -v sinfo >/dev/null 2>&1; then
    idle_8gpu_nodes="$(sinfo -h -p l40s-8gpu -t idle -o '%N' 2>/dev/null | head -n 1 || true)"
  fi

  if [ -n "$idle_8gpu_nodes" ]; then
    echo "$EVAL_SBATCH_8GPU"
  else
    echo "$EVAL_SBATCH_4GPU"
  fi
}

{
  echo "[$(date -Is)] Waiting for $CHECKPOINT_STEP_DIR/model/.metadata"
  while [ ! -f "$CHECKPOINT_STEP_DIR/model/.metadata" ]; do
    sleep "$POLL_SECONDS"
  done
  echo "[$(date -Is)] Found checkpoint metadata; waiting ${STABILIZE_SECONDS}s for writes to settle"
  sleep "$STABILIZE_SECONDS"
  selected_eval_sbatch="$(choose_eval_sbatch)"
  echo "[$(date -Is)] Submitting L40S eval job with $selected_eval_sbatch"
  sbatch \
    --export=ALL,CHECKPOINT_STEP_DIR="$CHECKPOINT_STEP_DIR",CHECKPOINT_LABEL=step50 \
    "$selected_eval_sbatch"
} >>"$LOG_PATH" 2>&1
