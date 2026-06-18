#!/usr/bin/env bash
#SBATCH --job-name=eval-q3-4b-swe-s50
#SBATCH --partition=l40s-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-eval-qwen3-4b-swe260612-step50-l40s-8g-%j.out
#SBATCH --error=logs/slurm-eval-qwen3-4b-swe260612-step50-l40s-8g-%j.err

# =============================================================================
# Execution setting: SLURM-GPU (L40S, 8 GPUs) -- thin wrapper.
# Purpose: request an 8-GPU L40S allocation and exec the served-checkpoint
# driver run_qwen3_4b_swe260612_step50_l40s_eval.sh with EVAL_GPU_COUNT=8. The
# matching 4-GPU wrapper is slurm_qwen3_4b_swe260612_step50_l40s_4gpu.sbatch;
# the warm-wait variant is slurm_qwen3_4b_swe260612_warm_wait_step50_l40s_8gpu.sbatch.
# Key env vars: EVAL_GPU_COUNT (default 8) plus everything the driver accepts.
# Prerequisites: see the driver header (servable HF checkpoint, eval venv, Docker).
# =============================================================================
set -euo pipefail

DEFAULT_REPO_ROOT="/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$DEFAULT_REPO_ROOT}}"
if [ ! -d "$REPO_ROOT/eval/benchmarks/swebench_multilingual" ]; then
  REPO_ROOT="$DEFAULT_REPO_ROOT"
fi

export EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-8}"
exec "$REPO_ROOT/eval/benchmarks/swebench_multilingual/run_qwen3_4b_swe260612_step50_l40s_eval.sh"
