#!/usr/bin/env bash
#SBATCH --job-name=eval-q3-4b-swe-s50
#SBATCH --partition=l40s-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-eval-qwen3-4b-swe260612-step50-l40s-4g-%j.out
#SBATCH --error=logs/slurm-eval-qwen3-4b-swe260612-step50-l40s-4g-%j.err

# =============================================================================
# Execution setting: SLURM-GPU (L40S, 4 GPUs) -- thin wrapper.
# Purpose: request a 4-GPU L40S allocation and exec the served-checkpoint driver
# run_qwen3_4b_swe260612_step50_l40s_eval.sh with EVAL_GPU_COUNT=4. The
# matching 8-GPU wrapper is slurm_qwen3_4b_swe260612_step50_l40s_8gpu.sbatch.
# Key env vars: EVAL_GPU_COUNT (default 4) plus everything the driver accepts
# (CHECKPOINT_STEP_DIR, CHECKPOINT_LABEL, BENCHMARK, ...).
# Prerequisites: see the driver header (servable HF checkpoint, eval venv, Docker).
# =============================================================================
set -euo pipefail

DEFAULT_REPO_ROOT="/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$DEFAULT_REPO_ROOT}}"
if [ ! -d "$REPO_ROOT/eval/benchmarks/swebench_multilingual" ]; then
  REPO_ROOT="$DEFAULT_REPO_ROOT"
fi
export EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-4}"
exec "$REPO_ROOT/eval/benchmarks/swebench_multilingual/run_qwen3_4b_swe260612_step50_l40s_eval.sh"
