#!/usr/bin/env bash
#SBATCH --job-name=eval-q3-vl-8b-swe
#SBATCH --partition=l40s-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-eval-qwen3-vl-8b-swe260612-l40s-8g-%j.out
#SBATCH --error=logs/slurm-eval-qwen3-vl-8b-swe260612-l40s-8g-%j.err

set -euo pipefail

DEFAULT_REPO_ROOT="/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$DEFAULT_REPO_ROOT}}"
if [ ! -d "$REPO_ROOT/eval/benchmarks/swebench_multilingual" ]; then
  REPO_ROOT="$DEFAULT_REPO_ROOT"
fi

export EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-8}"
exec "$REPO_ROOT/eval/benchmarks/swebench_multilingual/run_qwen3_vl_8b_swe260612_l40s_eval.sh"
