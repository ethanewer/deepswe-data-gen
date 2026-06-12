#!/usr/bin/env bash
#SBATCH --job-name=qwen3-4b-swe260612-4g
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=850G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-qwen3-4b-swe260612-h200-4g-%j.out
#SBATCH --error=logs/slurm-qwen3-4b-swe260612-h200-4g-%j.err

set -euo pipefail

DEFAULT_REPO_ROOT="/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$DEFAULT_REPO_ROOT}}"
if [ ! -x "$REPO_ROOT/sft/qwen3-sft/scripts/run_qwen3_4b_thinking_swe260612_highquality_sft_8gpu.sh" ]; then
  REPO_ROOT="$DEFAULT_REPO_ROOT"
fi
QWEN_DIR="$REPO_ROOT/sft/qwen3-sft"
cd "$QWEN_DIR"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set by Slurm; refusing to guess assigned GPUs on a shared node." >&2
  exit 1
fi

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Repo root: ${REPO_ROOT}"
echo "Qwen SFT dir: ${QWEN_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

export REQUIRED_LOCAL_GPUS=4
export NPROC_PER_NODE=4

# Preserve the 8-GPU recipe's effective batch while reducing per-GPU memory:
# 8 GPU: local 2 * 8 GPUs * accum 1 = 16 packed sequences/update.
# 4 GPU: local 1 * 4 GPUs * accum 4 = 16 packed sequences/update.
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen3_4b_thinking_swe260612_highquality_65k_toolcall_only_h200_4gpu_sft/}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260612_highquality_65k_toolcall_only_h200_4gpu_sft}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"
export MAX_STEPS="${MAX_STEPS:-300}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000)))}"

exec ./scripts/run_qwen3_4b_thinking_swe260612_highquality_sft_8gpu.sh "$@"
