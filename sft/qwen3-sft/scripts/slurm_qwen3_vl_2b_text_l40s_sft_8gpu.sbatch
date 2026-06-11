#!/usr/bin/env bash
#SBATCH --job-name=qwen3-vl-2b-text
#SBATCH --partition=l40s-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:l40s:8
#SBATCH --mem=1200G
#SBATCH --time=04:00:00
#SBATCH --output=/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft/logs/slurm-qwen3-vl-2b-text-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft/logs/slurm-qwen3-vl-2b-text-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft}"
cd "$ROOT_DIR"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-$ROOT_DIR/data/smoke_raw}"
export VAL_RAW_ROOT="${VAL_RAW_ROOT:-$ROOT_DIR/data/smoke_raw}"
export MAX_STEPS="${MAX_STEPS:-25}"
export VALIDATION_ENABLED="${VALIDATION_ENABLED:-false}"
export CHECKPOINT_ENABLED="${CHECKPOINT_ENABLED:-false}"
export RUN_NAME="${RUN_NAME:-qwen3_vl_2b_text_l40s_40k_smoke_${SLURM_JOB_ID:-manual}}"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname -f 2>/dev/null || hostname)"
echo "run_name=$RUN_NAME"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | sed -n '1,8p'

exec "$ROOT_DIR/scripts/run_qwen3_vl_2b_text_l40s_sft_8gpu.sh"
