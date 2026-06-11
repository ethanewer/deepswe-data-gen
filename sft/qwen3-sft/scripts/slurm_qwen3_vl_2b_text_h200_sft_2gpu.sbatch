#!/usr/bin/env bash
#SBATCH --job-name=qwen3-vl-2b-h200-2g
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=500G
#SBATCH --time=08:00:00
#SBATCH --output=/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft/logs/slurm-qwen3-vl-2b-h200-2g-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft/logs/slurm-qwen3-vl-2b-h200-2g-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft}"
cd "$ROOT_DIR"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export REQUIRED_LOCAL_GPUS=2
export NPROC_PER_NODE=2

export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260611/highquality-3x-duplicate-reasoning-90pct}"
export VAL_RAW_ROOT="${VAL_RAW_ROOT:-$ROOT_DIR/data/smoke_raw}"
export PACK_SIZE="${PACK_SIZE:-40960}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-8}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export LR="${LR:-5.0e-6}"
export MIN_LR="${MIN_LR:-5.0e-7}"
export MAX_STEPS="${MAX_STEPS:-400}"
export VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-100}"
export VALIDATION_ENABLED="${VALIDATION_ENABLED:-false}"
export CHECKPOINT_ENABLED="${CHECKPOINT_ENABLED:-true}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints/qwen3_vl_2b_text_h200_2gpu_40k_hq3x_reasoning90_lr5e6_s400_${SLURM_JOB_ID:-manual}}"
export RUN_NAME="${RUN_NAME:-qwen3_vl_2b_text_h200_2gpu_40k_hq3x_reasoning90_lr5e6_s400_${SLURM_JOB_ID:-manual}}"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname -f 2>/dev/null || hostname)"
echo "run_name=$RUN_NAME"
echo "train_raw_root=$TRAIN_RAW_ROOT"
echo "pack_size=$PACK_SIZE"
echo "local_batch_size=$LOCAL_BATCH_SIZE"
echo "grad_accum_steps=$GRAD_ACCUM_STEPS"
echo "max_steps=$MAX_STEPS"
echo "checkpoint_dir=$CHECKPOINT_DIR"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

exec "$ROOT_DIR/scripts/run_qwen3_vl_2b_text_h200_sft.sh"
