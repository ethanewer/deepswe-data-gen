#!/usr/bin/env bash
#SBATCH --job-name=qwen3-4b-l40s-sft
#SBATCH --partition=l40s-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:l40s:8
#SBATCH --mem=1200G
#SBATCH --time=04:00:00
#SBATCH --output=/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft/logs/slurm-qwen3-4b-l40s-sft-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft/logs/slurm-qwen3-4b-l40s-sft-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft}"
cd "$ROOT_DIR"

export REQUIRE_L40S="${REQUIRE_L40S:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CONFIG="${CONFIG:-configs/qwen3_4b_thinking_l40s_32k_online_packed_sft_8gpu.yaml}"
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/code-swe-terminal-agentic-sft}"
export VAL_RAW_ROOT="${VAL_RAW_ROOT:-$ROOT_DIR/data/smoke_raw}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2}"
export PACK_SIZE="${PACK_SIZE:-32768}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export ENABLE_COMPILE="${ENABLE_COMPILE:-false}"
export ENABLE_FSDP2_PREFETCH="${ENABLE_FSDP2_PREFETCH:-false}"
export MAX_STEPS="${MAX_STEPS:-1000}"
export VALIDATION_ENABLED="${VALIDATION_ENABLED:-false}"
export CHECKPOINT_ENABLED="${CHECKPOINT_ENABLED:-false}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_l40s_32k_sft_${SLURM_JOB_ID:-manual}}"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname -f 2>/dev/null || hostname)"
echo "root=$ROOT_DIR"
echo "run_name=$RUN_NAME"
echo "train_raw_root=$TRAIN_RAW_ROOT"
echo "pack_size=$PACK_SIZE"
echo "grad_accum_steps=$GRAD_ACCUM_STEPS"
echo "max_steps=$MAX_STEPS"
echo "chat_template=$CHAT_TEMPLATE"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | sed -n '1,8p'

exec "$ROOT_DIR/scripts/run_qwen3_4b_thinking_l40s_sft_8gpu.sh"
