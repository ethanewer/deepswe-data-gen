#!/usr/bin/env bash
#SBATCH --job-name=q3vl8b-v75-h200
#SBATCH --partition=h200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=1800G
#SBATCH --time=06:00:00
#SBATCH --output=/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft/logs/slurm-qwen3-vl-8b-v75-h200-2node-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft/logs/slurm-qwen3-vl-8b-v75-h200-2node-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft}"
cd "$ROOT_DIR"

if [ "${SLURM_NNODES:-0}" -ne 2 ]; then
  echo "This recipe must run on exactly 2 H200 nodes; got SLURM_NNODES=${SLURM_NNODES:-unset}" >&2
  exit 1
fi

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Missing .venv. Run ./scripts/setup_nemo_automodel_env.sh first." >&2
  exit 1
fi

DEFAULT_AUTOMODEL_ROOT="$ROOT_DIR/third_party/Automodel"
SHARED_AUTOMODEL_ROOT="/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft/third_party/Automodel"
if [ -z "${AUTOMODEL_ROOT:-}" ] && [ ! -d "$DEFAULT_AUTOMODEL_ROOT/nemo_automodel" ] && [ -d "$SHARED_AUTOMODEL_ROOT/nemo_automodel" ]; then
  AUTOMODEL_ROOT="$SHARED_AUTOMODEL_ROOT"
else
  AUTOMODEL_ROOT="${AUTOMODEL_ROOT:-$DEFAULT_AUTOMODEL_ROOT}"
fi
if [ ! -d "$AUTOMODEL_ROOT/nemo_automodel" ]; then
  echo "Missing nemo_automodel under AUTOMODEL_ROOT=$AUTOMODEL_ROOT" >&2
  exit 1
fi

export PYTHONPATH="$AUTOMODEL_ROOT:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

SHARED_HF_HOME="/wbl-fast/usrs/ee/code-swe-data/.cache/huggingface"
if [ -d "$SHARED_HF_HOME/hub" ]; then
  export HF_HOME="${HF_HOME:-$SHARED_HF_HOME}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$SHARED_HF_HOME/hub}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$SHARED_HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SHARED_HF_HOME/hub}"
fi

export QWEN3_VL_TEXT_USE_NATIVE_RMSNORM="${QWEN3_VL_TEXT_USE_NATIVE_RMSNORM:-1}"
export QWEN3_VL_TEXT_DISABLE_MLP_COMPILE="${QWEN3_VL_TEXT_DISABLE_MLP_COMPILE:-1}"
export QWEN3_VL_TEXT_MLP_CHUNK_TOKENS="${QWEN3_VL_TEXT_MLP_CHUNK_TOKENS:-65536}"

CONFIG="${CONFIG:-configs/qwen3_vl_8b_thinking_text_swe260617_v75_hf_processed_65k_sft_2node_h200.yaml}"
MODEL="${MODEL:-$ROOT_DIR/data/qwen3_vl_8b_text_checkpoint}"
TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/qwen3-4b-thinking-sft-v75-verification-strictpassed-cap4-processed/data}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2}"
PACK_SIZE="${PACK_SIZE:-65536}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-100}"
CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"
LR="${LR:-1.0e-6}"
MIN_LR="${MIN_LR:-1.0e-7}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-25}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
RUN_NAME="${RUN_NAME:-qwen3_vl_8b_thinking_text_swe260617_v75_hf_processed_65k_lr${LR}_s${MAX_STEPS}_2node_h200_sft_${SLURM_JOB_ID:-manual}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/ewer/qwen3-sft-local/outputs/${RUN_NAME}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
GLOBAL_TOKENS="$((GLOBAL_BATCH_SIZE * PACK_SIZE))"
export VENV_PYTHON CONFIG MODEL TRAIN_RAW_ROOT CHAT_TEMPLATE
export PACK_SIZE GLOBAL_BATCH_SIZE LOCAL_BATCH_SIZE MAX_STEPS CKPT_EVERY_STEPS VAL_EVERY_STEPS
export LR MIN_LR LR_WARMUP_STEPS WEIGHT_DECAY CHECKPOINT_DIR GPUS_PER_NODE

if [ "$GPUS_PER_NODE" -ne 8 ]; then
  echo "This H200 recipe expects 8 GPUs per node; got GPUS_PER_NODE=$GPUS_PER_NODE" >&2
  exit 1
fi
if [ "$GLOBAL_TOKENS" -ne 1048576 ]; then
  echo "Expected 1,048,576 tokens/update, got $GLOBAL_TOKENS" >&2
  exit 1
fi
if [ ! -d "$TRAIN_RAW_ROOT" ]; then
  echo "Missing TRAIN_RAW_ROOT=$TRAIN_RAW_ROOT" >&2
  exit 1
fi
if [ ! -f "$CHAT_TEMPLATE" ]; then
  echo "Missing chat template: $CHAT_TEMPLATE" >&2
  exit 1
fi
if [ ! -f "$MODEL/model.safetensors.index.json" ]; then
  "$VENV_PYTHON" scripts/prepare_qwen3_vl_text_checkpoint.py --output-dir "$MODEL"
fi

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"
export MASTER_ADDR MASTER_PORT

echo "job_id=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NODELIST"
echo "master=$MASTER_ADDR:$MASTER_PORT"
echo "run_name=$RUN_NAME"
echo "config=$CONFIG"
echo "model=$MODEL"
echo "train_raw_root=$TRAIN_RAW_ROOT"
echo "checkpoint_dir=$CHECKPOINT_DIR"
echo "lr=$LR min_lr=$MIN_LR warmup=$LR_WARMUP_STEPS weight_decay=$WEIGHT_DECAY"
echo "pack_size=$PACK_SIZE global_batch_size=$GLOBAL_BATCH_SIZE global_tokens=$GLOBAL_TOKENS"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c '
  set -euo pipefail
  echo "rank_node=${SLURM_PROCID} host=$(hostname -f 2>/dev/null || hostname)"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | sed -n "1,8p"
  exec "$VENV_PYTHON" -m torch.distributed.run \
    --nnodes "$SLURM_NNODES" \
    --nproc-per-node "$GPUS_PER_NODE" \
    --node-rank "$SLURM_PROCID" \
    --rdzv-id "$SLURM_JOB_ID" \
    --rdzv-backend c10d \
    --rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT" \
    -m nemo_automodel.cli.app \
    "$CONFIG" \
    --model.pretrained_model_name_or_path "$MODEL" \
    --dataset.raw_root "$TRAIN_RAW_ROOT" \
    --dataset.chat_template_path "$CHAT_TEMPLATE" \
    --dataset.sequence_length "$PACK_SIZE" \
    --step_scheduler.global_batch_size "$GLOBAL_BATCH_SIZE" \
    --step_scheduler.local_batch_size "$LOCAL_BATCH_SIZE" \
    --step_scheduler.max_steps "$MAX_STEPS" \
    --step_scheduler.ckpt_every_steps "$CKPT_EVERY_STEPS" \
    --step_scheduler.val_every_steps "$VAL_EVERY_STEPS" \
    --dataloader.batch_size "$LOCAL_BATCH_SIZE" \
    --optimizer.lr "$LR" \
    --optimizer.weight_decay "$WEIGHT_DECAY" \
    --lr_scheduler.min_lr "$MIN_LR" \
    --lr_scheduler.lr_warmup_steps "$LR_WARMUP_STEPS" \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint.model_save_format safetensors \
    --checkpoint.save_consolidated every \
    --checkpoint.v4_compatible true \
    --checkpoint.dcp_process_group_backend gloo
'
