#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f .venv/bin/activate ]; then
  echo "Missing .venv. Run ./scripts/setup_nemo_automodel_env.sh first." >&2
  exit 1
fi

source .venv/bin/activate
VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/.venv/bin/python}"
AUTOMODEL_BIN="${AUTOMODEL_BIN:-$ROOT_DIR/.venv/bin/automodel}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export PYTHONPATH="$ROOT_DIR/third_party/Automodel${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-configs/qwen36_27b_fp8_lora_sft_single_gpu.yaml}"
TRAIN_DATA="${TRAIN_DATA:-data/smoke_train.jsonl}"
VAL_DATA="${VAL_DATA:-data/smoke_val.jsonl}"
PACK_SIZE="${PACK_SIZE:-8192}"
SEQ_LENGTH="${SEQ_LENGTH:-$PACK_SIZE}"
MAX_STEPS="${MAX_STEPS:-10}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen36_27b_fp8_lora_sft/}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
CHECKPOINT_ENABLED="${CHECKPOINT_ENABLED:-true}"

if [ "$CUDA_VISIBLE_DEVICES" != "0" ]; then
  echo "This single-GPU recipe is pinned to GPU 0. CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

nvidia-smi -i 0 --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

exec "$VENV_PYTHON" "$AUTOMODEL_BIN" "$CONFIG" \
  --model.attn_implementation "$ATTN_IMPLEMENTATION" \
  --dataset.path_or_dataset_id "$TRAIN_DATA" \
  --validation_dataset.path_or_dataset_id "$VAL_DATA" \
  --dataset.seq_length "$SEQ_LENGTH" \
  --validation_dataset.seq_length "$SEQ_LENGTH" \
  --packed_sequence.packed_sequence_size "$PACK_SIZE" \
  --step_scheduler.max_steps "$MAX_STEPS" \
  --checkpoint.enabled "$CHECKPOINT_ENABLED" \
  --checkpoint.checkpoint_dir "$CHECKPOINT_DIR"
