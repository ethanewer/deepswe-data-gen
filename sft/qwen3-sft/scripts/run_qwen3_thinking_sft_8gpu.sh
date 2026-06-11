#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. Run ./scripts/setup_nemo_automodel_env.sh first." >&2
  exit 1
fi
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
AUTOMODEL_BIN="$ROOT_DIR/.venv/bin/automodel"

export PYTHONPATH="$ROOT_DIR/third_party/Automodel:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#visible_gpus[@]}}"
REQUIRED_LOCAL_GPUS="${REQUIRED_LOCAL_GPUS:-8}"
if [ "$NPROC_PER_NODE" -ne "$REQUIRED_LOCAL_GPUS" ] || [ "${#visible_gpus[@]}" -ne "$REQUIRED_LOCAL_GPUS" ]; then
  echo "This recipe must use exactly $REQUIRED_LOCAL_GPUS local GPUs. CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
  if [ "$gpu_count" -lt "$REQUIRED_LOCAL_GPUS" ]; then
    echo "Expected at least $REQUIRED_LOCAL_GPUS local GPUs from nvidia-smi, saw $gpu_count" >&2
    exit 1
  fi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | sed -n "1,${REQUIRED_LOCAL_GPUS}p"
fi

MODEL_SIZE="${MODEL_SIZE:-4b}"
case "$MODEL_SIZE" in
  4b)
    DEFAULT_CONFIG="configs/qwen3_4b_thinking_online_packed_sft_8gpu.yaml"
    DEFAULT_MODEL="Qwen/Qwen3-4B-Thinking-2507"
    DEFAULT_CHECKPOINT_DIR="checkpoints/qwen3_4b_thinking_online_packed_sft/"
    DEFAULT_RUN_PREFIX="qwen3_4b_thinking_sft"
    DEFAULT_PACK_SIZE="131072"
    DEFAULT_LOCAL_BATCH_SIZE="1"
    DEFAULT_GRAD_ACCUM_STEPS="1"
    DEFAULT_ENABLE_COMPILE="true"
    DEFAULT_FSDP2_BACKWARD_PREFETCH_DEPTH="3"
    DEFAULT_FSDP2_FORWARD_PREFETCH_DEPTH="1"
    ;;
  8b)
    DEFAULT_CONFIG="configs/qwen3_8b_thinking_online_packed_sft_8gpu.yaml"
    DEFAULT_MODEL="$ROOT_DIR/data/qwen3_vl_8b_text_checkpoint"
    DEFAULT_CHECKPOINT_DIR="checkpoints/qwen3_8b_thinking_online_packed_sft/"
    DEFAULT_RUN_PREFIX="qwen3_8b_thinking_sft"
    DEFAULT_PACK_SIZE="65536"
    DEFAULT_LOCAL_BATCH_SIZE="1"
    DEFAULT_GRAD_ACCUM_STEPS="2"
    DEFAULT_ENABLE_COMPILE="true"
    DEFAULT_FSDP2_BACKWARD_PREFETCH_DEPTH="2"
    DEFAULT_FSDP2_FORWARD_PREFETCH_DEPTH="1"
    ;;
  *)
    echo "MODEL_SIZE must be 4b or 8b; got $MODEL_SIZE" >&2
    exit 1
    ;;
esac

CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
MODEL="${MODEL:-$DEFAULT_MODEL}"
TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/code-swe-terminal-agentic-sft}"
VAL_RAW_ROOT="${VAL_RAW_ROOT:-$ROOT_DIR/data/smoke_raw}"
CHAT_TEMPLATE_SOURCE="${CHAT_TEMPLATE_SOURCE:-file}"
if [ "$CHAT_TEMPLATE_SOURCE" = "tokenizer" ]; then
  CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
else
  CHAT_TEMPLATE="${CHAT_TEMPLATE:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2}"
fi

PACK_SIZE="${PACK_SIZE:-$DEFAULT_PACK_SIZE}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-$DEFAULT_LOCAL_BATCH_SIZE}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$DEFAULT_GRAD_ACCUM_STEPS}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((LOCAL_BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM_STEPS))}"
GLOBAL_TOKENS="$((GLOBAL_BATCH_SIZE * PACK_SIZE))"
MAX_STEPS="${MAX_STEPS:-1000}"
LR="${LR:-1.0e-5}"
MIN_LR="${MIN_LR:-$LR}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MULTIPROCESSING_CONTEXT="${MULTIPROCESSING_CONTEXT:-fork}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
OVERLENGTH_STRATEGY="${OVERLENGTH_STRATEGY:-}"
REQUIRE_ASSISTANT_REASONING_FOR_LOSS="${REQUIRE_ASSISTANT_REASONING_FOR_LOSS:-}"
REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS="${REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS:-}"
ENABLE_COMPILE="${ENABLE_COMPILE:-$DEFAULT_ENABLE_COMPILE}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
ENABLE_FSDP2_PREFETCH="${ENABLE_FSDP2_PREFETCH:-true}"
FSDP2_BACKWARD_PREFETCH_DEPTH="${FSDP2_BACKWARD_PREFETCH_DEPTH:-$DEFAULT_FSDP2_BACKWARD_PREFETCH_DEPTH}"
FSDP2_FORWARD_PREFETCH_DEPTH="${FSDP2_FORWARD_PREFETCH_DEPTH:-$DEFAULT_FSDP2_FORWARD_PREFETCH_DEPTH}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_3}"
OUTPUT_HIDDEN_STATES="${OUTPUT_HIDDEN_STATES:-true}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"
CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-1000}"
CHECKPOINT_ENABLED="${CHECKPOINT_ENABLED:-false}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DEFAULT_CHECKPOINT_DIR}"
CHECKPOINT_MODEL_SAVE_FORMAT="${CHECKPOINT_MODEL_SAVE_FORMAT:-torch_save}"
CHECKPOINT_SAVE_CONSOLIDATED="${CHECKPOINT_SAVE_CONSOLIDATED:-false}"
CHECKPOINT_V4_COMPATIBLE="${CHECKPOINT_V4_COMPATIBLE:-false}"
RESTORE_FROM="${RESTORE_FROM:-}"
VALIDATION_ENABLED="${VALIDATION_ENABLED:-false}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_PREFIX}_pack${PACK_SIZE}_gbs${GLOBAL_BATCH_SIZE}}"

if [ "$MODEL_SIZE" = "8b" ] && [ "$ENABLE_COMPILE" = "true" ]; then
  export QWEN3_VL_TEXT_USE_NATIVE_RMSNORM="${QWEN3_VL_TEXT_USE_NATIVE_RMSNORM:-1}"
  export QWEN3_VL_TEXT_DISABLE_MLP_COMPILE="${QWEN3_VL_TEXT_DISABLE_MLP_COMPILE:-1}"
  export QWEN3_VL_TEXT_MLP_CHUNK_TOKENS="${QWEN3_VL_TEXT_MLP_CHUNK_TOKENS:-65536}"
fi

if [ "$MODEL_SIZE" = "8b" ] && [ "$MODEL" = "$ROOT_DIR/data/qwen3_vl_8b_text_checkpoint" ] && [ ! -f "$MODEL/model.safetensors.index.json" ]; then
  "$VENV_PYTHON" scripts/prepare_qwen3_vl_text_checkpoint.py --output-dir "$MODEL"
fi

if [ "$GLOBAL_TOKENS" -lt 1000000 ] || [ "$GLOBAL_TOKENS" -gt 5000000 ]; then
  echo "Global token batch must be between 1M and 5M; got $GLOBAL_TOKENS" >&2
  exit 1
fi

echo "Run: $RUN_NAME"
echo "Packed sequence length: $PACK_SIZE"
echo "Global packed sequences: $GLOBAL_BATCH_SIZE"
echo "Global tokens/update: $GLOBAL_TOKENS"
echo "Compile: $ENABLE_COMPILE"
echo "FSDP2 prefetch: $ENABLE_FSDP2_PREFETCH B${FSDP2_BACKWARD_PREFETCH_DEPTH}/F${FSDP2_FORWARD_PREFETCH_DEPTH}"
if [ -n "$OVERLENGTH_STRATEGY" ]; then
  echo "Overlength strategy: $OVERLENGTH_STRATEGY"
fi
if [ -n "$REQUIRE_ASSISTANT_REASONING_FOR_LOSS" ] || [ -n "$REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS" ]; then
  echo "Assistant loss requirements: reasoning=${REQUIRE_ASSISTANT_REASONING_FOR_LOSS:-config} tool_calls=${REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS:-config}"
fi
if [ "$CHAT_TEMPLATE_SOURCE" = "tokenizer" ]; then
  echo "Chat template: tokenizer default"
else
  echo "Chat template: $CHAT_TEMPLATE"
fi
if [ "$MODEL_SIZE" = "8b" ]; then
  echo "8B text memory path: native_rmsnorm=${QWEN3_VL_TEXT_USE_NATIVE_RMSNORM:-0} disable_mlp_compile=${QWEN3_VL_TEXT_DISABLE_MLP_COMPILE:-0} mlp_chunk_tokens=${QWEN3_VL_TEXT_MLP_CHUNK_TOKENS:-0}"
fi

CONFIG_FOR_RUN="$CONFIG"
if [ "$VALIDATION_ENABLED" = "false" ]; then
  CONFIG_FOR_RUN="$(mktemp -t qwen3_thinking_sft_no_val.XXXXXX.yaml)"
  "$VENV_PYTHON" - "$CONFIG" "$CONFIG_FOR_RUN" <<'PY'
import sys

import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
for key in list(cfg):
    if key.startswith("validation_dataset") or key == "validation_dataloader":
        del cfg[key]
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  trap 'rm -f "$CONFIG_FOR_RUN"' EXIT
fi

args=(
  "$CONFIG_FOR_RUN"
  --model.pretrained_model_name_or_path "$MODEL"
  --model.attn_implementation "$ATTN_IMPLEMENTATION"
  --model.output_hidden_states "$OUTPUT_HIDDEN_STATES"
  --dataset.raw_root "$TRAIN_RAW_ROOT"
  --dataset.sequence_length "$PACK_SIZE"
  --step_scheduler.global_batch_size "$GLOBAL_BATCH_SIZE"
  --step_scheduler.local_batch_size "$LOCAL_BATCH_SIZE"
  --step_scheduler.max_steps "$MAX_STEPS"
  --step_scheduler.val_every_steps "$VAL_EVERY_STEPS"
  --step_scheduler.ckpt_every_steps "$CKPT_EVERY_STEPS"
  --dataloader.batch_size "$LOCAL_BATCH_SIZE"
  --dataloader.num_workers "$NUM_WORKERS"
  --dataloader.multiprocessing_context "$MULTIPROCESSING_CONTEXT"
  --dataloader.persistent_workers "$PERSISTENT_WORKERS"
  --dataloader.prefetch_factor "$PREFETCH_FACTOR"
  --distributed.enable_compile "$ENABLE_COMPILE"
  --distributed.activation_checkpointing "$ACTIVATION_CHECKPOINTING"
  --distributed.enable_fsdp2_prefetch "$ENABLE_FSDP2_PREFETCH"
  --distributed.fsdp2_backward_prefetch_depth "$FSDP2_BACKWARD_PREFETCH_DEPTH"
  --distributed.fsdp2_forward_prefetch_depth "$FSDP2_FORWARD_PREFETCH_DEPTH"
  --optimizer.lr "$LR"
  --lr_scheduler.min_lr "$MIN_LR"
  --checkpoint.enabled "$CHECKPOINT_ENABLED"
  --checkpoint.checkpoint_dir "$CHECKPOINT_DIR"
  --checkpoint.model_save_format "$CHECKPOINT_MODEL_SAVE_FORMAT"
  --checkpoint.save_consolidated "$CHECKPOINT_SAVE_CONSOLIDATED"
  --checkpoint.v4_compatible "$CHECKPOINT_V4_COMPATIBLE"
)

if [ -n "$OVERLENGTH_STRATEGY" ]; then
  args+=(--dataset.overlength_strategy "$OVERLENGTH_STRATEGY")
fi

if [ -n "$REQUIRE_ASSISTANT_REASONING_FOR_LOSS" ]; then
  args+=(--dataset.require_assistant_reasoning_for_loss "$REQUIRE_ASSISTANT_REASONING_FOR_LOSS")
fi

if [ -n "$REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS" ]; then
  args+=(--dataset.require_assistant_tool_calls_for_loss "$REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS")
fi

if [ "$CHAT_TEMPLATE_SOURCE" != "tokenizer" ]; then
  args+=(--dataset.chat_template_path "$CHAT_TEMPLATE")
fi

if [ -n "$RESTORE_FROM" ]; then
  args+=(--checkpoint.restore_from "$RESTORE_FROM")
fi

if [ "$VALIDATION_ENABLED" = "true" ]; then
  args+=(
    --validation_dataset.raw_root "$VAL_RAW_ROOT"
    --validation_dataset.sequence_length "$PACK_SIZE"
  )
  if [ -n "$OVERLENGTH_STRATEGY" ]; then
    args+=(--validation_dataset.overlength_strategy "$OVERLENGTH_STRATEGY")
  fi
  if [ -n "$REQUIRE_ASSISTANT_REASONING_FOR_LOSS" ]; then
    args+=(--validation_dataset.require_assistant_reasoning_for_loss "$REQUIRE_ASSISTANT_REASONING_FOR_LOSS")
  fi
  if [ -n "$REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS" ]; then
    args+=(--validation_dataset.require_assistant_tool_calls_for_loss "$REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS")
  fi
  if [ "$CHAT_TEMPLATE_SOURCE" != "tokenizer" ]; then
    args+=(--validation_dataset.chat_template_path "$CHAT_TEMPLATE")
  fi
fi

if "$VENV_PYTHON" "$AUTOMODEL_BIN" --help 2>&1 | grep -q -- "--nproc-per-node"; then
  exec "$VENV_PYTHON" "$AUTOMODEL_BIN" --nproc-per-node "$NPROC_PER_NODE" "${args[@]}"
fi

exec "$VENV_PYTHON" "$AUTOMODEL_BIN" "${args[@]}"
