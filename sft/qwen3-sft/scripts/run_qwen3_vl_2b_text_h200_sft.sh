#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODEL_SIZE=4b
export CONFIG="${CONFIG:-configs/qwen3_vl_2b_text_h200_40k_online_packed_sft_4gpu.yaml}"
export MODEL="${MODEL:-$ROOT_DIR/data/qwen3_vl_2b_text_checkpoint}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen3_vl_2b_text_h200_40k_online_packed_sft/}"
export RUN_NAME="${RUN_NAME:-qwen3_vl_2b_text_h200_40k_sft}"

visible="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a visible_gpus <<< "$visible"
gpu_count="${NPROC_PER_NODE:-${#visible_gpus[@]}}"
export REQUIRED_LOCAL_GPUS="${REQUIRED_LOCAL_GPUS:-$gpu_count}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$REQUIRED_LOCAL_GPUS}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((REQUIRED_LOCAL_GPUS - 1))")"
  export CUDA_VISIBLE_DEVICES
fi

export PACK_SIZE="${PACK_SIZE:-40960}"
if [ "$PACK_SIZE" -lt 32768 ]; then
  echo "PACK_SIZE must be at least 32768 for the H200 Qwen3-VL 2B text recipe; got $PACK_SIZE" >&2
  exit 1
fi

export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-8}"
if [ "$REQUIRED_LOCAL_GPUS" -le 2 ]; then
  export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
else
  export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
fi

export ENABLE_COMPILE="${ENABLE_COMPILE:-false}"
export ENABLE_FSDP2_PREFETCH="${ENABLE_FSDP2_PREFETCH:-true}"
export FSDP2_BACKWARD_PREFETCH_DEPTH="${FSDP2_BACKWARD_PREFETCH_DEPTH:-3}"
export FSDP2_FORWARD_PREFETCH_DEPTH="${FSDP2_FORWARD_PREFETCH_DEPTH:-1}"
export OVERLENGTH_STRATEGY="${OVERLENGTH_STRATEGY:-truncate}"
export REQUIRE_ASSISTANT_REASONING_FOR_LOSS="${REQUIRE_ASSISTANT_REASONING_FOR_LOSS:-true}"
export REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS="${REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS:-true}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_3}"
export CHAT_TEMPLATE_SOURCE="${CHAT_TEMPLATE_SOURCE:-file}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2}"

export QWEN3_VL_TEXT_USE_NATIVE_RMSNORM="${QWEN3_VL_TEXT_USE_NATIVE_RMSNORM:-1}"
export QWEN3_VL_TEXT_DISABLE_MLP_COMPILE="${QWEN3_VL_TEXT_DISABLE_MLP_COMPILE:-1}"
export QWEN3_VL_TEXT_MLP_CHUNK_TOKENS="${QWEN3_VL_TEXT_MLP_CHUNK_TOKENS:-40960}"

if [ ! -f "$MODEL/model.safetensors.index.json" ]; then
  "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/prepare_qwen3_vl_text_checkpoint.py" \
    --repo-id Qwen/Qwen3-VL-2B-Thinking \
    --output-dir "$MODEL"
fi

exec "$SCRIPT_DIR/run_qwen3_thinking_sft_8gpu.sh" "$@"
