#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_SIZE=4b
export CONFIG="${CONFIG:-configs/qwen3_4b_thinking_l40s_32k_online_packed_sft_8gpu.yaml}"
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen3_4b_thinking_l40s_32k_online_packed_sft/}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_l40s_32k_sft}"

export PACK_SIZE="${PACK_SIZE:-32768}"
if [ "$PACK_SIZE" -lt 32768 ]; then
  echo "PACK_SIZE must be at least 32768 for the L40S Qwen3 4B recipe; got $PACK_SIZE" >&2
  exit 1
fi
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export ENABLE_COMPILE="${ENABLE_COMPILE:-false}"
export ENABLE_FSDP2_PREFETCH="${ENABLE_FSDP2_PREFETCH:-false}"
export FSDP2_BACKWARD_PREFETCH_DEPTH="${FSDP2_BACKWARD_PREFETCH_DEPTH:-2}"
export FSDP2_FORWARD_PREFETCH_DEPTH="${FSDP2_FORWARD_PREFETCH_DEPTH:-1}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_3}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2}"

if command -v nvidia-smi >/dev/null 2>&1 && [ "${REQUIRE_L40S:-0}" = "1" ]; then
  if ! nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1,8p' | grep -qi 'L40S'; then
    echo "REQUIRE_L40S=1 but the first visible GPUs are not L40S." >&2
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | sed -n '1,8p' >&2
    exit 1
  fi
fi

exec "$SCRIPT_DIR/run_qwen3_thinking_sft_8gpu.sh" "$@"
