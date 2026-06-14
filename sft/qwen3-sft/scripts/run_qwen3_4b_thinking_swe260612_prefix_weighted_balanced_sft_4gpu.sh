#!/usr/bin/env bash
set -euo pipefail

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES must be set for the local 4-GPU wrapper." >&2
  echo "Example: CUDA_VISIBLE_DEVICES=0,1,2,6 $0" >&2
  exit 1
fi

export REQUIRED_LOCAL_GPUS=4
export NPROC_PER_NODE=4

# Match the 8-GPU recipe's global batch:
# 8 GPU: local 2 * 8 GPUs * accum 1 = 16 packed sequences/update.
# 4 GPU: local 1 * 4 GPUs * accum 4 = 16 packed sequences/update.
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen3_4b_thinking_swe260612_prefix_weighted_v15_contextguard_65k_assistant_h200_4gpu_sft/}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260612_prefix_weighted_v15_contextguard_65k_assistant_h200_4gpu_sft}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + ($$ % 20000)))}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen3_4b_thinking_swe260612_prefix_weighted_balanced_sft_8gpu.sh" "$@"
