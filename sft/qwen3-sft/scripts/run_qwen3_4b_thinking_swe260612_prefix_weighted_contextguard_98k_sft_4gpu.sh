#!/usr/bin/env bash
set -euo pipefail

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES must be set for the local 4-GPU wrapper." >&2
  echo "Example: CUDA_VISIBLE_DEVICES=0,1,2,6 $0" >&2
  exit 1
fi

export REQUIRED_LOCAL_GPUS=4
export NPROC_PER_NODE=4

# Longer-pack candidate for H200 4-GPU slices. Use 12 packed sequences/update
# so the token batch stays close to the 65k x 16-sequence baseline.
export PACK_SIZE="${PACK_SIZE:-98304}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-3}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-12}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen3_4b_thinking_swe260612_prefix_weighted_v18_contextguard_98k_assistant_h200_4gpu_sft/}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260612_prefix_weighted_v18_contextguard_98k_assistant_h200_4gpu_sft}"
export LR="${LR:-1.0e-6}"
export MIN_LR="${MIN_LR:-1.0e-7}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-20}"
export MAX_STEPS="${MAX_STEPS:-200}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + ($$ % 20000)))}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen3_4b_thinking_swe260612_prefix_weighted_balanced_sft_8gpu.sh" "$@"
