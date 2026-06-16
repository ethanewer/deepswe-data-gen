#!/usr/bin/env bash
set -euo pipefail

# Conservative 4-GPU follow-up to v42. This keeps the same strict 07:45
# raw-source, mini-swe-aligned, prefix-weighted data and v28 initialization,
# but halves the corrective LR while preserving the 8-GPU recipe's global
# packed-sequence batch size:
#   8 GPU v42: local 2 * 8 GPUs * accum 1 = 16 packs/update
#   4 GPU v43: local 1 * 4 GPUs * accum 4 = 16 packs/update
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES must be set for the local 4-GPU wrapper." >&2
  echo "Example: CUDA_VISIBLE_DEVICES=2,3,4,5 $0" >&2
  exit 1
fi

export REQUIRED_LOCAL_GPUS=4
export NPROC_PER_NODE=4
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"

export LR="${LR:-5.0e-8}"
export MIN_LR="${MIN_LR:-5.0e-9}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-5}"
export MAX_STEPS="${MAX_STEPS:-100}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen3_4b_thinking_swe260616_v43_raw0745_prefixweighted_from_v28_65k_lr5e8_s100_assistant_h200_4gpu_sft/}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260616_v43_raw0745_prefixweighted_from_v28_65k_lr5e8_s100_assistant_h200_4gpu_sft}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + ($$ % 20000)))}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen3_4b_thinking_swe260616_v42_raw0745_prefixweighted_from_v28_65k_lr1e7_sft_8gpu.sh" "$@"
