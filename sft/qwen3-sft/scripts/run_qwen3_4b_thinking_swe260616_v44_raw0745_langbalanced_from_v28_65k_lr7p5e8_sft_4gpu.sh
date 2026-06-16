#!/usr/bin/env bash
set -euo pipefail

# Local 4-GPU candidate for the 07:45 raw-source data. This keeps the stable
# v43 4-GPU geometry while switching from action-only prefix weighting to the
# v44 language-balanced prefix mix. The LR is between v42 and v43 so the run can
# adapt to the new mix without taking as large a corrective step as v42.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES must be set for the local 4-GPU wrapper." >&2
  echo "Example: CUDA_VISIBLE_DEVICES=0,1,2,3 $0" >&2
  exit 1
fi

export REQUIRED_LOCAL_GPUS=4
export NPROC_PER_NODE=4
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"

export MODEL="${MODEL:-/scratch/ewer/qwen3-sft-local/checkpoints/qwen3_4b_v28_s199_consolidated}"
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260616/swerebench-rawplus-exact-0745-miniswe-passed-prefix-language-balanced-v1/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/ewer/qwen3-sft-local/outputs/qwen3_4b_thinking_swe260616_v44_raw0745_langbalanced_from_v28_65k_lr7p5e8_s150_assistant_h200_4gpu_sft}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260616_v44_raw0745_langbalanced_from_v28_65k_lr7p5e8_s150_assistant_h200_4gpu_sft}"

export PACK_SIZE="${PACK_SIZE:-65536}"
export MAX_STEPS="${MAX_STEPS:-150}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"
export VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"

export LR="${LR:-7.5e-8}"
export MIN_LR="${MIN_LR:-7.5e-9}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-5}"

export CHECKPOINT_ENABLED="${CHECKPOINT_ENABLED:-true}"
export CHECKPOINT_MODEL_SAVE_FORMAT="${CHECKPOINT_MODEL_SAVE_FORMAT:-safetensors}"
export CHECKPOINT_SAVE_CONSOLIDATED="${CHECKPOINT_SAVE_CONSOLIDATED:-final}"
export CHECKPOINT_V4_COMPATIBLE="${CHECKPOINT_V4_COMPATIBLE:-true}"
export VALIDATION_ENABLED="${VALIDATION_ENABLED:-false}"

export ENABLE_COMPILE="${ENABLE_COMPILE:-true}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
export ENABLE_FSDP2_PREFETCH="${ENABLE_FSDP2_PREFETCH:-true}"
export FSDP2_BACKWARD_PREFETCH_DEPTH="${FSDP2_BACKWARD_PREFETCH_DEPTH:-3}"
export FSDP2_FORWARD_PREFETCH_DEPTH="${FSDP2_FORWARD_PREFETCH_DEPTH:-1}"

export OVERLENGTH_STRATEGY="${OVERLENGTH_STRATEGY:-split}"
export SHUFFLE_JSONL_ROWS="${SHUFFLE_JSONL_ROWS:-true}"
export DATASET_SEED="${DATASET_SEED:-61644}"
export REQUIRE_ASSISTANT_REASONING_FOR_LOSS="${REQUIRE_ASSISTANT_REASONING_FOR_LOSS:-true}"
export REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS="${REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS:-true}"
export DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS="${DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS:-true}"
export ASSISTANT_LOSS_TARGET="${ASSISTANT_LOSS_TARGET:-assistant}"
export REJECT_MANUAL_PATCH_TARGETS="${REJECT_MANUAL_PATCH_TARGETS:-true}"
export REJECT_UNVERIFIED_SUBMIT_TARGETS="${REJECT_UNVERIFIED_SUBMIT_TARGETS:-true}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

MODEL_SIZE=4b exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen3_thinking_sft_8gpu.sh" "$@"
