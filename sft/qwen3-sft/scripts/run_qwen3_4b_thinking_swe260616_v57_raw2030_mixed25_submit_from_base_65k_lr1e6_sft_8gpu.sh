#!/usr/bin/env bash
set -euo pipefail

# Larger mixed-quality recipe for the 2026-06-16 20:30 raw2030 source.
#
# Data view:
#   swerebench-raw2030-targeted-limitations-mixed25-miniswe-aligned
#
# Selection policy:
#   - all strict passed rows from raw2030
#   - as many high-quality non-passing rows as the selector can supply toward
#     a 25% passrate target; current clean-candidate supply gives ~26%
#   - failed rows are ranked, not random: submitted, non-empty-patch,
#     high-reasoning structural traces with infra, empty-patch, not-submitted,
#     low-reasoning, huge-patch, and huge-trajectory rows excluded
#
# Loss policy:
#   - keep full trajectories in context
#   - do not use prefix expansion
#   - train on selected non-passing submit turns as well as passed submit turns
#   - still mask concrete bad targets: manual patch writing, unverified submits,
#     assistant turns without reasoning, and assistant turns without tool calls

export CONFIG="${CONFIG:-configs/qwen3_4b_thinking_swe260612_highquality_65k_online_packed_sft_8gpu.yaml}"
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260616/swerebench-raw2030-targeted-limitations-mixed25-miniswe-aligned/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/ewer/qwen3-sft-local/outputs/qwen3_4b_thinking_swe260616_v57_raw2030_mixed25_submit_from_base_65k_lr1e6_s1400_assistant_h200_8gpu_sft}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260616_v57_raw2030_mixed25_submit_from_base_65k_lr1e6_s1400_assistant_h200_8gpu_sft}"

export PACK_SIZE="${PACK_SIZE:-65536}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export MAX_STEPS="${MAX_STEPS:-1400}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"
export VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"

export LR="${LR:-1.0e-6}"
export MIN_LR="${MIN_LR:-1.0e-7}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-50}"

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
export SHUFFLE_FILES="${SHUFFLE_FILES:-false}"
export SHUFFLE_JSONL_ROWS="${SHUFFLE_JSONL_ROWS:-false}"
export DATASET_SEED="${DATASET_SEED:-61657}"
export REQUIRE_ASSISTANT_REASONING_FOR_LOSS="${REQUIRE_ASSISTANT_REASONING_FOR_LOSS:-true}"
export REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS="${REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS:-true}"
export DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS="${DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS:-true}"
export ASSISTANT_LOSS_TARGET="${ASSISTANT_LOSS_TARGET:-assistant}"
export REJECT_MANUAL_PATCH_TARGETS="${REJECT_MANUAL_PATCH_TARGETS:-true}"
export REJECT_UNVERIFIED_SUBMIT_TARGETS="${REJECT_UNVERIFIED_SUBMIT_TARGETS:-true}"
export REJECT_NONPASSING_SUBMIT_TARGETS="${REJECT_NONPASSING_SUBMIT_TARGETS:-false}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

MODEL_SIZE=4b exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen3_thinking_sft_8gpu.sh" "$@"
