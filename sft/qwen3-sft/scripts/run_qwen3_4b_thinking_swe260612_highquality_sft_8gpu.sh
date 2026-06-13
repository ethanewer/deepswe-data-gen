#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/qwen3_4b_thinking_swe260612_highquality_65k_online_packed_sft_8gpu.yaml}"
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/highquality-1x-duplicate-reasoning-90pct-30k-full-miniswe-aligned/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen3_4b_thinking_swe260612_miniswe_aligned_65k_reasoning_toolcall_sft/}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260612_miniswe_aligned_65k_reasoning_toolcall_sft}"

export PACK_SIZE="${PACK_SIZE:-65536}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export MAX_STEPS="${MAX_STEPS:-600}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"
export VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"

export LR="${LR:-5.0e-6}"
export MIN_LR="${MIN_LR:-5.0e-7}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-20}"

export CHECKPOINT_ENABLED="${CHECKPOINT_ENABLED:-true}"
export CHECKPOINT_MODEL_SAVE_FORMAT="${CHECKPOINT_MODEL_SAVE_FORMAT:-torch_save}"
export CHECKPOINT_SAVE_CONSOLIDATED="${CHECKPOINT_SAVE_CONSOLIDATED:-false}"
export VALIDATION_ENABLED="${VALIDATION_ENABLED:-false}"

export ENABLE_COMPILE="${ENABLE_COMPILE:-true}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
export ENABLE_FSDP2_PREFETCH="${ENABLE_FSDP2_PREFETCH:-true}"
export FSDP2_BACKWARD_PREFETCH_DEPTH="${FSDP2_BACKWARD_PREFETCH_DEPTH:-3}"
export FSDP2_FORWARD_PREFETCH_DEPTH="${FSDP2_FORWARD_PREFETCH_DEPTH:-1}"

export OVERLENGTH_STRATEGY="${OVERLENGTH_STRATEGY:-split}"
export REQUIRE_ASSISTANT_REASONING_FOR_LOSS="${REQUIRE_ASSISTANT_REASONING_FOR_LOSS:-true}"
export REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS="${REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS:-true}"
export DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS="${DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS:-true}"
export ASSISTANT_LOSS_TARGET="${ASSISTANT_LOSS_TARGET:-assistant}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

MODEL_SIZE=4b exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen3_thinking_sft_8gpu.sh" "$@"
