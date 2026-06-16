#!/usr/bin/env bash
set -euo pipefail

# Phase-1 8B run for the 2026-06-16 final SWE-rebench source mix. It ports the
# best 4B prefix-weighted recipe to the prepared Qwen3-VL-8B text checkpoint and
# stops after 50 steps for the first train/eval alternation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/data/qwen3_vl_8b_text_checkpoint}"
if [ "$BASE_MODEL" = "$ROOT_DIR/data/qwen3_vl_8b_text_checkpoint" ] && [ ! -f "$BASE_MODEL/model.safetensors.index.json" ]; then
  "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/prepare_qwen3_vl_text_checkpoint.py" --output-dir "$BASE_MODEL"
fi

export CONFIG="${CONFIG:-configs/qwen3_8b_thinking_online_packed_sft_8gpu.yaml}"
export MODEL="${MODEL:-$BASE_MODEL}"
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260616/swerebench-final-20260616-0428utc-miniswe-passed-prefix-weighted-v1/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen3_8b_thinking_swe260616_v28_finaldata_base_prefixweighted_phase1_65k_lr5e7_s50_assistant_h200_8gpu_nocompile_sft/}"
export RUN_NAME="${RUN_NAME:-qwen3_8b_thinking_swe260616_v28_finaldata_base_prefixweighted_phase1_65k_lr5e7_s50_assistant_h200_8gpu_nocompile_sft}"

export PACK_SIZE="${PACK_SIZE:-65536}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export MAX_STEPS="${MAX_STEPS:-50}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"
export VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"

export LR="${LR:-5.0e-7}"
export MIN_LR="${MIN_LR:-5.0e-8}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"

export CHECKPOINT_ENABLED="${CHECKPOINT_ENABLED:-true}"
export CHECKPOINT_MODEL_SAVE_FORMAT="${CHECKPOINT_MODEL_SAVE_FORMAT:-torch_save}"
export CHECKPOINT_SAVE_CONSOLIDATED="${CHECKPOINT_SAVE_CONSOLIDATED:-false}"
export VALIDATION_ENABLED="${VALIDATION_ENABLED:-false}"

export ENABLE_COMPILE="${ENABLE_COMPILE:-false}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
export ENABLE_FSDP2_PREFETCH="${ENABLE_FSDP2_PREFETCH:-true}"
export FSDP2_BACKWARD_PREFETCH_DEPTH="${FSDP2_BACKWARD_PREFETCH_DEPTH:-2}"
export FSDP2_FORWARD_PREFETCH_DEPTH="${FSDP2_FORWARD_PREFETCH_DEPTH:-1}"

export OVERLENGTH_STRATEGY="${OVERLENGTH_STRATEGY:-split}"
export SHUFFLE_JSONL_ROWS="${SHUFFLE_JSONL_ROWS:-true}"
export DATASET_SEED="${DATASET_SEED:-57575}"
export REQUIRE_ASSISTANT_REASONING_FOR_LOSS="${REQUIRE_ASSISTANT_REASONING_FOR_LOSS:-true}"
export REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS="${REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS:-true}"
export DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS="${DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS:-true}"
export ASSISTANT_LOSS_TARGET="${ASSISTANT_LOSS_TARGET:-assistant}"
export REJECT_MANUAL_PATCH_TARGETS="${REJECT_MANUAL_PATCH_TARGETS:-true}"
export REJECT_UNVERIFIED_SUBMIT_TARGETS="${REJECT_UNVERIFIED_SUBMIT_TARGETS:-true}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

MODEL_SIZE=8b exec "$SCRIPT_DIR/run_qwen3_thinking_sft_8gpu.sh" "$@"
