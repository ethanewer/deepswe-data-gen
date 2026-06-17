#!/usr/bin/env bash
set -euo pipefail

# Larger base-start ramped-passrate recipe for the 2026-06-17 v5 compaction-aware
# mixed25 cleanpatch view, with parser-error recovery turns masked rather than
# dropping whole trajectories that contain Mini-SWE tool-call parser errors.
#
# Motivation:
#   v67 is a strict all-tool-error row-drop ablation. v68 keeps more data at a
#   50% passrate. This recipe tests the same turn-level parser-error masking on
#   the larger 27.9% passrate view so training can see substantially more high
#   quality non-passing process traces before the ramp reaches denser passed
#   submit supervision.
#
# Data view:
#   swerebench-v5-compaction-mixed25-cleanpatch-ramped-miniswe-aligned

export CONFIG="${CONFIG:-configs/qwen3_4b_thinking_swe260612_highquality_65k_online_packed_sft_8gpu.yaml}"
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617/swerebench-v5-compaction-mixed25-cleanpatch-ramped-miniswe-aligned/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/ewer/qwen3-sft-local/outputs/qwen3_4b_thinking_swe260617_v69_v5_mixed25_cleanpatch_toolerrmasked_ramped_process_from_base_65k_lr1e6_s1600_assistant_h200_8gpu_sft}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260617_v69_v5_mixed25_cleanpatch_toolerrmasked_ramped_process_from_base_65k_lr1e6_s1600_assistant_h200_8gpu_sft}"

export PACK_SIZE="${PACK_SIZE:-65536}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export MAX_STEPS="${MAX_STEPS:-1600}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"
export VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1600}"

export LR="${LR:-1.0e-6}"
export MIN_LR="${MIN_LR:-1.0e-7}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"

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
export DATASET_SEED="${DATASET_SEED:-61769}"
export REQUIRE_ASSISTANT_REASONING_FOR_LOSS="${REQUIRE_ASSISTANT_REASONING_FOR_LOSS:-true}"
export REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS="${REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS:-true}"
export DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS="${DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS:-true}"
export ASSISTANT_LOSS_TARGET="${ASSISTANT_LOSS_TARGET:-assistant}"
export REJECT_MANUAL_PATCH_TARGETS="${REJECT_MANUAL_PATCH_TARGETS:-true}"
export REJECT_UNVERIFIED_SUBMIT_TARGETS="${REJECT_UNVERIFIED_SUBMIT_TARGETS:-true}"
export REJECT_NONPASSING_SUBMIT_TARGETS="${REJECT_NONPASSING_SUBMIT_TARGETS:-true}"
export MASK_ASSISTANT_AFTER_TOOL_CALL_ERROR="${MASK_ASSISTANT_AFTER_TOOL_CALL_ERROR:-true}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

MODEL_SIZE=4b exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen3_thinking_sft_8gpu.sh" "$@"
