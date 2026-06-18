#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# v85: two-phase controlled scaling recipe.
#
# Phase 1 starts from base Qwen3-4B-Thinking and reproduces the original v54
# s49 window exactly. Phase 2 loads only the consolidated phase-1 s49 HF model
# directory, which resets AdamW/scheduler state, then continues on the same v54
# data with larger effective batch, much lower LR, and low weight decay.

export MODEL_SIZE=4b
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260616/swerebench-raw2030-targeted-limitations-strict-passed-miniswe-aligned/data}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2}"

export PHASE1_CONFIG="${PHASE1_CONFIG:-configs/qwen3_4b_thinking_swe260616_v85_phase1_exact_v54_s49_65k_sft_8gpu.yaml}"
export PHASE2_CONFIG="${PHASE2_CONFIG:-configs/qwen3_4b_thinking_swe260616_v85_phase2_s49_optimizer_reset_lowlr_65k_sft_8gpu.yaml}"

export PHASE1_CHECKPOINT_DIR="${PHASE1_CHECKPOINT_DIR:-/scratch/ewer/qwen3-sft-local/outputs/qwen3_4b_thinking_swe260616_v85_phase1_exact_v54_s49_from_base_65k_s50_assistant_h200_8gpu_sft}"
export PHASE2_CHECKPOINT_DIR="${PHASE2_CHECKPOINT_DIR:-/scratch/ewer/qwen3-sft-local/outputs/qwen3_4b_thinking_swe260616_v85_phase2_s49_optimizer_reset_lowlr_65k_s125_assistant_h200_8gpu_sft}"
export PHASE1_STEP49_MODEL="${PHASE1_STEP49_MODEL:-$PHASE1_CHECKPOINT_DIR/epoch_0_step_49/model/consolidated}"

export PACK_SIZE="${PACK_SIZE:-65536}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-2}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export VALIDATION_ENABLED="${VALIDATION_ENABLED:-false}"
export CHECKPOINT_ENABLED="${CHECKPOINT_ENABLED:-true}"
export CHECKPOINT_MODEL_SAVE_FORMAT="${CHECKPOINT_MODEL_SAVE_FORMAT:-safetensors}"
export CHECKPOINT_SAVE_CONSOLIDATED="${CHECKPOINT_SAVE_CONSOLIDATED:-final}"
export CHECKPOINT_V4_COMPATIBLE="${CHECKPOINT_V4_COMPATIBLE:-true}"
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
export REJECT_MANUAL_PATCH_TARGETS="${REJECT_MANUAL_PATCH_TARGETS:-true}"
export REJECT_UNVERIFIED_SUBMIT_TARGETS="${REJECT_UNVERIFIED_SUBMIT_TARGETS:-true}"
export REJECT_NONPASSING_SUBMIT_TARGETS="${REJECT_NONPASSING_SUBMIT_TARGETS:-true}"
export DATASET_REPEAT="${DATASET_REPEAT:-true}"
export MASK_TOOL_CALL_ERROR_RECOVERY="${MASK_TOOL_CALL_ERROR_RECOVERY:-false}"
export MASK_MANUAL_PATCH_ARTIFACT_TURNS="${MASK_MANUAL_PATCH_ARTIFACT_TURNS:-false}"
export ENABLE_TURN_LOSS_WEIGHTS="${ENABLE_TURN_LOSS_WEIGHTS:-false}"
export MASK_NONPASSING_SUBMIT_TURNS="${MASK_NONPASSING_SUBMIT_TURNS:-false}"
export MASK_EMPTY_PATCH_SUBMIT_TURNS="${MASK_EMPTY_PATCH_SUBMIT_TURNS:-false}"

if [ "$ASSISTANT_LOSS_TARGET" != "assistant" ]; then
  echo "v85 must train full assistant loss spans; got ASSISTANT_LOSS_TARGET=$ASSISTANT_LOSS_TARGET" >&2
  exit 1
fi

if [ ! -f "$TRAIN_RAW_ROOT/shard-000.jsonl" ]; then
  echo "Missing v54 raw training view: $TRAIN_RAW_ROOT/shard-000.jsonl" >&2
  exit 1
fi

run_phase1() {
  export CONFIG="$PHASE1_CONFIG"
  export MODEL="Qwen/Qwen3-4B-Thinking-2507"
  export CHECKPOINT_DIR="$PHASE1_CHECKPOINT_DIR"
  export RUN_NAME="qwen3_4b_thinking_swe260616_v85_phase1_exact_v54_s49_from_base_65k_s50_assistant_h200_8gpu_sft"
  export GLOBAL_BATCH_SIZE=16
  export GRAD_ACCUM_STEPS=1
  export MAX_STEPS=50
  export CKPT_EVERY_STEPS=50
  export VAL_EVERY_STEPS=1000
  export LR=1.0e-6
  export MIN_LR=1.0e-7
  export WEIGHT_DECAY=0.01
  export LR_WARMUP_STEPS=25

  "$SCRIPT_DIR/run_qwen3_thinking_sft_8gpu.sh"
}

run_phase2() {
  if [ ! -f "$PHASE1_STEP49_MODEL/model.safetensors.index.json" ]; then
    echo "Missing consolidated phase-1 s49 model: $PHASE1_STEP49_MODEL" >&2
    exit 1
  fi

  export CONFIG="$PHASE2_CONFIG"
  export MODEL="$PHASE1_STEP49_MODEL"
  export CHECKPOINT_DIR="$PHASE2_CHECKPOINT_DIR"
  export RUN_NAME="qwen3_4b_thinking_swe260616_v85_phase2_s49_optimizer_reset_lowlr_65k_s125_assistant_h200_8gpu_sft"
  export GLOBAL_BATCH_SIZE=32
  export GRAD_ACCUM_STEPS=2
  export MAX_STEPS=125
  export CKPT_EVERY_STEPS=25
  export VAL_EVERY_STEPS=1000
  export LR=1.0e-7
  export MIN_LR=1.0e-8
  export WEIGHT_DECAY=1.0e-4
  export LR_WARMUP_STEPS=5

  "$SCRIPT_DIR/run_qwen3_thinking_sft_8gpu.sh"
}

if [ "${V85_SKIP_PHASE1:-false}" = "true" ]; then
  echo "Skipping phase 1; using existing phase-1 model: $PHASE1_STEP49_MODEL"
else
  run_phase1
fi

run_phase2
