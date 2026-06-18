#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# v79: base-start reproduction of the actual saved v54 winning recipe, with
# only the post-s49 LR schedule changed to test whether it scales past s49.

export MODEL_SIZE=4b
export CONFIG="${CONFIG:-configs/qwen3_4b_thinking_swe260612_v79_v54_actual_wsd_cooldown_65k_sft_8gpu.yaml}"
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/highquality-1x-duplicate-reasoning-90pct-30k-full-miniswe-aligned-passed/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/ewer/qwen3-sft-local/outputs/qwen3_4b_thinking_swe260612_v79_v54_actual_wsd_cooldown_from_base_65k_s150_assistant_h200_8gpu_sft}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260612_v79_v54_actual_wsd_cooldown_from_base_65k_s150_assistant_h200_8gpu_sft}"

export PACK_SIZE="${PACK_SIZE:-65536}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export MAX_STEPS="${MAX_STEPS:-150}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-25}"
export VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"

export LR="${LR:-2.0e-6}"
export MIN_LR="${MIN_LR:-1.0e-7}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"

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

export NUM_WORKERS="${NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2}"

if [ "$MODEL" != "Qwen/Qwen3-4B-Thinking-2507" ]; then
  echo "v79 must start from base Qwen/Qwen3-4B-Thinking-2507; got MODEL=$MODEL" >&2
  exit 1
fi

if [ "$ASSISTANT_LOSS_TARGET" != "assistant" ]; then
  echo "v79 must train full assistant loss spans; got ASSISTANT_LOSS_TARGET=$ASSISTANT_LOSS_TARGET" >&2
  exit 1
fi

if [ "$GLOBAL_BATCH_SIZE" -ne $((LOCAL_BATCH_SIZE * 8 * GRAD_ACCUM_STEPS)) ]; then
  echo "GLOBAL_BATCH_SIZE must equal LOCAL_BATCH_SIZE * 8 * GRAD_ACCUM_STEPS for this 8-GPU recipe" >&2
  exit 1
fi

if [ ! -f "$TRAIN_RAW_ROOT/shard-000.jsonl" ]; then
  echo "Missing v54 saved-config raw training view: $TRAIN_RAW_ROOT/shard-000.jsonl" >&2
  exit 1
fi

exec "$SCRIPT_DIR/run_qwen3_thinking_sft_8gpu.sh" "$@"
