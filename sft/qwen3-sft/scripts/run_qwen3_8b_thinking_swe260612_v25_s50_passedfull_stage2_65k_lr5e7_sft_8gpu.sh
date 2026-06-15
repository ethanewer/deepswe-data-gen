#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

START_STEP_DIR="${START_STEP_DIR:-$ROOT_DIR/checkpoints/qwen3_8b_thinking_swe260612_v23_highquality_exact_contextguard_65k_s300_assistant_h200_8gpu_sft/epoch_0_step_49}"
START_MODEL="${START_MODEL:-$START_STEP_DIR/model/consolidated}"
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/data/qwen3_vl_8b_text_checkpoint}"

if [ "$BASE_MODEL" = "$ROOT_DIR/data/qwen3_vl_8b_text_checkpoint" ] && [ ! -f "$BASE_MODEL/model.safetensors.index.json" ]; then
  "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/prepare_qwen3_vl_text_checkpoint.py" --output-dir "$BASE_MODEL"
fi

if [ ! -f "$START_MODEL/.complete" ] || ! compgen -G "$START_MODEL/*.safetensors" >/dev/null; then
  echo "Consolidating 8B start checkpoint $START_STEP_DIR/model -> $START_MODEL"
  tmp_model="${START_MODEL}.tmp.$$"
  rm -rf "$tmp_model"
  mkdir -p "$tmp_model"
  "$ROOT_DIR/.venv/bin/python" \
    "$REPO_ROOT/eval/benchmarks/swebench_multilingual/export_dcp_torchsave_to_hf.py" \
    --model-name "$BASE_MODEL" \
    --input-dir "$START_STEP_DIR/model" \
    --output-dir "$tmp_model" \
    --dtype bf16
  rm -rf "$START_MODEL"
  mv "$tmp_model" "$START_MODEL"
  touch "$START_MODEL/.complete"
fi

export CONFIG="${CONFIG:-configs/qwen3_8b_thinking_online_packed_sft_8gpu.yaml}"
export MODEL="${MODEL:-$START_MODEL}"
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/highquality-1x-duplicate-reasoning-90pct-30k-full-miniswe-aligned-passed-singlejsonl/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen3_8b_thinking_swe260612_v25_s50_passedfull_stage2_65k_lr5e7_s200_assistant_h200_8gpu_sft/}"
export RUN_NAME="${RUN_NAME:-qwen3_8b_thinking_swe260612_v25_s50_passedfull_stage2_65k_lr5e7_s200_assistant_h200_8gpu_sft}"

export PACK_SIZE="${PACK_SIZE:-65536}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export MAX_STEPS="${MAX_STEPS:-200}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"
export VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"

export LR="${LR:-5.0e-7}"
export MIN_LR="${MIN_LR:-5.0e-8}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"

export CHECKPOINT_ENABLED="${CHECKPOINT_ENABLED:-true}"
export CHECKPOINT_MODEL_SAVE_FORMAT="${CHECKPOINT_MODEL_SAVE_FORMAT:-torch_save}"
export CHECKPOINT_SAVE_CONSOLIDATED="${CHECKPOINT_SAVE_CONSOLIDATED:-false}"
export VALIDATION_ENABLED="${VALIDATION_ENABLED:-false}"

export ENABLE_COMPILE="${ENABLE_COMPILE:-true}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
export ENABLE_FSDP2_PREFETCH="${ENABLE_FSDP2_PREFETCH:-true}"
export FSDP2_BACKWARD_PREFETCH_DEPTH="${FSDP2_BACKWARD_PREFETCH_DEPTH:-2}"
export FSDP2_FORWARD_PREFETCH_DEPTH="${FSDP2_FORWARD_PREFETCH_DEPTH:-1}"

export OVERLENGTH_STRATEGY="${OVERLENGTH_STRATEGY:-split}"
export SHUFFLE_JSONL_ROWS="${SHUFFLE_JSONL_ROWS:-true}"
export DATASET_SEED="${DATASET_SEED:-55555}"
export REQUIRE_ASSISTANT_REASONING_FOR_LOSS="${REQUIRE_ASSISTANT_REASONING_FOR_LOSS:-true}"
export REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS="${REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS:-true}"
export DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS="${DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS:-true}"
export ASSISTANT_LOSS_TARGET="${ASSISTANT_LOSS_TARGET:-assistant}"
export REJECT_MANUAL_PATCH_TARGETS="${REJECT_MANUAL_PATCH_TARGETS:-true}"
export REJECT_UNVERIFIED_SUBMIT_TARGETS="${REJECT_UNVERIFIED_SUBMIT_TARGETS:-true}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

MODEL_SIZE=8b exec "$SCRIPT_DIR/run_qwen3_thinking_sft_8gpu.sh" "$@"
