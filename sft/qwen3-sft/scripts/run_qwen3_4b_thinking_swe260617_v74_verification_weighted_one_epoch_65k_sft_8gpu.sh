#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

export MODEL_SIZE=4b
export CONFIG="${CONFIG:-configs/qwen3_4b_thinking_swe260617_v74_verification_weighted_one_epoch_65k_sft_8gpu.yaml}"
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617/swerebench-verification-enhanced-v74-mixed50-cleanpatch-provenance-miniswe-aligned/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen3_4b_thinking_swe260617_v74_verification_weighted_submitmaskfix_one_epoch_65k_sft/}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260617_v74_verification_weighted_submitmaskfix_one_epoch_65k_sft}"

export PACK_SIZE="${PACK_SIZE:-65536}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"
export VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"

export LR="${LR:-1.0e-5}"
export MIN_LR="${MIN_LR:-1.0e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-4}"

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
export DATASET_REPEAT="${DATASET_REPEAT:-false}"

export MASK_TOOL_CALL_ERROR_RECOVERY="${MASK_TOOL_CALL_ERROR_RECOVERY:-true}"
export MASK_MANUAL_PATCH_ARTIFACT_TURNS="${MASK_MANUAL_PATCH_ARTIFACT_TURNS:-true}"
export ENABLE_TURN_LOSS_WEIGHTS="${ENABLE_TURN_LOSS_WEIGHTS:-true}"
export READ_LOSS_WEIGHT="${READ_LOSS_WEIGHT:-0.5}"
export WRITE_LOSS_WEIGHT="${WRITE_LOSS_WEIGHT:-1.0}"
export TEST_LOSS_WEIGHT="${TEST_LOSS_WEIGHT:-1.0}"
export VERIFY_LOSS_WEIGHT="${VERIFY_LOSS_WEIGHT:-1.5}"
export SUBMIT_LOSS_WEIGHT="${SUBMIT_LOSS_WEIGHT:-2.0}"
export DEFAULT_LOSS_WEIGHT="${DEFAULT_LOSS_WEIGHT:-1.0}"
export NONPASSING_LOSS_MULTIPLIER="${NONPASSING_LOSS_MULTIPLIER:-0.75}"
export MASK_NONPASSING_SUBMIT_TURNS="${MASK_NONPASSING_SUBMIT_TURNS:-false}"
export MASK_EMPTY_PATCH_SUBMIT_TURNS="${MASK_EMPTY_PATCH_SUBMIT_TURNS:-true}"

export NUM_WORKERS="${NUM_WORKERS:-2}"
export COUNT_PROCESSES="${COUNT_PROCESSES:-16}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2}"

if [ "$ASSISTANT_LOSS_TARGET" != "assistant" ]; then
  echo "v74 must use full assistant loss spans; got ASSISTANT_LOSS_TARGET=$ASSISTANT_LOSS_TARGET" >&2
  exit 1
fi

if [ "$GLOBAL_BATCH_SIZE" -ne $((LOCAL_BATCH_SIZE * 8 * GRAD_ACCUM_STEPS)) ]; then
  echo "GLOBAL_BATCH_SIZE must equal LOCAL_BATCH_SIZE * 8 * GRAD_ACCUM_STEPS for this 8-GPU recipe" >&2
  exit 1
fi

if [ ! -x "$ROOT_DIR/.venv/bin/python" ]; then
  echo "Missing .venv. Run ./scripts/setup_nemo_automodel_env.sh first." >&2
  exit 1
fi

if [ -z "${MAX_STEPS:-}" ] || [ -z "${PAD_TO_PACK_COUNT:-}" ]; then
  mkdir -p "$CHECKPOINT_DIR"
  EPOCH_MANIFEST="${EPOCH_MANIFEST:-$CHECKPOINT_DIR/one_epoch_pack_count.json}"
  export PYTHONPATH="$ROOT_DIR/third_party/Automodel:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  count_args=(
    -m qwen_agentic_sft.online_packed_dataset count-shards
    --model "$MODEL" \
    --raw-root "$TRAIN_RAW_ROOT" \
    --chat-template "$CHAT_TEMPLATE" \
    --sequence-length "$PACK_SIZE" \
    --overlength-strategy "$OVERLENGTH_STRATEGY" \
    --require-assistant-reasoning-for-loss \
    --require-assistant-tool-calls-for-loss \
    --drop-assistant-content-for-tool-calls \
    --assistant-loss-target "$ASSISTANT_LOSS_TARGET" \
    --mask-tool-call-error-recovery \
    --mask-manual-patch-artifact-turns \
    --enable-turn-loss-weights \
    --read-loss-weight "$READ_LOSS_WEIGHT" \
    --write-loss-weight "$WRITE_LOSS_WEIGHT" \
    --test-loss-weight "$TEST_LOSS_WEIGHT" \
    --verify-loss-weight "$VERIFY_LOSS_WEIGHT" \
    --submit-loss-weight "$SUBMIT_LOSS_WEIGHT" \
    --default-loss-weight "$DEFAULT_LOSS_WEIGHT" \
    --nonpassing-loss-multiplier "$NONPASSING_LOSS_MULTIPLIER" \
    --world-size 8 \
    --num-workers "$NUM_WORKERS" \
    --local-batch-size "$LOCAL_BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --packs-per-worker-multiple 4 \
    --count-processes "$COUNT_PROCESSES" \
    --output-json "$EPOCH_MANIFEST"
  )
  if [ "$MASK_NONPASSING_SUBMIT_TURNS" = "true" ]; then
    count_args+=(--mask-nonpassing-submit-turns)
  fi
  if [ "$MASK_EMPTY_PATCH_SUBMIT_TURNS" = "true" ]; then
    count_args+=(--mask-empty-patch-submit-turns)
  fi
  "$ROOT_DIR/.venv/bin/python" "${count_args[@]}"

  eval "$("$ROOT_DIR/.venv/bin/python" - "$EPOCH_MANIFEST" <<'PY'
import json
import math
import shlex
import sys

path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
max_steps = int(data["max_steps"])
pad_to_pack_count = int(data["pad_to_pack_count"])
warmup = max(1, math.ceil(max_steps * 0.10))
print("export MAX_STEPS=" + shlex.quote(str(max_steps)))
print("export PAD_TO_PACK_COUNT=" + shlex.quote(str(pad_to_pack_count)))
print("export LR_WARMUP_STEPS=" + shlex.quote(str(warmup)))
PY
  )"
fi

if [ -z "${LR_WARMUP_STEPS:-}" ]; then
  LR_WARMUP_STEPS="$("$ROOT_DIR/.venv/bin/python" - <<'PY'
import math
import os

print(max(1, math.ceil(int(os.environ["MAX_STEPS"]) * 0.10)))
PY
  )"
  export LR_WARMUP_STEPS
fi

exec "$SCRIPT_DIR/run_qwen3_thinking_sft_8gpu.sh" "$@"
