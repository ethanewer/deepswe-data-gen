#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Reproducible v75 recipe.
# Source dataset, explicitly:
#   eewer/swerebench-traces-raw-source-verification-enhanced-20260617
#
# Set PREPARE_DATA=true to rebuild the strict-passed/cap-4 mini-swe view before
# launching training. The preparation step writes manifests that preserve the
# dataset ID, local source mirror, selected UUIDs, task-cap policy, and final
# row order.

export MODEL_SIZE=4b
export SOURCE_DATASET_ID="${SOURCE_DATASET_ID:-eewer/swerebench-traces-raw-source-verification-enhanced-20260617}"
export SOURCE_LOCAL_ROOT="${SOURCE_LOCAL_ROOT:-/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/swerebench-traces-raw-source-verification-enhanced-20260617}"
export DATA_VIEW_ROOT="${DATA_VIEW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617/swerebench-verification-enhanced-v75-strictpassed-cap4-miniswe-aligned-spread}"
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-$DATA_VIEW_ROOT/data}"
export ALLOWLIST_ROOT="${ALLOWLIST_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617/swerebench-verification-enhanced-v75-strictpassed-cap4-allowlist}"

export CONFIG="${CONFIG:-configs/qwen3_4b_thinking_swe260617_v75_verification_strictpassed_cap4_65k_sft_8gpu.yaml}"
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/ewer/qwen3-sft-local/outputs/qwen3_4b_thinking_swe260617_v75_verification_strictpassed_cap4_from_base_65k_lr1e6_s350_assistant_h200_8gpu_sft}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260617_v75_verification_strictpassed_cap4_from_base_65k_lr1e6_s350_assistant_h200_8gpu_sft}"

export PACK_SIZE="${PACK_SIZE:-65536}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export MAX_STEPS="${MAX_STEPS:-350}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"
export VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"

export LR="${LR:-1.0e-6}"
export MIN_LR="${MIN_LR:-1.0e-7}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-25}"

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
export DATASET_REPEAT="${DATASET_REPEAT:-true}"

export MASK_TOOL_CALL_ERROR_RECOVERY="${MASK_TOOL_CALL_ERROR_RECOVERY:-false}"
export MASK_MANUAL_PATCH_ARTIFACT_TURNS="${MASK_MANUAL_PATCH_ARTIFACT_TURNS:-true}"
export ENABLE_TURN_LOSS_WEIGHTS="${ENABLE_TURN_LOSS_WEIGHTS:-false}"
export MASK_NONPASSING_SUBMIT_TURNS="${MASK_NONPASSING_SUBMIT_TURNS:-false}"
export MASK_EMPTY_PATCH_SUBMIT_TURNS="${MASK_EMPTY_PATCH_SUBMIT_TURNS:-true}"

export NUM_WORKERS="${NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2}"

if [ "$MODEL" != "Qwen/Qwen3-4B-Thinking-2507" ]; then
  echo "v75 must start from base Qwen/Qwen3-4B-Thinking-2507; got MODEL=$MODEL" >&2
  exit 1
fi

if [ "$SOURCE_DATASET_ID" != "eewer/swerebench-traces-raw-source-verification-enhanced-20260617" ]; then
  echo "v75 must use eewer/swerebench-traces-raw-source-verification-enhanced-20260617; got SOURCE_DATASET_ID=$SOURCE_DATASET_ID" >&2
  exit 1
fi

if [ "$ASSISTANT_LOSS_TARGET" != "assistant" ]; then
  echo "v75 must train full assistant loss spans; got ASSISTANT_LOSS_TARGET=$ASSISTANT_LOSS_TARGET" >&2
  exit 1
fi

if [ "$GLOBAL_BATCH_SIZE" -ne $((LOCAL_BATCH_SIZE * 8 * GRAD_ACCUM_STEPS)) ]; then
  echo "GLOBAL_BATCH_SIZE must equal LOCAL_BATCH_SIZE * 8 * GRAD_ACCUM_STEPS for this 8-GPU recipe" >&2
  exit 1
fi

if [ "${PREPARE_DATA:-false}" = "true" ]; then
  SOURCE_DATASET_ID="$SOURCE_DATASET_ID" \
  SOURCE_LOCAL_ROOT="$SOURCE_LOCAL_ROOT" \
  ALLOWLIST_ROOT="$ALLOWLIST_ROOT" \
  FINAL_VIEW_ROOT="$DATA_VIEW_ROOT" \
  "$SCRIPT_DIR/prepare_qwen3_4b_v75_verification_strictpassed_cap4_data.sh"
fi

if [ ! -f "$ALLOWLIST_ROOT/manifest.json" ]; then
  echo "Missing allowlist manifest: $ALLOWLIST_ROOT/manifest.json" >&2
  echo "Run PREPARE_DATA=true $0 first." >&2
  exit 1
fi

if [ ! -f "$DATA_VIEW_ROOT/manifest.json" ] || [ ! -d "$TRAIN_RAW_ROOT" ]; then
  echo "Missing final data view under $DATA_VIEW_ROOT" >&2
  echo "Run PREPARE_DATA=true $0 first." >&2
  exit 1
fi

"$ROOT_DIR/.venv/bin/python" - "$ALLOWLIST_ROOT/manifest.json" "$DATA_VIEW_ROOT/manifest.json" "$SOURCE_DATASET_ID" <<'PY'
import json
import sys

allowlist_path, data_view_path, expected_id = sys.argv[1:4]
allowlist = json.load(open(allowlist_path, "r", encoding="utf-8"))
data_view = json.load(open(data_view_path, "r", encoding="utf-8"))
if allowlist.get("dataset_id") != expected_id:
    raise SystemExit(
        f"allowlist manifest dataset_id={allowlist.get('dataset_id')!r}, expected {expected_id!r}"
    )
policy = allowlist.get("selection_policy") or {}
if policy.get("max_pass_per_task") != 4:
    raise SystemExit(f"expected max_pass_per_task=4, got {policy.get('max_pass_per_task')!r}")
if allowlist.get("counts", {}).get("selected_max_rollouts_per_task", 0) > 4:
    raise SystemExit("allowlist has a task with more than 4 selected passing rollouts")
rows = int(data_view.get("rows_written") or 0)
if rows < 8_000:
    raise SystemExit(f"final data view has unexpectedly few rows: {rows}")
print(
    "v75 data manifest ok: "
    f"dataset_id={allowlist['dataset_id']} "
    f"selected={allowlist['counts']['selected_pass_traces']} "
    f"final_rows={rows} "
    f"unique_tasks={allowlist['counts']['selected_unique_tasks']}"
)
PY

exec "$SCRIPT_DIR/run_qwen3_thinking_sft_8gpu.sh" "$@"
