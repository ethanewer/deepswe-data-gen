#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. Run ./scripts/setup_nemo_automodel_env.sh first." >&2
  exit 1
fi

export SOURCE_DATASET_ID="${SOURCE_DATASET_ID:-eewer/swerebench-traces-raw-source-verification-enhanced-20260617}"
export SOURCE_LOCAL_ROOT="${SOURCE_LOCAL_ROOT:-/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/swerebench-traces-raw-source-verification-enhanced-20260617}"
export OUTPUT_BASE="${OUTPUT_BASE:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617}"
export ALLOWLIST_ROOT="${ALLOWLIST_ROOT:-$OUTPUT_BASE/swerebench-verification-enhanced-v75-strictpassed-cap4-allowlist}"
export RAW_VIEW_ROOT="${RAW_VIEW_ROOT:-$OUTPUT_BASE/swerebench-verification-enhanced-v75-strictpassed-cap4-miniswe-aligned-raworder}"
export FINAL_VIEW_ROOT="${FINAL_VIEW_ROOT:-$OUTPUT_BASE/swerebench-verification-enhanced-v75-strictpassed-cap4-miniswe-aligned-spread}"
export MAX_PASS_PER_TASK="${MAX_PASS_PER_TASK:-4}"
export TARGET_PASS_TRACES="${TARGET_PASS_TRACES:-12000}"
export SHARDS="${SHARDS:-64}"
export OVERWRITE="${OVERWRITE:-true}"

overwrite_args=()
if [ "$OVERWRITE" = "true" ]; then
  overwrite_args+=(--overwrite)
fi

echo "Source dataset id: $SOURCE_DATASET_ID"
echo "Source local root: $SOURCE_LOCAL_ROOT"
echo "Allowlist root: $ALLOWLIST_ROOT"
echo "Raw transformed view: $RAW_VIEW_ROOT"
echo "Final spread view: $FINAL_VIEW_ROOT"
echo "Max passing rollouts per task: $MAX_PASS_PER_TASK"
echo "Target passing traces: $TARGET_PASS_TRACES"

.venv/bin/python scripts/build_swerebench_verification_enhanced_strict_pass_allowlist.py \
  --dataset-id "$SOURCE_DATASET_ID" \
  --local-root "$SOURCE_LOCAL_ROOT" \
  --output-root "$ALLOWLIST_ROOT" \
  --max-pass-per-task "$MAX_PASS_PER_TASK" \
  --target-pass-traces "$TARGET_PASS_TRACES" \
  "${overwrite_args[@]}"

.venv/bin/python scripts/build_swe260612_miniswe_raw.py \
  --input-root "$SOURCE_LOCAL_ROOT" \
  --output-root "$RAW_VIEW_ROOT" \
  --allow-uuid-file "$ALLOWLIST_ROOT/selected_pass_uuids.txt" \
  --drop-manual-patch-context-rows \
  --shards "$SHARDS" \
  "${overwrite_args[@]}"

.venv/bin/python scripts/spread_miniswe_rows_by_task.py \
  --input-root "$RAW_VIEW_ROOT" \
  --training-order-jsonl "$ALLOWLIST_ROOT/selected_pass_records_training_order.jsonl" \
  --output-root "$FINAL_VIEW_ROOT" \
  --shards "$SHARDS" \
  "${overwrite_args[@]}"

echo "Final training data root: $FINAL_VIEW_ROOT/data"
