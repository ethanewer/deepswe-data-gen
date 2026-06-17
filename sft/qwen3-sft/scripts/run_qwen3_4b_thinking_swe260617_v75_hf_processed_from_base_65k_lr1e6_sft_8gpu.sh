#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Reproducible v75 recipe from the uploaded processed dataset.
# Dataset, explicitly:
#   eewer/qwen3-4b-thinking-sft-v75-verification-strictpassed-cap4-processed
#
# The uploaded rows have the v75 assistant-loss message policy already
# materialized into the `messages` column:
# - assistant tool-call content is reasoning-only,
# - assistant turns that should not receive loss have `loss: false`,
# - the original v75 filtering, task cap, lineage de-duplication, and long
#   duplicate trajectory filtering have already been applied.
#
# Therefore this wrapper disables the runtime message-shaping flags. The trainer
# still reads `loss: false` from each assistant message when building labels.

export MODEL_SIZE=4b
export HF_DATASET_ID="${HF_DATASET_ID:-eewer/qwen3-4b-thinking-sft-v75-verification-strictpassed-cap4-processed}"
export HF_DATASET_LOCAL_ROOT="${HF_DATASET_LOCAL_ROOT:-/wbl-fast/usrs/ee/code-swe-data/runtime/hf_datasets/qwen3-4b-thinking-sft-v75-verification-strictpassed-cap4-processed}"
export TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-$HF_DATASET_LOCAL_ROOT/data}"
export DOWNLOAD_PYTHON="${DOWNLOAD_PYTHON:-$ROOT_DIR/.venv/bin/python}"
export VALIDATE_ONLY="${VALIDATE_ONLY:-false}"

export CONFIG="${CONFIG:-configs/qwen3_4b_thinking_swe260617_v75_verification_strictpassed_cap4_65k_sft_8gpu.yaml}"
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/ewer/qwen3-sft-local/outputs/qwen3_4b_thinking_swe260617_v75_hf_processed_from_base_65k_lr1e6_s350_assistant_h200_8gpu_sft}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260617_v75_hf_processed_from_base_65k_lr1e6_s350_assistant_h200_8gpu_sft}"

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
export ASSISTANT_LOSS_TARGET="${ASSISTANT_LOSS_TARGET:-assistant}"
export DATASET_REPEAT="${DATASET_REPEAT:-true}"

# Important: these were already materialized into the uploaded dataset.
export REQUIRE_ASSISTANT_REASONING_FOR_LOSS="${REQUIRE_ASSISTANT_REASONING_FOR_LOSS:-false}"
export REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS="${REQUIRE_ASSISTANT_TOOL_CALLS_FOR_LOSS:-false}"
export DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS="${DROP_ASSISTANT_CONTENT_FOR_TOOL_CALLS:-false}"
export MASK_TOOL_CALL_ERROR_RECOVERY="${MASK_TOOL_CALL_ERROR_RECOVERY:-false}"
export MASK_MANUAL_PATCH_ARTIFACT_TURNS="${MASK_MANUAL_PATCH_ARTIFACT_TURNS:-false}"
export ENABLE_TURN_LOSS_WEIGHTS="${ENABLE_TURN_LOSS_WEIGHTS:-false}"
export MASK_NONPASSING_SUBMIT_TURNS="${MASK_NONPASSING_SUBMIT_TURNS:-false}"
export MASK_EMPTY_PATCH_SUBMIT_TURNS="${MASK_EMPTY_PATCH_SUBMIT_TURNS:-false}"

export NUM_WORKERS="${NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2}"

if [ "$MODEL" != "Qwen/Qwen3-4B-Thinking-2507" ]; then
  echo "v75 HF processed recipe must start from base Qwen/Qwen3-4B-Thinking-2507; got MODEL=$MODEL" >&2
  exit 1
fi

if [ "$ASSISTANT_LOSS_TARGET" != "assistant" ]; then
  echo "v75 HF processed recipe must train full assistant spans; got ASSISTANT_LOSS_TARGET=$ASSISTANT_LOSS_TARGET" >&2
  exit 1
fi

if [ "$GLOBAL_BATCH_SIZE" -ne $((LOCAL_BATCH_SIZE * 8 * GRAD_ACCUM_STEPS)) ]; then
  echo "GLOBAL_BATCH_SIZE must equal LOCAL_BATCH_SIZE * 8 * GRAD_ACCUM_STEPS for this 8-GPU recipe" >&2
  exit 1
fi

if [ ! -f "$HF_DATASET_LOCAL_ROOT/manifest.json" ] || [ ! -d "$TRAIN_RAW_ROOT" ]; then
  mkdir -p "$HF_DATASET_LOCAL_ROOT"
  "$DOWNLOAD_PYTHON" - "$HF_DATASET_ID" "$HF_DATASET_LOCAL_ROOT" <<'PY'
import sys

repo_id, local_dir = sys.argv[1:3]
try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required to download the processed dataset. "
        "Install it or set HF_DATASET_LOCAL_ROOT to an existing snapshot."
    ) from exc

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
PY
fi

"$ROOT_DIR/.venv/bin/python" - "$HF_DATASET_LOCAL_ROOT/manifest.json" "$HF_DATASET_ID" <<'PY'
import json
import sys

manifest_path, expected_repo = sys.argv[1:3]
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
rows = int(manifest.get("rows_written") or 0)
if rows != 8858:
    raise SystemExit(f"expected processed v75 row count 8858, got {rows}")
policy = manifest.get("policy") or {}
required = {
    "require_assistant_reasoning_for_loss": True,
    "require_assistant_tool_calls_for_loss": True,
    "drop_assistant_content_for_tool_calls": True,
    "mask_manual_patch_artifact_turns": True,
    "mask_empty_patch_submit_turns": True,
}
for key, value in required.items():
    if policy.get(key) is not value:
        raise SystemExit(f"manifest policy {key}={policy.get(key)!r}, expected {value!r}")
print(
    "v75 HF processed manifest ok: "
    f"repo={expected_repo} rows={rows} source={manifest.get('source_dataset_id')}"
)
PY

if [ "$VALIDATE_ONLY" = "true" ]; then
  echo "VALIDATE_ONLY=true; not launching training."
  exit 0
fi

exec "$SCRIPT_DIR/run_qwen3_thinking_sft_8gpu.sh" "$@"
