#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAX_STEPS="${MAX_STEPS:-24}"
TRAIN_RAW_ROOT="${TRAIN_RAW_ROOT:-$ROOT_DIR/data/smoke_raw}"
MODEL_SIZE="${MODEL_SIZE:-4b}"

run_case() {
  local pack_size="$1"
  local local_batch="$2"
  local grad_accum="$3"
  local enable_compile="$4"
  local name="$5"
  local enable_fsdp2_prefetch="${6:-true}"
  local fsdp2_backward_prefetch_depth="${7:-3}"
  local fsdp2_forward_prefetch_depth="${8:-1}"
  PACK_SIZE="$pack_size" \
  LOCAL_BATCH_SIZE="$local_batch" \
  GRAD_ACCUM_STEPS="$grad_accum" \
  ENABLE_COMPILE="$enable_compile" \
  ENABLE_FSDP2_PREFETCH="$enable_fsdp2_prefetch" \
  FSDP2_BACKWARD_PREFETCH_DEPTH="$fsdp2_backward_prefetch_depth" \
  FSDP2_FORWARD_PREFETCH_DEPTH="$fsdp2_forward_prefetch_depth" \
  VALIDATION_ENABLED=false \
  MAX_STEPS="$MAX_STEPS" \
  TRAIN_RAW_ROOT="$TRAIN_RAW_ROOT" \
  VAL_RAW_ROOT="$TRAIN_RAW_ROOT" \
  RUN_NAME="$name" \
  MODEL_SIZE="$MODEL_SIZE" ./scripts/run_qwen3_thinking_sft_8gpu.sh
}

run_case 131072 1 1 true qwen3_${MODEL_SIZE}_pack131k_lbs1_ga1_compile_fsdp_prefetch_b3f1 true 3 1
run_case 131072 1 1 true qwen3_${MODEL_SIZE}_pack131k_lbs1_ga1_compile_no_prefetch false 2 1
run_case 131072 1 1 false qwen3_${MODEL_SIZE}_pack131k_lbs1_ga1 false 2 1
run_case 262144 1 1 false qwen3_${MODEL_SIZE}_pack262k_lbs1_ga1 false 2 1
run_case 65536 4 1 false qwen3_${MODEL_SIZE}_pack65k_lbs4_ga1 false 2 1
run_case 65536 2 1 false qwen3_${MODEL_SIZE}_pack65k_lbs2_ga1 false 2 1
