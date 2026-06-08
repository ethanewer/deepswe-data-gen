#!/usr/bin/env bash
set -euo pipefail

export LOCAL_MODEL_SERVING_ROOT="${LOCAL_MODEL_SERVING_ROOT:-/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving}"

# Keep every cache/temp path on /wbl-fast. Do not let HF, pip, torch, or
# tokenizer libraries fall back to /home.
export HF_HOME="$LOCAL_MODEL_SERVING_ROOT/hf_home"
export HUGGINGFACE_HUB_CACHE="$LOCAL_MODEL_SERVING_ROOT/hf_cache"
export HF_XET_CACHE="$LOCAL_MODEL_SERVING_ROOT/hf_xet_cache"
export TRANSFORMERS_CACHE="$LOCAL_MODEL_SERVING_ROOT/transformers_cache"
export XDG_CACHE_HOME="$LOCAL_MODEL_SERVING_ROOT/xdg_cache"
export PIP_CACHE_DIR="$LOCAL_MODEL_SERVING_ROOT/pip_cache"
export TORCH_HOME="$LOCAL_MODEL_SERVING_ROOT/torch_home"
export TRITON_CACHE_DIR="$LOCAL_MODEL_SERVING_ROOT/triton_cache"
export TMPDIR="$LOCAL_MODEL_SERVING_ROOT/tmp"

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

export KIMI_MODEL_PATH="${KIMI_MODEL_PATH:-$LOCAL_MODEL_SERVING_ROOT/models/moonshotai_Kimi-K2.6.snapshot}"
export MIMO_MODEL_PATH="${MIMO_MODEL_PATH:-$LOCAL_MODEL_SERVING_ROOT/models/XiaomiMiMo_MiMo-V2.5.snapshot}"

mkdir -p \
  "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_XET_CACHE" "$TRANSFORMERS_CACHE" \
  "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
  "$TMPDIR" "$LOCAL_MODEL_SERVING_ROOT/models" "$LOCAL_MODEL_SERVING_ROOT/logs"

if [[ -f /wbl-fast/usrs/ee/code-swe-data/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /wbl-fast/usrs/ee/code-swe-data/.env
  set +a
fi
