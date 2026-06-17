#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

echo "root=$LOCAL_MODEL_SERVING_ROOT"
echo "hf_home=$HF_HOME"
echo "hf_cache=$HUGGINGFACE_HUB_CACHE"
echo "hf_xet_cache=$HF_XET_CACHE"
echo

for model_path in \
  "$KIMI_MODEL_PATH" \
  "$KIMI27_CODE_MODEL_PATH" \
  "$MIMO_MODEL_PATH" \
  "$QWEN36_MODEL_PATH" \
  "$QWEN36_MOE_FP8_MODEL_PATH"; do
  if [[ -L "$model_path" ]]; then
    echo "$model_path -> $(readlink "$model_path")"
  elif [[ -e "$model_path" ]]; then
    echo "$model_path exists but is not a symlink"
  else
    echo "$model_path missing"
  fi
done

echo
echo "download env:"
printf 'HF_HOME=%s\n' "$HF_HOME"
printf 'HUGGINGFACE_HUB_CACHE=%s\n' "$HUGGINGFACE_HUB_CACHE"
printf 'HF_XET_CACHE=%s\n' "$HF_XET_CACHE"
printf 'TRANSFORMERS_CACHE=%s\n' "$TRANSFORMERS_CACHE"
printf 'XDG_CACHE_HOME=%s\n' "$XDG_CACHE_HOME"
printf 'PIP_CACHE_DIR=%s\n' "$PIP_CACHE_DIR"
printf 'TORCH_HOME=%s\n' "$TORCH_HOME"
printf 'TRITON_CACHE_DIR=%s\n' "$TRITON_CACHE_DIR"
printf 'TMPDIR=%s\n' "$TMPDIR"
