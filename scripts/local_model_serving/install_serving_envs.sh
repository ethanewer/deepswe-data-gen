#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

PYTHON_BASE="${PYTHON_BASE:-/wbl-fast/usrs/ee/code-swe-data/runtime/cpython-3.12.13-linux-x86_64-gnu/bin/python3}"

create_venv() {
  local venv_dir="$1"
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$PYTHON_BASE" -m venv "$venv_dir"
  fi
  "$venv_dir/bin/python" -m pip install --upgrade pip setuptools wheel
}

create_venv "$LOCAL_MODEL_SERVING_ROOT/venv-vllm"
"$LOCAL_MODEL_SERVING_ROOT/venv-vllm/bin/python" -m pip install \
  "vllm==0.19.1"
"$LOCAL_MODEL_SERVING_ROOT/venv-vllm/bin/python" -m pip install \
  "transformers>=4.57.1,<5.0.0"

create_venv "$LOCAL_MODEL_SERVING_ROOT/venv-sglang"
"$LOCAL_MODEL_SERVING_ROOT/venv-sglang/bin/python" -m pip install \
  --pre "sglang[all]" decord
"$LOCAL_MODEL_SERVING_ROOT/venv-sglang/bin/python" -m pip install \
  --force-reinstall torch==2.11.0 torchaudio==2.11.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu128
"$LOCAL_MODEL_SERVING_ROOT/venv-sglang/bin/python" -m pip install \
  "transformers==5.6.0" "huggingface-hub>=1.10.0" "kernels<0.15" \
  "numpy<2.4,>=1.25" "setuptools<82,>=80" "cuda-pathfinder>=1.4.2"

if [[ "${INSTALL_DEEPEP:-0}" == "1" ]]; then
  "$LOCAL_MODEL_SERVING_ROOT/venv-sglang/bin/python" -m pip install \
    --no-build-isolation deep_ep
fi
