#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTOMODEL_DIR="${AUTOMODEL_DIR:-$ROOT_DIR/third_party/Automodel}"
AUTOMODEL_REF="${AUTOMODEL_REF:-main}"
UPDATE_AUTOMODEL="${UPDATE_AUTOMODEL:-0}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
INSTALL_FLASH_ATTN3="${INSTALL_FLASH_ATTN3:-1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
INSTALL_TE="${INSTALL_TE:-0}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv first or put it on PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "$AUTOMODEL_DIR")"

if [ ! -d "$AUTOMODEL_DIR/.git" ]; then
  git clone https://github.com/NVIDIA-NeMo/Automodel.git "$AUTOMODEL_DIR"
fi

if [ "$UPDATE_AUTOMODEL" = "1" ]; then
  git -C "$AUTOMODEL_DIR" fetch origin "$AUTOMODEL_REF"
  git -C "$AUTOMODEL_DIR" checkout "$AUTOMODEL_REF"
  git -C "$AUTOMODEL_DIR" pull --ff-only origin "$AUTOMODEL_REF"
else
  echo "Using existing AutoModel checkout at $AUTOMODEL_DIR. Set UPDATE_AUTOMODEL=1 to update it."
fi

AUTOMODEL_PATCH="$ROOT_DIR/patches/automodel-qwen3-vl-text.patch"
if [ -f "$AUTOMODEL_PATCH" ]; then
  if git -C "$AUTOMODEL_DIR" apply --check "$AUTOMODEL_PATCH"; then
    git -C "$AUTOMODEL_DIR" apply "$AUTOMODEL_PATCH"
  elif git -C "$AUTOMODEL_DIR" apply --reverse --check "$AUTOMODEL_PATCH"; then
    echo "AutoModel Qwen3-VL text patch already applied."
  else
    echo "AutoModel Qwen3-VL text patch does not apply cleanly: $AUTOMODEL_PATCH" >&2
    exit 1
  fi
fi

if [ -d .venv ] && [ "${RECREATE_VENV:-0}" != "1" ]; then
  uv venv --python "$PYTHON_BIN" --allow-existing .venv
else
  uv venv --python "$PYTHON_BIN" --clear .venv
fi
source .venv/bin/activate
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

uv pip install --upgrade pip setuptools wheel packaging ninja psutil
uv pip install --torch-backend "$UV_TORCH_BACKEND" --prerelease=allow -e "$AUTOMODEL_DIR[vlm]"

# Runtime dependencies used by the online raw-data stream.
uv pip install --torch-backend "$UV_TORCH_BACKEND" --prerelease=allow pyarrow jinja2

# Qwen hybrid-attention support used by recent Qwen3.x checkpoints.
uv pip install --torch-backend "$UV_TORCH_BACKEND" --prerelease=allow --no-build-isolation \
  "causal-conv1d" \
  "mamba-ssm" \
  "flash-linear-attention>=0.4.2"

if [ "$INSTALL_FLASH_ATTN3" = "1" ]; then
  ./scripts/install_flash_attn3_prebuilt.sh
fi

if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
  uv pip install --torch-backend "$UV_TORCH_BACKEND" --prerelease=allow --no-build-isolation "flash-attn<=2.8.3"
fi

if [ "$INSTALL_TE" = "1" ]; then
  uv pip install --torch-backend "$UV_TORCH_BACKEND" --prerelease=allow "transformer-engine[pytorch]>=2.14.1" "onnxscript>=0.5.6"
fi

uv pip install --torch-backend "$UV_TORCH_BACKEND" cut-cross-entropy

python - <<'PY'
import importlib.util

mods = [
    "torch",
    "flash_attn_interface",
    "fla",
    "cut_cross_entropy",
    "nemo_automodel",
    "transformers",
    "pyarrow",
    "qwen_agentic_sft",
]
for mod in mods:
    print(f"{mod}: {bool(importlib.util.find_spec(mod))}")
PY

echo
echo "Environment ready. Activate it with:"
echo "  source .venv/bin/activate"
