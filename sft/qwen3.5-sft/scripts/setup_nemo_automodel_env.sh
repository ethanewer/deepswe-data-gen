#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTOMODEL_DIR="${AUTOMODEL_DIR:-$ROOT_DIR/third_party/Automodel}"
AUTOMODEL_REF="${AUTOMODEL_REF:-main}"
UPDATE_AUTOMODEL="${UPDATE_AUTOMODEL:-0}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
INSTALL_TE="${INSTALL_TE:-0}"
INSTALL_FLASH_ATTN3="${INSTALL_FLASH_ATTN3:-1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
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

python - <<PY
from pathlib import Path

path = Path("$AUTOMODEL_DIR") / "nemo_automodel" / "_transformers" / "kernel_patches.py"
text = path.read_text()
if "HAS_FA3" not in text:
    text = text.replace(
        'HAS_FA, _ = safe_import("flash_attn")\nDEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2" if HAS_FA else "sdpa"',
        'HAS_FA, _ = safe_import("flash_attn")\nHAS_FA3, _ = safe_import("flash_attn_interface")\nDEFAULT_ATTN_IMPLEMENTATION = "flash_attention_3" if HAS_FA3 else "flash_attention_2" if HAS_FA else "sdpa"',
    )
    text = text.replace(
        """assert HAS_FA, "Flash Attention is not available"
            attn_implementation = "flash_attention_2"
            logger.warning(
                "Packed sequence is supported only with Flash Attention. "
                "Setting model's attn_implementation to flash_attention_2"
            )""",
        """if HAS_FA3:
                attn_implementation = "flash_attention_3"
            else:
                assert HAS_FA, "Flash Attention is not available"
                attn_implementation = "flash_attention_2"
            logger.warning(
                "Packed sequence is supported only with Flash Attention. "
                "Setting model's attn_implementation to %s",
                attn_implementation,
            )""",
    )
    path.write_text(text)
PY

if [ -d .venv ] && [ "${RECREATE_VENV:-0}" != "1" ]; then
  uv venv --python "$PYTHON_BIN" --allow-existing .venv
else
  uv venv --python "$PYTHON_BIN" --clear .venv
fi
source .venv/bin/activate

uv pip install --upgrade pip setuptools wheel packaging ninja psutil

# Install the base package first so torch is present before source-built CUDA
# extensions such as flash-attn run their setup.py imports.
uv pip install --torch-backend "$UV_TORCH_BACKEND" --prerelease=allow -e "$AUTOMODEL_DIR[vlm]"

# Qwen3.5/Qwen3.6 uses the Qwen3.5 hybrid stack: full attention plus Gated
# DeltaNet. These packages cover the fast linear-attention path.
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

# Used by the config to avoid materializing the huge [seq, vocab] logits tensor
# for Qwen3.6's 248k-token vocab.
uv pip install --torch-backend "$UV_TORCH_BACKEND" cut-cross-entropy

python - <<'PY'
import importlib.util

mods = [
    "torch",
    "torchao",
    "flash_attn_interface",
    "flash_attn",
    "fla",
    "cut_cross_entropy",
    "nemo_automodel",
    "transformers",
]
for mod in mods:
    print(f"{mod}: {bool(importlib.util.find_spec(mod))}")
print(f"transformer_engine: {bool(importlib.util.find_spec('transformer_engine'))}")
PY

echo
echo "Environment ready. Activate it with:"
echo "  source .venv/bin/activate"
