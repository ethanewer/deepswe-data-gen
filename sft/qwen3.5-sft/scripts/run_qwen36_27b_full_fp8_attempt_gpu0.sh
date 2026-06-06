#!/usr/bin/env bash
set -euo pipefail

if [ "${I_UNDERSTAND_FULL_SFT_WILL_OOM:-0}" != "1" ]; then
  echo "Refusing to launch by default: full 27B AdamW SFT is expected to OOM on one H200." >&2
  echo "Set I_UNDERSTAND_FULL_SFT_WILL_OOM=1 to run the reference attempt anyway." >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f .venv/bin/activate ]; then
  echo "Missing .venv. Run ./scripts/setup_nemo_automodel_env.sh first." >&2
  exit 1
fi

source .venv/bin/activate
VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/.venv/bin/python}"
AUTOMODEL_BIN="${AUTOMODEL_BIN:-$ROOT_DIR/.venv/bin/automodel}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export PYTHONPATH="$ROOT_DIR/third_party/Automodel${PYTHONPATH:+:$PYTHONPATH}"

if [ "$CUDA_VISIBLE_DEVICES" != "0" ]; then
  echo "This single-GPU recipe is pinned to GPU 0. CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

exec "$VENV_PYTHON" "$AUTOMODEL_BIN" configs/qwen36_27b_full_fp8_training_single_gpu_attempt.yaml
