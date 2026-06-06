#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -x .venv/bin/python ]; then
  PYTHON_BIN=".venv/bin/python"
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
RAW_ROOT="${RAW_ROOT:-$ROOT_DIR/data/smoke_raw}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2}"
PACK_SIZE="${PACK_SIZE:-131072}"
MAX_EXAMPLES="${MAX_EXAMPLES:-128}"
MAX_PACKS="${MAX_PACKS:-2}"

exec "$PYTHON_BIN" -m qwen_agentic_sft.online_packed_dataset inspect \
  --model "$MODEL" \
  --raw-root "$RAW_ROOT" \
  --chat-template "$CHAT_TEMPLATE" \
  --sequence-length "$PACK_SIZE" \
  --max-examples "$MAX_EXAMPLES" \
  --max-packs "$MAX_PACKS"
