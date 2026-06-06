#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -x .venv/bin/python ]; then
  PYTHON_BIN=".venv/bin/python"
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

RAW_ROOT="${RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/code-swe-terminal-agentic-sft}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/data/smoke_raw}"
ROWS_PER_DATASET="${ROWS_PER_DATASET:-3}"
MAX_EVENT_LOG_LINES="${MAX_EVENT_LOG_LINES:-400}"

exec "$PYTHON_BIN" -m qwen_agentic_sft.data build-smoke-raw \
  --raw-root "$RAW_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --rows-per-dataset "$ROWS_PER_DATASET" \
  --max-event-log-lines "$MAX_EVENT_LOG_LINES" \
  --overwrite
