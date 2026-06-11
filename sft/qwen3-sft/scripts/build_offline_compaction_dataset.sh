#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x .venv/bin/python ]; then
  PYTHON_BIN=".venv/bin/python"
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

DATASET="${DATASET:-eewer/agent-traces-openai-style-all}"
HF_JSONL_FILE="${HF_JSONL_FILE:-all_traces_openai_style.jsonl}"
SPLIT="${SPLIT:-train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/data/agent_traces_compacted}"
MODE="${MODE:-included}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-65536}"
BOUNDARY_TOKENS="${BOUNDARY_TOKENS:-49152}"
SUMMARY_TOKEN_BUDGET="${SUMMARY_TOKEN_BUDGET:-1536}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ROOT_DIR/../../eval/chat_templates/qwen3_thinking_acc.jinja2}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"

exec "$PYTHON_BIN" -m qwen_agentic_sft.offline_compaction build \
  --dataset "$DATASET" \
  --hf-jsonl-file "$HF_JSONL_FILE" \
  --split "$SPLIT" \
  --output-root "$OUTPUT_ROOT" \
  --source-name "agent_traces_${MODE}_compaction" \
  --mode "$MODE" \
  --model "$MODEL" \
  --chat-template "$CHAT_TEMPLATE" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --boundary-tokens "$BOUNDARY_TOKENS" \
  --summary-token-budget "$SUMMARY_TOKEN_BUDGET" \
  --max-examples "$MAX_EXAMPLES" \
  --overwrite
