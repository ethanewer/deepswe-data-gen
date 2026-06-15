#!/usr/bin/env bash
set -euo pipefail

# Rebuild the v34 clean broad mix into many physical JSONL shards. Online
# packing shards by file when a raw root has multiple JSONLs, so the single
# large passed-prefix file from v34 must be split before distributed training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_ROOT="${INPUT_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/qwen3-4b-thinking-v34-clean-broad-prefix-editanchors-mix}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/qwen3-4b-thinking-v35-clean-broad-balanced-prefix-editanchors-mix}"
SHARDS_PER_SOURCE="${SHARDS_PER_SOURCE:-128}"

python "$SCRIPT_DIR/reshard_raw_jsonl_root.py" \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --shards-per-source "$SHARDS_PER_SOURCE" \
  "$@"
