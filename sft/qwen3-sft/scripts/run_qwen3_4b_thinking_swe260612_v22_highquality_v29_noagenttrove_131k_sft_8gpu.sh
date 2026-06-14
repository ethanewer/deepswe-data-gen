#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_qwen3_4b_thinking_swe260612_v23_highquality_exact_131k_sft_8gpu.sh" "$@"
