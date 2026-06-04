#!/usr/bin/env bash
set -euo pipefail

# Launch the DeepSWE easiest-5 eval through Pier/mini-swe-agent against the
# local round-robin OpenAI-compatible proxy.

TASKS_DIR="${TASKS_DIR:-/tmp/deep-swe/tasks}"
JOBS_DIR="${JOBS_DIR:-runs/pier-jobs}"
MODEL="${MODEL:-openai/Qwen/Qwen3.6-27B-FP8}"
JOB_PREFIX="${JOB_PREFIX:-qwen3.6-27b-fp8-easiest5-c10-ctx131k-tools-maxout4096}"
N_ATTEMPTS="${N_ATTEMPTS:-5}"
N_CONCURRENT="${N_CONCURRENT:-10}"
OPENAI_BASE_URL_VALUE="${OPENAI_BASE_URL_VALUE:-http://172.17.0.1.nip.io:8000/v1}"
OPENAI_API_BASE_VALUE="${OPENAI_API_BASE_VALUE:-$OPENAI_BASE_URL_VALUE}"
OPENAI_API_KEY_VALUE="${OPENAI_API_KEY_VALUE:-dummy}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
DEFAULT_PIER_BIN=""
if [[ -x ".venv-swe-uv/bin/pier" ]]; then
  DEFAULT_PIER_BIN=".venv-swe-uv/bin/pier"
else
  DEFAULT_PIER_BIN="$(command -v pier || true)"
fi
PIER_BIN="${PIER_BIN:-$DEFAULT_PIER_BIN}"

if [[ -z "$PIER_BIN" ]]; then
  echo "pier is not installed or not on PATH. Set PIER_BIN=/path/to/pier." >&2
  exit 2
fi

JOB_NAME="${JOB_NAME:-${JOB_PREFIX}-$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$JOBS_DIR"
printf '%s\n' "$JOB_NAME" | tee /tmp/deepswe_active_job_name.txt

OPENAI_BASE_URL="$OPENAI_BASE_URL_VALUE" \
OPENAI_API_BASE="$OPENAI_API_BASE_VALUE" \
OPENAI_API_KEY="$OPENAI_API_KEY_VALUE" \
"$PIER_BIN" run \
  -p "$TASKS_DIR" \
  --agent mini-swe-agent \
  --model "$MODEL" \
  --jobs-dir "$JOBS_DIR" \
  --job-name "$JOB_NAME" \
  --n-attempts "$N_ATTEMPTS" \
  --n-concurrent "$N_CONCURRENT" \
  --yes \
  -i true-myth-iterable-collection-combinators \
  -i happy-dom-abort-pending-body-reads \
  -i wazero-multi-module-snapshots \
  -i psd-tools-blend-range-api \
  -i ytt-jsonpath-query-api \
  --agent-kwarg "model_kwargs={\"tool_choice\":\"required\",\"parallel_tool_calls\":false,\"max_tool_calls\":1,\"max_output_tokens\":${MAX_OUTPUT_TOKENS}}" \
  --agent-env "MSWEA_API_KEY=$OPENAI_API_KEY_VALUE" \
  --agent-env "OPENAI_BASE_URL=$OPENAI_BASE_URL_VALUE" \
  --agent-env "OPENAI_API_BASE=$OPENAI_API_BASE_VALUE"
