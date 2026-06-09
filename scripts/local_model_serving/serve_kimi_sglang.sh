#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

PYTHON_BIN="${PYTHON_BIN:-$SGLANG_VENV/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18000}"
if [[ -z "${MODEL_LOADER_EXTRA_CONFIG:-}" ]]; then
  MODEL_LOADER_EXTRA_CONFIG='{"enable_multithread_load":"true","num_threads":64}'
fi

exec "$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$KIMI_MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME:-kimi-k2.6}" \
  --host "$HOST" \
  --port "$PORT" \
  --log-level-http warning \
  --enable-cache-report \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size "${TP_SIZE:-8}" \
  --decode-log-interval "${DECODE_LOG_INTERVAL:-1}" \
  --trust-remote-code \
  --watchdog-timeout 1000000 \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-128}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-16384}" \
  --model-loader-extra-config "$MODEL_LOADER_EXTRA_CONFIG" \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --context-length "${CONTEXT_LENGTH:-128000}" \
  --collect-tokens-histogram \
  --enable-metrics \
  --allow-auto-truncate \
  --enable-metrics-for-all-schedulers \
  --quantization compressed-tensors \
  --skip-server-warmup \
  --disable-tokenizer-batch-decode \
  --attention-backend "${ATTENTION_BACKEND:-flashinfer}" \
  --enforce-disable-flashinfer-allreduce-fusion \
  --disable-flashinfer-autotune \
  "$@"
