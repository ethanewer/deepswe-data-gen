#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

PYTHON_BIN="${PYTHON_BIN:-$SGLANG_VENV/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-19001}"
ENABLE_DEEPEP="${ENABLE_DEEPEP:-0}"
if [[ -z "${MODEL_LOADER_EXTRA_CONFIG:-}" ]]; then
  MODEL_LOADER_EXTRA_CONFIG='{"enable_multithread_load":"true","num_threads":64}'
fi
ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-}"
if [[ -z "$ENABLE_DP_ATTENTION" ]]; then
  if [[ "${DP_SIZE:-2}" -gt 1 ]]; then
    ENABLE_DP_ATTENTION=1
  else
    ENABLE_DP_ATTENTION=0
  fi
fi

export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-256}"

deepep_args=()
if [[ "$ENABLE_DEEPEP" == "1" ]]; then
  deepep_args+=(--moe-a2a-backend deepep --deepep-mode auto --moe-dense-tp-size 1)
fi

dp_attention_args=()
if [[ "$ENABLE_DP_ATTENTION" == "1" ]]; then
  dp_attention_args+=(--enable-dp-attention)
fi

exec "$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$MIMO_MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME:-mimo-v2.5}" \
  --host "$HOST" \
  --port "$PORT" \
  --log-level-http warning \
  --enable-cache-report \
  --pp-size 1 \
  --dp-size "${DP_SIZE:-2}" \
  --tp-size "${TP_SIZE:-8}" \
  "${dp_attention_args[@]}" \
  --decode-log-interval "${DECODE_LOG_INTERVAL:-1}" \
  --page-size 1 \
  --trust-remote-code \
  --watchdog-timeout 1000000 \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.65}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-128}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-16384}" \
  --model-loader-extra-config "$MODEL_LOADER_EXTRA_CONFIG" \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  --tool-call-parser mimo \
  --context-length "${CONTEXT_LENGTH:-262144}" \
  --collect-tokens-histogram \
  --enable-metrics \
  --load-balance-method round_robin \
  --allow-auto-truncate \
  --enable-metrics-for-all-schedulers \
  --quantization fp8 \
  --skip-server-warmup \
  --enable-dp-lm-head \
  --disable-tokenizer-batch-decode \
  --attention-backend "${ATTENTION_BACKEND:-triton}" \
  "${deepep_args[@]}" \
  "$@"
