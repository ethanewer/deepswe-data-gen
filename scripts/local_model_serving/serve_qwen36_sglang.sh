#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

PYTHON_BIN="${PYTHON_BIN:-$SGLANG_VENV/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-20000}"
if [[ -z "${MODEL_LOADER_EXTRA_CONFIG:-}" ]]; then
  MODEL_LOADER_EXTRA_CONFIG='{"enable_multithread_load":"true","num_threads":64}'
fi
ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-}"
if [[ -z "$ENABLE_DP_ATTENTION" ]]; then
  if [[ "${DP_SIZE:-1}" -gt 1 ]]; then
    ENABLE_DP_ATTENTION=1
  else
    ENABLE_DP_ATTENTION=0
  fi
fi

dp_attention_args=()
if [[ "$ENABLE_DP_ATTENTION" == "1" ]]; then
  dp_attention_args+=(--enable-dp-attention)
fi

max_total_tokens_args=()
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
  max_total_tokens_args+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi

cuda_graph_args=()
if [[ -n "${CUDA_GRAPH_MAX_BS:-}" ]]; then
  cuda_graph_args+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
fi
if [[ -n "${CUDA_GRAPH_BS:-}" ]]; then
  # shellcheck disable=SC2206
  cuda_graph_bs_values=($CUDA_GRAPH_BS)
  cuda_graph_args+=(--cuda-graph-bs "${cuda_graph_bs_values[@]}")
fi

exec "$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$QWEN36_MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME:-qwen3.6-27b}" \
  --host "$HOST" \
  --port "$PORT" \
  --log-level-http warning \
  --enable-cache-report \
  --pp-size 1 \
  --dp-size "${DP_SIZE:-1}" \
  --tp-size "${TP_SIZE:-8}" \
  "${dp_attention_args[@]}" \
  --decode-log-interval "${DECODE_LOG_INTERVAL:-1}" \
  --trust-remote-code \
  --watchdog-timeout 1000000 \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.90}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-16}" \
  "${max_total_tokens_args[@]}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-16384}" \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS:-65536}" \
  --model-loader-extra-config "$MODEL_LOADER_EXTRA_CONFIG" \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
  --context-length "${CONTEXT_LENGTH:-131072}" \
  --collect-tokens-histogram \
  --enable-metrics \
  --load-balance-method round_robin \
  --allow-auto-truncate \
  --enable-metrics-for-all-schedulers \
  --skip-server-warmup \
  --disable-tokenizer-batch-decode \
  --attention-backend "${ATTENTION_BACKEND:-flashinfer}" \
  --linear-attn-backend "${LINEAR_ATTN_BACKEND:-triton}" \
  "${cuda_graph_args[@]}" \
  "$@"
