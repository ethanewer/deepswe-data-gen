#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

VLLM_BIN="${VLLM_BIN:-$LOCAL_MODEL_SERVING_ROOT/venv-vllm/bin/vllm}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18000}"

exec "$VLLM_BIN" serve "$KIMI_MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME:-kimi-k2.6}" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "${TP_SIZE:-8}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --max-num-seqs "${MAX_NUM_SEQS:-256}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
  --enable-prefix-caching \
  --generation-config vllm \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB:-0}" \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --disable-uvicorn-access-log \
  --uvicorn-log-level warning \
  "$@"
