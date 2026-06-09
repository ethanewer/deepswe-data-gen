#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SERVING_BACKEND="${SERVING_BACKEND:-vllm}"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

VLLM_BIN="${VLLM_BIN:-$VLLM_VENV/bin/vllm}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-20011}"

max_num_seqs_args=()
if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
  max_num_seqs_args+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

max_num_batched_tokens_args=()
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  max_num_batched_tokens_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

speculative_config_args=()
if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
  speculative_config_args+=(--speculative-config "$SPECULATIVE_CONFIG")
fi

expert_parallel_args=()
if [[ "${ENABLE_EXPERT_PARALLEL:-0}" == "1" ]]; then
  expert_parallel_args+=(--enable-expert-parallel)
fi

prefix_cache_args=()
if [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
  prefix_cache_args+=(--enable-prefix-caching)
fi

exec "$VLLM_BIN" serve "$QWEN36_MOE_FP8_MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME:-qwen3.6-35b-a3b-fp8}" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "${TP_SIZE:-8}" \
  --max-model-len "${MAX_MODEL_LEN:-${CONTEXT_LENGTH:-131072}}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
  --language-model-only \
  "${expert_parallel_args[@]}" \
  "${prefix_cache_args[@]}" \
  "${max_num_seqs_args[@]}" \
  "${max_num_batched_tokens_args[@]}" \
  "${speculative_config_args[@]}" \
  "$@"
