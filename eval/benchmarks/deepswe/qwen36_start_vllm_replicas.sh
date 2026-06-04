#!/usr/bin/env bash
set -euo pipefail

# Start one vLLM OpenAI-compatible server per GPU, leaving GPU 0 free.
# The commands intentionally run in the foreground unless wrapped by a process
# manager such as tmux, screen, nohup, or this script with RUN_IN_BACKGROUND=1.

MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
GPUS_CSV="${GPUS_CSV:-1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VENV_BIN="${VENV_BIN:-.venv-swe-uv/bin}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

start_one() {
  local gpu="$1"
  local port="$((BASE_PORT + gpu))"
  echo "Starting $MODEL on GPU $gpu at :$port"
  CUDA_VISIBLE_DEVICES="$gpu" "$VENV_BIN/vllm" serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$port" \
    --tensor-parallel-size 1 \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
}

if [[ "$RUN_IN_BACKGROUND" == "1" ]]; then
  mkdir -p runs/deepswe-local-vllm
  for gpu in "${GPUS[@]}"; do
    log="runs/deepswe-local-vllm/vllm-gpu${gpu}.log"
    start_one "$gpu" >"$log" 2>&1 &
    echo "$!" >"runs/deepswe-local-vllm/vllm-gpu${gpu}.pid"
  done
  echo "Started ${#GPUS[@]} vLLM servers in background."
  echo "Logs: runs/deepswe-local-vllm/"
else
  if [[ "${#GPUS[@]}" -ne 1 ]]; then
    echo "Set RUN_IN_BACKGROUND=1 or GPUS_CSV=<single_gpu>." >&2
    exit 2
  fi
  start_one "${GPUS[0]}"
fi
