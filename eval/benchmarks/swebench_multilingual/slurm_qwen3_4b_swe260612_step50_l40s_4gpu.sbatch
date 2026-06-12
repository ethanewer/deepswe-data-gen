#!/usr/bin/env bash
#SBATCH --job-name=eval-q3-4b-swe-s50
#SBATCH --partition=l40s-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-eval-qwen3-4b-swe260612-step50-l40s-4g-%j.out
#SBATCH --error=logs/slurm-eval-qwen3-4b-swe260612-step50-l40s-4g-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs runs/serving_configs runs/serving runs/swebench_ml

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "Missing eval Python at $PYTHON" >&2
  exit 1
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/sft/qwen3-sft/checkpoints/qwen3_4b_thinking_swe260612_highquality_65k_online_packed_sft_fixed_h200_4gpu}"
CHECKPOINT_STEP_DIR="${CHECKPOINT_STEP_DIR:-$CHECKPOINT_DIR/epoch_0_step_49}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-step50}"
CONTEXT_LABEL="${CONTEXT_LABEL:-65k}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
CONSOLIDATED_DIR="${CONSOLIDATED_DIR:-$CHECKPOINT_STEP_DIR/model/consolidated}"

if [ ! -d "$CHECKPOINT_STEP_DIR/model" ]; then
  echo "Checkpoint model directory does not exist: $CHECKPOINT_STEP_DIR/model" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/sft/qwen3-sft/third_party/Automodel:$REPO_ROOT/sft/qwen3-sft/src${PYTHONPATH:+:$PYTHONPATH}"

if ! compgen -G "$CONSOLIDATED_DIR/*.safetensors" >/dev/null; then
  echo "Consolidating $CHECKPOINT_STEP_DIR/model -> $CONSOLIDATED_DIR"
  mkdir -p "$CONSOLIDATED_DIR"
  "$REPO_ROOT/sft/qwen3-sft/.venv/bin/python" \
    "$REPO_ROOT/sft/qwen3-sft/third_party/Automodel/tools/offline_hf_consolidation.py" \
    --backend gloo \
    --model-name "$MODEL_NAME" \
    --input-dir "$CHECKPOINT_STEP_DIR/model" \
    --output-dir "$CONSOLIDATED_DIR" \
    --cast-dtype bf16
fi

BASE_PORT=$((20000 + (${SLURM_JOB_ID:-0} % 20000)))
PROXY_PORT="$BASE_PORT"
BACKEND_PORT_0=$((BASE_PORT + 1))
BACKEND_PORT_1=$((BASE_PORT + 2))
BACKEND_PORT_2=$((BASE_PORT + 3))
BACKEND_PORT_3=$((BASE_PORT + 4))

RUN_NAME="qwen3-4b-thinking-swe260612-fixed-${CHECKPOINT_LABEL}-${CONTEXT_LABEL}-l40s-4gpu-swebench-ml-p30-${SLURM_JOB_ID:-manual}"
CONFIG_PATH="$REPO_ROOT/runs/serving_configs/${RUN_NAME}.json"
SERVE_DIR="$REPO_ROOT/runs/serving/$RUN_NAME"
OUTPUT_DIR="$REPO_ROOT/runs/swebench_ml/$RUN_NAME"

"$PYTHON" - "$CONFIG_PATH" "$CONSOLIDATED_DIR" "$MODEL_NAME" "$PROXY_PORT" "$BACKEND_PORT_0" "$BACKEND_PORT_1" "$BACKEND_PORT_2" "$BACKEND_PORT_3" <<'PY'
import json
import sys

config_path, model_dir, model_name, proxy_port, *backend_ports = sys.argv[1:]
payload = {
    "model": model_dir,
    "description": "Qwen3-4B Thinking SWE260612 fixed SFT checkpoint on 4 L40S GPUs.",
    "sources": [model_name, "eval/chat_templates/qwen3_thinking_acc.jinja2"],
    "serve": {
        "gpus": [0, 1, 2, 3],
        "map_gpus_from_cuda_visible_devices": True,
        "backend_ports": [int(port) for port in backend_ports],
        "backend_host": "0.0.0.0",
        "backend_base_url_host": "127.0.0.1",
        "proxy_host": "0.0.0.0",
        "proxy_port": int(proxy_port),
        "served_model_name": model_name,
        "tensor_parallel_size": 1,
        "max_model_len": 65536,
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": True,
        "chat_template": "eval/chat_templates/qwen3_thinking_acc.jinja2",
        "chat_template_content_format": "string",
        "vllm_args": [
            "--reasoning-parser",
            "deepseek_r1",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
        ],
    },
}
with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
print(config_path)
PY

cleanup() {
  if [ -f "$SERVE_DIR/manifest.json" ]; then
    "$PYTHON" - "$SERVE_DIR/manifest.json" <<'PY' || true
import json
import os
import signal
import sys
import time

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
pids = [int(item["pid"]) for item in manifest.get("processes", []) if item.get("pid")]
for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid in pids:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    time.sleep(5 if sig == signal.SIGTERM else 1)
PY
  fi
}
trap cleanup EXIT

"$PYTHON" -m eval.model.serve_from_config \
  --config "$CONFIG_PATH" \
  --run-dir "$SERVE_DIR" \
  --background \
  --health-timeout 1800

EXTRA_BODY='{"top_p":0.95,"top_k":20,"min_p":0,"presence_penalty":0}'
"$PYTHON" -m eval.benchmarks.swebench_multilingual.run \
  --harness mini-swe-agent \
  --output "$OUTPUT_DIR" \
  --run-id "$RUN_NAME" \
  --generation-workers 4 \
  --eval-workers 4 \
  --model "$MODEL_NAME" \
  --litellm-model "openai/$MODEL_NAME" \
  --api-base "http://127.0.0.1:${PROXY_PORT}/v1" \
  --no-require-api-key \
  --temperature 0.6 \
  --max-tokens 16384 \
  --extra-body-json "$EXTRA_BODY"
