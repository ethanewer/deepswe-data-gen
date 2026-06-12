#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REPO_ROOT="/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$DEFAULT_REPO_ROOT}}"
if [ ! -d "$REPO_ROOT/eval/benchmarks/swebench_multilingual" ]; then
  REPO_ROOT="$DEFAULT_REPO_ROOT"
fi
cd "$REPO_ROOT"
mkdir -p logs runs/serving_configs runs/serving runs/swebench_ml

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "Missing eval Python at $PYTHON" >&2
  exit 1
fi

EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-4}"
if ! [[ "$EVAL_GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_GPU_COUNT must be a positive integer; got $EVAL_GPU_COUNT" >&2
  exit 1
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/sft/qwen3-sft/checkpoints/qwen3_4b_thinking_swe260612_highquality_65k_online_packed_sft_fixed_h200_4gpu}"
CHECKPOINT_STEP_DIR="${CHECKPOINT_STEP_DIR:-$CHECKPOINT_DIR/epoch_0_step_49}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-step50}"
CONTEXT_LABEL="${CONTEXT_LABEL:-65k}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
CONSOLIDATED_DIR="${CONSOLIDATED_DIR:-$CHECKPOINT_STEP_DIR/model/consolidated}"
CONSOLIDATED_READY="$CONSOLIDATED_DIR/.complete"
CONSOLIDATED_LOCK="$CHECKPOINT_STEP_DIR/model/.consolidate.lock"

if [ ! -d "$CHECKPOINT_STEP_DIR/model" ]; then
  echo "Checkpoint model directory does not exist: $CHECKPOINT_STEP_DIR/model" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/sft/qwen3-sft/third_party/Automodel:$REPO_ROOT/sft/qwen3-sft/src${PYTHONPATH:+:$PYTHONPATH}"

if ! compgen -G "$CONSOLIDATED_DIR/*.safetensors" >/dev/null || [ ! -f "$CONSOLIDATED_READY" ]; then
  echo "Consolidating $CHECKPOINT_STEP_DIR/model -> $CONSOLIDATED_DIR"
  (
    flock -x 9
    if ! compgen -G "$CONSOLIDATED_DIR/*.safetensors" >/dev/null || [ ! -f "$CONSOLIDATED_READY" ]; then
      TMP_CONSOLIDATED_DIR="${CONSOLIDATED_DIR}.tmp.${SLURM_JOB_ID:-manual}.$$"
      rm -rf "$TMP_CONSOLIDATED_DIR"
      mkdir -p "$TMP_CONSOLIDATED_DIR"
      if [ -d "$CHECKPOINT_STEP_DIR/model/.hf_metadata" ]; then
        "$REPO_ROOT/sft/qwen3-sft/.venv/bin/python" \
          "$REPO_ROOT/sft/qwen3-sft/third_party/Automodel/tools/offline_hf_consolidation.py" \
          --backend gloo \
          --model-name "$MODEL_NAME" \
          --input-dir "$CHECKPOINT_STEP_DIR/model" \
          --output-dir "$TMP_CONSOLIDATED_DIR" \
          --cast-dtype bf16
      else
        "$REPO_ROOT/sft/qwen3-sft/.venv/bin/python" \
          "$REPO_ROOT/eval/benchmarks/swebench_multilingual/export_dcp_torchsave_to_hf.py" \
          --model-name "$MODEL_NAME" \
          --input-dir "$CHECKPOINT_STEP_DIR/model" \
          --output-dir "$TMP_CONSOLIDATED_DIR" \
          --dtype bf16
      fi
      rm -rf "$CONSOLIDATED_DIR"
      mv "$TMP_CONSOLIDATED_DIR" "$CONSOLIDATED_DIR"
      touch "$CONSOLIDATED_READY"
    fi
  ) 9>"$CONSOLIDATED_LOCK"
fi

BASE_PORT=$((20000 + (${SLURM_JOB_ID:-0} % 20000)))
PROXY_PORT="$BASE_PORT"
BACKEND_PORTS=()
for offset in $(seq 1 "$EVAL_GPU_COUNT"); do
  BACKEND_PORTS+=("$((BASE_PORT + offset))")
done

RUN_NAME="qwen3-4b-thinking-swe260612-fixed-${CHECKPOINT_LABEL}-${CONTEXT_LABEL}-l40s-${EVAL_GPU_COUNT}gpu-swebench-ml-p30-${SLURM_JOB_ID:-manual}"
CONFIG_PATH="$REPO_ROOT/runs/serving_configs/${RUN_NAME}.json"
SERVE_DIR="$REPO_ROOT/runs/serving/$RUN_NAME"
OUTPUT_DIR="$REPO_ROOT/runs/swebench_ml/$RUN_NAME"

"$PYTHON" - "$CONFIG_PATH" "$CONSOLIDATED_DIR" "$MODEL_NAME" "$PROXY_PORT" "${BACKEND_PORTS[@]}" <<'PY'
import json
import sys

config_path, model_dir, model_name, proxy_port, *backend_ports = sys.argv[1:]
gpu_count = len(backend_ports)
payload = {
    "model": model_dir,
    "description": f"Qwen3-4B Thinking SWE260612 fixed SFT checkpoint on {gpu_count} L40S GPUs.",
    "sources": [model_name, "eval/chat_templates/qwen3_thinking_acc.jinja2"],
    "serve": {
        "gpus": list(range(gpu_count)),
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

EVAL_WORKERS="${EVAL_WORKERS:-$EVAL_GPU_COUNT}"
EXTRA_BODY='{"top_p":0.95,"top_k":20,"min_p":0,"presence_penalty":0}'
"$PYTHON" -m eval.benchmarks.swebench_multilingual.run \
  --harness mini-swe-agent \
  --output "$OUTPUT_DIR" \
  --run-id "$RUN_NAME" \
  --generation-workers "$EVAL_WORKERS" \
  --eval-workers "$EVAL_WORKERS" \
  --model "$MODEL_NAME" \
  --litellm-model "openai/$MODEL_NAME" \
  --api-base "http://127.0.0.1:${PROXY_PORT}/v1" \
  --no-require-api-key \
  --temperature 0.6 \
  --max-tokens 16384 \
  --extra-body-json "$EXTRA_BODY"
