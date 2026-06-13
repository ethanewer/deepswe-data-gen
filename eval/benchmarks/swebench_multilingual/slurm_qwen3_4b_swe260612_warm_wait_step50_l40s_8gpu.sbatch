#!/usr/bin/env bash
#SBATCH --job-name=eval-q3-4b-warm-s50
#SBATCH --partition=l40s-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-eval-qwen3-4b-swe260612-warm-step50-l40s-8g-%j.out
#SBATCH --error=logs/slurm-eval-qwen3-4b-swe260612-warm-step50-l40s-8g-%j.err

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

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/sft/qwen3-sft/checkpoints/qwen3_4b_thinking_swe260612_miniswe_aligned_65k_toolcall_only_h200_4gpu_sft}"
CHECKPOINT_STEP_DIR="${CHECKPOINT_STEP_DIR:-$CHECKPOINT_DIR/epoch_0_step_49}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-step50}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-8}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STABILIZE_SECONDS="${STABILIZE_SECONDS:-120}"
SMOKE_MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-65536}"
SMOKE_GPU_MEMORY_UTILIZATION="${SMOKE_GPU_MEMORY_UTILIZATION:-0.9}"
SMOKE_REQUIRE_TOOL_CALL="${SMOKE_REQUIRE_TOOL_CALL:-false}"
TRAIN_LOG_GLOB="${TRAIN_LOG_GLOB:-$REPO_ROOT/logs/slurm-qwen3-4b-swe260612-h200-4g-*.err}"

export PYTHONPATH="$REPO_ROOT/sft/qwen3-sft/third_party/Automodel:$REPO_ROOT/sft/qwen3-sft/src${PYTHONPATH:+:$PYTHONPATH}"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "checkpoint_step_dir=$CHECKPOINT_STEP_DIR"
echo "eval_gpu_count=$EVAL_GPU_COUNT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

BASE_PORT=$((10000 + (${SLURM_JOB_ID:-0} % 10000)))
SMOKE_PROXY_PORT="$BASE_PORT"
SMOKE_BACKEND_PORTS=()
for offset in $(seq 1 "$EVAL_GPU_COUNT"); do
  SMOKE_BACKEND_PORTS+=("$((BASE_PORT + offset))")
done

SMOKE_RUN_NAME="qwen3-4b-thinking-base-l40s-${EVAL_GPU_COUNT}gpu-smoke-${SLURM_JOB_ID:-manual}"
SMOKE_CONFIG_PATH="$REPO_ROOT/runs/serving_configs/${SMOKE_RUN_NAME}.json"
SMOKE_SERVE_DIR="$REPO_ROOT/runs/serving/$SMOKE_RUN_NAME"

"$PYTHON" - "$SMOKE_CONFIG_PATH" "$BASE_MODEL_NAME" "$SMOKE_PROXY_PORT" "$SMOKE_MAX_MODEL_LEN" "$SMOKE_GPU_MEMORY_UTILIZATION" "${SMOKE_BACKEND_PORTS[@]}" <<'PY'
import json
import sys

config_path, model_name, proxy_port, max_model_len, gpu_memory_utilization, *backend_ports = sys.argv[1:]
gpu_count = len(backend_ports)
payload = {
    "model": model_name,
    "description": f"Base Qwen3-4B Thinking smoke serve on {gpu_count} L40S GPUs.",
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
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(gpu_memory_utilization),
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

cleanup_smoke() {
  if [ -f "$SMOKE_SERVE_DIR/manifest.json" ]; then
    "$PYTHON" - "$SMOKE_SERVE_DIR/manifest.json" <<'PY' || true
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
trap cleanup_smoke EXIT

echo "Starting base-model smoke serve"
"$PYTHON" -m eval.model.serve_from_config \
  --config "$SMOKE_CONFIG_PATH" \
  --run-dir "$SMOKE_SERVE_DIR" \
  --background \
  --health-timeout 1800

echo "Running base-model smoke chat request"
"$PYTHON" - "$BASE_MODEL_NAME" "$SMOKE_PROXY_PORT" "$SMOKE_REQUIRE_TOOL_CALL" <<'PY'
import json
import sys
import urllib.request

model_name, proxy_port, require_tool_call = sys.argv[1:]
payload = {
    "model": model_name,
    "messages": [
        {"role": "system", "content": "You may call tools when useful."},
        {"role": "user", "content": "Use the bash tool to print the current working directory."},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a bash command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ],
    "tool_choice": "auto",
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 512,
}
request = urllib.request.Request(
    f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.loads(response.read().decode("utf-8"))
message = result["choices"][0]["message"]
tool_calls = message.get("tool_calls") or []
print(json.dumps({"smoke_tool_call_count": len(tool_calls), "message": message}, indent=2)[:4000])
if require_tool_call.lower() == "true" and not tool_calls:
    raise SystemExit("smoke request did not produce a tool call")
PY

echo "Stopping base-model smoke serve"
cleanup_smoke
trap - EXIT
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

echo "Waiting for checkpoint metadata: $CHECKPOINT_STEP_DIR/model/.metadata"
while [ ! -f "$CHECKPOINT_STEP_DIR/model/.metadata" ]; do
  latest_train_log="$(find "$REPO_ROOT/logs" -maxdepth 1 -type f -name "$(basename "$TRAIN_LOG_GLOB")" -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR == 1 {print $2}')"
  latest_step=""
  if [ -n "$latest_train_log" ]; then
    latest_step="$(tail -n 120 "$latest_train_log" 2>/dev/null | grep -E 'step [0-9]+ \\|' | tail -n 1 || true)"
  fi
  echo "[$(date -Is)] checkpoint not ready; latest_step=${latest_step:-unknown}"
  sleep "$POLL_SECONDS"
done

echo "[$(date -Is)] Found checkpoint metadata; waiting ${STABILIZE_SECONDS}s for writes to settle"
sleep "$STABILIZE_SECONDS"

echo "Starting real step-50 eval on warmed L40S allocation"
export CHECKPOINT_STEP_DIR
export CHECKPOINT_LABEL
export EVAL_GPU_COUNT
exec "$REPO_ROOT/eval/benchmarks/swebench_multilingual/run_qwen3_4b_swe260612_step50_l40s_eval.sh"
