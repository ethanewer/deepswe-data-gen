#!/usr/bin/env bash
#SBATCH --job-name=swe-qwen3vl2b
#SBATCH --partition=l40s-8gpu
#SBATCH --gres=gpu:l40s:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-swebench-ml-%j.out
#SBATCH --error=logs/slurm-swebench-ml-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
MODEL_CONFIG="${MODEL_CONFIG:-eval/serving/configs/qwen3_vl_2b_text_start_8gpu_40k.json}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-start}"
CONTEXT_LABEL="${CONTEXT_LABEL:-40k}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="${RUN_NAME:-qwen3-vl-2b-text-${CHECKPOINT_LABEL}-${CONTEXT_LABEL}-swebench-ml-p30-${RUN_SUFFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/swebench_ml/${RUN_NAME}}"
SERVE_DIR="${SERVE_DIR:-runs/serving/${RUN_NAME}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Thinking}"
LITELLM_MODEL="${LITELLM_MODEL:-openai/${MODEL_NAME}}"
API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
GENERATION_WORKERS="${GENERATION_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
SERVE_HEALTH_TIMEOUT="${SERVE_HEALTH_TIMEOUT:-2400}"
EXTRA_BODY_JSON="${EXTRA_BODY_JSON:-{\"top_p\":0.95,\"top_k\":20,\"min_p\":0,\"presence_penalty\":0}}"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

mkdir -p "$OUTPUT_DIR" "$SERVE_DIR" logs

cleanup() {
  local manifest="$SERVE_DIR/manifest.json"
  if [[ -f "$manifest" ]]; then
    "$PYTHON" - "$manifest" <<'PY'
import json
import os
import signal
import sys
import time

manifest = json.loads(open(sys.argv[1]).read())
processes = manifest.get("processes", [])

for entry in processes:
    pid = entry.get("pid")
    if not pid:
        continue
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

deadline = time.time() + 30
while time.time() < deadline:
    alive = []
    for entry in processes:
        pid = entry.get("pid")
        if not pid:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        alive.append(pid)
    if not alive:
        break
    time.sleep(1)

for entry in processes:
    pid = entry.get("pid")
    if not pid:
        continue
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
PY
  fi
}
trap cleanup EXIT

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
start_seconds="$(date +%s)"

"$PYTHON" -m eval.model.serve_from_config \
  --config "$MODEL_CONFIG" \
  --run-dir "$SERVE_DIR" \
  --background \
  --health-timeout "$SERVE_HEALTH_TIMEOUT"

"$PYTHON" - <<'PY'
import time
import urllib.request

deadline = time.time() + 120
last_error = ""
while time.time() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
            if response.status == 200:
                raise SystemExit(0)
    except Exception as exc:
        last_error = str(exc)
    time.sleep(2)
raise SystemExit(f"proxy health check failed: {last_error}")
PY

set +e
"$PYTHON" -m eval.benchmarks.swebench_multilingual.run \
  --harness mini-swe-agent \
  --output "$OUTPUT_DIR" \
  --run-id "$RUN_NAME" \
  --generation-workers "$GENERATION_WORKERS" \
  --eval-workers "$EVAL_WORKERS" \
  --model "$MODEL_NAME" \
  --api-base "$API_BASE" \
  --api-key-env OPENAI_API_KEY \
  --litellm-model "$LITELLM_MODEL" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --extra-body-json "$EXTRA_BODY_JSON" \
  --no-require-api-key
status="$?"
set -e

ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
end_seconds="$(date +%s)"

"$PYTHON" - "$OUTPUT_DIR/timing.json" <<PY
import json
import pathlib

path = pathlib.Path("$OUTPUT_DIR/timing.json")
payload = {
    "run_name": "$RUN_NAME",
    "checkpoint_label": "$CHECKPOINT_LABEL",
    "context_label": "$CONTEXT_LABEL",
    "model_config": "$MODEL_CONFIG",
    "output_dir": "$OUTPUT_DIR",
    "serve_dir": "$SERVE_DIR",
    "started_at": "$started_at",
    "ended_at": "$ended_at",
    "wall_time_seconds": int("$end_seconds") - int("$start_seconds"),
    "exit_status": int("$status"),
    "generation_workers": int("$GENERATION_WORKERS"),
    "eval_workers": int("$EVAL_WORKERS"),
    "max_tokens": int("$MAX_TOKENS"),
    "temperature": float("$TEMPERATURE"),
}
path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2), flush=True)
PY

exit "$status"
