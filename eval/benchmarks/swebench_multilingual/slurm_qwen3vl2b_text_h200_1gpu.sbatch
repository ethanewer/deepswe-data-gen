#!/usr/bin/env bash
#SBATCH --job-name=swe-q3vl2b-h200
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-swebench-ml-h200-1g-%j.out
#SBATCH --error=logs/slurm-swebench-ml-h200-1g-%j.err

# =============================================================================
# Execution setting: SLURM-GPU (H200, 1 GPU). (Despite the historical
# "run_local_served" name, this is a SLURM batch job, hence the slurm_ prefix.)
#
# Purpose: single-H200 variant of slurm_qwen3vl2b_text_8gpu.sbatch -- serve a
# Qwen3-VL-2B text checkpoint from a serving config (one vLLM backend; proxy
# port read from the config), then run SWE-bench Multilingual predictive-30 via
# mini-swe-agent + official swebench Docker evaluation, and write timing.json.
#
# Key env vars: MODEL_CONFIG (eval/serving/configs/qwen3_vl_2b_text_*h200_1gpu*.json;
# default the h200 1gpu 40k start config), MODEL_NAME, CHECKPOINT_LABEL,
# CONTEXT_LABEL, MAX_TOKENS, GENERATION_WORKERS (default 1), EVAL_WORKERS,
# OUTPUT_DIR/SERVE_DIR/RUN_NAME.
# Prerequisites: SLURM H200 partition; eval venv at $PYTHON; Docker reachable.
# =============================================================================
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen}"
SCRIPT_DIR="$REPO_ROOT/eval/benchmarks/swebench_multilingual"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
MODEL_CONFIG="${MODEL_CONFIG:-eval/serving/configs/qwen3_vl_2b_text_start_h200_1gpu_40k.json}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-start}"
CONTEXT_LABEL="${CONTEXT_LABEL:-40k}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="${RUN_NAME:-qwen3-vl-2b-text-${CHECKPOINT_LABEL}-${CONTEXT_LABEL}-h200-1gpu-swebench-ml-p30-${RUN_SUFFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/swebench_ml/${RUN_NAME}}"
SERVE_DIR="${SERVE_DIR:-runs/serving/${RUN_NAME}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Thinking}"
LITELLM_MODEL="${LITELLM_MODEL:-openai/${MODEL_NAME}}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
GENERATION_WORKERS="${GENERATION_WORKERS:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
SERVE_HEALTH_TIMEOUT="${SERVE_HEALTH_TIMEOUT:-2400}"
EXTRA_BODY_JSON="${EXTRA_BODY_JSON:-{\"top_p\":0.95,\"top_k\":20,\"min_p\":0,\"presence_penalty\":0}}"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

PROXY_PORT="$("$PYTHON" - "$MODEL_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    cfg = json.load(f)
print(cfg["serve"].get("proxy_port", 8000))
PY
)"
API_BASE="${API_BASE:-http://127.0.0.1:${PROXY_PORT}/v1}"

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

"$PYTHON" - "$PROXY_PORT" <<'PY'
import sys
import time
import urllib.request

port = sys.argv[1]
deadline = time.time() + 120
last_error = ""
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
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
path = pathlib.Path("$OUTPUT_DIR/timing.json")
path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2), flush=True)
PY

exit "$status"
