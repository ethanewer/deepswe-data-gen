#!/usr/bin/env bash
# =============================================================================
# Execution setting: SLURM-GPU (L40S) -- served-checkpoint driver
#
# Purpose: serve a ready Hugging Face SFT checkpoint (Qwen3-4B-Thinking,
# swe260612 miniswe-aligned) with local vLLM (one backend per GPU behind a
# round-robin proxy) and run the SWE-bench Multilingual (or Verified) predictive
# subset through the mini-swe-agent harness + official swebench Docker
# evaluation. This is the inner driver exec'd by the slurm_* wrappers and by the
# warm_wait wrapper; it can also be run directly inside an interactive SLURM
# allocation.
#
# NeMo note: the retired NeMo DCP-consolidation path was REMOVED. This script
# now serves CHECKPOINT_STEP_DIR directly as a ready HF checkpoint (ms-swift
# output, *.safetensors). It no longer puts Automodel/src on PYTHONPATH and no
# longer runs offline_hf_consolidation.py / export_dcp_torchsave_to_hf.py.
#
# Key env vars:
#   REPO_ROOT            repo root (defaults to SLURM_SUBMIT_DIR or fixed path)
#   PYTHON               eval venv python (default $REPO_ROOT/.venv/bin/python)
#   EVAL_GPU_COUNT       vLLM backends / GPUs (default 4)
#   BASELINE_MODEL       true -> serve the HF base MODEL_NAME instead of a ckpt
#   CHECKPOINT_DIR       SFT run dir; CHECKPOINT_STEP_DIR=its step subdir
#   CHECKPOINT_STEP_DIR  servable HF checkpoint dir (contains *.safetensors)
#   MODEL_NAME           base/served model id (default Qwen/Qwen3-4B-Thinking-2507)
#   BENCHMARK            multilingual | verified
#   CONTEXT_LABEL/MAX_TOKENS/TEMPERATURE/GENERATION_STEP_LIMIT/RUN_SUFFIX
#
# Prerequisites: GPU allocation; eval venv at $PYTHON; Docker reachable for the
# official swebench evaluation; a servable HF checkpoint at CHECKPOINT_STEP_DIR
# (unless BASELINE_MODEL=true).
# =============================================================================
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

BASELINE_MODEL="${BASELINE_MODEL:-false}"
# CHECKPOINT_DIR/CHECKPOINT_STEP_DIR must point at a servable HF checkpoint dir
# (ms-swift output, containing *.safetensors). It is served as-is; no NeMo DCP
# consolidation is performed.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/sft/qwen3/checkpoints/qwen3_4b_thinking_swe260612_miniswe_aligned_passed_65k_reasoning_toolcall_h200_4gpu_sft}"
CHECKPOINT_STEP_DIR="${CHECKPOINT_STEP_DIR:-$CHECKPOINT_DIR/epoch_0_step_49}"
if [ "$BASELINE_MODEL" = "true" ]; then
  CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-base}"
else
  CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-step50}"
fi
CONTEXT_LABEL="${CONTEXT_LABEL:-65k}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
BENCHMARK="${BENCHMARK:-multilingual}"
case "$BENCHMARK" in
  multilingual)
    DEFAULT_BENCHMARK_MODULE="eval.benchmarks.swebench_multilingual.run"
    DEFAULT_INSTANCE_IDS_PATH="$REPO_ROOT/eval/benchmarks/swebench_multilingual/predictive_30_instance_ids.txt"
    DEFAULT_BENCHMARK_OUTPUT_ROOT="$REPO_ROOT/runs/swebench_ml"
    DEFAULT_BENCHMARK_LABEL="swebench-ml-p30"
    ;;
  verified)
    DEFAULT_BENCHMARK_MODULE="eval.benchmarks.swebench_verified.run"
    DEFAULT_INSTANCE_IDS_PATH="$REPO_ROOT/eval/benchmarks/swebench_verified/predictive_20_instance_ids.txt"
    DEFAULT_BENCHMARK_OUTPUT_ROOT="$REPO_ROOT/runs/swebench_verified"
    DEFAULT_BENCHMARK_LABEL="swebench-verified-p20"
    ;;
  *)
    echo "BENCHMARK must be multilingual or verified; got $BENCHMARK" >&2
    exit 1
    ;;
esac
BENCHMARK_MODULE="${BENCHMARK_MODULE:-$DEFAULT_BENCHMARK_MODULE}"
INSTANCE_IDS_PATH="${INSTANCE_IDS_PATH:-$DEFAULT_INSTANCE_IDS_PATH}"
BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-$DEFAULT_BENCHMARK_OUTPUT_ROOT}"
BENCHMARK_LABEL="${BENCHMARK_LABEL:-$DEFAULT_BENCHMARK_LABEL}"
# Keep enough headroom for long mini-swe-agent histories under the 65k vLLM cap.
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.6}"
GENERATION_STEP_LIMIT="${GENERATION_STEP_LIMIT:-250}"
SKIP_EVALUATION="${SKIP_EVALUATION:-false}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
DEVICE_LABEL="${DEVICE_LABEL:-l40s}"
ENABLE_DOCKER_STDIO_PROXY="${ENABLE_DOCKER_STDIO_PROXY:-true}"
DOCKER_PROXY_PID=""
DOCKER_PROXY_DIR=""
TRIAL_LOCK=""
TRIAL_LOCK_ROOT=""
TRIAL_COMPLETE=""

if [ "$BASELINE_MODEL" != "true" ] && ! compgen -G "$CHECKPOINT_STEP_DIR/*.safetensors" >/dev/null; then
  echo "No servable HF checkpoint (*.safetensors) found in: $CHECKPOINT_STEP_DIR" >&2
  exit 1
fi

if [ "$BASELINE_MODEL" = "true" ]; then
  MODEL_SOURCE="$MODEL_NAME"
else
  # Serve the ready HF checkpoint directory directly (ms-swift output).
  MODEL_SOURCE="$CHECKPOINT_STEP_DIR"
fi

BASE_PORT=$((20000 + (${SLURM_JOB_ID:-0} % 20000)))
PROXY_PORT="$BASE_PORT"
BACKEND_PORT_GAP="${BACKEND_PORT_GAP:-100}"
BACKEND_PORTS=()
for offset in $(seq 1 "$EVAL_GPU_COUNT"); do
  BACKEND_PORTS+=("$((BASE_PORT + BACKEND_PORT_GAP + offset))")
done

if [ "$BASELINE_MODEL" = "true" ]; then
  DEFAULT_RUN_STEM="qwen3-4b-thinking-base-${CHECKPOINT_LABEL}-${CONTEXT_LABEL}"
else
  DEFAULT_RUN_STEM="qwen3-4b-thinking-swe260612-miniswe-${CHECKPOINT_LABEL}-${CONTEXT_LABEL}"
fi
RUN_STEM="${RUN_STEM:-$DEFAULT_RUN_STEM}"
RUN_NAME="${RUN_STEM}-${DEVICE_LABEL}-${EVAL_GPU_COUNT}gpu-${BENCHMARK_LABEL}${RUN_SUFFIX:+-$RUN_SUFFIX}-${SLURM_JOB_ID:-manual}"
CONFIG_PATH="$REPO_ROOT/runs/serving_configs/${RUN_NAME}.json"
SERVE_DIR="$REPO_ROOT/runs/serving/$RUN_NAME"
OUTPUT_DIR="$BENCHMARK_OUTPUT_ROOT/$RUN_NAME"
SERVE_CACHE_DIR="${SERVE_CACHE_DIR:-/tmp/q3eval-${SLURM_JOB_ID:-manual}-real}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SERVE_CACHE_DIR/hf_datasets}"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$BENCHMARK_OUTPUT_ROOT"

if [ "${ENABLE_EVAL_TRIAL_LOCK:-true}" = "true" ] && [ -n "$RUN_SUFFIX" ]; then
  TRIAL_LOCK_BASE="${EVAL_TRIAL_LOCK_DIR:-$REPO_ROOT/runs/eval_trial_locks}"
  TRIAL_LOCK_KEY="${EVAL_TRIAL_LOCK_KEY:-${CHECKPOINT_LABEL}-${BENCHMARK_LABEL}-${RUN_SUFFIX}}"
  SAFE_TRIAL_LOCK_KEY="$(printf '%s' "$TRIAL_LOCK_KEY" | tr -cs 'A-Za-z0-9_.=-' '_' | sed 's/^_*//; s/_*$//')"
  SAFE_TRIAL_LOCK_KEY="${SAFE_TRIAL_LOCK_KEY:0:180}"
  TRIAL_LOCK_ROOT="$TRIAL_LOCK_BASE/$SAFE_TRIAL_LOCK_KEY"
  TRIAL_LOCK="$TRIAL_LOCK_ROOT/lock"
  TRIAL_COMPLETE="$TRIAL_LOCK_ROOT/complete"
  mkdir -p "$TRIAL_LOCK_ROOT"
  if [ -f "$TRIAL_COMPLETE" ]; then
    echo "Eval trial already complete; skipping: $TRIAL_LOCK_KEY"
    cat "$TRIAL_COMPLETE"
    exit 0
  fi
  if ! mkdir "$TRIAL_LOCK" 2>/dev/null; then
    echo "Eval trial already claimed by another job; skipping: $TRIAL_LOCK_KEY"
    cat "$TRIAL_LOCK/owner" 2>/dev/null || true
    exit 0
  fi
  {
    echo "run_name=$RUN_NAME"
    echo "job_id=${SLURM_JOB_ID:-manual}"
    echo "host=$(hostname)"
    echo "started_at=$(date -Is)"
  } >"$TRIAL_LOCK/owner"
fi

"$PYTHON" - "$CONFIG_PATH" "$MODEL_SOURCE" "$MODEL_NAME" "$PROXY_PORT" "$SERVE_CACHE_DIR" "${BACKEND_PORTS[@]}" <<'PY'
import json
import sys

config_path, model_dir, model_name, proxy_port, serve_cache_dir, *backend_ports = sys.argv[1:]
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
        "env": {
            "TORCHINDUCTOR_CACHE_DIR": f"{serve_cache_dir}/backend-{{backend_index}}/torchinductor",
            "TRITON_CACHE_DIR": f"{serve_cache_dir}/backend-{{backend_index}}/triton",
            "CUDA_CACHE_PATH": f"{serve_cache_dir}/backend-{{backend_index}}/cuda",
            "VLLM_CACHE_ROOT": f"{serve_cache_dir}/backend-{{backend_index}}/vllm",
            "TMPDIR": f"{serve_cache_dir}/backend-{{backend_index}}/tmp",
        },
        "vllm_args": [
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
  if [ -n "$DOCKER_PROXY_PID" ]; then
    kill "$DOCKER_PROXY_PID" 2>/dev/null || true
    wait "$DOCKER_PROXY_PID" 2>/dev/null || true
  fi
  if [ -n "$DOCKER_PROXY_DIR" ]; then
    rm -rf "$DOCKER_PROXY_DIR"
  fi
  if [ -n "$TRIAL_LOCK" ] && [ -d "$TRIAL_LOCK" ]; then
    rm -rf "$TRIAL_LOCK"
  fi
}
trap cleanup EXIT

"$PYTHON" -m eval.model.serve_from_config \
  --config "$CONFIG_PATH" \
  --run-dir "$SERVE_DIR" \
  --background \
  --health-timeout 1800

EVAL_WORKERS="${EVAL_WORKERS:-$EVAL_GPU_COUNT}"
GENERATION_WORKERS="${GENERATION_WORKERS:-$EVAL_GPU_COUNT}"
if [ "${ENABLE_STRICT_MSWEA_GUARDS:-false}" = "true" ]; then
  export PYDEPS_OVERLAY="${PYDEPS_OVERLAY:-/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-miniswe-upstream-a85bf5e-fixed-20260610T2238}"
  if [ ! -d "$PYDEPS_OVERLAY/minisweagent" ]; then
    echo "Missing mini-swe-agent overlay at PYDEPS_OVERLAY=$PYDEPS_OVERLAY" >&2
    exit 1
  fi
  export PYTHONPATH="$PYDEPS_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
  export EXTRA_BODY_TOP_P="${EXTRA_BODY_TOP_P:-0.95}"
  export EXTRA_BODY_TOP_K="${EXTRA_BODY_TOP_K:-20}"
  export EXTRA_BODY_MIN_P="${EXTRA_BODY_MIN_P:-0}"
  export EXTRA_BODY_PRESENCE_PENALTY="${EXTRA_BODY_PRESENCE_PENALTY:-1}"
  export EXTRA_BODY_REPETITION_PENALTY="${EXTRA_BODY_REPETITION_PENALTY:-1.1}"
  export EXTRA_BODY_FREQUENCY_PENALTY="${EXTRA_BODY_FREQUENCY_PENALTY:-0.2}"
  export MSWEA_REPEAT_GUARD_THRESHOLD="${MSWEA_REPEAT_GUARD_THRESHOLD:-2}"
  export MSWEA_REPEAT_GUARD_BLOCK_AFTER="${MSWEA_REPEAT_GUARD_BLOCK_AFTER:-3}"
  export MSWEA_CONTEXT_GUARD_MAX_CHARS="${MSWEA_CONTEXT_GUARD_MAX_CHARS:-220000}"
  export MSWEA_CONTEXT_GUARD_MIN_CALLS="${MSWEA_CONTEXT_GUARD_MIN_CALLS:-10}"
  export MSWEA_OUTPUT_MAX_CHARS="${MSWEA_OUTPUT_MAX_CHARS:-3500}"
  export MSWEA_OUTPUT_HEAD_CHARS="${MSWEA_OUTPUT_HEAD_CHARS:-1800}"
  export MSWEA_OUTPUT_TAIL_CHARS="${MSWEA_OUTPUT_TAIL_CHARS:-1300}"
  export MSWEA_HISTORY_COMPACT_MAX_CHARS="${MSWEA_HISTORY_COMPACT_MAX_CHARS:-145000}"
  export MSWEA_HISTORY_COMPACT_KEEP_RECENT="${MSWEA_HISTORY_COMPACT_KEEP_RECENT:-18}"
  export MSWEA_HISTORY_COMPACT_MSG_CHARS="${MSWEA_HISTORY_COMPACT_MSG_CHARS:-1100}"
  export MSWEA_HISTORY_COMPACT_HEAD_CHARS="${MSWEA_HISTORY_COMPACT_HEAD_CHARS:-600}"
  export MSWEA_HISTORY_COMPACT_TAIL_CHARS="${MSWEA_HISTORY_COMPACT_TAIL_CHARS:-400}"
  export MSWEA_NO_EDIT_WARNING_CALLS="${MSWEA_NO_EDIT_WARNING_CALLS:-20}"
  export MSWEA_NO_EDIT_WARNING_INTERVAL="${MSWEA_NO_EDIT_WARNING_INTERVAL:-15}"
  export MSWEA_FORCE_SUBMIT_AFTER_CALLS="${MSWEA_FORCE_SUBMIT_AFTER_CALLS:-75}"
  export MSWEA_STOP_NO_EDIT_AFTER_CALLS="${MSWEA_STOP_NO_EDIT_AFTER_CALLS:-90}"
  export MSWEA_VALIDATE_SUBMISSION="${MSWEA_VALIDATE_SUBMISSION:-true}"
  export MSWEA_MIN_SUBMISSION_CHARS="${MSWEA_MIN_SUBMISSION_CHARS:-80}"
  export MSWEA_REQUIRE_DIFF_SUBMISSION="${MSWEA_REQUIRE_DIFF_SUBMISSION:-true}"
  export MSWEA_REQUIRE_SOURCE_EDIT="${MSWEA_REQUIRE_SOURCE_EDIT:-true}"
  export MSWEA_MAX_CHANGED_FILES="${MSWEA_MAX_CHANGED_FILES:-8}"
  export MSWEA_VALIDATE_GIT_APPLY="${MSWEA_VALIDATE_GIT_APPLY:-true}"
  export MSWEA_VALIDATE_SMOKE="${MSWEA_VALIDATE_SMOKE:-true}"
  export MSWEA_VALIDATE_SMOKE_TIMEOUT="${MSWEA_VALIDATE_SMOKE_TIMEOUT:-120}"
  echo "Enabled strict mini-swe-agent eval guards via ENABLE_STRICT_MSWEA_GUARDS=true"
  "$PYTHON" - <<'PY'
import minisweagent.run.benchmarks.utils.common as common

if not hasattr(common.ProgressTrackingAgent, "_validate_submission_from_env"):
    raise SystemExit(f"strict mini-swe-agent overlay not active: {common.__file__}")
print(f"Using strict mini-swe-agent overlay: {common.__file__}")
PY
fi
if [ -n "${EXTRA_BODY_JSON:-}" ]; then
  EXTRA_BODY="$EXTRA_BODY_JSON"
else
  EXTRA_BODY="$("$PYTHON" - <<'PY'
import json
import os


def number_env(name: str, default: float | int | None = None):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        numeric = float(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be numeric; got {value!r}") from exc
    return int(numeric) if numeric.is_integer() else numeric


body = {
    "top_p": number_env("EXTRA_BODY_TOP_P", 0.95),
    "top_k": number_env("EXTRA_BODY_TOP_K", 20),
    "min_p": number_env("EXTRA_BODY_MIN_P", 0),
    "presence_penalty": number_env("EXTRA_BODY_PRESENCE_PENALTY", 0),
}
optional_keys = {
    "EXTRA_BODY_REPETITION_PENALTY": "repetition_penalty",
    "EXTRA_BODY_FREQUENCY_PENALTY": "frequency_penalty",
}
for env_name, json_key in optional_keys.items():
    value = number_env(env_name)
    if value is not None:
        body[json_key] = value
print(json.dumps(body, separators=(",", ":")))
PY
)"
fi
if [ "$ENABLE_DOCKER_STDIO_PROXY" = "true" ]; then
  DOCKER_PROXY_DIR="$(mktemp -d "/tmp/swebench-docker-proxy.${SLURM_JOB_ID:-manual}.XXXXXX")"
  DOCKER_PROXY_SOCKET="$DOCKER_PROXY_DIR/docker.sock"
  "$PYTHON" "$REPO_ROOT/eval/benchmarks/swebench_multilingual/docker_stdio_proxy.py" \
    "$DOCKER_PROXY_SOCKET" \
    >"$DOCKER_PROXY_DIR/proxy.out" \
    2>"$DOCKER_PROXY_DIR/proxy.err" &
  DOCKER_PROXY_PID="$!"
  for _ in $(seq 1 100); do
    if [ -S "$DOCKER_PROXY_SOCKET" ]; then
      break
    fi
    sleep 0.1
  done
  if [ ! -S "$DOCKER_PROXY_SOCKET" ]; then
    echo "Docker stdio proxy did not create socket: $DOCKER_PROXY_SOCKET" >&2
    exit 1
  fi
  export DOCKER_HOST="unix://$DOCKER_PROXY_SOCKET"
  "$PYTHON" - <<'PY'
import docker

client = docker.from_env()
print(f"Docker SDK proxy API version: {client.version().get('ApiVersion')}")
PY
fi
RUN_ARGS=(
  "$PYTHON" -m "$BENCHMARK_MODULE"
  --harness mini-swe-agent \
  --output "$OUTPUT_DIR" \
  --run-id "$RUN_NAME" \
  --instance-ids "$INSTANCE_IDS_PATH" \
  --generation-workers "$GENERATION_WORKERS" \
  --generation-step-limit "$GENERATION_STEP_LIMIT" \
  --eval-workers "$EVAL_WORKERS" \
  --model "$MODEL_NAME" \
  --litellm-model "openai/$MODEL_NAME" \
  --api-base "http://127.0.0.1:${PROXY_PORT}/v1" \
  --no-require-api-key \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --extra-body-json "$EXTRA_BODY"
)
if [ "$SKIP_EVALUATION" = "true" ]; then
  RUN_ARGS+=(--skip-evaluation)
fi
"${RUN_ARGS[@]}"
if [ -n "$TRIAL_COMPLETE" ]; then
  {
    echo "run_name=$RUN_NAME"
    echo "output_dir=$OUTPUT_DIR"
    echo "job_id=${SLURM_JOB_ID:-manual}"
    echo "host=$(hostname)"
    echo "completed_at=$(date -Is)"
  } >"$TRIAL_COMPLETE.tmp"
  mv "$TRIAL_COMPLETE.tmp" "$TRIAL_COMPLETE"
fi
