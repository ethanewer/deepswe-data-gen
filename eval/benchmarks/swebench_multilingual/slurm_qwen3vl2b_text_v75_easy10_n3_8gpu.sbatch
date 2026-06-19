#!/usr/bin/env bash
#SBATCH --job-name=eval-2b-v75-easy10-n3
#SBATCH --partition=l40s-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1200G
#SBATCH --time=08:00:00
#SBATCH --output=/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/logs/slurm-eval-2b-v75-easy10-n3-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/logs/slurm-eval-2b-v75-easy10-n3-%j.err

# Serve the local 2B v75 SFT checkpoint ONCE on 8 L40S GPUs (8 vLLM backends behind
# a round-robin proxy), then run N trials of the SWE-bench Multilingual easy-10
# subset through mini-swe-agent + official swebench Docker evaluation, sequentially
# against the warm server (per-instance images build once, reused across trials).
# Serving: qwen3_thinking_acc chat template + hermes tool parser + NO reasoning parser.
# The model emits its own <think>...</think>; with no reasoning parser it stays in the
# response `content`, so mini-swe-agent resends it and the template re-renders it into
# the chat history -- reasoning is retained every turn, matching how the model was trained.
# (The qwen3 reasoning parser broke tool-call parsing in vLLM 0.19.1; deepseek_r1 dropped it.)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen}"   # has .venv + harness + docker infra
cd "$REPO_ROOT"
mkdir -p logs runs/serving_configs runs/serving runs/swebench_ml

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || { echo "Missing eval python at $PYTHON" >&2; exit 1; }

CHECKPOINT_STEP_DIR="${CHECKPOINT_STEP_DIR:-/wbl-fast/usrs/ee/clean-20260619/deepswe-data-gen/sft/qwen3/checkpoints/qwen3_2b_v75_l40s_lr1e-5_s100/v0-20260620-025531/checkpoint-100}"
INSTANCE_IDS_PATH="${INSTANCE_IDS_PATH:-$REPO_ROOT/eval/benchmarks/swebench_multilingual/swebench_multilingual_easy_10_instance_ids.txt}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Thinking}"   # served name + base for chat template/tokenizer (cached)
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-v75-s100}"
TRIALS="${TRIALS:-3}"
EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"  # match the 64K training context; 40K overflowed long agentic trajectories
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.6}"
GENERATION_STEP_LIMIT="${GENERATION_STEP_LIMIT:-250}"
GENERATION_WORKERS="${GENERATION_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
EXTRA_BODY_JSON="${EXTRA_BODY_JSON:-{\"top_p\":0.95,\"top_k\":20,\"min_p\":0,\"presence_penalty\":0}}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_STEM="${RUN_STEM:-qwen3-2b-thinking-v75-${CHECKPOINT_LABEL}-40k-l40s-${EVAL_GPU_COUNT}gpu-swebench-ml-easy10}"

compgen -G "$CHECKPOINT_STEP_DIR/*.safetensors" >/dev/null || { echo "No *.safetensors in $CHECKPOINT_STEP_DIR" >&2; exit 1; }
[ -f "$INSTANCE_IDS_PATH" ] || { echo "Missing instance ids $INSTANCE_IDS_PATH" >&2; exit 1; }

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

echo "job_id=${SLURM_JOB_ID:-manual} node=$(hostname)"
echo "checkpoint=$CHECKPOINT_STEP_DIR"
echo "instance_ids=$INSTANCE_IDS_PATH  trials=$TRIALS  gpus=$EVAL_GPU_COUNT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

BASE_PORT=$((20000 + (${SLURM_JOB_ID:-0} % 20000)))
PROXY_PORT="$BASE_PORT"
BACKEND_PORTS=()
for offset in $(seq 1 "$EVAL_GPU_COUNT"); do BACKEND_PORTS+=("$((BASE_PORT + 100 + offset))"); done

RUN_NAME="${RUN_STEM}-${RUN_SUFFIX}"
CONFIG_PATH="$REPO_ROOT/runs/serving_configs/${RUN_NAME}.json"
SERVE_DIR="$REPO_ROOT/runs/serving/$RUN_NAME"
SERVE_CACHE_DIR="/tmp/q3eval-${SLURM_JOB_ID:-manual}-2b"

# Serve config: local checkpoint weights, hermes tool parser, NO reasoning parser (reasoning kept in content).
"$PYTHON" - "$CONFIG_PATH" "$CHECKPOINT_STEP_DIR" "$MODEL_NAME" "$PROXY_PORT" "$MAX_MODEL_LEN" "$SERVE_CACHE_DIR" "${BACKEND_PORTS[@]}" <<'PY'
import json, sys, os
config_path, model_dir, model_name, proxy_port, max_model_len, serve_cache_dir, *backend_ports = sys.argv[1:]
# REASONING_PARSER is empty by default (no reasoning parser -> <think> stays in content,
# retained in history). Set it (e.g. deepseek_r1) only for controlled comparisons.
_rp = os.environ.get("REASONING_PARSER", "").strip()
_vargs = (["--reasoning-parser", _rp] if _rp else []) + ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
gpu_count = len(backend_ports)
payload = {
    "model": model_dir,
    "description": f"Qwen3-VL-2B-thinking-text v75 SFT checkpoint on {gpu_count} L40S GPUs.",
    "sources": [model_name],
    "serve": {
        "gpus": list(range(gpu_count)),
        "map_gpus_from_cuda_visible_devices": True,
        "backend_ports": [int(p) for p in backend_ports],
        "backend_host": "0.0.0.0",
        "backend_base_url_host": "127.0.0.1",
        "proxy_host": "0.0.0.0",
        "proxy_port": int(proxy_port),
        "served_model_name": model_name,
        "tensor_parallel_size": 1,
        "max_model_len": int(max_model_len),
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
        # hermes tool parser matches the trained Hermes <tool_call>{json} format. NO reasoning
        # parser on purpose: the model emits its own <think>...</think>, and with no parser it
        # stays in `content`, so mini-swe-agent resends it and the qwen3_thinking_acc template
        # re-renders it into history -- reasoning is retained in every turn, matching training.
        "vllm_args": _vargs,
    },
}
open(config_path, "w").write(json.dumps(payload, indent=2) + "\n")
print(config_path)
PY

DOCKER_PROXY_PID=""; DOCKER_PROXY_DIR=""
cleanup() {
  if [ -f "$SERVE_DIR/manifest.json" ]; then
    "$PYTHON" - "$SERVE_DIR/manifest.json" <<'PY' || true
import json, os, signal, sys, time
m = json.load(open(sys.argv[1])); pids = [int(p["pid"]) for p in m.get("processes", []) if p.get("pid")]
for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid in pids:
        try: os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError): pass
    time.sleep(5 if sig == signal.SIGTERM else 1)
PY
  fi
  [ -n "$DOCKER_PROXY_PID" ] && { kill "$DOCKER_PROXY_PID" 2>/dev/null || true; }
  [ -n "$DOCKER_PROXY_DIR" ] && rm -rf "$DOCKER_PROXY_DIR"
}
trap cleanup EXIT

"$PYTHON" -m eval.model.serve_from_config --config "$CONFIG_PATH" --run-dir "$SERVE_DIR" --background --health-timeout 1800

# Wait for proxy health.
"$PYTHON" - "$PROXY_PORT" <<'PY'
import sys, time, urllib.request
port = sys.argv[1]; deadline = time.time() + 300; err = ""
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            if r.status == 200: raise SystemExit(0)
    except SystemExit: raise
    except Exception as e: err = str(e)
    time.sleep(3)
raise SystemExit(f"proxy health failed: {err}")
PY
echo "proxy healthy on :$PROXY_PORT"

# Docker stdio proxy (official swebench eval talks to docker through this).
DOCKER_PROXY_DIR="$(mktemp -d "/tmp/swebench-docker-proxy.${SLURM_JOB_ID:-manual}.XXXXXX")"
DOCKER_PROXY_SOCKET="$DOCKER_PROXY_DIR/docker.sock"
"$PYTHON" "$REPO_ROOT/eval/benchmarks/swebench_multilingual/docker_stdio_proxy.py" "$DOCKER_PROXY_SOCKET" \
  >"$DOCKER_PROXY_DIR/proxy.out" 2>"$DOCKER_PROXY_DIR/proxy.err" &
DOCKER_PROXY_PID="$!"
for _ in $(seq 1 100); do [ -S "$DOCKER_PROXY_SOCKET" ] && break; sleep 0.1; done
[ -S "$DOCKER_PROXY_SOCKET" ] || { echo "docker proxy socket missing" >&2; exit 1; }
export DOCKER_HOST="unix://$DOCKER_PROXY_SOCKET"

overall=0
for trial in $(seq 1 "$TRIALS"); do
  OUT="$REPO_ROOT/runs/swebench_ml/${RUN_NAME}-t${trial}"
  echo "===== easy-10 trial ${trial}/${TRIALS} -> $OUT ====="
  set +e
  "$PYTHON" -m eval.benchmarks.swebench_multilingual.run \
    --harness mini-swe-agent \
    --output "$OUT" \
    --run-id "${RUN_NAME}-t${trial}" \
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
    --extra-body-json "$EXTRA_BODY_JSON"
  st=$?
  set -e
  echo "trial ${trial} exit=${st}"
  [ "$st" -ne 0 ] && overall="$st"
done
echo "all trials done; overall_exit=$overall"
exit "$overall"
