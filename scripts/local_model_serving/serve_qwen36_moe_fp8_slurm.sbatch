#!/usr/bin/env bash
#SBATCH --job-name=serve-qwen36-moe-fp8
#SBATCH --partition=l40s-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:l40s:8
#SBATCH --mem=1200G
#SBATCH --time=12:00:00
#SBATCH --output=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/logs/slurm-qwen36-moe-fp8-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/logs/slurm-qwen36-moe-fp8-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen}"
SHARED_ROOT="${SHARED_ROOT:-/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving}"

export LOCAL_MODEL_SERVING_ROOT="${LOCAL_MODEL_SERVING_ROOT:-/scratch/${USER}/local_model_serving/qwen36-moe-fp8-${SLURM_JOB_ID}}"
export SGLANG_VENV="${SGLANG_VENV:-$SHARED_ROOT/venv-sglang}"
export QWEN36_MOE_FP8_MODEL_PATH="${QWEN36_MOE_FP8_MODEL_PATH:-$SHARED_ROOT/models/Qwen_Qwen3.6-35B-A3B-FP8.snapshot}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-20010}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-35b-a3b-fp8}"

# The official FP8 checkpoint uses 128-wide block quantization. TP8 splits one
# shared-expert projection to width 64 in SGLang, so the L40S default is TP4/DP2.
# DP attention remains opt-in because the verified Qwen3.6 hybrid-attention path
# crashed during FlashInfer paged prefill with DP attention enabled.
export TP_SIZE="${TP_SIZE:-4}"
export DP_SIZE="${DP_SIZE:-2}"
export ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-0}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-600000}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-65536}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
export LINEAR_ATTN_BACKEND="${LINEAR_ATTN_BACKEND:-triton}"
export MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-triton}"
export DECODE_LOG_INTERVAL="${DECODE_LOG_INTERVAL:-1}"
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
export CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 2 4 8 16 32}"
if [[ -z "${MODEL_LOADER_EXTRA_CONFIG:-}" ]]; then
  export MODEL_LOADER_EXTRA_CONFIG='{"enable_multithread_load":"true","num_threads":64}'
fi

mkdir -p "$LOCAL_MODEL_SERVING_ROOT" "$SHARED_ROOT/logs"

cd "$REPO_DIR"

echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname -f 2>/dev/null || hostname)"
echo "base_url=http://$(hostname -f 2>/dev/null || hostname):$PORT/v1"
echo "local_root=$LOCAL_MODEL_SERVING_ROOT"
echo "model_path=$QWEN36_MOE_FP8_MODEL_PATH"
echo "venv=$SGLANG_VENV"
nvidia-smi

exec "$REPO_DIR/scripts/local_model_serving/serve_qwen36_moe_fp8_sglang.sh"
