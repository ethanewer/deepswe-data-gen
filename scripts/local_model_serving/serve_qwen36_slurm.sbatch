#!/usr/bin/env bash
#SBATCH --job-name=serve-qwen3.6-27b
#SBATCH --partition=l40s-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:l40s:8
#SBATCH --mem=1200G
#SBATCH --time=12:00:00
#SBATCH --output=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/logs/slurm-qwen36-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/logs/slurm-qwen36-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen}"
SHARED_ROOT="${SHARED_ROOT:-/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving}"

export LOCAL_MODEL_SERVING_ROOT="${LOCAL_MODEL_SERVING_ROOT:-/scratch/${USER}/local_model_serving/qwen36-${SLURM_JOB_ID}}"
export SGLANG_VENV="${SGLANG_VENV:-$SHARED_ROOT/venv-sglang}"
export QWEN36_MODEL_PATH="${QWEN36_MODEL_PATH:-$SHARED_ROOT/models/Qwen_Qwen3.6-27B.snapshot}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-20000}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b}"

# TP8 uses all 8 L40S GPUs and keeps enough KV/linear-attention cache for
# parallel 131K-token sequences. DP attention is intentionally off by default:
# TP8/DP2 accepted the config but crashed on first generation in FlashInfer
# paged prefill for Qwen3.6's hybrid attention path.
export TP_SIZE="${TP_SIZE:-8}"
export DP_SIZE="${DP_SIZE:-1}"
export ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-0}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-800000}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-65536}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
export LINEAR_ATTN_BACKEND="${LINEAR_ATTN_BACKEND:-triton}"
export DECODE_LOG_INTERVAL="${DECODE_LOG_INTERVAL:-1}"
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}"
export CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 2 4 8 16}"
if [[ -z "${MODEL_LOADER_EXTRA_CONFIG:-}" ]]; then
  export MODEL_LOADER_EXTRA_CONFIG='{"enable_multithread_load":"true","num_threads":64}'
fi

mkdir -p "$LOCAL_MODEL_SERVING_ROOT" "$SHARED_ROOT/logs"

cd "$REPO_DIR"

echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname -f 2>/dev/null || hostname)"
echo "base_url=http://$(hostname -f 2>/dev/null || hostname):$PORT/v1"
echo "local_root=$LOCAL_MODEL_SERVING_ROOT"
echo "model_path=$QWEN36_MODEL_PATH"
echo "venv=$SGLANG_VENV"
nvidia-smi

exec "$REPO_DIR/scripts/local_model_serving/serve_qwen36_sglang.sh"
