#!/usr/bin/env bash
#SBATCH --job-name=serve-qwen36-moe-vllm
#SBATCH --partition=l40s-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:l40s:8
#SBATCH --mem=1200G
#SBATCH --time=12:00:00
#SBATCH --output=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/logs/slurm-qwen36-moe-vllm-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/logs/slurm-qwen36-moe-vllm-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen}"
SHARED_ROOT="${SHARED_ROOT:-/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving}"

export LOCAL_MODEL_SERVING_ROOT="${LOCAL_MODEL_SERVING_ROOT:-/scratch/${USER}/local_model_serving/qwen36-moe-fp8-vllm-${SLURM_JOB_ID}}"
export VLLM_VENV="${VLLM_VENV:-$SHARED_ROOT/venv-vllm}"
export QWEN36_MOE_FP8_MODEL_PATH="${QWEN36_MOE_FP8_MODEL_PATH:-$SHARED_ROOT/models/Qwen_Qwen3.6-35B-A3B-FP8.snapshot}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-20011}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-35b-a3b-fp8}"

export TP_SIZE="${TP_SIZE:-8}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1048576}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

mkdir -p "$LOCAL_MODEL_SERVING_ROOT" "$SHARED_ROOT/logs"

cd "$REPO_DIR"

echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname -f 2>/dev/null || hostname)"
echo "base_url=http://$(hostname -f 2>/dev/null || hostname):$PORT/v1"
echo "local_root=$LOCAL_MODEL_SERVING_ROOT"
echo "model_path=$QWEN36_MOE_FP8_MODEL_PATH"
echo "venv=$VLLM_VENV"
nvidia-smi

exec "$REPO_DIR/scripts/local_model_serving/serve_qwen36_moe_fp8_vllm.sh"
