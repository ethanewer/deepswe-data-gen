#!/usr/bin/env bash
#SBATCH --job-name=serve-kimi-k2.7-code
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=1800G
#SBATCH --time=12:00:00
#SBATCH --output=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/logs/slurm-kimi-k2.7-code-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/logs/slurm-kimi-k2.7-code-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen}"
SHARED_ROOT="${SHARED_ROOT:-/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving}"

export LOCAL_MODEL_SERVING_ROOT="${LOCAL_MODEL_SERVING_ROOT:-/scratch/${USER}/local_model_serving/kimi-k2.7-code-${SLURM_JOB_ID}}"
export VLLM_VENV="${VLLM_VENV:-$SHARED_ROOT/venv-vllm}"
export SGLANG_VENV="${SGLANG_VENV:-$SHARED_ROOT/venv-sglang}"
export KIMI27_CODE_MODEL_PATH="${KIMI27_CODE_MODEL_PATH:-$SHARED_ROOT/models/moonshotai_Kimi-K2.7-Code.snapshot}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-18010}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.7-code}"
export TP_SIZE="${TP_SIZE:-8}"
export KIMI27_CODE_BACKEND="${KIMI27_CODE_BACKEND:-vllm}"
export SERVING_BACKEND="$KIMI27_CODE_BACKEND"

mkdir -p "$LOCAL_MODEL_SERVING_ROOT" "$SHARED_ROOT/logs"

cd "$REPO_DIR"

echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname -f 2>/dev/null || hostname)"
echo "base_url=http://$(hostname -f 2>/dev/null || hostname):$PORT/v1"
echo "local_root=$LOCAL_MODEL_SERVING_ROOT"
echo "model_path=$KIMI27_CODE_MODEL_PATH"
echo "backend=$KIMI27_CODE_BACKEND"
echo "vllm_venv=$VLLM_VENV"
echo "sglang_venv=$SGLANG_VENV"
nvidia-smi

case "$KIMI27_CODE_BACKEND" in
  vllm)
    exec "$REPO_DIR/scripts/local_model_serving/serve_kimi27_code_vllm.sh"
    ;;
  sglang)
    exec "$REPO_DIR/scripts/local_model_serving/serve_kimi27_code_sglang.sh"
    ;;
  *)
    echo "unsupported KIMI27_CODE_BACKEND=$KIMI27_CODE_BACKEND" >&2
    exit 2
    ;;
esac
