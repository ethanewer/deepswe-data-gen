#!/usr/bin/env bash
#SBATCH --job-name=serve-mimo-v2.5
#SBATCH --partition=rtx6000pro-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:rtxproserver6000:4
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --output=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/logs/slurm-mimo-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/logs/slurm-mimo-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen}"
SHARED_ROOT="${SHARED_ROOT:-/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving}"

export LOCAL_MODEL_SERVING_ROOT="${LOCAL_MODEL_SERVING_ROOT:-/scratch/${USER}/local_model_serving/mimo-${SLURM_JOB_ID}}"
export SGLANG_VENV="${SGLANG_VENV:-$SHARED_ROOT/venv-sglang}"
export MIMO_MODEL_PATH="${MIMO_MODEL_PATH:-$SHARED_ROOT/models/XiaomiMiMo_MiMo-V2.5.snapshot}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-19001}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mimo-v2.5}"

# Free Slurm capacity is often 4x RTX PRO 6000. MiMo requires effective
# attention TP size 4, so TP4/DP1 is the target layout for this script.
export TP_SIZE="${TP_SIZE:-4}"
export DP_SIZE="${DP_SIZE:-1}"
export ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-0}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
export DECODE_LOG_INTERVAL="${DECODE_LOG_INTERVAL:-1}"
if [[ -z "${MODEL_LOADER_EXTRA_CONFIG:-}" ]]; then
  export MODEL_LOADER_EXTRA_CONFIG='{"enable_multithread_load":"true","num_threads":64}'
fi

mkdir -p "$LOCAL_MODEL_SERVING_ROOT" "$SHARED_ROOT/logs"

cd "$REPO_DIR"

echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname -f 2>/dev/null || hostname)"
echo "base_url=http://$(hostname -f 2>/dev/null || hostname):$PORT/v1"
echo "local_root=$LOCAL_MODEL_SERVING_ROOT"
echo "model_path=$MIMO_MODEL_PATH"
echo "venv=$SGLANG_VENV"
nvidia-smi

exec "$REPO_DIR/scripts/local_model_serving/serve_mimo_sglang.sh"
