#!/usr/bin/env bash
#SBATCH --job-name=q3-4b-l40s
#SBATCH --partition=l40s-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:l40s:8
#SBATCH --mem=1400G
#SBATCH --time=10:00:00
#SBATCH --output=/wbl-fast/usrs/ee/clean-20260619/deepswe-data-gen/sft/qwen3/logs/slurm-q3-4b-l40s-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/clean-20260619/deepswe-data-gen/sft/qwen3/logs/slurm-q3-4b-l40s-%j.err

set -euo pipefail

# Single-node 8xL40S wrapper around run_qwen3_4b_swift_local_l40s.sh. SLURM grabs a
# dedicated l40s-8gpu node; the launcher then does `docker run --gpus all` on it.
export ROOT_DIR="${ROOT_DIR:-/wbl-fast/usrs/ee/clean-20260619/deepswe-data-gen/sft/qwen3}"
export LR="${LR:-1e-5}"
export MAX_STEPS="${MAX_STEPS:-250}"
export SAVE_STEPS="${SAVE_STEPS:-50}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_v75_l40s_lr${LR}_s${MAX_STEPS}_${SLURM_JOB_ID:-manual}}"
export CONTAINER_NAME="${CONTAINER_NAME:-train_4b_l40s_${SLURM_JOB_ID:-manual}}"

cd "$ROOT_DIR"
echo "job_id=${SLURM_JOB_ID:-none} node=$(hostname) run_name=$RUN_NAME"
echo "lr=$LR max_steps=$MAX_STEPS save_steps=$SAVE_STEPS"
exec bash scripts/run_qwen3_4b_swift_local_l40s.sh
