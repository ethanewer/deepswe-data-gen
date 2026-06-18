#!/usr/bin/env bash
#SBATCH --job-name=q3-8b-swift
#SBATCH --partition=h200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=1800G
#SBATCH --time=06:00:00
#SBATCH --output=/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3/logs/slurm-qwen3-8b-swift-2node-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3/logs/slurm-qwen3-8b-swift-2node-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3}"
cd "$ROOT_DIR"
mkdir -p logs checkpoints

if [ "${SLURM_NNODES:-0}" -ne 2 ]; then
  echo "This SWIFT sweep recipe expects exactly 2 H200 nodes; got SLURM_NNODES=${SLURM_NNODES:-unset}" >&2
  exit 1
fi

DOCKER_IMAGE="${DOCKER_IMAGE:-modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1}"
MODEL="${MODEL:-$ROOT_DIR/data/qwen3_vl_8b_text_as_qwen3_checkpoint}"
TRAIN_DATASET="${TRAIN_DATASET:-$ROOT_DIR/data/qwen3_v75_swift_messages/train.jsonl}"
LR="${LR:-1e-5}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-50}"
PACKING_LENGTH="${PACKING_LENGTH:-65536}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
FSDP_CONFIG="${FSDP_CONFIG:-$ROOT_DIR/configs/qwen3_swift_fsdp_65k_memory_first.json}"
RUN_NAME="${RUN_NAME:-qwen3_8b_swift_lr${LR}_s${MAX_STEPS}_2node_h200_${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/checkpoints/$RUN_NAME}"

WORLD_SIZE=$((SLURM_NNODES * 8))
# DP-only (FSDP full_shard, no tensor parallelism): every rank is a data-parallel replica.
DP_SIZE=$WORLD_SIZE
GLOBAL_PACKS=$((DP_SIZE * PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS))
GLOBAL_TOKENS=$((GLOBAL_PACKS * PACKING_LENGTH))
if [ "$GLOBAL_TOKENS" -ne 1048576 ]; then
  echo "Expected 1,048,576 tokens/update, got $GLOBAL_TOKENS" >&2
  exit 1
fi
if [ ! -f "$TRAIN_DATASET" ]; then
  echo "Missing TRAIN_DATASET=$TRAIN_DATASET. Run materialize_swift_messages_dataset.py first." >&2
  exit 1
fi
if [ ! -f "$MODEL/model.safetensors.index.json" ]; then
  echo "Missing model index under MODEL=$MODEL" >&2
  exit 1
fi

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((30000 + SLURM_JOB_ID % 10000))}"
export MASTER_ADDR MASTER_PORT

echo "job_id=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NODELIST"
echo "container_backend=docker"
echo "docker_image=$DOCKER_IMAGE"
echo "master=$MASTER_ADDR:$MASTER_PORT"
echo "model=$MODEL"
echo "train_dataset=$TRAIN_DATASET"
echo "output_dir=$OUTPUT_DIR"
echo "lr=$LR max_steps=$MAX_STEPS save_steps=$SAVE_STEPS"
echo "packing_length=$PACKING_LENGTH world_size=$WORLD_SIZE dp_size=$DP_SIZE global_packs=$GLOBAL_PACKS global_tokens=$GLOBAL_TOKENS"
echo "parallelism=fsdp full_shard (data-parallel, no tensor parallelism)"
echo "fsdp_config=$FSDP_CONFIG"

export ROOT_DIR DOCKER_IMAGE MODEL TRAIN_DATASET LR MAX_STEPS SAVE_STEPS PACKING_LENGTH
export PER_DEVICE_BATCH_SIZE GRAD_ACCUM_STEPS WARMUP_RATIO WEIGHT_DECAY OUTPUT_DIR
export MASTER_ADDR MASTER_PORT
export FSDP_CONFIG

srun --ntasks="$SLURM_NNODES" \
  --ntasks-per-node=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  bash -lc '
set -euo pipefail
exec docker run --rm \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /wbl-fast:/wbl-fast \
  -v /home/ewer:/home/ewer \
  -v /scratch:/scratch \
  -w "$ROOT_DIR" \
  -e ROOT_DIR \
  -e MODEL \
  -e TRAIN_DATASET \
  -e LR \
  -e MAX_STEPS \
  -e SAVE_STEPS \
  -e PACKING_LENGTH \
  -e PER_DEVICE_BATCH_SIZE \
  -e GRAD_ACCUM_STEPS \
  -e WARMUP_RATIO \
  -e WEIGHT_DECAY \
  -e OUTPUT_DIR \
  -e FSDP_CONFIG \
  -e MASTER_ADDR \
  -e MASTER_PORT \
  -e SLURM_NNODES \
  -e SLURM_NODEID \
  "$DOCKER_IMAGE" \
  bash scripts/run_qwen3_swift_inside_container.sh
'
