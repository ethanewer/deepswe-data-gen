#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3}"
cd "$ROOT_DIR"
mkdir -p logs checkpoints

DOCKER_IMAGE="${DOCKER_IMAGE:-modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1}"
MODEL="${MODEL:-/wbl-fast/usrs/ee/code-swe-data/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea}"
TRAIN_DATASET="${TRAIN_DATASET:-$ROOT_DIR/data/qwen3_v75_swift_messages/train.jsonl}"
LR="${LR:-1e-5}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-50}"
PACKING_LENGTH="${PACKING_LENGTH:-65536}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
FSDP_CONFIG="${FSDP_CONFIG:-$ROOT_DIR/configs/qwen3_swift_fsdp_65k_memory_first.json}"
# L40S is PCIe-only (no NVLink). ZeRO-2 (shard_grad_op) keeps parameters resident
# on every rank and shards only gradients + optimizer state, removing the ZeRO-3
# per-layer parameter all-gather; it measured ~8% faster than full_shard ZeRO-3
# (~13.1k vs ~11.9k TPS). The cost is memory: a 4B model in bf16 (~8 GiB) stays
# resident, so the peak is ~43.7 GiB on the 46 GiB card -- only ~2 GiB free. That
# fits and is deterministic (every pack is exactly 64K tokens), but if you hit an
# OOM (other GPU users, a longer PACKING_LENGTH, a bigger model) fall back to the
# higher-headroom ZeRO-3 (~38.9 GiB peak) with FSDP_SHARDING="full_shard auto_wrap".
FSDP_SHARDING="${FSDP_SHARDING:-shard_grad_op auto_wrap}"
SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((30000 + ($$ % 10000)))}"

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_v75_msswift_dp8_lr${LR}_s${MAX_STEPS}_local_l40s_${RUN_STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/checkpoints/$RUN_NAME}"
CONTAINER_NAME="${CONTAINER_NAME:-swift_${RUN_NAME}}"

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if [ "$NNODES" -ne 1 ]; then
  echo "This local wrapper expects NNODES=1; got NNODES=$NNODES" >&2
  exit 1
fi
if [ "$NODE_RANK" -ne 0 ]; then
  echo "This local wrapper expects NODE_RANK=0; got NODE_RANK=$NODE_RANK" >&2
  exit 1
fi
if [ "$NPROC_PER_NODE" -ne 8 ]; then
  echo "This local L40S recipe expects 8 GPUs; got NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  exit 1
fi
# Sequence parallelism forms groups of SEQUENCE_PARALLEL_SIZE ranks that jointly
# process one packed sequence, so the data-parallel width shrinks accordingly.
if [ $((WORLD_SIZE % SEQUENCE_PARALLEL_SIZE)) -ne 0 ]; then
  echo "WORLD_SIZE=$WORLD_SIZE not divisible by SEQUENCE_PARALLEL_SIZE=$SEQUENCE_PARALLEL_SIZE" >&2
  exit 1
fi
DP_SIZE=$((WORLD_SIZE / SEQUENCE_PARALLEL_SIZE))
GLOBAL_PACKS=$((DP_SIZE * PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS))
GLOBAL_TOKENS=$((GLOBAL_PACKS * PACKING_LENGTH))
# Keep the H200 recipe's global batch: 16 packed sequences / 1,048,576 tokens per update.
if [ "$GLOBAL_PACKS" -ne 16 ]; then
  echo "Expected 16 packed sequences/update, got $GLOBAL_PACKS" >&2
  exit 1
fi
if [ "$GLOBAL_TOKENS" -ne 1048576 ]; then
  echo "Expected 1,048,576 tokens/update, got $GLOBAL_TOKENS" >&2
  exit 1
fi
if [ ! -f "$TRAIN_DATASET" ]; then
  echo "Missing TRAIN_DATASET=$TRAIN_DATASET. Run materialize_swift_messages_dataset.py first." >&2
  exit 1
fi
# MODEL may be a local checkpoint dir or a Hub repo id. Only validate local absolute paths.
if [[ "$MODEL" == /* ]] && [ ! -f "$MODEL/model.safetensors.index.json" ]; then
  echo "Missing model index under local MODEL=$MODEL" >&2
  exit 1
fi
if [ ! -f "$FSDP_CONFIG" ]; then
  echo "Missing FSDP_CONFIG=$FSDP_CONFIG" >&2
  exit 1
fi

echo "container_backend=docker"
echo "docker_image=$DOCKER_IMAGE"
echo "container_name=$CONTAINER_NAME"
echo "host=$(hostname -f 2>/dev/null || hostname)"
echo "master=$MASTER_ADDR:$MASTER_PORT"
echo "model=$MODEL"
echo "train_dataset=$TRAIN_DATASET"
echo "output_dir=$OUTPUT_DIR"
echo "lr=$LR max_steps=$MAX_STEPS save_steps=$SAVE_STEPS"
echo "packing_length=$PACKING_LENGTH world_size=$WORLD_SIZE dp_size=$DP_SIZE global_packs=$GLOBAL_PACKS global_tokens=$GLOBAL_TOKENS"
echo "parallelism=fsdp '$FSDP_SHARDING' (data-parallel, no tensor parallelism), sequence_parallel_size=$SEQUENCE_PARALLEL_SIZE"
echo "fsdp_config=$FSDP_CONFIG"

export ROOT_DIR MODEL TRAIN_DATASET LR MAX_STEPS SAVE_STEPS PACKING_LENGTH
export PER_DEVICE_BATCH_SIZE GRAD_ACCUM_STEPS WARMUP_RATIO WEIGHT_DECAY OUTPUT_DIR
export MASTER_ADDR MASTER_PORT
export FSDP_CONFIG FSDP_SHARDING SEQUENCE_PARALLEL_SIZE
export NPROC_PER_NODE NNODES NODE_RANK

exec docker run --rm \
  --name "$CONTAINER_NAME" \
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
  -e FSDP_SHARDING \
  -e SEQUENCE_PARALLEL_SIZE \
  -e MASTER_ADDR \
  -e MASTER_PORT \
  -e NPROC_PER_NODE \
  -e NNODES \
  -e NODE_RANK \
  "$DOCKER_IMAGE" \
  bash scripts/run_qwen3_swift_inside_container.sh
