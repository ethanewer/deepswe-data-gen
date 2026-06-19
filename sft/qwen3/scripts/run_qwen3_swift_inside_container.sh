#!/usr/bin/env bash
set -euo pipefail

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_HOME="${HF_HOME:-/wbl-fast/usrs/ee/code-swe-data/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
# Resolve Hub model ids from HuggingFace (the eewer/* bases live there and the
# caches above are HF). Without this the ModelScope image defaults to the
# ModelScope hub and 404s on a HF-only repo id. Local checkpoint paths (e.g. the
# 4B recipe) bypass hub resolution entirely, so this is a no-op for them.
export USE_HF="${USE_HF:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-${SLURM_NNODES:?SLURM_NNODES is required}}"
export NODE_RANK="${NODE_RANK:-${SLURM_NODEID:?SLURM_NODEID is required}}"
export FSDP_CONFIG="${FSDP_CONFIG:-$ROOT_DIR/configs/qwen3_swift_fsdp_65k_memory_first.json}"
# Sharding/parallelism knobs. Defaults reproduce the original H200 recipe exactly
# (ZeRO-3 full_shard, data-parallel, no sequence parallelism). The L40S launchers
# override FSDP_SHARDING to ZeRO-2 (shard_grad_op) to avoid the per-layer parameter
# all-gather that is ruinous on PCIe-only L40S boxes (no NVLink).
export FSDP_SHARDING="${FSDP_SHARDING:-full_shard auto_wrap}"
export SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-1}"
export PACKING="${PACKING:-true}"
# Training duration: prefer epochs. If NUM_EPOCHS is set, train that many epochs
# (max_steps disabled); otherwise fall back to a fixed MAX_STEPS. Epoch-based is the
# default mode for recipes (1 epoch typical, 2 as a targeted choice).
export NUM_EPOCHS="${NUM_EPOCHS:-}"

echo "node_rank=$NODE_RANK host=$(hostname -f 2>/dev/null || hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | sed -n "1,8p"

common_args=(
  --model "$MODEL" \
  --model_type qwen3 \
  --template qwen3 \
  --train_type full \
  --dataset "$TRAIN_DATASET" \
  --load_from_cache_file false \
  --do_train true \
  --split_dataset_ratio 0 \
  --eval_strategy no \
  --do_eval false \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --per_device_eval_batch_size 1 \
  --learning_rate "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio "$WARMUP_RATIO" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --packing "$PACKING" \
  --packing_length "$PACKING_LENGTH" \
  --max_length "$PACKING_LENGTH" \
  --truncation_strategy right \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 4 \
  --save_only_model true \
  --logging_steps 1 \
  --dataloader_num_workers 8 \
  --dataset_num_proc 8 \
  --output_dir "$OUTPUT_DIR" \
  --use_liger_kernel true \
  --attn_impl flash_attn
)

# FSDP fully-sharded data parallelism, no tensor parallelism. Every rank is a
# data-parallel replica whose parameters/gradients/optimizer state are sharded
# across ranks. With FSDP_SHARDING="full_shard auto_wrap" this is ZeRO-3 (the
# H200 default); with "shard_grad_op auto_wrap" it is ZeRO-2 (params resident,
# only gradients + optimizer state sharded), which removes the per-layer param
# all-gather and is much faster on PCIe-only nodes. This recipe is intentionally
# DP-only: there is no tensor-parallel code path. SEQUENCE_PARALLEL_SIZE>1 adds
# DeepSpeed-Ulysses sequence parallelism (composes with FSDP).
distributed_args=(
  --fsdp "$FSDP_SHARDING" \
  --fsdp_config "$FSDP_CONFIG" \
  --no_gradient_checkpointing
)
if [ "${SEQUENCE_PARALLEL_SIZE:-1}" -gt 1 ]; then
  distributed_args+=(--sequence_parallel_size "$SEQUENCE_PARALLEL_SIZE")
fi

# Training duration: epochs if NUM_EPOCHS set (recipe default), else fixed steps.
if [ -n "${NUM_EPOCHS:-}" ]; then
  duration_args=(--num_train_epochs "$NUM_EPOCHS" --max_steps -1)
  duration_desc="num_train_epochs=$NUM_EPOCHS"
else
  duration_args=(--max_steps "$MAX_STEPS")
  duration_desc="max_steps=$MAX_STEPS"
fi

echo "parallelism=fsdp '$FSDP_SHARDING' (data-parallel, no tensor parallelism), sequence_parallel_size=$SEQUENCE_PARALLEL_SIZE"
echo "packing=$PACKING packing_length=$PACKING_LENGTH"
echo "fsdp_config=$FSDP_CONFIG"
echo "duration: $duration_desc"

swift sft "${common_args[@]}" "${distributed_args[@]}" "${duration_args[@]}"
