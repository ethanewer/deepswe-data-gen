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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-${SLURM_NNODES:?SLURM_NNODES is required}}"
export NODE_RANK="${NODE_RANK:-${SLURM_NODEID:?SLURM_NODEID is required}}"
export FSDP_CONFIG="${FSDP_CONFIG:-$ROOT_DIR/configs/qwen3_swift_fsdp_65k_memory_first.json}"

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
  --packing true \
  --packing_length "$PACKING_LENGTH" \
  --max_length "$PACKING_LENGTH" \
  --truncation_strategy right \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 4 \
  --save_only_model true \
  --logging_steps 1 \
  --max_steps "$MAX_STEPS" \
  --dataloader_num_workers 8 \
  --dataset_num_proc 8 \
  --output_dir "$OUTPUT_DIR" \
  --use_liger_kernel true \
  --attn_impl flash_attn
)

# FSDP fully-sharded data parallelism, no tensor parallelism. Every rank is a
# data-parallel replica whose parameters/gradients/optimizer state are sharded
# across ranks (ZeRO-3 equivalent). This recipe is intentionally DP-only: there
# is no DeepSpeed or tensor-parallel code path.
distributed_args=(
  --fsdp "full_shard auto_wrap" \
  --fsdp_config "$FSDP_CONFIG" \
  --no_gradient_checkpointing
)

echo "parallelism=fsdp full_shard (data-parallel, no tensor parallelism)"
echo "fsdp_config=$FSDP_CONFIG"

swift sft "${common_args[@]}" "${distributed_args[@]}"
