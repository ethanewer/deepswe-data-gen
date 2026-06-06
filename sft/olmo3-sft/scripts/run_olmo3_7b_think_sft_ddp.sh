#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OLMO_SFT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

OLMO_CORE_DIR=${OLMO_CORE_DIR:-/wbl-fast/usrs/mk/OLMo-core}
TORCHRUN=${TORCHRUN:-"${OLMO_CORE_DIR}/.venv/bin/torchrun"}

export PYTHONPATH="${OLMO_CORE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OLMO_PROCESS_GROUP_TIMEOUT_MINUTES=${OLMO_PROCESS_GROUP_TIMEOUT_MINUTES:-30}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE=${#visible_gpus[@]}
fi

SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-65536}
LOCAL_BATCH_SEQS=${LOCAL_BATCH_SEQS:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
GLOBAL_BATCH_SEQS=${GLOBAL_BATCH_SEQS:-$((LOCAL_BATCH_SEQS * NPROC_PER_NODE * GRAD_ACCUM_STEPS))}
MAX_STEPS=${MAX_STEPS:-1000}
DATA_PARALLEL=${DATA_PARALLEL:-fsdp}
DATA_MODE=${DATA_MODE:-offline}
RUN_NAME=${RUN_NAME:-"olmo3_7b_think_sft_s${SEQUENCE_LENGTH}_lbs${LOCAL_BATCH_SEQS}_gbs${GLOBAL_BATCH_SEQS}_${DATA_PARALLEL}${NPROC_PER_NODE}"}

SAVE_FOLDER=${SAVE_FOLDER:-"${OLMO_SFT_DIR}/runs/${RUN_NAME}"}
WORK_DIR=${WORK_DIR:-"${OLMO_SFT_DIR}/work"}
METRICS_JSONL=${METRICS_JSONL:-"${OLMO_SFT_DIR}/results/${RUN_NAME}.jsonl"}
LOAD_PATH=${LOAD_PATH:-/wbl-fast/usrs/mk/data/checkpoints/Olmo-3-1025-7B-stage3-step11921/model_and_optim}
DATASET_DIR=${DATASET_DIR:-/wbl-fast/usrs/ee/code-swe-data/data/tokenized/code-swe-terminal-agentic-sft-olmo3-65k}
TOKEN_IDS_GLOB=${TOKEN_IDS_GLOB:-"${DATASET_DIR}/**/token_ids_part_*.npy"}
LABEL_MASK_GLOB=${LABEL_MASK_GLOB:-"${DATASET_DIR}/**/labels_mask_*.npy"}
RAW_ROOT=${RAW_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/code-swe-terminal-agentic-sft}
ONLINE_CACHE_DIR=${ONLINE_CACHE_DIR:-"${WORK_DIR}/online-cache/${RUN_NAME}"}
ONLINE_TOKENIZER=${ONLINE_TOKENIZER:-/wbl-fast/usrs/mk/data/tokenizers/Olmo-3-7B-Think-SFT}
ONLINE_PYTHON=${ONLINE_PYTHON:-/wbl-fast/usrs/mk/open-instruct/.venv/bin/python}
ONLINE_MIN_READY_BATCHES=${ONLINE_MIN_READY_BATCHES:-2}
ONLINE_CACHE_PART_INSTANCES=${ONLINE_CACHE_PART_INSTANCES:-128}
ONLINE_POLL_INTERVAL=${ONLINE_POLL_INTERVAL:-5}
ONLINE_MAX_ROWS_PER_FILE=${ONLINE_MAX_ROWS_PER_FILE:-0}
ONLINE_MAX_EXAMPLES=${ONLINE_MAX_EXAMPLES:-0}
ONLINE_TRUNCATE_OVERLENGTH=${ONLINE_TRUNCATE_OVERLENGTH:-false}
ONLINE_SKIP_ROOT=${ONLINE_SKIP_ROOT:-}
ONLINE_DOC_LENS=${ONLINE_DOC_LENS:-true}

ATTN_BACKEND=${ATTN_BACKEND:-flash_2}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-full}
PACKING_WORKERS=${PACKING_WORKERS:-0}
COMPILE_MODEL=${COMPILE_MODEL:-true}
COMPILE_OPTIM=${COMPILE_OPTIM:-false}
OPTIM=${OPTIM:-adamw}
ADAMW_FUSED=${ADAMW_FUSED:-true}
ADAMW_FOREACH=${ADAMW_FOREACH:-}
SKIP_LOAD=${SKIP_LOAD:-false}
REDUCE_DTYPE=${REDUCE_DTYPE:-float32}
FSDP_WRAPPING_STRATEGY=${FSDP_WRAPPING_STRATEGY:-full}
FSDP_PREFETCH_FACTOR=${FSDP_PREFETCH_FACTOR:-0}
DDP_BUCKET_CAP_MB=${DDP_BUCKET_CAP_MB:-100}
DDP_OPTIMIZE_MODE=${DDP_OPTIMIZE_MODE:-default}

mkdir -p "${OLMO_SFT_DIR}/runs" "${OLMO_SFT_DIR}/results" "${OLMO_SFT_DIR}/logs" "${WORK_DIR}"

args=(
  "${OLMO_SFT_DIR}/scripts/benchmark_olmo3_sft.py"
  --save-folder "${SAVE_FOLDER}"
  --work-dir "${WORK_DIR}"
  --metrics-jsonl "${METRICS_JSONL}"
  --sequence-length "${SEQUENCE_LENGTH}"
  --local-batch-seqs "${LOCAL_BATCH_SEQS}"
  --global-batch-seqs "${GLOBAL_BATCH_SEQS}"
  --max-steps "${MAX_STEPS}"
  --data-parallel "${DATA_PARALLEL}"
  --data-mode "${DATA_MODE}"
  --attn-backend "${ATTN_BACKEND}"
  --activation-checkpointing "${ACTIVATION_CHECKPOINTING}"
  --packing-workers "${PACKING_WORKERS}"
  --optim "${OPTIM}"
  --reduce-dtype "${REDUCE_DTYPE}"
  --fsdp-wrapping-strategy "${FSDP_WRAPPING_STRATEGY}"
  --fsdp-prefetch-factor "${FSDP_PREFETCH_FACTOR}"
  --token-ids-glob "${TOKEN_IDS_GLOB}"
  --label-mask-glob "${LABEL_MASK_GLOB}"
  --ddp-bucket-cap-mb "${DDP_BUCKET_CAP_MB}"
  --ddp-optimize-mode "${DDP_OPTIMIZE_MODE}"
)

if [[ "${DATA_MODE}" == "online" ]]; then
  args+=(
    --raw-root "${RAW_ROOT}"
    --online-cache-dir "${ONLINE_CACHE_DIR}"
    --online-tokenizer "${ONLINE_TOKENIZER}"
    --online-python "${ONLINE_PYTHON}"
    --online-min-ready-batches "${ONLINE_MIN_READY_BATCHES}"
    --online-cache-part-instances "${ONLINE_CACHE_PART_INSTANCES}"
    --online-poll-interval "${ONLINE_POLL_INTERVAL}"
    --online-max-rows-per-file "${ONLINE_MAX_ROWS_PER_FILE}"
    --online-max-examples "${ONLINE_MAX_EXAMPLES}"
  )
  if [[ "${ONLINE_TRUNCATE_OVERLENGTH}" == "true" ]]; then
    args+=(--online-truncate-overlength)
  fi
  if [[ "${ONLINE_DOC_LENS}" != "true" ]]; then
    args+=(--no-online-doc-lens)
  fi
  if [[ -n "${ONLINE_SKIP_ROOT}" ]]; then
    IFS=':' read -r -a online_skip_roots <<< "${ONLINE_SKIP_ROOT}"
    for skip_root in "${online_skip_roots[@]}"; do
      args+=(--online-skip-root "${skip_root}")
    done
  fi
fi

if [[ "${COMPILE_MODEL}" == "true" ]]; then
  args+=(--compile-model)
else
  args+=(--no-compile-model)
fi

if [[ "${COMPILE_OPTIM}" == "true" ]]; then
  args+=(--compile-optim)
else
  args+=(--no-compile-optim)
fi

if [[ "${ADAMW_FUSED}" == "true" ]]; then
  args+=(--adamw-fused)
else
  args+=(--no-adamw-fused)
fi

if [[ -n "${ADAMW_FOREACH}" ]]; then
  if [[ "${ADAMW_FOREACH}" == "true" ]]; then
    args+=(--adamw-foreach)
  else
    args+=(--no-adamw-foreach)
  fi
fi

if [[ "${SKIP_LOAD}" == "true" ]]; then
  args+=(--skip-load)
else
  args+=(--load-path "${LOAD_PATH}")
fi

exec "${TORCHRUN}" --standalone --nproc-per-node="${NPROC_PER_NODE}" "${args[@]}"
