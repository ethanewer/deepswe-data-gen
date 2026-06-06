#!/usr/bin/env bash
#SBATCH --job-name=code-swe-tok
#SBATCH --partition=m7i-cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=24:00:00
#SBATCH --output=/wbl-fast/usrs/ee/code-swe-data/sft/olmo3-sft/logs/tokenize_workers/slurm/%x_%A_%a.log

set -u

OUTPUT_ROOT="${OUTPUT_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/tokenized/code-swe-terminal-agentic-sft-olmo3-65k-smallest-first}"
EXCLUDE_ROOT="${EXCLUDE_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/tokenized/code-swe-terminal-agentic-sft-olmo3-65k}"
SCRIPT="${SCRIPT:-/wbl-fast/usrs/ee/code-swe-data/sft/olmo3-sft/scripts/prepare_code_swe_sft_data.py}"
PYTHON="${PYTHON:-/wbl-fast/usrs/mk/open-instruct/.venv/bin/python}"
QUEUE="${QUEUE:-/wbl-fast/usrs/ee/code-swe-data/sft/olmo3-sft/logs/tokenize_workers/queue.20260604T152827Z.txt}"
LOCK="${QUEUE}.lock"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
FILTER_CHUNKSIZE="${FILTER_CHUNKSIZE:-64}"
TASK_TAG="${SLURM_ARRAY_JOB_ID:-manual}.${SLURM_ARRAY_TASK_ID:-0}"
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-${OUTPUT_ROOT}/.job-hf-cache/${TASK_TAG}}"

cleanup_local_cache() {
    if [ -n "${LOCAL_CACHE_ROOT:-}" ] && [ -d "${LOCAL_CACHE_ROOT}" ]; then
        rm -rf "${LOCAL_CACHE_ROOT}"
    fi
}
trap cleanup_local_cache EXIT

mkdir -p "$(dirname "${QUEUE}")" \
    /wbl-fast/usrs/ee/code-swe-data/sft/olmo3-sft/logs/tokenize_workers/slurm \
    "${LOCAL_CACHE_ROOT}/datasets" \
    "${LOCAL_CACHE_ROOT}/hf_home" \
    "${LOCAL_CACHE_ROOT}/xdg"

echo "[slurm ${TASK_TAG}] host=$(hostname) cpus=${CPUS} queue=${QUEUE}"
echo "[slurm ${TASK_TAG}] output_root=${OUTPUT_ROOT}"
echo "[slurm ${TASK_TAG}] hf_datasets_cache=${LOCAL_CACHE_ROOT}/datasets"

pop_shard() {
    flock "${LOCK}" bash -c '
        queue="$1"
        first="$(head -n 1 "$queue" 2>/dev/null || true)"
        [ -n "$first" ] || exit 1
        tail -n +2 "$queue" > "$queue.tmp.$$"
        mv "$queue.tmp.$$" "$queue"
        printf "%s\n" "$first"
    ' bash "${QUEUE}" || true
}

requeue_shard() {
    local shard="$1"
    flock "${LOCK}" bash -c '
        queue="$1"
        shard="$2"
        success_dir="$3"
        [ -f "$success_dir/_SUCCESS" ] && exit 0
        if grep -qx "$shard" "$queue" 2>/dev/null; then
            exit 0
        fi
        printf "%s\n" "$shard" >> "$queue"
    ' bash "${QUEUE}" "${shard}" "${OUTPUT_ROOT}/shard_$(printf '%05d' "${shard}")"
}

while true; do
    shard="$(pop_shard)"
    if [ -z "${shard}" ]; then
        echo "[slurm ${TASK_TAG}] queue empty"
        break
    fi

    shard_dir="${OUTPUT_ROOT}/shard_$(printf '%05d' "${shard}")"
    if [ -f "${shard_dir}/_SUCCESS" ]; then
        printf '[slurm %s] shard %05d already complete\n' "${TASK_TAG}" "${shard}"
        continue
    fi

    printf '[slurm %s] starting shard %05d\n' "${TASK_TAG}" "${shard}"
    CUDA_VISIBLE_DEVICES= \
    TOKENIZERS_PARALLELISM=false \
    HF_DATASETS_CACHE="${LOCAL_CACHE_ROOT}/datasets" \
    HF_HOME="${LOCAL_CACHE_ROOT}/hf_home" \
    XDG_CACHE_HOME="${LOCAL_CACHE_ROOT}/xdg" \
    BEAKER_ASSIGNED_CPU_COUNT="${CPUS}" \
    OI_NUM_PROC_OVERRIDE="${CPUS}" \
    "${PYTHON}" "${SCRIPT}" \
        --output-root "${OUTPUT_ROOT}" \
        --exclude-tokenized-root "${EXCLUDE_ROOT}" \
        --manifest-mode dataset_size_asc \
        --filter-workers "${CPUS}" \
        --filter-chunksize "${FILTER_CHUNKSIZE}" \
        --shard-index "${shard}"
    rc=$?

    if [ "${rc}" -ne 0 ]; then
        printf '[slurm %s] shard %05d failed rc=%s; requeueing\n' "${TASK_TAG}" "${shard}" "${rc}"
        requeue_shard "${shard}"
        exit "${rc}"
    fi

    printf '[slurm %s] finished shard %05d\n' "${TASK_TAG}" "${shard}"
done

echo "[slurm ${TASK_TAG}] done"
