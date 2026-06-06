#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/tokenized/code-swe-terminal-agentic-sft-olmo3-65k-smallest-first}"
EXCLUDE_ROOT="${EXCLUDE_ROOT:-/wbl-fast/usrs/ee/code-swe-data/data/tokenized/code-swe-terminal-agentic-sft-olmo3-65k}"
SCRIPT="${SCRIPT:-/wbl-fast/usrs/ee/code-swe-data/sft/olmo3-sft/scripts/prepare_code_swe_sft_data.py}"
PYTHON="${PYTHON:-/wbl-fast/usrs/mk/open-instruct/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-/wbl-fast/usrs/ee/code-swe-data/sft/olmo3-sft/logs/tokenize_workers}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CORES_PER_WORKER="${CORES_PER_WORKER:-48}"
CORE_OFFSET="${CORE_OFFSET:-0}"
WORKER_ID_OFFSET="${WORKER_ID_OFFSET:-0}"
FILTER_CHUNKSIZE="${FILTER_CHUNKSIZE:-64}"

mkdir -p "$LOG_DIR"

QUEUE="${QUEUE:-$LOG_DIR/queue.$(date -u +%Y%m%dT%H%M%SZ).txt}"
LOCK="$QUEUE.lock"

if [ ! -s "$QUEUE" ]; then
    "$PYTHON" - "$OUTPUT_ROOT" > "$QUEUE" <<'PY'
import sys
from pathlib import Path

manifest = Path(sys.argv[1]) / "manifest.tsv"
seen = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line:
        continue
    shard = int(line.split("\t", 1)[0])
    if shard not in seen:
        seen.append(shard)
for shard in seen:
    print(shard)
PY
fi

echo "[launcher] queue=$QUEUE workers=$NUM_WORKERS cores_per_worker=$CORES_PER_WORKER core_offset=$CORE_OFFSET worker_id_offset=$WORKER_ID_OFFSET"

worker_loop() {
    local worker_id="$1"
    local local_worker_id="$2"
    local start_core=$((CORE_OFFSET + local_worker_id * CORES_PER_WORKER))
    local end_core=$((start_core + CORES_PER_WORKER - 1))
    local core_range="${start_core}-${end_core}"
    local worker_log="$LOG_DIR/worker_${worker_id}.$(date -u +%Y%m%dT%H%M%SZ).log"

    {
        echo "[worker $worker_id] cores=$core_range"
        while true; do
            local shard=""
            shard="$(
                flock "$LOCK" bash -c '
                    queue="$1"
                    first="$(head -n 1 "$queue" 2>/dev/null || true)"
                    [ -n "$first" ] || exit 1
                    tail -n +2 "$queue" > "$queue.tmp"
                    mv "$queue.tmp" "$queue"
                    printf "%s\n" "$first"
                ' bash "$QUEUE" || true
            )"
            [ -n "$shard" ] || break
            local local_cache_root="${LOCAL_CACHE_ROOT:-${OUTPUT_ROOT}/.job-hf-cache/local_${worker_id}_${shard}}"
            rm -rf "$local_cache_root"
            mkdir -p "$local_cache_root/datasets" "$local_cache_root/hf_home" "$local_cache_root/xdg"
            printf '[worker %s] starting shard %05d\n' "$worker_id" "$shard"
            env \
                CUDA_VISIBLE_DEVICES= \
                TOKENIZERS_PARALLELISM=false \
                HF_DATASETS_CACHE="$local_cache_root/datasets" \
                HF_HOME="$local_cache_root/hf_home" \
                XDG_CACHE_HOME="$local_cache_root/xdg" \
                BEAKER_ASSIGNED_CPU_COUNT="$CORES_PER_WORKER" \
                OI_NUM_PROC_OVERRIDE="$CORES_PER_WORKER" \
                taskset -c "$core_range" \
                "$PYTHON" "$SCRIPT" \
                    --output-root "$OUTPUT_ROOT" \
                    --exclude-tokenized-root "$EXCLUDE_ROOT" \
                    --manifest-mode dataset_size_asc \
                    --filter-workers "$CORES_PER_WORKER" \
                    --filter-chunksize "$FILTER_CHUNKSIZE" \
                    --shard-index "$shard"
            rm -rf "$local_cache_root"
            printf '[worker %s] finished shard %05d\n' "$worker_id" "$shard"
        done
        echo "[worker $worker_id] done"
    } >> "$worker_log" 2>&1
}

for local_worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
    worker_id=$((WORKER_ID_OFFSET + local_worker_id))
    worker_loop "$worker_id" "$local_worker_id" &
done

wait
echo "[launcher] done"
