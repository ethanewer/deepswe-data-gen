# Code SWE Tokenization Progress

Last updated: 2026-06-05 18:12 UTC

## Current State

Tokenization is intentionally stopped. All stopped jobs were `ewer` CPU-only `m7i-cpu2` tokenization jobs. The unrelated `h200` job `sft-filtering-test` was left running.

Output root:

`/wbl-fast/usrs/ee/code-swe-data/data/tokenized/code-swe-terminal-agentic-sft-olmo3-65k-smallest-first`

Raw-data exclusion remains:

`/wbl-fast/usrs/ee/code-swe-data/data/code-swe-terminal-agentic-sft/AlienKevin__SWE-ZERO-12M-trajectories`

## Completion

- Manifest shards: `131`
- Complete shards with `_SUCCESS`: `120`
- Remaining shards: `11`
- Remaining shard IDs: `6, 7, 9, 10, 11, 21, 23, 87, 115, 126, 130`

## Resume Queues

Default queue:

`/wbl-fast/usrs/ee/code-swe-data/sft/olmo3-sft/logs/tokenize_workers/queue.20260604T152827Z.txt`

Queued shards:

`11, 130`

Refill queue:

`/wbl-fast/usrs/ee/code-swe-data/sft/olmo3-sft/logs/tokenize_workers/queue.refill.20260604T220446Z.txt`

Queued shards:

`126, 23, 87, 115, 7, 6, 9`

Note: shard `10` and shard `21` still lack `_SUCCESS` but were not in the queues at stop time. Recheck before resuming; they may have partial output from interrupted workers and should be explicitly queued if still incomplete.

## Jobs Stopped

The following CPU-only Slurm jobs were requeued then canceled:

- `200303_1` (`code-swe-tok-refill`): current shard `115`
- `200303_3` (`code-swe-tok-refill`): current shard `87`
- `200302_0` (`code-swe-tok`): current shard `11`
- `200303_0` (`code-swe-tok-refill`): current shard `23`
- `200303_2` (`code-swe-tok-refill`): current shard `126`

At verification time, `squeue -u ewer` showed no remaining `code-swe-tok*` jobs.

## Cache Safety

The Hugging Face cache issue was cleaned up before stopping:

- `/home/ewer/.cache/huggingface` is now only symlinks and was `4K` at verification.
- `/home/ewer/.cache/huggingface/datasets` points to `/wbl-fast/usrs/ee/code-swe-data/.cache/huggingface/datasets`.
- `/home/ewer/.cache/huggingface/hub` points to `/wbl-fast/usrs/ee/code-swe-data/.cache/huggingface/hub`.
- `/home/ewer/.cache/huggingface/xet` points to `/wbl-fast/usrs/ee/code-swe-data/.cache/huggingface/xet`.
- Tokenization worker scripts now set `HF_DATASETS_CACHE`, `HF_HOME`, and `XDG_CACHE_HOME` under the tokenized output root `.job-hf-cache` and remove that job cache on exit.
- `.job-hf-cache` was `144K` after the tokenization jobs exited, indicating cleanup ran.

## Resume Guidance

Before resuming:

1. Confirm no tokenization jobs are already running:
   `squeue -u ewer | grep code-swe-tok`
2. Recompute incomplete shards from `_SUCCESS` files and compare with the two queues.
3. Ensure shards `10` and `21` are queued if they still lack `_SUCCESS`.
4. Resume only on compute nodes with the patched Slurm worker:
   `sft/olmo3-sft/scripts/slurm_tokenize_queue_worker.sh`

Avoid using local jobs for this workload unless there is a deliberate reason and cache paths are explicitly verified.
