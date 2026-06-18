#!/usr/bin/env python3
"""Reorder Mini-SWE JSONL shards while preserving the first worker file.

The online packer sees 64 files with 16 rank/worker streams as:

  worker s: shard-s, shard-(s+16), shard-(s+32), shard-(s+48)

This script copies the first 16 shards unchanged, then deterministically
reorders the remaining rows so each worker stream sees a task-spread sequence
after its preserved prefix shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def data_root(root: Path) -> Path:
    return root / "data" if (root / "data").is_dir() else root


def row_uuid(row: dict[str, Any]) -> str:
    outcome = row.get("source_outcome")
    if isinstance(outcome, dict) and outcome.get("uuid"):
        return str(outcome["uuid"])
    if row.get("uuid"):
        return str(row["uuid"])
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def row_task_id(row: dict[str, Any]) -> str:
    outcome = row.get("source_outcome")
    if isinstance(outcome, dict) and outcome.get("task_id"):
        return str(outcome["task_id"])
    if row.get("task_id"):
        return str(row["task_id"])
    return "unknown"


def stable_key(seed: int, *parts: str) -> str:
    payload = "\0".join([str(seed), *parts])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def count_jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def task_spread_order(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row_task_id(row)].append(row)

    queues: dict[str, deque[dict[str, Any]]] = {}
    for task_id, task_rows in by_task.items():
        task_rows.sort(key=lambda row: stable_key(seed, task_id, row_uuid(row)))
        queues[task_id] = deque(task_rows)

    task_ids = sorted(queues, key=lambda task_id: stable_key(seed, task_id))
    ordered: list[dict[str, Any]] = []
    while task_ids:
        next_task_ids: list[str] = []
        for task_id in task_ids:
            queue = queues[task_id]
            ordered.append(queue.popleft())
            if queue:
                next_task_ids.append(task_id)
        task_ids = next_task_ids
    return ordered


def split_contiguous(rows: list[dict[str, Any]], parts: int) -> list[list[dict[str, Any]]]:
    if parts < 1:
        raise ValueError("parts must be >= 1")
    base, extra = divmod(len(rows), parts)
    chunks: list[list[dict[str, Any]]] = []
    start = 0
    for index in range(parts):
        size = base + (1 if index < extra else 0)
        chunks.append(rows[start : start + size])
        start += size
    return chunks


def summarize_shards(paths: list[Path]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for path in paths:
        rows = 0
        task_counts: dict[str, int] = defaultdict(int)
        for row in load_jsonl(path):
            rows += 1
            task_counts[row_task_id(row)] += 1
        stats.append(
            {
                "shard": path.name,
                "rows": rows,
                "unique_tasks": len(task_counts),
                "max_rows_per_task": max(task_counts.values(), default=0),
            }
        )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preserve-shards", type=int, default=16)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--seed", type=int, default=61682)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_data = data_root(args.input_root)
    files = sorted(in_data.glob("shard-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no shard-*.jsonl files under {in_data}")
    if args.preserve_shards <= 0 or args.preserve_shards >= len(files):
        raise ValueError("--preserve-shards must be between 1 and input shard count - 1")
    if args.streams <= 0:
        raise ValueError("--streams must be positive")
    remaining_count = len(files) - args.preserve_shards
    if remaining_count % args.streams != 0:
        raise ValueError(
            f"remaining shard count {remaining_count} must be divisible by streams {args.streams}"
        )

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    out_data = args.output_root / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    preserved_files = files[: args.preserve_shards]
    for source in preserved_files:
        shutil.copy2(source, out_data / source.name)

    later_rows: list[dict[str, Any]] = []
    for source in files[args.preserve_shards :]:
        later_rows.extend(load_jsonl(source))
    ordered_rows = task_spread_order(later_rows, seed=args.seed)

    stream_rows: list[list[dict[str, Any]]] = [[] for _ in range(args.streams)]
    for index, row in enumerate(ordered_rows):
        stream_rows[index % args.streams].append(row)

    shard_groups = remaining_count // args.streams
    for stream_index, rows in enumerate(stream_rows):
        chunks = split_contiguous(rows, shard_groups)
        for group_index, chunk in enumerate(chunks):
            shard_index = args.preserve_shards + stream_index + group_index * args.streams
            write_jsonl(out_data / f"shard-{shard_index:03d}.jsonl", chunk)

    output_files = sorted(out_data.glob("shard-*.jsonl"))
    manifest = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "input_shards": len(files),
        "output_shards": len(output_files),
        "preserve_shards": args.preserve_shards,
        "streams": args.streams,
        "post_prefix_shard_groups_per_stream": shard_groups,
        "preserved_rows": sum(count_jsonl_rows(path) for path in preserved_files),
        "reordered_rows": len(ordered_rows),
        "total_rows": sum(count_jsonl_rows(path) for path in output_files),
        "ordering": (
            "first preserve_shards copied unchanged; remaining rows grouped by task, "
            "stable-shuffled within task, round-robin across tasks, then assigned to "
            "rank/worker streams and split contiguously across each stream's later files"
        ),
        "shards": summarize_shards(output_files),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
