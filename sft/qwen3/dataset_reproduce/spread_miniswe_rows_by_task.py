#!/usr/bin/env python3
"""Rewrite transformed mini-swe rows in deterministic task-spread order."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def iter_jsonl_files(root: Path) -> list[Path]:
    data_root = root / "data" if (root / "data").is_dir() else root
    files = sorted(data_root.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no .jsonl files under {data_root}")
    return files


def row_uuid(row: dict[str, Any]) -> str:
    source_outcome = row.get("source_outcome")
    if isinstance(source_outcome, dict) and source_outcome.get("uuid"):
        return str(source_outcome["uuid"])
    if row.get("uuid"):
        return str(row["uuid"])
    return ""


def row_task_id(row: dict[str, Any]) -> str:
    source_outcome = row.get("source_outcome")
    if isinstance(source_outcome, dict) and source_outcome.get("task_id"):
        return str(source_outcome["task_id"])
    if row.get("task_id"):
        return str(row["task_id"])
    return ""


def load_training_order(path: Path) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            uuid = str(row.get("uuid") or "")
            if not uuid:
                raise ValueError(f"missing uuid in {path}:{line_number}")
            if uuid not in seen:
                order.append(uuid)
                seen.add(uuid)
    return order


def load_rows(input_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows_by_uuid: dict[str, dict[str, Any]] = {}
    duplicate_uuids = 0
    missing_uuid = 0
    files = iter_jsonl_files(input_root)
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                uuid = row_uuid(row)
                if not uuid:
                    missing_uuid += 1
                    continue
                if uuid in rows_by_uuid:
                    duplicate_uuids += 1
                    continue
                rows_by_uuid[uuid] = row
    return rows_by_uuid, {
        "input_files": [str(path) for path in files],
        "input_rows_with_uuid": len(rows_by_uuid),
        "input_duplicate_uuids": duplicate_uuids,
        "input_missing_uuid_rows": missing_uuid,
    }


def assistant_sequence_fingerprint(row: dict[str, Any]) -> tuple[str, int, int]:
    assistant_messages = [
        message
        for message in row.get("messages", [])
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    payload = json.dumps(assistant_messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(assistant_messages), len(payload)


def drop_duplicate_long_assistant_sequences(
    rows: list[dict[str, Any]],
    *,
    min_turns: int,
    min_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: dict[str, str] = {}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        digest, turns, chars = assistant_sequence_fingerprint(row)
        uuid = row_uuid(row)
        if turns < min_turns or chars < min_chars:
            kept.append(row)
            continue
        first_uuid = seen.get(digest)
        if first_uuid is not None:
            dropped.append(
                {
                    "uuid": uuid,
                    "task_id": row_task_id(row),
                    "duplicate_of_uuid": first_uuid,
                    "assistant_sequence_sha256": digest,
                    "assistant_turns": turns,
                    "assistant_sequence_chars": chars,
                }
            )
            continue
        seen[digest] = uuid
        kept.append(row)
    return kept, {
        "duplicate_long_assistant_sequence_min_turns": min_turns,
        "duplicate_long_assistant_sequence_min_chars": min_chars,
        "dropped_duplicate_long_assistant_sequences": len(dropped),
        "duplicate_long_assistant_sequence_sample": dropped[:25],
    }


def write_shards(rows: list[dict[str, Any]], data_dir: Path, shards: int) -> list[dict[str, Any]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    handles = [
        (data_dir / f"shard-{idx:03d}.jsonl").open("w", encoding="utf-8")
        for idx in range(shards)
    ]
    try:
        for index, row in enumerate(rows):
            handles[index % shards].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        for handle in handles:
            handle.close()

    stats: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("shard-*.jsonl")):
        rows_in_file = 0
        task_counts: dict[str, int] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows_in_file += 1
                task_id = row_task_id(json.loads(line))
                task_counts[task_id] = task_counts.get(task_id, 0) + 1
        if rows_in_file:
            stats.append(
                {
                    "shard": path.name,
                    "rows": rows_in_file,
                    "unique_tasks": len(task_counts),
                    "max_rows_per_task": max(task_counts.values(), default=0),
                }
            )
        else:
            path.unlink()
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--training-order-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--drop-duplicate-long-assistant-sequences", action="store_true")
    parser.add_argument("--duplicate-long-assistant-min-turns", type=int, default=4)
    parser.add_argument("--duplicate-long-assistant-min-chars", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be >= 1")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    order = load_training_order(args.training_order_jsonl)
    rows_by_uuid, input_stats = load_rows(args.input_root)
    ordered_rows: list[dict[str, Any]] = []
    missing_ordered_uuids: list[str] = []
    used: set[str] = set()
    for uuid in order:
        row = rows_by_uuid.get(uuid)
        if row is None:
            missing_ordered_uuids.append(uuid)
            continue
        ordered_rows.append(row)
        used.add(uuid)

    extra_rows = [row for uuid, row in sorted(rows_by_uuid.items()) if uuid not in used]
    ordered_rows.extend(extra_rows)
    assistant_dedupe_stats: dict[str, Any] = {
        "drop_duplicate_long_assistant_sequences": args.drop_duplicate_long_assistant_sequences,
        "duplicate_long_assistant_sequence_min_turns": args.duplicate_long_assistant_min_turns,
        "duplicate_long_assistant_sequence_min_chars": args.duplicate_long_assistant_min_chars,
        "dropped_duplicate_long_assistant_sequences": 0,
        "duplicate_long_assistant_sequence_sample": [],
    }
    if args.drop_duplicate_long_assistant_sequences:
        ordered_rows, assistant_dedupe_stats = drop_duplicate_long_assistant_sequences(
            ordered_rows,
            min_turns=args.duplicate_long_assistant_min_turns,
            min_chars=args.duplicate_long_assistant_min_chars,
        )
        assistant_dedupe_stats["drop_duplicate_long_assistant_sequences"] = True
    shard_stats = write_shards(ordered_rows, args.output_root / "data", args.shards)

    manifest = {
        "input_root": str(args.input_root),
        "training_order_jsonl": str(args.training_order_jsonl),
        "output_root": str(args.output_root),
        "ordering": "selected UUID order from allowlist; rows missing after transform skipped; extras appended by UUID",
        "rows_written": len(ordered_rows),
        "ordered_rows_written": len(ordered_rows) - len(extra_rows),
        "extra_rows_appended": len(extra_rows),
        "missing_ordered_uuids": len(missing_ordered_uuids),
        "missing_ordered_uuid_sample": missing_ordered_uuids[:25],
        "assistant_sequence_deduplication": assistant_dedupe_stats,
        "input_stats": input_stats,
        "shards": shard_stats,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
