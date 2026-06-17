#!/usr/bin/env python3
"""Build a ramped-passrate view from a Mini-SWE-aligned JSONL view.

The online packed trainer streams files in sorted order when
``shuffle_files=false``.  A pure pass-last layout can delay all verified submit
targets until late in training, making early and mid checkpoints poor patch
format teachers.  This builder keeps a curriculum by increasing passrate over
successive shards while interleaving passed and non-passing rows inside each
shard.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617/"
    "swerebench-v5-compaction-mixed50-miniswe-aligned"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617/"
    "swerebench-v5-compaction-mixed50-ramped-miniswe-aligned"
)


RowRecord = tuple[Path, int, dict[str, Any], str]


def is_trueish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "passed", "pass"}
    return False


def row_passed(row: dict[str, Any]) -> bool:
    source_outcome = row.get("source_outcome")
    if isinstance(source_outcome, dict) and "passed" in source_outcome:
        return is_trueish(source_outcome.get("passed"))
    return is_trueish(row.get("passed"))


def iter_rows(data_root: Path):
    for path in sorted(data_root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for row_in_file, line in enumerate(handle):
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                yield path, row_in_file, row, text


def balanced_counts(total: int, shards: int) -> list[int]:
    base = total // shards
    extra = total % shards
    return [base + (1 if idx < extra else 0) for idx in range(shards)]


def passrate_schedule(actual_passrate: float, shards: int, min_passrate: float | None, max_passrate: float | None) -> list[float]:
    if min_passrate is None and max_passrate is None:
        low = max(0.05, min(actual_passrate * 0.4, actual_passrate - 0.05))
        high = (2.0 * actual_passrate) - low
    elif min_passrate is None:
        high = float(max_passrate)
        low = (2.0 * actual_passrate) - high
    elif max_passrate is None:
        low = float(min_passrate)
        high = (2.0 * actual_passrate) - low
    else:
        low = float(min_passrate)
        high = float(max_passrate)

    if not (0.0 <= low <= 1.0 and 0.0 <= high <= 1.0):
        raise ValueError(f"invalid passrate range: min={low}, max={high}")
    if high < low:
        raise ValueError(f"max passrate must be >= min passrate: min={low}, max={high}")
    if shards == 1:
        return [actual_passrate]
    return [low + (high - low) * (idx / (shards - 1)) for idx in range(shards)]


def allocate_pass_counts(row_counts: list[int], rates: list[float], total_passed: int) -> list[int]:
    raw = [min(float(count), max(0.0, count * rate)) for count, rate in zip(row_counts, rates, strict=True)]
    floors = [int(math.floor(value)) for value in raw]
    counts = [min(count, floor) for count, floor in zip(row_counts, floors, strict=True)]
    remaining = total_passed - sum(counts)
    if remaining < 0:
        # Remove from the smallest fractional remainders first.
        order = sorted(range(len(raw)), key=lambda idx: (raw[idx] - floors[idx], -idx))
        for idx in order:
            if remaining == 0:
                break
            take = min(counts[idx], -remaining)
            counts[idx] -= take
            remaining += take
    else:
        order = sorted(range(len(raw)), key=lambda idx: (raw[idx] - floors[idx], idx), reverse=True)
        while remaining > 0:
            changed = False
            for idx in order:
                if counts[idx] >= row_counts[idx]:
                    continue
                counts[idx] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
            if not changed:
                raise RuntimeError("could not allocate all passed rows")
    if sum(counts) != total_passed:
        raise RuntimeError(f"pass allocation mismatch: {sum(counts)} != {total_passed}")
    return counts


def interleave_rows(passed: list[RowRecord], failed: list[RowRecord]) -> list[RowRecord]:
    total = len(passed) + len(failed)
    if not passed:
        return list(failed)
    if not failed:
        return list(passed)

    pass_slots = {
        min(total - 1, math.floor((idx + 0.5) * total / len(passed)))
        for idx in range(len(passed))
    }
    # Collisions are rare but possible for tiny totals. Fill missing slots from
    # the end so the early-shard passrate remains conservative.
    cursor = total - 1
    while len(pass_slots) < len(passed):
        if cursor not in pass_slots:
            pass_slots.add(cursor)
        cursor -= 1

    out: list[RowRecord] = []
    pass_idx = 0
    fail_idx = 0
    for pos in range(total):
        if pos in pass_slots and pass_idx < len(passed):
            out.append(passed[pass_idx])
            pass_idx += 1
        elif fail_idx < len(failed):
            out.append(failed[fail_idx])
            fail_idx += 1
        else:
            out.append(passed[pass_idx])
            pass_idx += 1
    return out


def write_shard(path: Path, rows: list[RowRecord]) -> Counter[str]:
    stats: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for _source_path, _row_in_file, row, text in rows:
            handle.write(text + "\n")
            stats["rows"] += 1
            source_outcome = row.get("source_outcome") or {}
            language = row.get("language") or source_outcome.get("language") or "unknown"
            stats[f"language:{language}"] += 1
            stats["passed" if row_passed(row) else "failed"] += 1
    return stats


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_data_root = args.input_root / "data" if (args.input_root / "data").is_dir() else args.input_root
    if not input_data_root.is_dir():
        raise FileNotFoundError(f"input data root not found: {input_data_root}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)

    failed_rows: list[RowRecord] = []
    passed_rows: list[RowRecord] = []
    input_files: set[str] = set()
    for path, row_in_file, row, text in iter_rows(input_data_root):
        input_files.add(str(path))
        if row_passed(row):
            passed_rows.append((path, row_in_file, row, text))
        else:
            failed_rows.append((path, row_in_file, row, text))

    total_rows = len(failed_rows) + len(passed_rows)
    if total_rows == 0:
        raise RuntimeError(f"no rows found in {input_data_root}")
    passed_fraction = len(passed_rows) / total_rows
    row_counts = balanced_counts(total_rows, args.shards)
    rates = passrate_schedule(passed_fraction, args.shards, args.min_passrate, args.max_passrate)
    pass_counts = allocate_pass_counts(row_counts, rates, len(passed_rows))

    output_data_root = args.output_root / "data"
    output_data_root.mkdir(parents=True, exist_ok=True)
    passed_cursor = 0
    failed_cursor = 0
    shard_stats: list[dict[str, Any]] = []
    aggregate: Counter[str] = Counter()
    for shard_idx, (row_count, pass_count, target_rate) in enumerate(zip(row_counts, pass_counts, rates, strict=True)):
        fail_count = row_count - pass_count
        shard_passed = passed_rows[passed_cursor : passed_cursor + pass_count]
        shard_failed = failed_rows[failed_cursor : failed_cursor + fail_count]
        if len(shard_passed) != pass_count or len(shard_failed) != fail_count:
            raise RuntimeError("row allocation exceeded source rows")
        passed_cursor += pass_count
        failed_cursor += fail_count
        rows = interleave_rows(shard_passed, shard_failed)
        stats = write_shard(output_data_root / f"shard-{shard_idx:03d}.jsonl", rows)
        aggregate.update(stats)
        shard_stats.append(
            {
                "shard": shard_idx,
                "rows": row_count,
                "passed": pass_count,
                "failed": fail_count,
                "actual_passrate": pass_count / row_count if row_count else 0.0,
                "target_passrate": target_rate,
            }
        )

    if passed_cursor != len(passed_rows) or failed_cursor != len(failed_rows):
        raise RuntimeError("not all rows were written")

    manifest = {
        "input_root": str(args.input_root),
        "input_data_root": str(input_data_root),
        "input_files": sorted(input_files),
        "output_root": str(args.output_root),
        "ordering": "ramped passrate across sorted shards; passed and non-passing rows interleaved within each shard",
        "shards": args.shards,
        "rows_total": total_rows,
        "rows_failed": len(failed_rows),
        "rows_passed": len(passed_rows),
        "actual_passrate": passed_fraction,
        "min_shard_passrate": min(item["actual_passrate"] for item in shard_stats),
        "max_shard_passrate": max(item["actual_passrate"] for item in shard_stats),
        "aggregate_stats": dict(sorted(aggregate.items())),
        "shard_stats": shard_stats,
        "trainer_requirements": {
            "shuffle_files": False,
            "shuffle_jsonl_rows": False,
            "clean_comparison": "start from base checkpoint; no prefix expansion; mask non-passing submit targets",
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--min-passrate", type=float, default=None)
    parser.add_argument("--max-passrate", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    manifest = build(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
