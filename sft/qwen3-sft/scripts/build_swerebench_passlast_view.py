#!/usr/bin/env python3
"""Build a pass-last ordered view from an existing Mini-SWE-aligned JSONL view.

The trainer shards files by rank/worker and then streams each assigned file list
in sorted order when ``shuffle_files=false``.  This script creates early shards
from non-passing rows and late shards from passing rows so every worker sees
process/tool-use traces before verified end-to-end traces in a single clean run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260616/"
    "swerebench-raw2030-targeted-limitations-mixed50-miniswe-aligned"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260616/"
    "swerebench-raw2030-targeted-limitations-mixed50-passlast-miniswe-aligned"
)


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


def open_shards(output_data_root: Path, shards: int):
    output_data_root.mkdir(parents=True, exist_ok=True)
    return [
        (output_data_root / f"shard-{idx:03d}.jsonl").open("w", encoding="utf-8")
        for idx in range(shards)
    ]


def write_group(rows: list[tuple[Path, int, dict[str, Any], str]], handles: list[Any], start: int, stop: int) -> Counter[str]:
    stats: Counter[str] = Counter()
    width = stop - start
    if width <= 0:
        raise ValueError("shard range must be non-empty")
    for idx, (_path, _row_in_file, row, text) in enumerate(rows):
        shard = start + (idx % width)
        handles[shard].write(text + "\n")
        stats["rows"] += 1
        source_outcome = row.get("source_outcome") or {}
        language = row.get("language") or source_outcome.get("language") or "unknown"
        stats[f"language:{language}"] += 1
    return stats


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_data_root = args.input_root / "data" if (args.input_root / "data").is_dir() else args.input_root
    if not input_data_root.is_dir():
        raise FileNotFoundError(f"input data root not found: {input_data_root}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)

    failed_rows: list[tuple[Path, int, dict[str, Any], str]] = []
    passed_rows: list[tuple[Path, int, dict[str, Any], str]] = []
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
    passed_shards = args.passed_shards
    if passed_shards is None:
        passed_shards = max(1, round(args.shards * passed_fraction))
    if passed_shards <= 0 or passed_shards >= args.shards:
        raise ValueError("--passed-shards must leave at least one failed and one passed shard")
    failed_shards = args.shards - passed_shards

    handles = open_shards(args.output_root / "data", args.shards)
    try:
        failed_stats = write_group(failed_rows, handles, 0, failed_shards)
        passed_stats = write_group(passed_rows, handles, failed_shards, args.shards)
    finally:
        for handle in handles:
            handle.close()

    manifest = {
        "input_root": str(args.input_root),
        "input_data_root": str(input_data_root),
        "input_files": sorted(input_files),
        "output_root": str(args.output_root),
        "ordering": "failed shards first, passed shards last",
        "shards": args.shards,
        "failed_shards": failed_shards,
        "passed_shards": passed_shards,
        "rows_total": total_rows,
        "rows_failed": len(failed_rows),
        "rows_passed": len(passed_rows),
        "actual_passrate": passed_fraction,
        "failed_stats": dict(sorted(failed_stats.items())),
        "passed_stats": dict(sorted(passed_stats.items())),
        "trainer_requirements": {
            "shuffle_files": False,
            "shuffle_jsonl_rows": False,
            "clean_comparison": "start from base checkpoint; do not use as a continuation artifact",
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
    parser.add_argument("--passed-shards", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    manifest = build(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
