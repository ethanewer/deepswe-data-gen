#!/usr/bin/env python3
"""Build v36 passed-prefix plus empty-diff recovery rows.

The output is one physically resharded JSONL source. This avoids online packer
file-count weighting artifacts from roots that contain one huge prefix file and
many small recovery shards.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, TextIO


DEFAULT_PREFIX_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "highquality-1x-duplicate-reasoning-90pct-30k-full-miniswe-aligned-passed-prefix-weighted-v2/data"
)
DEFAULT_RECOVERY_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "qwen3-4b-current-empty-diff-recovery-v1/empty_diff_recovery_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "qwen3-4b-thinking-v36-v28-prefix-emptydiffx4-balanced-mix"
)


def jsonl_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def iter_lines(files: Iterable[Path]) -> Iterable[str]:
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line


def write_line(handles: list[TextIO], line: str, row_index: int) -> int:
    handles[row_index % len(handles)].write(line)
    return row_index + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-root", type=Path, default=DEFAULT_PREFIX_ROOT)
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--empty-repeats", type=int, default=4)
    parser.add_argument("--prefix-per-empty", type=int, default=3)
    parser.add_argument("--shards", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prefix_files = jsonl_files(args.prefix_root)
    recovery_files = jsonl_files(args.recovery_root)
    if not prefix_files:
        raise FileNotFoundError(f"no prefix JSONL files under {args.prefix_root}")
    if not recovery_files:
        raise FileNotFoundError(f"no recovery JSONL files under {args.recovery_root}")
    if args.empty_repeats < 0:
        raise ValueError("--empty-repeats must be non-negative")
    if args.prefix_per_empty < 1:
        raise ValueError("--prefix-per-empty must be at least 1")
    if args.shards < 1:
        raise ValueError("--shards must be positive")

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    mixed_dir = args.output_root / "mixed"
    mixed_dir.mkdir(parents=True, exist_ok=True)

    handles = [
        (mixed_dir / f"shard-{idx:03d}.jsonl").open("w", encoding="utf-8")
        for idx in range(args.shards)
    ]
    prefix_rows = 0
    recovery_rows = 0
    total_rows = 0
    recovery_iter = iter_lines(recovery_files)
    recovery_repeat_index = 0

    def next_recovery_line() -> str | None:
        nonlocal recovery_iter, recovery_repeat_index
        while recovery_repeat_index < args.empty_repeats:
            try:
                return next(recovery_iter)
            except StopIteration:
                recovery_repeat_index += 1
                if recovery_repeat_index >= args.empty_repeats:
                    return None
                recovery_iter = iter_lines(recovery_files)
        return None

    try:
        since_recovery = 0
        for line in iter_lines(prefix_files):
            total_rows = write_line(handles, line, total_rows)
            prefix_rows += 1
            since_recovery += 1
            if since_recovery >= args.prefix_per_empty:
                recovery_line = next_recovery_line()
                if recovery_line is not None:
                    total_rows = write_line(handles, recovery_line, total_rows)
                    recovery_rows += 1
                    since_recovery = 0
        while True:
            recovery_line = next_recovery_line()
            if recovery_line is None:
                break
            total_rows = write_line(handles, recovery_line, total_rows)
            recovery_rows += 1
    finally:
        for handle in handles:
            handle.close()

    nonempty = 0
    total_bytes = 0
    for path in mixed_dir.glob("shard-*.jsonl"):
        size = path.stat().st_size
        if size == 0:
            path.unlink()
        else:
            nonempty += 1
            total_bytes += size

    manifest = {
        "output_root": str(args.output_root),
        "selection": (
            "v36: v28-style passed-prefix rows plus 4x current empty-diff "
            "recovery rows, interleaved and physically resharded for online packing."
        ),
        "prefix_root": str(args.prefix_root),
        "recovery_root": str(args.recovery_root),
        "empty_repeats": args.empty_repeats,
        "prefix_per_empty": args.prefix_per_empty,
        "rows": {
            "prefix": prefix_rows,
            "empty_diff_recovery": recovery_rows,
            "total": total_rows,
        },
        "shards": nonempty,
        "bytes": total_bytes,
        "rough_tokens_bytes_div4": total_bytes // 4,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
