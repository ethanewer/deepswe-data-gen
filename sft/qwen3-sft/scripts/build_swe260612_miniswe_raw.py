#!/usr/bin/env python3
"""Build an eval-aligned mini-swe-agent raw view of the 260612 synthetic data."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_swebench_ml_sft_mix import BASH_TOOL, adapt_to_bash_tool, json_dumps
from qwen_agentic_sft.data import iter_normalized_examples_from_files


DEFAULT_INPUT_JSONL = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "highquality-1x-duplicate-reasoning-90pct-30k-full/data/train.jsonl"
)


def open_shards(output_dir: Path, shards: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        (output_dir / f"shard-{idx:03d}.jsonl").open("w", encoding="utf-8")
        for idx in range(shards)
    ]


def aggregate_stats(total: dict[str, int], stats: dict[str, int]) -> None:
    for key, value in stats.items():
        total[key] = total.get(key, 0) + int(value)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)

    data_dir = args.output_root / "data"
    handles = open_shards(data_dir, args.shards)
    rows_seen = 0
    rows_written = 0
    rows_skipped = 0
    transform_stats: dict[str, int] = {}
    try:
        for example in iter_normalized_examples_from_files(
            [args.input_jsonl],
            parquet_batch_size=args.parquet_batch_size,
            max_examples=args.max_rows,
        ):
            rows_seen += 1
            transformed, stats = adapt_to_bash_tool(
                example,
                reasoning_tool_boundary=True,
                strict_prompt=True,
                require_submit=True,
                tool_observation_roles=True,
                single_tool_calls=args.single_tool_calls,
            )
            aggregate_stats(transform_stats, stats)
            if transformed is None:
                rows_skipped += 1
                continue
            row = {
                "messages": transformed["messages"],
                "tools": transformed.get("tools", BASH_TOOL),
                "source": "swe260612_highquality_miniswe_aligned",
                "source_note": (
                    "260612 high-quality synthetic SWE traces normalized to the "
                    "strict mini-swe-agent prompt, XML tool observations, and "
                    "reasoning/tool-call assistant format."
                ),
            }
            handles[rows_written % args.shards].write(json_dumps(row) + "\n")
            rows_written += 1
            if args.log_every and rows_seen % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "rows_seen": rows_seen,
                            "rows_written": rows_written,
                            "rows_skipped": rows_skipped,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.close()

    nonempty_shards = 0
    for path in data_dir.glob("shard-*.jsonl"):
        if path.stat().st_size:
            nonempty_shards += 1
        else:
            path.unlink()

    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "output_root": str(args.output_root),
        "transform": (
            "reasoning_tool_boundary_strict_miniswe_toolobs"
            + ("_single" if args.single_tool_calls else "")
            + "_submit"
        ),
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "rows_skipped": rows_skipped,
        "shards": nonempty_shards,
        "requested_shards": args.shards,
        "transform_stats": transform_stats,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--parquet-batch-size", type=int, default=128)
    parser.add_argument("--single-tool-calls", action="store_true")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
