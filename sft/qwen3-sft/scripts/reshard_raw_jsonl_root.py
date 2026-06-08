#!/usr/bin/env python3
"""Reshard a normalized raw JSONL root into more files.

This preserves raw chat rows and does not tokenize. It is useful for distributed
online packing, where rank/worker sharding happens at file granularity.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def source_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


def iter_jsonl_files(source: Path) -> list[Path]:
    return sorted(source.rglob("*.jsonl"))


def reshard_source(source: Path, output_source: Path, shards: int) -> dict[str, object]:
    output_source.mkdir(parents=True, exist_ok=True)
    handles = [
        (output_source / f"shard-{idx:03d}.jsonl").open("w", encoding="utf-8")
        for idx in range(shards)
    ]
    rows = 0
    try:
        for path in iter_jsonl_files(source):
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    handles[rows % shards].write(line)
                    rows += 1
    finally:
        for handle in handles:
            handle.close()

    nonempty = 0
    for path in output_source.glob("shard-*.jsonl"):
        if path.stat().st_size == 0:
            path.unlink()
        else:
            nonempty += 1
    return {
        "name": source.name,
        "rows": rows,
        "input_files": len(iter_jsonl_files(source)),
        "output_files": nonempty,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards-per-source", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        reshard_source(source, args.output_root / source.name, args.shards_per_source)
        for source in source_dirs(args.input_root)
    ]
    input_manifest = args.input_root / "manifest.json"
    manifest = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "shards_per_source": args.shards_per_source,
        "sources": summaries,
    }
    if input_manifest.exists():
        manifest["input_manifest"] = json.loads(input_manifest.read_text(encoding="utf-8"))
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
