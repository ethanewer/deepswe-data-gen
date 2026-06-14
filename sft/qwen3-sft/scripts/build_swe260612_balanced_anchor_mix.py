#!/usr/bin/env python3
"""Build a balanced SWE260612 mix with safe action anchors.

The v9 empty-diff recipe over-sampled recovery rows whose target is always a
`git diff` command. This builder keeps the current 260612 weighted prefix data,
keeps only a small amount of empty-diff recovery, and adds older v38 anchor
shards that do not teach direct synthetic-diff submission. The v38
patch-recovery bucket is excluded by default because its targets are all
rejected by the online SWE loss policy. The v1 edit/submit prefix anchor is
also excluded by default because step-50 evals showed it encourages manual
`diff --git` submissions with fabricated metadata instead of source edits.
The builder writes symlinks instead of copying large JSONL files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "qwen3-4b-thinking-prefix-weighted-v12-cleanbalanced-mix"
)
DEFAULT_WEIGHTED_V2 = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "highquality-1x-duplicate-reasoning-90pct-30k-full-miniswe-aligned-passed-prefix-weighted-v2/data"
)
DEFAULT_EMPTY_DIFF = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "qwen3-4b-thinking-prefix-weighted-v2-emptydiff-mix/empty_diff_recovery_v1"
)
DEFAULT_V38_ANCHOR = Path(
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft/data/"
    "swebench_ml_v38_clean_sourceedit_shortthought_raw_sharded"
)
DEFAULT_EXCLUDED_V38_ANCHORS = (
    "v1_edit_submit_prefix_anchor_v38",
    "v23_patch_recovery_v38",
)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def jsonl_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def require_root(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not jsonl_files(path):
        raise FileNotFoundError(f"{label} contains no JSONL files: {path}")


def symlink_files(source_root: Path, output_root: Path, source_name: str) -> dict[str, Any]:
    files = jsonl_files(source_root)
    if not files:
        raise FileNotFoundError(f"no JSONL files found under {source_root}")
    out_dir = output_root / source_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(files):
        suffix = source.suffix
        destination = out_dir / f"shard-{index:05d}{suffix}"
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source.resolve())
    return {
        "name": source_name,
        "source_root": str(source_root),
        "files": len(files),
        "output": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weighted-v2-root", type=Path, default=DEFAULT_WEIGHTED_V2)
    parser.add_argument("--empty-diff-root", type=Path, default=DEFAULT_EMPTY_DIFF)
    parser.add_argument("--v38-anchor-root", type=Path, default=DEFAULT_V38_ANCHOR)
    parser.add_argument("--empty-diff-copies", type=int, default=1)
    parser.add_argument(
        "--exclude-v38-anchor",
        action="append",
        default=list(DEFAULT_EXCLUDED_V38_ANCHORS),
        help="v38 anchor subdirectory name to skip; may be repeated",
    )
    parser.add_argument(
        "--include-all-v38-anchors",
        action="store_true",
        help="ignore the default v38 anchor exclusion list",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.empty_diff_copies < 0:
        raise ValueError("--empty-diff-copies must be non-negative")
    require_root(args.weighted_v2_root, "weighted v2 root")
    require_root(args.v38_anchor_root, "v38 anchor root")
    if args.empty_diff_copies:
        require_root(args.empty_diff_root, "empty-diff root")

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = []
    sources.append(symlink_files(args.weighted_v2_root, args.output_root, "weighted_v2"))
    for copy_index in range(args.empty_diff_copies):
        sources.append(
            symlink_files(
                args.empty_diff_root,
                args.output_root,
                f"empty_diff_recovery_v1_x{copy_index}",
            )
        )
    excluded_v38_anchors = set() if args.include_all_v38_anchors else set(args.exclude_v38_anchor or [])
    skipped_sources: list[dict[str, Any]] = []
    for anchor_dir in sorted(path for path in args.v38_anchor_root.iterdir() if path.is_dir()):
        if anchor_dir.name in excluded_v38_anchors:
            skipped_sources.append(
                {
                    "name": f"v38_{anchor_dir.name}",
                    "source_root": str(anchor_dir),
                    "reason": "excluded by --exclude-v38-anchor",
                }
            )
            continue
        sources.append(
            symlink_files(
                anchor_dir,
                args.output_root,
                f"v38_{anchor_dir.name}",
            )
        )

    manifest = {
        "output_root": str(args.output_root),
        "selection": (
            "260612 weighted prefix data plus one-copy empty-diff recovery and "
            "safe v38 anchors, excluding v38 buckets that are rejected by the "
            "online SWE loss policy or that teach direct synthetic-diff "
            "submission. This avoids the v9 x12 diff-target skew while "
            "preserving explicit recovery examples."
        ),
        "empty_diff_copies": args.empty_diff_copies,
        "excluded_v38_anchors": sorted(excluded_v38_anchors),
        "sources": sources,
        "skipped_sources": skipped_sources,
        "files": sum(int(source["files"]) for source in sources),
    }
    (args.output_root / "manifest.json").write_text(json_dumps(manifest) + "\n", encoding="utf-8")
    print(json_dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
