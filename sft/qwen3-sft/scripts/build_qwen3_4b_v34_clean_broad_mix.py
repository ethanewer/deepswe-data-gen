#!/usr/bin/env python3
"""Build the qwen3-4b v34 broad clean continuation mix.

This mix uses the 260612 passed-prefix weighted rows as the bulk source and
adds the v33 edit/anti-repeat anchors. The online loss policy is responsible
for dropping any row whose context contains a manual patch.txt write.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_PREFIX_WEIGHTED = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "highquality-1x-duplicate-reasoning-90pct-30k-full-miniswe-aligned-passed-prefix-weighted-v2/data"
)
DEFAULT_EDIT_ANCHOR = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "qwen3-4b-thinking-v33-v28-edit-anchor-stage3-nomanual-mix"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "qwen3-4b-thinking-v34-clean-broad-prefix-editanchors-mix"
)


def jsonl_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def require_jsonl_root(root: Path, label: str) -> list[Path]:
    files = jsonl_files(root)
    if not files:
        raise FileNotFoundError(f"{label} has no JSONL files: {root}")
    return files


def symlink_source(files: list[Path], *, input_root: Path, output_root: Path, source_name: str) -> dict[str, Any]:
    source_dir = output_root / source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for index, source in enumerate(files):
        rel_parent = source.parent.relative_to(input_root)
        safe_parent = "_".join(rel_parent.parts) if rel_parent.parts else "root"
        destination = source_dir / f"{safe_parent}-{index:05d}.jsonl"
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source.resolve())
        total_bytes += source.stat().st_size
    return {
        "source_name": source_name,
        "input_root": str(input_root),
        "files": len(files),
        "bytes": total_bytes,
        "rough_tokens_bytes_div4": total_bytes // 4,
        "output": str(source_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-weighted-root", type=Path, default=DEFAULT_PREFIX_WEIGHTED)
    parser.add_argument("--edit-anchor-root", type=Path, default=DEFAULT_EDIT_ANCHOR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prefix_files = require_jsonl_root(args.prefix_weighted_root, "prefix weighted root")
    anchor_files = require_jsonl_root(args.edit_anchor_root, "edit anchor root")

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    sources = [
        symlink_source(
            prefix_files,
            input_root=args.prefix_weighted_root,
            output_root=args.output_root,
            source_name="passed_prefix_weighted_v2",
        ),
        symlink_source(
            anchor_files,
            input_root=args.edit_anchor_root,
            output_root=args.output_root,
            source_name="v33_edit_antirepeat_anchor",
        ),
    ]
    manifest = {
        "output_root": str(args.output_root),
        "selection": (
            "v34 broad clean continuation mix: passed 260612 prefix-weighted rows "
            "as bulk data plus v33 edit/anti-repeat anchors. Manual patch writes "
            "are dropped online by reject_manual_patch_targets=true."
        ),
        "sources": sources,
        "files": sum(source["files"] for source in sources),
        "bytes": sum(source["bytes"] for source in sources),
        "rough_tokens_bytes_div4": sum(source["bytes"] for source in sources) // 4,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
