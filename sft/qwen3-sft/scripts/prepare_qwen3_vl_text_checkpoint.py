#!/usr/bin/env python3
"""Create a local text-only view of a Qwen3-VL safetensors checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


TEXT_PREFIX = "model.language_model."
KEEP_PREFIXES = (TEXT_PREFIX, "lm_head.")
DEFAULT_REPO = "Qwen/Qwen3-VL-8B-Thinking"


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)
    except OSError:
        shutil.copy2(src, dst)


def build_view(repo_id: str, output_dir: Path, revision: str | None, local_files_only: bool) -> None:
    base_patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "*.tiktoken",
        "model.safetensors.index.json",
    ]
    snapshot = Path(
        snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=base_patterns,
            local_files_only=local_files_only,
        )
    )
    index_path = snapshot / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    full_weight_map = index["weight_map"]
    text_weight_map = {k: v for k, v in full_weight_map.items() if k.startswith(KEEP_PREFIXES)}
    if not text_weight_map:
        raise RuntimeError(f"no text weights found in {index_path}")

    shard_names = sorted(set(text_weight_map.values()))
    snapshot = Path(
        snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=base_patterns + shard_names,
            local_files_only=local_files_only,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in snapshot.iterdir():
        if path.name.endswith(".safetensors"):
            continue
        if path.name == "model.safetensors.index.json":
            continue
        if path.is_file():
            _link_or_copy(path, output_dir / path.name)

    for shard_name in shard_names:
        _link_or_copy(snapshot / shard_name, output_dir / shard_name)

    filtered_index = dict(index)
    filtered_index["weight_map"] = text_weight_map
    if "metadata" in filtered_index:
        filtered_index["metadata"] = dict(filtered_index["metadata"])
        filtered_index["metadata"]["source_repo"] = repo_id
        filtered_index["metadata"]["text_only_view"] = "true"
    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(filtered_index, f, indent=2, sort_keys=True)
        f.write("\n")

    print(
        f"Wrote {output_dir} with {len(text_weight_map)} text tensors across {len(shard_names)} shard files",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    build_view(args.repo_id, args.output_dir, args.revision, args.local_files_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
