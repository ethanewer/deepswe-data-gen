#!/usr/bin/env python3
"""Overlay trained Qwen3-VL text weights onto the original full VL checkpoint.

Training uses a text-only view of ``Qwen/Qwen3-VL-8B-Thinking`` so the vision
tower is never loaded. vLLM already serves the original full VL architecture, so
for evaluation we build a lightweight merged checkpoint directory:

* config/tokenizer/preprocessor files come from the original full checkpoint
* unchanged vision tensors point at the original full checkpoint shards
* trained language-model/lm-head tensors point at the SFT checkpoint shards

The output is a standard safetensors-indexed HF checkpoint and usually consists
mostly of symlinks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover - only needed for single-shard fallback
    safe_open = None


DEFAULT_BASE_REPO = "Qwen/Qwen3-VL-8B-Thinking"
TEXT_PREFIX = "model.language_model."
NATIVE_TEXT_PREFIX = "model."
KEEP_NON_WEIGHT_SUFFIXES = {
    ".json",
    ".txt",
    ".model",
    ".tiktoken",
}


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(os.path.relpath(src, start=dst.parent))
    except OSError:
        shutil.copy2(src, dst)


def copy_metadata_files(src_dir: Path, output_dir: Path, *, copy: bool) -> None:
    for path in src_dir.iterdir():
        if not path.is_file() and not path.is_symlink():
            continue
        if path.name.endswith(".safetensors") or path.name == "model.safetensors.index.json":
            continue
        if path.suffix in KEEP_NON_WEIGHT_SUFFIXES or path.name in {
            "chat_template.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
            "merges.txt",
        }:
            link_or_copy(path.resolve(), output_dir / path.name, copy=copy)


def load_safetensors_index(model_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = dict(index["weight_map"])
        return index, weight_map

    single_shard = model_dir / "model.safetensors"
    if not single_shard.exists():
        raise FileNotFoundError(f"No safetensors index or model.safetensors found in {model_dir}")
    if safe_open is None:
        raise RuntimeError("safetensors is required to inspect a single-shard checkpoint")
    with safe_open(single_shard, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    index = {"metadata": {"format": "pt"}, "weight_map": {key: single_shard.name for key in keys}}
    return index, dict(index["weight_map"])


def resolve_base_checkpoint(repo_id_or_path: str, revision: str | None, local_files_only: bool) -> Path:
    candidate = Path(repo_id_or_path)
    if candidate.exists():
        return candidate.resolve()
    return Path(
        snapshot_download(
            repo_id_or_path,
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.tiktoken",
                "*.safetensors",
                "model.safetensors.index.json",
            ],
            local_files_only=local_files_only,
        )
    )


def normalize_text_key(key: str) -> str | None:
    if key.startswith(TEXT_PREFIX) or key.startswith("lm_head."):
        return key
    if key.startswith(NATIVE_TEXT_PREFIX):
        return f"{TEXT_PREFIX}{key[len(NATIVE_TEXT_PREFIX):]}"
    return None


def build_merge(
    *,
    base_model: str,
    trained_text_checkpoint: Path,
    output_dir: Path,
    revision: str | None,
    local_files_only: bool,
    copy_shards: bool,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    base_dir = resolve_base_checkpoint(base_model, revision, local_files_only)
    base_index, base_weight_map = load_safetensors_index(base_dir)
    _trained_index, trained_weight_map = load_safetensors_index(trained_text_checkpoint)

    copy_metadata_files(base_dir, output_dir, copy=copy_shards)

    for shard_name in sorted(set(base_weight_map.values())):
        link_or_copy(base_dir / shard_name, output_dir / shard_name, copy=copy_shards)

    trained_shard_map: dict[str, str] = {}
    for shard_name in sorted(set(trained_weight_map.values())):
        src = trained_text_checkpoint / shard_name
        dst_name = f"trained-text-{shard_name}"
        trained_shard_map[shard_name] = dst_name
        link_or_copy(src, output_dir / dst_name, copy=copy_shards)

    merged_weight_map = dict(base_weight_map)
    replaced: list[str] = []
    ignored: list[str] = []
    missing_in_base: list[str] = []
    for trained_key, shard_name in sorted(trained_weight_map.items()):
        merged_key = normalize_text_key(trained_key)
        if merged_key is None:
            ignored.append(trained_key)
            continue
        if merged_key not in base_weight_map:
            missing_in_base.append(merged_key)
        merged_weight_map[merged_key] = trained_shard_map[shard_name]
        replaced.append(merged_key)

    merged_index = dict(base_index)
    merged_index["weight_map"] = merged_weight_map
    metadata = dict(merged_index.get("metadata") or {})
    metadata.update(
        {
            "base_model": str(base_model),
            "trained_text_checkpoint": str(trained_text_checkpoint),
            "text_overlay": "true",
            "text_tensors_replaced": str(len(replaced)),
        }
    )
    merged_index["metadata"] = metadata
    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(merged_index, f, indent=2, sort_keys=True)
        f.write("\n")

    manifest = {
        "base_dir": str(base_dir),
        "trained_text_checkpoint": str(trained_text_checkpoint),
        "output_dir": str(output_dir),
        "base_tensors": len(base_weight_map),
        "trained_tensors": len(trained_weight_map),
        "text_tensors_replaced": len(replaced),
        "ignored_trained_tensors": ignored,
        "trained_text_keys_missing_in_base": missing_in_base,
        "copy_shards": copy_shards,
    }
    with (output_dir / "merge_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if missing_in_base:
        raise RuntimeError(f"{len(missing_in_base)} trained text keys were not present in the base checkpoint")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_REPO)
    parser.add_argument("--trained-text-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--copy-shards", action="store_true", help="Copy shards instead of symlinking them.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_merge(
        base_model=args.base_model,
        trained_text_checkpoint=args.trained_text_checkpoint.resolve(),
        output_dir=args.output_dir.resolve(),
        revision=args.revision,
        local_files_only=args.local_files_only,
        copy_shards=args.copy_shards,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
