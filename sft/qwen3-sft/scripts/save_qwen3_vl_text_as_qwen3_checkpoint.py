#!/usr/bin/env python3
"""Save a Qwen3-VL text checkpoint as a standard Qwen3 causal-LM checkpoint.

Qwen3-VL text weights are stored under ``model.language_model.*`` and the
checkpoint config has ``model_type: qwen3_vl``.  That makes generic Qwen3 SFT
trainers instantiate the full VL class, including unused vision modules.  This
script materializes a true text-only checkpoint by:

* writing a normal ``model_type: qwen3`` config from the VL ``text_config``;
* renaming safetensors keys from ``model.language_model.*`` to ``model.*``;
* preserving ``lm_head.weight`` and tokenizer/generation files.

The output is intentionally a real safetensors checkpoint, not just a rewritten
index, because HF loaders request tensor names from inside the safetensors file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file


TEXT_PREFIX = "model.language_model."
TOKENIZER_FILE_NAMES = {
    "chat_template.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}


def link_or_copy(src: Path, dst: Path, *, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(os.path.relpath(src.resolve(), start=dst.parent))
    except OSError:
        shutil.copy2(src, dst)


def load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            return dict(json.load(f)["weight_map"])

    single_shard = model_dir / "model.safetensors"
    if not single_shard.exists():
        raise FileNotFoundError(f"No safetensors checkpoint found under {model_dir}")
    with safe_open(single_shard, framework="pt", device="cpu") as f:
        return {key: single_shard.name for key in f.keys()}


def rename_tensor_key(key: str) -> str | None:
    if key.startswith(TEXT_PREFIX):
        return f"model.{key[len(TEXT_PREFIX):]}"
    if key.startswith("model.") or key.startswith("lm_head."):
        return key
    return None


def qwen3_config_from_vl_config(vl_config: dict[str, Any]) -> dict[str, Any]:
    text_config = dict(vl_config.get("text_config") or {})
    if not text_config:
        raise ValueError("input config does not contain text_config")

    rope_theta = text_config.get("rope_theta", 5000000)
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": text_config.get("attention_bias", False),
        "attention_dropout": text_config.get("attention_dropout", 0.0),
        "bos_token_id": text_config.get("bos_token_id"),
        "eos_token_id": text_config.get("eos_token_id"),
        "head_dim": text_config.get("head_dim"),
        "hidden_act": text_config.get("hidden_act", "silu"),
        "hidden_size": text_config["hidden_size"],
        "initializer_range": text_config.get("initializer_range", 0.02),
        "intermediate_size": text_config["intermediate_size"],
        "max_position_embeddings": text_config.get("max_position_embeddings", 262144),
        "max_window_layers": text_config.get("num_hidden_layers"),
        "model_type": "qwen3",
        "num_attention_heads": text_config["num_attention_heads"],
        "num_hidden_layers": text_config["num_hidden_layers"],
        "num_key_value_heads": text_config.get("num_key_value_heads"),
        "rms_norm_eps": text_config.get("rms_norm_eps", 1.0e-6),
        "rope_scaling": None,
        "rope_theta": rope_theta,
        "sliding_window": None,
        "tie_word_embeddings": vl_config.get("tie_word_embeddings", False),
        "torch_dtype": text_config.get("torch_dtype") or text_config.get("dtype", "bfloat16"),
        "transformers_version": vl_config.get("transformers_version"),
        "use_cache": text_config.get("use_cache", True),
        "use_sliding_window": False,
        "vocab_size": text_config["vocab_size"],
    }


def copy_tokenizer_files(input_dir: Path, output_dir: Path, *, copy_files: bool) -> list[str]:
    copied: list[str] = []
    for name in sorted(TOKENIZER_FILE_NAMES):
        src = input_dir / name
        if src.exists() or src.is_symlink():
            link_or_copy(src, output_dir / name, copy_files=copy_files)
            copied.append(name)
    return copied


def convert_checkpoint(
    *,
    input_dir: Path,
    output_dir: Path,
    copy_files: bool,
    overwrite: bool,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with (input_dir / "config.json").open("r", encoding="utf-8") as f:
        vl_config = json.load(f)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(qwen3_config_from_vl_config(vl_config), f, indent=2, sort_keys=False)
        f.write("\n")

    copied_tokenizer_files = copy_tokenizer_files(input_dir, output_dir, copy_files=copy_files)
    input_weight_map = load_weight_map(input_dir)

    shard_to_keys: dict[str, list[tuple[str, str]]] = {}
    ignored_keys: list[str] = []
    for old_key, shard_name in sorted(input_weight_map.items()):
        new_key = rename_tensor_key(old_key)
        if new_key is None:
            ignored_keys.append(old_key)
            continue
        shard_to_keys.setdefault(shard_name, []).append((old_key, new_key))

    output_weight_map: dict[str, str] = {}
    total_size = 0
    shard_manifest: list[dict[str, Any]] = []
    output_shards = [(name, pairs) for name, pairs in sorted(shard_to_keys.items()) if pairs]
    for shard_idx, (source_shard_name, key_pairs) in enumerate(output_shards, start=1):
        source_shard = input_dir / source_shard_name
        output_shard_name = f"model-{shard_idx:05d}-of-{len(output_shards):05d}.safetensors"
        output_tensors = {}
        with safe_open(source_shard, framework="pt", device="cpu") as sf:
            for old_key, new_key in key_pairs:
                tensor = sf.get_tensor(old_key)
                output_tensors[new_key] = tensor
                output_weight_map[new_key] = output_shard_name
                total_size += tensor.numel() * tensor.element_size()
        save_file(output_tensors, output_dir / output_shard_name, metadata={"format": "pt"})
        shard_manifest.append(
            {
                "source_shard": source_shard_name,
                "output_shard": output_shard_name,
                "tensors": len(output_tensors),
            }
        )

    index = {
        "metadata": {
            "format": "pt",
            "source_checkpoint": str(input_dir),
            "source_model_type": vl_config.get("model_type"),
            "converted_model_type": "qwen3",
            "total_size": total_size,
        },
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "copied_tokenizer_files": copied_tokenizer_files,
        "ignored_input_keys": ignored_keys,
        "output_tensors": len(output_weight_map),
        "output_shards": shard_manifest,
        "total_size": total_size,
    }
    with (output_dir / "qwen3_conversion_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-files", action="store_true", help="Copy tokenizer files instead of symlinking.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_checkpoint(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        copy_files=args.copy_files,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
