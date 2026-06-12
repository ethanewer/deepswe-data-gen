#!/usr/bin/env python3
"""Export a torch.distributed.checkpoint model directory to HF safetensors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_model(model_name: str, dtype: torch.dtype, trust_remote_code: bool) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    for attr in ("dtype", "torch_dtype"):
        if hasattr(config, attr):
            setattr(config, attr, dtype)

    kwargs = {"trust_remote_code": trust_remote_code}
    last_type_error: TypeError | None = None
    for dtype_kwarg in ("dtype", "torch_dtype"):
        try:
            return AutoModelForCausalLM.from_config(config, **kwargs, **{dtype_kwarg: dtype})
        except TypeError as exc:
            last_type_error = exc

    try:
        model = AutoModelForCausalLM.from_config(config, **kwargs)
    except TypeError:
        if last_type_error is not None:
            raise last_type_error
        raise
    return model.to(dtype=dtype)


def load_dcp_weights(model: torch.nn.Module, input_dir: Path) -> list[str]:
    reader = FileSystemReader(str(input_dir))
    metadata = reader.read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata)
    model_state = model.state_dict()
    model_keys = set(model_state)

    unexpected = sorted(checkpoint_keys - model_keys)
    if unexpected:
        preview = ", ".join(unexpected[:10])
        raise RuntimeError(f"Checkpoint has {len(unexpected)} key(s) absent from model state dict: {preview}")

    missing = sorted(model_keys - checkpoint_keys)
    state_to_load = {key: model_state[key] for key in sorted(checkpoint_keys)}
    dcp.load(state_to_load, storage_reader=reader, no_dist=True)

    # The Qwen3 thinking checkpoints tie lm_head.weight to model.embed_tokens.weight
    # and therefore do not store a separate lm_head tensor in torch_save DCP.
    model.load_state_dict(state_to_load, strict=False)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return missing


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not (input_dir / ".metadata").is_file():
        raise FileNotFoundError(f"Missing DCP metadata file: {input_dir / '.metadata'}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    dtype = DTYPES[args.dtype]
    model = build_model(args.model_name, dtype, args.trust_remote_code)
    missing = load_dcp_weights(model, input_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(output_dir)

    (output_dir / "export_metadata.json").write_text(
        json.dumps(
            {
                "source": str(input_dir),
                "model_name": args.model_name,
                "dtype": str(dtype),
                "missing_model_keys": missing,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Exported {input_dir} to {output_dir}")
    if missing:
        print(f"Model keys not present in DCP checkpoint: {len(missing)}")
        print("\n".join(missing[:20]))


if __name__ == "__main__":
    main()
