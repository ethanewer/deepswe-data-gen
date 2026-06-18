#!/usr/bin/env python3
"""Materialize the exact processed v75 Qwen3-4B SFT dataset for HF upload."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qwen_agentic_sft.data import iter_jsonl_rows, json_dumps, normalize_row
from qwen_agentic_sft.online_packed_dataset import (
    apply_assistant_loss_policy,
    assistant_has_reasoning,
    assistant_has_valid_tool_calls,
    assistant_turn_action,
)


DEFAULT_INPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617/"
    "swerebench-verification-enhanced-v75-strictpassed-cap4-miniswe-aligned-spread/data"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "qwen3-4b-thinking-sft-v75-verification-strictpassed-cap4-processed"
)
DEFAULT_SOURCE_DATASET_ID = "eewer/swerebench-traces-raw-source-verification-enhanced-20260617"
DEFAULT_REPO_ID = "eewer/qwen3-4b-thinking-sft-v75-verification-strictpassed-cap4-processed"
DEFAULT_CHAT_TEMPLATE = (
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/"
    "qwen3_thinking_acc.jinja2"
)

SOURCE_MAPPING_KEYS = (
    "uuid",
    "task_id",
    "source_shard",
    "source_row_index",
    "row_source",
    "source_group",
    "language",
    "teacher",
    "difficulty",
    "agent_exit_status",
    "model_patch_bytes",
    "percent_messages_with_reasoning",
    "api_calls",
    "assistant_message_count",
    "trajectory_bytes",
    "trajectory_chars",
    "recommended_for_compaction_training",
    "quality_rule_version",
    "quality_flags",
    "failure_class",
    "compaction_original_row_id",
    "compaction_original_row_path",
    "prompt_repair_source_raw_compacted_uuid",
    "prompt_repair_source_firstturn_uuid",
    "prompt_repair_v4_uuid",
    "verification_modification_family",
    "verification_source_uuid",
    "verification_original_row_uuid",
)


class ZstdJsonlWriter:
    def __init__(self, path: Path, *, level: int):
        self.path = path
        self.level = int(level)
        self.process: subprocess.Popen[bytes] | None = None
        self.stdin: Any = None

    def __enter__(self) -> "ZstdJsonlWriter":
        self.process = subprocess.Popen(
            ["zstd", "-q", f"-{self.level}", "-T0", "-f", "-o", str(self.path), "-"],
            stdin=subprocess.PIPE,
        )
        assert self.process.stdin is not None
        self.stdin = self.process.stdin
        return self

    def write_row(self, row: dict[str, Any]) -> None:
        self.stdin.write((json_dumps(row) + "\n").encode("utf-8"))

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self.process is not None
        self.stdin.close()
        return_code = self.process.wait()
        if return_code != 0 and exc_type is None:
            raise RuntimeError(f"zstd failed while writing {self.path} with exit code {return_code}")


def source_value(example: dict[str, Any], key: str) -> Any:
    if key in example:
        return example[key]
    outcome = example.get("source_outcome")
    if isinstance(outcome, dict) and key in outcome:
        return outcome[key]
    metadata = example.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    return None


def build_output_row(
    example: dict[str, Any],
    *,
    source_dataset_id: str,
    source_training_view: str,
    training_index: int,
    training_shard: str,
    training_row_index: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "messages": example["messages"],
        "source_dataset_id": source_dataset_id,
        "source_training_view": source_training_view,
        "training_index": training_index,
        "training_shard": training_shard,
        "training_row_index": training_row_index,
        "passed": bool(source_value(example, "passed")),
    }
    tools = example.get("tools")
    if tools not in (None, "", [], {}):
        out["tools"] = tools
    reward = source_value(example, "reward")
    if reward not in (None, ""):
        out["reward"] = reward
    source = source_value(example, "source")
    if source not in (None, ""):
        out["source"] = source
    source_note = source_value(example, "source_note")
    if source_note not in (None, ""):
        out["source_note"] = source_note
    for key in SOURCE_MAPPING_KEYS:
        value = source_value(example, key)
        if value not in (None, "", [], {}):
            out["source_uuid" if key == "uuid" else key] = value
    return out


def iter_input_files(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    files = sorted(input_root.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no .jsonl shards found under {input_root}")
    return files


def update_stats(stats: dict[str, Any], example: dict[str, Any]) -> None:
    messages = example.get("messages") or []
    stats["messages"] += len(messages)
    for message in messages:
        role = message.get("role")
        stats["roles"][str(role)] += 1
        if role != "assistant":
            continue
        stats["assistant_turns"] += 1
        action = assistant_turn_action(message)
        stats["assistant_actions"][action] += 1
        if assistant_has_reasoning(message):
            stats["assistant_turns_with_reasoning"] += 1
        if assistant_has_valid_tool_calls(message):
            stats["assistant_turns_with_valid_tool_calls"] += 1
        if message.get("loss") is False:
            stats["assistant_turns_masked"] += 1
        else:
            stats["assistant_turns_trainable"] += 1


def write_readme(output_root: Path, manifest: dict[str, Any], repo_id: str) -> None:
    readme = f"""---
license: other
pretty_name: Qwen3 4B Thinking SFT v75 Processed Training View
task_categories:
- text-generation
language:
- en
tags:
- code
- software-engineering
- agent-traces
- swe-bench
- qwen3
size_categories:
- 1K<n<10K
---

# Qwen3 4B Thinking SFT v75 Processed Training View

This dataset is the processed and filtered training view used by the v75
Qwen3-4B-Thinking SFT recipe. It starts from
`{manifest["source_dataset_id"]}`, applies the v75 strict-passed cap-4
selection, mini-swe-agent alignment, lineage de-duplication, long duplicate
trajectory filtering, and the exact assistant-loss message policy used by the
training run.

Rows are compressed JSONL.zst files under `data/`. Each row contains a
top-level `messages` column, optional `tools`, and scalar source mapping fields
such as `source_uuid`, `task_id`, `source_shard`, and `source_row_index` when
available.

Important counts:

- Rows: {manifest["rows_written"]:,}
- Input training shards: {manifest["input_files"]:,}
- Assistant turns: {manifest["assistant_turns"]:,}
- Trainable assistant turns: {manifest["assistant_turns_trainable"]:,}
- Masked assistant turns: {manifest["assistant_turns_masked"]:,}

This dataset is consumed by the ms-swift recipe: convert it to swift messages
JSONL with `sft/qwen3/scripts/materialize_swift_messages_dataset.py`, then train
with `sft/qwen3/scripts/run_qwen3_4b_swift_local_h200.sh` (4B) or
`sft/qwen3/scripts/slurm_qwen3_8b_swift_2node_h200.sbatch` (8B).

Default repo id: `{repo_id}`.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def write_dataset_info(output_root: Path) -> None:
    info = {
        "configs": [
            {
                "config_name": "default",
                "data_files": [
                    {
                        "split": "train",
                        "path": "data/*.jsonl.zst",
                    }
                ],
            }
        ]
    }
    (output_root / "dataset_info.json").write_text(json_dumps(info) + "\n", encoding="utf-8")
    (output_root / ".gitattributes").write_text("*.zst filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")


def upload_folder(output_root: Path, repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for upload. Install it in a temporary "
            "environment or run without --upload."
        ) from exc
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_root),
        commit_message="Upload processed qwen3-4b v75 SFT training dataset",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    data_dir = args.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "source_dataset_id": args.source_dataset_id,
        "source_training_view": str(args.input_root),
        "chat_template_path": args.chat_template_path,
        "rows_seen": 0,
        "rows_written": 0,
        "rows_dropped_normalization": 0,
        "input_files": 0,
        "output_files": 0,
        "messages": 0,
        "assistant_turns": 0,
        "assistant_turns_with_reasoning": 0,
        "assistant_turns_with_valid_tool_calls": 0,
        "assistant_turns_trainable": 0,
        "assistant_turns_masked": 0,
        "roles": Counter(),
        "assistant_actions": Counter(),
        "policy": {
            "require_assistant_reasoning_for_loss": True,
            "require_assistant_tool_calls_for_loss": True,
            "drop_assistant_content_for_tool_calls": True,
            "assistant_loss_target": "assistant",
            "mask_tool_call_error_recovery": False,
            "mask_manual_patch_artifact_turns": True,
            "enable_turn_loss_weights": False,
            "mask_nonpassing_submit_turns": False,
            "mask_empty_patch_submit_turns": True,
        },
    }

    training_index = 0
    for input_file in iter_input_files(args.input_root):
        stats["input_files"] += 1
        output_file = data_dir / f"{input_file.stem}.jsonl.zst"
        stats["output_files"] += 1
        with ZstdJsonlWriter(output_file, level=args.zstd_level) as writer:
            for training_row_index, raw_row in enumerate(iter_jsonl_rows(input_file)):
                stats["rows_seen"] += 1
                example = normalize_row(raw_row)
                if example is None:
                    stats["rows_dropped_normalization"] += 1
                    continue
                example = apply_assistant_loss_policy(
                    example,
                    require_assistant_reasoning_for_loss=True,
                    require_assistant_tool_calls_for_loss=True,
                    drop_assistant_content_for_tool_calls=True,
                    mask_tool_call_error_recovery=False,
                    mask_manual_patch_artifact_turns=True,
                    enable_turn_loss_weights=False,
                    mask_nonpassing_submit_turns=False,
                    mask_empty_patch_submit_turns=True,
                )
                row = build_output_row(
                    example,
                    source_dataset_id=args.source_dataset_id,
                    source_training_view=str(args.input_root),
                    training_index=training_index,
                    training_shard=input_file.name,
                    training_row_index=training_row_index,
                )
                update_stats(stats, row)
                writer.write_row(row)
                stats["rows_written"] += 1
                training_index += 1

    stats["roles"] = dict(stats["roles"])
    stats["assistant_actions"] = dict(stats["assistant_actions"])
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json_dumps(stats) + "\n", encoding="utf-8")
    write_dataset_info(args.output_root)
    write_readme(args.output_root, stats, args.repo_id)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-dataset-id", default=DEFAULT_SOURCE_DATASET_ID)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--chat-template-path", default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--zstd-level", type=int, default=19)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("zstd") is None:
        raise SystemExit("zstd CLI is required to write compressed JSONL.zst shards")
    stats = build(args)
    print(
        "materialized processed dataset: "
        f"rows={stats['rows_written']} files={stats['output_files']} "
        f"masked_assistant_turns={stats['assistant_turns_masked']} "
        f"output={args.output_root}"
    )
    if args.upload:
        upload_folder(args.output_root, args.repo_id, private=args.private)
        print(f"uploaded dataset to hf://datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
