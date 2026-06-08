#!/usr/bin/env python3
"""Build prefix-target SFT rows from normalized chat trajectories.

Each emitted row contains one target assistant turn. Earlier assistant turns
remain in the prompt as eval-style history and are masked from loss with
``trainable: false``.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_jsonl_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.jsonl"))


def source_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


def assistant_indices(messages: list[dict[str, Any]]) -> list[int]:
    return [idx for idx, message in enumerate(messages) if message.get("role") == "assistant"]


def normalize_prefix_messages(messages: list[dict[str, Any]], target_idx: int) -> list[dict[str, Any]]:
    prefix = copy.deepcopy(messages[: target_idx + 1])
    for idx, message in enumerate(prefix):
        if message.get("role") != "assistant":
            continue
        if idx == target_idx:
            message["reasoning_content"] = message.get("reasoning_content") or "\n"
            message.pop("trainable", None)
            message.pop("loss", None)
        else:
            message.pop("reasoning_content", None)
            message["trainable"] = False
    return prefix


def convert_source(source: Path, output_source: Path, max_assistant_turns: int) -> dict[str, int | str]:
    output_source.mkdir(parents=True, exist_ok=True)
    output_path = output_source / "data.jsonl"
    rows_in = 0
    rows_out = 0
    assistant_targets = 0
    with output_path.open("w", encoding="utf-8") as out:
        for path in iter_jsonl_files(source):
            with path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    rows_in += 1
                    row = json.loads(line)
                    messages = row.get("messages") or []
                    targets = assistant_indices(messages)
                    if max_assistant_turns > 0:
                        targets = targets[:max_assistant_turns]
                    for turn_number, target_idx in enumerate(targets):
                        emitted = {
                            "messages": normalize_prefix_messages(messages, target_idx),
                            "metadata": {
                                "source_file": str(path),
                                "source_line": line_number,
                                "target_assistant_turn": turn_number,
                                "target_message_index": target_idx,
                            },
                        }
                        if "tools" in row:
                            emitted["tools"] = row["tools"]
                        out.write(json_dumps(emitted) + "\n")
                        rows_out += 1
                        assistant_targets += 1
    return {
        "name": source.name,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "assistant_targets": assistant_targets,
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-assistant-turns", type=int, default=8)
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
        convert_source(source, args.output_root / source.name.replace("_v12", "_prefix_v13"), args.max_assistant_turns)
        for source in source_dirs(args.input_root)
    ]
    manifest = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "max_assistant_turns": args.max_assistant_turns,
        "rows_in": sum(int(item["rows_in"]) for item in summaries),
        "rows_out": sum(int(item["rows_out"]) for item in summaries),
        "sources": summaries,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
