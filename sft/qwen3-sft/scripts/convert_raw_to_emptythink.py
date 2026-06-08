#!/usr/bin/env python3
"""Stream-convert raw chat JSONL data to eval-aligned empty-thinking tool rows."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_jsonl_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def convert_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    stats = {
        "assistant_turns": 0,
        "trainable_targets": 0,
        "masked_assistant_turns": 0,
    }
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        stats["assistant_turns"] += 1
        if message.get("trainable") is False or message.get("loss") is False:
            stats["masked_assistant_turns"] += 1
        else:
            stats["trainable_targets"] += 1
        message["content"] = ""
        message["reasoning_content"] = "\n"
    metadata = row.setdefault("metadata", {})
    metadata["target_format"] = "emptythink_all_assistant_tool_calls"
    return row, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tag", default="emptythink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    total = {
        "files": 0,
        "rows": 0,
        "assistant_turns": 0,
        "trainable_targets": 0,
        "masked_assistant_turns": 0,
    }
    sources: list[dict[str, Any]] = []

    for input_path in iter_jsonl_files(input_root):
        relative = input_path.relative_to(input_root)
        output_path = output_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_stats = {key: 0 for key in total if key != "files"}
        with input_path.open("r", encoding="utf-8") as inp, output_path.open("w", encoding="utf-8") as out:
            for line in inp:
                if not line.strip():
                    continue
                row = json.loads(line)
                row, stats = convert_row(row)
                out.write(json_dumps(row) + "\n")
                file_stats["rows"] += 1
                for key, value in stats.items():
                    file_stats[key] += value
        if file_stats["rows"]:
            total["files"] += 1
            for key, value in file_stats.items():
                total[key] += value
            sources.append({"input_file": str(input_path), "output_file": str(output_path), **file_stats})
        else:
            output_path.unlink(missing_ok=True)

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "tag": args.tag,
        "target_format": "emptythink_all_assistant_tool_calls",
        "totals": total,
        "sources": sources,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
