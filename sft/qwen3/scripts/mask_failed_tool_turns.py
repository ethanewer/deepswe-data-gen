#!/usr/bin/env python3
"""Mask assistant turns whose tool call immediately produced a failed returncode.

Input and output are ms-swift ``messages`` JSONL files. The script preserves row
metadata and message content, adding ``"loss": false`` only to an assistant
message followed by a rendered ``<tool_response>`` user message containing a
nonzero ``<returncode>``. ms-swift's default messages preprocessor honors this
per-message loss key without extra training flags.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RETURNCODE_RE = re.compile(r"<returncode>\s*([+-]?\d+)\s*</returncode>")


def _has_failed_returncode(content: str) -> bool:
    if "<tool_response>" not in content:
        return False
    for match in RETURNCODE_RE.finditer(content):
        try:
            if int(match.group(1)) != 0:
                return True
        except ValueError:
            return True
    return False


def _mask_row(row: dict[str, Any]) -> tuple[dict[str, Any], int]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("row missing messages list")

    masked = 0
    new_messages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} is not an object")
        new_message = dict(message)
        next_message = messages[index + 1] if index + 1 < len(messages) else None
        if (
            message.get("role") == "assistant"
            and isinstance(next_message, dict)
            and next_message.get("role") == "user"
            and _has_failed_returncode(str(next_message.get("content") or ""))
        ):
            if new_message.get("loss") is not False:
                masked += 1
            new_message["loss"] = False
        new_messages.append(new_message)

    out = dict(row)
    out["messages"] = new_messages
    return out, masked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input train.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="train.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name

    rows = 0
    rows_with_masks = 0
    assistant_turns_masked = 0
    with args.input.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                masked_row, masked = _mask_row(row)
            except Exception as exc:  # noqa: BLE001 - include row number for data prep failures.
                raise RuntimeError(f"failed to process {args.input}:{line_number}") from exc
            dst.write(json.dumps(masked_row, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows += 1
            if masked:
                rows_with_masks += 1
                assistant_turns_masked += masked

    manifest = {
        "source": str(args.input),
        "output": str(output_path),
        "format": "ms-swift messages jsonl",
        "policy": "loss_false_on_assistant_turn_immediately_before_nonzero_tool_returncode",
        "rows_written": rows,
        "rows_with_masks": rows_with_masks,
        "assistant_turns_masked": assistant_turns_masked,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
