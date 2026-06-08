#!/usr/bin/env python3
"""Build patch-recovery rows with exact mini-swe format-error prefixes.

The v26 recovery rows teach apply/diff/submit targets, but the real mini-swe
smoke often reaches those states only after one or more "No tool calls found"
format-error user messages. This builder duplicates each v26 row with repeated
format-error messages inserted immediately before the trainable assistant
target, while preserving the eval-aligned empty-think assistant format.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


MINI_SWE_FORMAT_ERROR = """Tool call error:

<error>
No tool calls found in the response. Every response MUST include at least one tool call.
</error>

Here is general guidance on how to submit correct toolcalls:

Every response needs to use the 'bash' tool at least once to execute commands.

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {"command": "your_command_here"}

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that).
"""


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_rows(root: Path):
    for path in sorted(root.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if line.strip():
                    yield path, line_number, json.loads(line)


def target_index(messages: list[dict[str, Any]]) -> int | None:
    targets = [
        idx
        for idx, message in enumerate(messages)
        if message.get("role") == "assistant"
        and message.get("trainable") is not False
        and message.get("loss") is not False
    ]
    return targets[-1] if targets else None


def command_from_assistant(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls:
        return ""
    function = calls[0].get("function", {}) if isinstance(calls[0], dict) else {}
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    return str(args.get("command") or "") if isinstance(args, dict) else ""


def command_category(command: str) -> str:
    text = command.strip().lower()
    if not text:
        return "none"
    if "complete_task_and_submit_final_output" in text:
        return "submit"
    if "git diff" in text:
        return "diff"
    if "apply_patch" in text or "python" in text and (".replace(" in text or "write_text" in text):
        return "edit"
    if "grep" in text or text.startswith("find "):
        return "search"
    if text.startswith("cat ") or text.startswith("sed ") or text.startswith("head ") or text.startswith("tail "):
        return "view"
    return "other"


def normalize_assistant(message: dict[str, Any]) -> None:
    message["content"] = ""
    message["reasoning_content"] = "\n"


def make_variant(row: dict[str, Any], *, error_count: int, assistant_format: str) -> dict[str, Any]:
    emitted = copy.deepcopy(row)
    messages = emitted.get("messages") or []
    idx = target_index(messages)
    if idx is None:
        raise ValueError("row has no trainable assistant target")

    for message in messages:
        if message.get("role") == "assistant" and assistant_format == "emptythink_all":
            normalize_assistant(message)

    if error_count:
        errors = [{"role": "user", "content": MINI_SWE_FORMAT_ERROR} for _ in range(error_count)]
        emitted["messages"] = messages[:idx] + errors + messages[idx:]

    metadata = emitted.setdefault("metadata", {})
    command = command_from_assistant(messages[idx])
    metadata["v27_format_error_recovery"] = {
        "error_count": error_count,
        "target_command": command,
        "target_category": command_category(command),
    }
    metadata["target_format"] = f"{assistant_format}_assistant_tool_calls_with_format_error_prefix"
    return emitted


def parse_error_counts(value: str) -> list[int]:
    counts = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        count = int(item)
        if count < 0:
            raise argparse.ArgumentTypeError("error counts must be non-negative")
        counts.append(count)
    if not counts:
        raise argparse.ArgumentTypeError("at least one error count is required")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--error-counts", type=parse_error_counts, default=parse_error_counts("0,1,3,6"))
    parser.add_argument("--assistant-format", choices=("emptythink_all", "visible"), default="emptythink_all")
    parser.add_argument("--seed", type=int, default=60609)
    parser.add_argument("--max-input-rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    out_dir = args.output_root / "format_error_patch_recovery_v27"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    stats = Counter()
    input_rows = 0
    for path, line_number, row in iter_rows(args.input_root):
        input_rows += 1
        if args.max_input_rows and input_rows > args.max_input_rows:
            break
        for error_count in args.error_counts:
            try:
                emitted = make_variant(row, error_count=error_count, assistant_format=args.assistant_format)
            except ValueError:
                stats["dropped_missing_target"] += 1
                continue
            meta = emitted.get("metadata") or {}
            target_kind = str(meta.get("target_kind", "unknown"))
            target_category = meta["v27_format_error_recovery"]["target_category"]
            stats[f"target_kind:{target_kind}"] += 1
            stats[f"target_category:{target_category}"] += 1
            stats[f"error_count:{error_count}"] += 1
            meta["v27_format_error_recovery"]["source_path"] = str(path)
            meta["v27_format_error_recovery"]["source_line"] = line_number
            rows.append(emitted)

    random.Random(args.seed).shuffle(rows)
    with (out_dir / "data.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json_dumps(row) + "\n")

    manifest = {
        "output_root": str(args.output_root),
        "input_root": str(args.input_root),
        "input_rows": input_rows,
        "rows": len(rows),
        "error_counts": args.error_counts,
        "assistant_format": args.assistant_format,
        "stats": dict(sorted(stats.items())),
        "selection": "v26 eval-aligned patch recovery rows with exact mini-swe no-tool format-error messages inserted before the trainable target",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
