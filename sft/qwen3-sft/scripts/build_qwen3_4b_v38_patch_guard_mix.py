#!/usr/bin/env python3
"""Build the qwen3-4b v38 patch-guard continuation mix.

The v28/v37 checkpoints usually emit valid mini-swe-agent bash calls, but they
often submit immediately after ``git diff ... > patch.txt``. This mix keeps the
best v28 passed-prefix data and adds targeted rows that teach:

1. after writing patch.txt, explicitly check that it is non-empty and inspect it;
2. if that check fails, inspect/recover instead of submitting.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TextIO


DEFAULT_PREFIX_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "highquality-1x-duplicate-reasoning-90pct-30k-full-miniswe-aligned-passed-prefix-weighted-v2/data"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "qwen3-4b-thinking-v38-v28-prefix-patchguard-mix"
)

BASH_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the shell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
]

PATCH_GUARD_COMMAND = "test -s patch.txt && sed -n '1,240p' patch.txt"
PATCH_GUARD_FAILURE = "<returncode>1</returncode>\n<output>\n</output>"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def jsonl_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def iter_lines(files: Iterable[Path]) -> Iterable[tuple[Path, int, str]]:
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield path, line_number, line


def assistant_command(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls or not isinstance(calls[0], dict):
        return ""
    function = calls[0].get("function") or {}
    args = function.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    if not isinstance(args, dict):
        return ""
    return str(args.get("command") or args.get("cmd") or "")


def assistant_target_index(messages: list[dict[str, Any]]) -> int | None:
    target = None
    for index, message in enumerate(messages):
        if (
            message.get("role") == "assistant"
            and message.get("trainable") is not False
            and message.get("loss") is not False
        ):
            target = index
    return target


def previous_assistant_index(messages: list[dict[str, Any]], before_index: int) -> int | None:
    for index in range(before_index - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    return None


def is_submit_command(command: str) -> bool:
    return "complete_task_and_submit_final_output" in command.lower()


def is_diff_to_patch(command: str) -> bool:
    text = command.lower()
    return "git diff" in text and "patch.txt" in text and (">" in text or "| tee" in text)


def leading_cd_prefix(command: str) -> str:
    match = re.match(r"\s*cd\s+([^;&|\n]+)\s*&&\s*", command)
    if not match:
        return ""
    return f"cd {match.group(1).strip()} && "


def patch_guard_command(diff_command: str) -> str:
    return f"{leading_cd_prefix(diff_command)}{PATCH_GUARD_COMMAND}"


def already_displays_patch(command: str) -> bool:
    text = command.lower()
    if "patch.txt" not in text:
        return False
    return bool(re.search(r"(^|[;&|]\s*)(cat|sed|head|tail|grep)\b[^;&|]*patch\.txt", text))


def extract_diff_paths(command: str) -> list[str]:
    before_redirect = re.split(
        r">\s*(?:/testbed/|\./)?patch\.txt|\|\s*tee\s+(?:-a\s+)?(?:/testbed/|\./)?patch\.txt",
        command,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    if "--" in before_redirect:
        after_marker = before_redirect.split("--", 1)[1]
    else:
        parts = re.split(r"\bgit\s+diff\b", before_redirect, flags=re.IGNORECASE)
        after_marker = parts[-1] if parts else before_redirect
    after_marker = re.split(r"&&|\|\||;", after_marker, maxsplit=1)[0]
    try:
        tokens = shlex.split(after_marker)
    except ValueError:
        tokens = after_marker.split()
    paths: list[str] = []
    for token in tokens:
        if not token or token.startswith("-"):
            continue
        if token in {"git", "diff", "patch.txt"}:
            continue
        if token.endswith("patch.txt"):
            continue
        paths.append(token)
    return paths[:4]


def recovery_command(diff_command: str) -> str:
    prefix = leading_cd_prefix(diff_command)
    paths = extract_diff_paths(diff_command)
    if paths:
        quoted = " ".join(shlex.quote(path) for path in paths)
        first = shlex.quote(paths[0])
        return f"{prefix}git status --short && git diff --stat && git diff -- {quoted} && sed -n '1,220p' {first}"
    return f"{prefix}git status --short && git diff --stat && find . -maxdepth 3 -type f | sed -n '1,160p'"


def mark_prefix_untrainable(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        if message.get("role") == "assistant":
            message["trainable"] = False
            message["loss"] = False


def bash_assistant(command: str, reasoning: str, *, trainable: bool) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "reasoning_content": f"\n{reasoning}\n",
        "tool_calls": [{"function": {"name": "bash", "arguments": {"command": command}}}],
    }
    if not trainable:
        message["trainable"] = False
        message["loss"] = False
    return message


def tool_output(content: str) -> dict[str, str]:
    return {"role": "tool", "content": content}


def build_guard_rows(row: dict[str, Any], *, line_key: tuple[str, int, int, str], copies: int) -> list[dict[str, Any]]:
    messages = row.get("messages") or []
    target_index = assistant_target_index(messages)
    if target_index is None:
        return []
    previous_index = previous_assistant_index(messages, target_index)
    if previous_index is None:
        return []
    previous_command = assistant_command(messages[previous_index])
    if not is_diff_to_patch(previous_command) or already_displays_patch(previous_command):
        return []
    target_command = assistant_command(messages[target_index])
    guard = patch_guard_command(previous_command)
    if target_command.strip() == guard:
        return []

    prefix = copy.deepcopy(messages[:target_index])
    mark_prefix_untrainable(prefix)
    tools = copy.deepcopy(row.get("tools") or BASH_TOOL)
    recovery = recovery_command(previous_command)
    rows: list[dict[str, Any]] = []
    for copy_index in range(copies):
        metadata = {
            "source": "v38_patch_guard_after_diff",
            "line_key": list(line_key),
            "copy_index": copy_index,
            "previous_diff_command": previous_command,
            "original_target_command": target_command,
            "target_command": guard,
            "target_format": "qwen3_thinking_acc_miniswe_tool_call",
        }
        rows.append(
            {
                "messages": copy.deepcopy(prefix)
                + [
                    bash_assistant(
                        guard,
                        "Before submitting, I need to confirm patch.txt is non-empty and inspect the actual diff.",
                        trainable=True,
                    )
                ],
                "tools": copy.deepcopy(tools),
                "metadata": metadata,
            }
        )
        rows.append(
            {
                "messages": copy.deepcopy(prefix)
                + [
                    bash_assistant(
                        guard,
                        "Before submitting, I need to confirm patch.txt is non-empty and inspect the actual diff.",
                        trainable=False,
                    ),
                    tool_output(PATCH_GUARD_FAILURE),
                    bash_assistant(
                        recovery,
                        "patch.txt is empty, so I need to inspect the working tree and source instead of submitting.",
                        trainable=True,
                    ),
                ],
                "tools": copy.deepcopy(tools),
                "metadata": {
                    **metadata,
                    "source": "v38_patch_guard_empty_recovery",
                    "target_command": recovery,
                    "failed_guard_command": guard,
                },
            }
        )
    return rows


def write_line(handles: list[TextIO], line: str, row_index: int) -> int:
    handles[row_index % len(handles)].write(line)
    return row_index + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-root", type=Path, default=DEFAULT_PREFIX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--guard-copies", type=int, default=8)
    parser.add_argument("--prefix-per-guard-burst", type=int, default=8)
    parser.add_argument("--shards", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prefix_files = jsonl_files(args.prefix_root)
    if not prefix_files:
        raise FileNotFoundError(f"no prefix JSONL files under {args.prefix_root}")
    if args.guard_copies < 0:
        raise ValueError("--guard-copies must be non-negative")
    if args.prefix_per_guard_burst < 1:
        raise ValueError("--prefix-per-guard-burst must be positive")
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    if args.max_rows < 0:
        raise ValueError("--max-rows must be non-negative")

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    mixed_dir = args.output_root / "mixed"
    mixed_dir.mkdir(parents=True, exist_ok=True)

    handles = [(mixed_dir / f"shard-{index:03d}.jsonl").open("w", encoding="utf-8") for index in range(args.shards)]
    counters: Counter[str] = Counter()
    total_rows = 0
    pending_guard_lines: list[str] = []
    seen_guard_prefixes: set[tuple[str, int, int, str]] = set()

    try:
        since_guard = 0
        for path, line_number, line in iter_lines(prefix_files):
            if args.max_rows and counters["base_prefix_rows"] >= args.max_rows:
                break
            total_rows = write_line(handles, line, total_rows)
            counters["base_prefix_rows"] += 1
            since_guard += 1

            row = json.loads(line)
            metadata = row.get("metadata") or {}
            target_turn = int(metadata.get("target_assistant_turn") or -1)
            target_index = assistant_target_index(row.get("messages") or [])
            previous_index = previous_assistant_index(row.get("messages") or [], target_index or 0)
            previous_command = ""
            if previous_index is not None:
                previous_command = assistant_command(row["messages"][previous_index])
            line_key = (
                str(metadata.get("source_file") or path),
                int(metadata.get("source_line") or line_number),
                target_turn - 1,
                previous_command,
            )
            if line_key not in seen_guard_prefixes:
                guard_rows = build_guard_rows(row, line_key=line_key, copies=args.guard_copies)
                if guard_rows:
                    seen_guard_prefixes.add(line_key)
                    counters["unique_guard_prefixes"] += 1
                    for guard_row in guard_rows:
                        source = str(guard_row.get("metadata", {}).get("source") or "guard")
                        counters[source] += 1
                        pending_guard_lines.append(json_dumps(guard_row) + "\n")

            if since_guard >= args.prefix_per_guard_burst and pending_guard_lines:
                guard_line = pending_guard_lines.pop(0)
                total_rows = write_line(handles, guard_line, total_rows)
                counters["guard_rows_written"] += 1
                since_guard = 0

        while pending_guard_lines:
            guard_line = pending_guard_lines.pop(0)
            total_rows = write_line(handles, guard_line, total_rows)
            counters["guard_rows_written"] += 1
    finally:
        for handle in handles:
            handle.close()

    nonempty = 0
    total_bytes = 0
    for path in mixed_dir.glob("shard-*.jsonl"):
        size = path.stat().st_size
        if size == 0:
            path.unlink()
        else:
            nonempty += 1
            total_bytes += size

    manifest = {
        "output_root": str(args.output_root),
        "prefix_root": str(args.prefix_root),
        "selection": (
            "v38: best v28 passed-prefix rows plus deduplicated patch guard rows. "
            "The added targets teach test -s/sed inspection after git diff writes "
            "patch.txt, and source inspection/recovery after an empty patch guard."
        ),
        "guard_copies": args.guard_copies,
        "prefix_per_guard_burst": args.prefix_per_guard_burst,
        "rows": {"total": total_rows, **dict(counters)},
        "shards": nonempty,
        "bytes": total_bytes,
        "rough_tokens_bytes_div4": total_bytes // 4,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
