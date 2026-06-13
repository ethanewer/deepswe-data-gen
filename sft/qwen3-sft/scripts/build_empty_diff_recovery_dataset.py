#!/usr/bin/env python3
"""Build recovery rows for empty patch.txt states from mini-swe trajectories.

These rows target the failure mode where the model writes `git diff ... >
patch.txt`, observes no patch content, and then submits the empty file. The
target action is deliberately an inspection command, not another patch write or
submit, so the model learns to recover before final submission.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import shlex
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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

NO_TOOL_ERROR_MARKER = "No tool calls found in the response"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_trajs(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file() and root.name.endswith(".traj.json"):
            yield root
        elif root.is_dir():
            yield from sorted(root.rglob("*.traj.json"))


def assistant_command(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls or not isinstance(calls[0], dict):
        return ""
    function = calls[0].get("function", {})
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    if not isinstance(args, dict):
        return ""
    return str(args.get("command") or "")


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    emitted: dict[str, Any] = {"role": role}
    if "content" in message:
        emitted["content"] = message.get("content") or ""
    if role == "assistant":
        emitted["content"] = message.get("content") or ""
        if "reasoning_content" in message:
            emitted["reasoning_content"] = message.get("reasoning_content") or "\n"
        if message.get("tool_calls"):
            emitted["tool_calls"] = copy.deepcopy(message["tool_calls"])
        emitted["trainable"] = False
        emitted["loss"] = False
    if role == "tool" and "tool_call_id" in message:
        emitted["tool_call_id"] = message["tool_call_id"]
    return emitted


def bash_assistant(command: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": "\npatch.txt is empty, so I need to inspect the working tree and source before submitting.\n",
        "tool_calls": [{"function": {"name": "bash", "arguments": {"command": command}}}],
    }


def empty_tool_output(message: dict[str, Any]) -> bool:
    if message.get("role") != "tool":
        return False
    content = message.get("content") or ""
    if "<returncode>0</returncode>" not in content:
        return False
    match = re.search(r"<output>\s*(.*?)\s*</output>", content, flags=re.DOTALL)
    if not match:
        return False
    return not match.group(1).strip()


def diff_to_patch_command(command: str) -> bool:
    text = command.lower()
    return "git diff" in text and "patch.txt" in text and ">" in text


def extract_diff_paths(command: str) -> list[str]:
    before_redirect = re.split(r">\s*patch\.txt|\|\s*tee\s+patch\.txt", command, maxsplit=1, flags=re.IGNORECASE)[0]
    after_marker = before_redirect
    if "--" in before_redirect:
        after_marker = before_redirect.split("--", 1)[1]
    else:
        after_marker = before_redirect.split("git diff", 1)[-1]
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
    return paths[:3]


def recovery_command(command: str) -> str:
    paths = extract_diff_paths(command)
    if paths:
        quoted = " ".join(shlex.quote(path) for path in paths)
        first = shlex.quote(paths[0])
        return f"git status --short && git diff --stat && git diff -- {quoted} && sed -n '1,240p' {first}"
    return "git status --short && git diff --stat && find . -maxdepth 3 -type f | sed -n '1,160p'"


def no_tool_error(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and NO_TOOL_ERROR_MARKER in (message.get("content") or "")


def build_rows_from_traj(path: Path, copies: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    messages = data.get("messages") or []
    instance_id = data.get("instance_id") or path.parent.name
    rows: list[dict[str, Any]] = []
    seen_prefixes: set[tuple[int, str]] = set()

    for idx, message in enumerate(messages[:-1]):
        if message.get("role") != "assistant":
            continue
        command = assistant_command(message)
        if not diff_to_patch_command(command):
            continue
        if not empty_tool_output(messages[idx + 1]):
            continue

        prefix_end = idx + 2
        if prefix_end < len(messages) and no_tool_error(messages[prefix_end]):
            prefix_end += 1
        target_command = recovery_command(command)
        key = (prefix_end, target_command)
        if key in seen_prefixes:
            continue
        seen_prefixes.add(key)

        prefix = [normalize_message(item) for item in messages[:prefix_end]]
        for copy_index in range(copies):
            rows.append(
                {
                    "messages": copy.deepcopy(prefix) + [bash_assistant(target_command)],
                    "tools": BASH_TOOL,
                    "metadata": {
                        "source": "current_empty_diff_recovery",
                        "instance_id": instance_id,
                        "traj_path": str(path),
                        "prefix_end": prefix_end,
                        "copy_index": copy_index,
                        "empty_diff_command": command,
                        "target_command": target_command,
                        "target_format": "eval_prefix_visible_reasoning_tool_call",
                    },
                }
            )
    return rows


def write_shards(rows: list[dict[str, Any]], output_dir: Path, shards: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = [(output_dir / f"shard-{idx:03d}.jsonl").open("w", encoding="utf-8") for idx in range(shards)]
    try:
        for idx, row in enumerate(rows):
            handles[idx % shards].write(json_dumps(row) + "\n")
    finally:
        for handle in handles:
            handle.close()
    for path in output_dir.glob("shard-*.jsonl"):
        if path.stat().st_size == 0:
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=50)
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--seed", type=int, default=52052)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    traj_count = 0
    rows_by_instance: Counter[str] = Counter()
    for traj in iter_trajs(args.traj_root):
        traj_count += 1
        built = build_rows_from_traj(traj, args.copies)
        rows.extend(built)
        for row in built:
            rows_by_instance[str(row["metadata"]["instance_id"])] += 1

    random.Random(args.seed).shuffle(rows)
    out_dir = args.output_root / "empty_diff_recovery_v1"
    write_shards(rows, out_dir, args.shards)
    manifest = {
        "output_root": str(args.output_root),
        "traj_roots": [str(path) for path in args.traj_root],
        "traj_count": traj_count,
        "rows": len(rows),
        "copies": args.copies,
        "shards": args.shards,
        "instances": len(rows_by_instance),
        "rows_by_instance": dict(rows_by_instance.most_common()),
        "selection": "Failed mini-swe prefixes where git diff wrote an empty patch.txt; target is inspect/recovery, never submit.",
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
