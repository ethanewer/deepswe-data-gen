#!/usr/bin/env python3
"""Materialize a SWIFT-friendly JSONL view of processed SWE SFT messages.

The uploaded v75 dataset stores assistant tool calls in the OpenAI-style
``message["tool_calls"]`` field. MS-SWIFT's standard messages preprocessor keeps
only role/content/loss inside each message, so feeding those rows directly would
drop every assistant tool call. This script serializes assistant tool calls into
the assistant content using the Qwen/Hermes ``<tool_call>`` format before SWIFT
does its normal packed SFT preprocessing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


def _iter_jsonl_zst(path: Path) -> Iterable[dict[str, Any]]:
    proc = subprocess.Popen(
        ["zstdcat", str(path)],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.strip()
            if line:
                yield json.loads(line)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        status = proc.wait()
        # A negative SIGPIPE status is expected when callers intentionally stop
        # early with --limit.
        if status not in (0, -13):
            raise RuntimeError(f"zstdcat failed for {path} with status {status}")


def _normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict) and "function" in tool_call:
        tool_call = tool_call["function"]
    if not isinstance(tool_call, dict):
        raise ValueError(f"tool_call is not a dict: {tool_call!r}")

    name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    if not name:
        raise ValueError(f"tool_call missing name: {tool_call!r}")
    return {"name": name, "arguments": arguments}


def _serialize_tool_calls(tool_calls: list[Any]) -> str:
    rendered = []
    for tool_call in tool_calls:
        payload = _normalize_tool_call(tool_call)
        arguments_json = json.dumps(payload["arguments"], ensure_ascii=False)
        rendered.append(
            "<tool_call>\n"
            + '{"name": '
            + json.dumps(str(payload["name"]), ensure_ascii=False)
            + ', "arguments": '
            + arguments_json
            + "}"
            + "\n</tool_call>"
        )
    return "\n".join(rendered)


def _assistant_content(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content")
    if reasoning and "</think>" not in content:
        content = f"<think>\n{str(reasoning).strip()}\n</think>\n\n{content.lstrip()}"

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        rendered_tool_calls = _serialize_tool_calls(tool_calls)
        if content:
            content = content.rstrip() + "\n" + rendered_tool_calls
        else:
            content = rendered_tool_calls
    return content


def _convert_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, int]]:
    messages = []
    stats = {
        "assistant_messages": 0,
        "assistant_messages_with_tool_calls": 0,
        "tool_calls_serialized": 0,
        "tool_messages": 0,
        "tool_messages_rendered_as_user": 0,
        "trailing_messages_stripped": 0,
        "rows_dropped_no_assistant_target": 0,
    }
    for message in row["messages"]:
        role = message.get("role")
        if role == "assistant":
            stats["assistant_messages"] += 1
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                stats["assistant_messages_with_tool_calls"] += 1
                stats["tool_calls_serialized"] += len(tool_calls)
            messages.append({"role": "assistant", "content": _assistant_content(message)})
        elif role == "tool":
            stats["tool_messages"] += 1
            stats["tool_messages_rendered_as_user"] += 1
            tool_block = f"<tool_response>\n{message.get('content') or ''}\n</tool_response>"
            if messages and messages[-1]["role"] == "user" and messages[-1]["content"].startswith("<tool_response>"):
                messages[-1]["content"] += "\n" + tool_block
            else:
                messages.append({"role": "user", "content": tool_block})
        elif role in {"system", "user"}:
            messages.append({"role": role, "content": message.get("content") or ""})
        else:
            raise ValueError(f"unsupported role {role!r}")

    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
        stats["trailing_messages_stripped"] += 1
    if not messages:
        stats["rows_dropped_no_assistant_target"] = 1
        return None, stats

    out: dict[str, Any] = {"messages": messages}
    if row.get("tools") is not None:
        out["tools"] = row["tools"]
    for key in [
        "source_dataset_id",
        "source_uuid",
        "task_id",
        "training_index",
        "training_shard",
        "training_row_index",
        "passed",
        "language",
        "teacher",
        "row_source",
    ]:
        if key in row:
            out[key] = row[key]
    return out, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="train.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.input_root / "data"
    input_files = sorted(data_dir.glob("*.jsonl.zst"))
    if not input_files:
        raise SystemExit(f"No .jsonl.zst shards under {data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    stats = {
        "rows_written": 0,
        "rows_dropped_no_assistant_target": 0,
        "input_files": len(input_files),
        "assistant_messages": 0,
        "assistant_messages_with_tool_calls": 0,
        "tool_calls_serialized": 0,
        "tool_messages": 0,
        "tool_messages_rendered_as_user": 0,
        "trailing_messages_stripped": 0,
    }
    with output_path.open("w", encoding="utf-8") as out:
        for input_file in input_files:
            for row in _iter_jsonl_zst(input_file):
                converted, row_stats = _convert_row(row)
                for key, value in row_stats.items():
                    stats[key] += value
                if converted is None:
                    continue
                out.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["rows_written"] += 1
                if args.limit and stats["rows_written"] >= args.limit:
                    break
            if args.limit and stats["rows_written"] >= args.limit:
                break

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.input_root),
                "output": str(output_path),
                "format": "ms-swift messages jsonl",
                "conversion": "assistant tool_calls serialized into assistant content",
                **stats,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
