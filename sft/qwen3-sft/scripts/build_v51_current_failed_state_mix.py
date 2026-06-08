#!/usr/bin/env python3
"""Build exact current failed-state correction rows for the v50 smoke loops."""

from __future__ import annotations

import argparse
import copy
import json
import random
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
                "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
]

TARGETS = {
    "apache__lucene-12196": {
        "recover": "find /testbed -type f -name \"MultiFieldQueryParser.java\" | head -20",
        "after_find": "sed -n '1,180p' /testbed/lucene/queryparser/src/java/org/apache/lucene/queryparser/classic/MultiFieldQueryParser.java",
        "find_output": "/testbed/lucene/queryparser/src/java/org/apache/lucene/queryparser/classic/MultiFieldQueryParser.java",
    },
    "redis__redis-10764": {
        "recover": "sed -n '3980,4100p' /testbed/src/t_zset.c",
        "after_find": "grep -n \"bzmpop\" /testbed/src/t_zset.c | head -20",
        "find_output": "<returncode>0</returncode>\n<output>\nvoid bzmpopCommand(client *c) {\n</output>",
    },
}


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_jsonl_files(root: Path) -> Iterable[Path]:
    yield from sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def align_emptythink(row: dict[str, Any]) -> dict[str, Any]:
    emitted = copy.deepcopy(row)
    emitted["tools"] = BASH_TOOL
    for message in emitted.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        message["content"] = ""
        message["reasoning_content"] = "\n"
    emitted.setdefault("metadata", {})["target_format"] = "emptythink_all_assistant_tool_calls"
    return emitted


def bash_assistant(command: str, *, trainable: bool) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "\n",
        "tool_calls": [{"function": {"name": "bash", "arguments": {"command": command}}}],
    }
    if not trainable:
        message["trainable"] = False
        message["loss"] = False
    return message


def tool_output(content: str) -> dict[str, str]:
    if content.startswith("<returncode>"):
        return {"role": "tool", "content": content}
    return {"role": "tool", "content": f"<returncode>0</returncode>\n<output>\n{content}\n</output>"}


def command_from_assistant(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls:
        return ""
    args = calls[0].get("function", {}).get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    return str(args.get("command") or "") if isinstance(args, dict) else ""


def target_category(command: str) -> str:
    text = command.lower()
    if "complete_task" in text:
        return "submit"
    if "git diff" in text:
        return "diff"
    if any(marker in text for marker in ("apply_patch", "git apply", "patch.txt", "sed -i", "perl -i", "cat >")):
        return "edit"
    if any(marker in text for marker in ("find ", "grep", "sed -n", "cat ", "head ", "tail ")):
        return "inspect"
    return "other"


def load_base_rows(root: Path, max_rows: int, seed: int) -> list[dict[str, Any]]:
    rows: list[tuple[float, dict[str, Any]]] = []
    rng = random.Random(seed)
    for path in iter_jsonl_files(root):
        for line_number, line in enumerate(path.open("r", encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = align_emptythink(json.loads(line))
            row.setdefault("metadata", {})["v51_mix"] = {
                "bucket": "base_v47",
                "source_file": str(path),
                "source_line": line_number,
            }
            rows.append((rng.random(), row))
    rows.sort(key=lambda item: item[0])
    return [row for _, row in rows[:max_rows]]


def failed_prefix_rows(traj_root: Path, copies: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id, target_spec in TARGETS.items():
        traj_path = traj_root / instance_id / f"{instance_id}.traj.json"
        data = json.loads(traj_path.read_text(encoding="utf-8"))
        messages = data["messages"]
        for repeat_count in (1, 2, 4, 8, 16):
            prefix = copy.deepcopy(messages[: 2 + (2 * repeat_count)])
            for message in prefix:
                if message.get("role") == "assistant":
                    message["content"] = ""
                    message["reasoning_content"] = "\n"
                    message["trainable"] = False
                    message["loss"] = False
            for copy_index in range(copies):
                row = {
                    "messages": prefix + [bash_assistant(target_spec["recover"], trainable=True)],
                    "tools": BASH_TOOL,
                    "metadata": {
                        "source": "v50_failed_state_exact_recovery",
                        "instance_id": instance_id,
                        "repeat_count": repeat_count,
                        "copy_index": copy_index,
                        "target_format": "emptythink_all_assistant_tool_calls",
                        "target_command": target_spec["recover"],
                    },
                }
                rows.append(row)

        prefix = copy.deepcopy(messages[:4])
        for message in prefix:
            if message.get("role") == "assistant":
                message["content"] = ""
                message["reasoning_content"] = "\n"
                message["trainable"] = False
                message["loss"] = False
        prefix += [
            bash_assistant(target_spec["recover"], trainable=False),
            tool_output(target_spec["find_output"]),
        ]
        for copy_index in range(copies):
            rows.append(
                {
                    "messages": copy.deepcopy(prefix)
                    + [bash_assistant(target_spec["after_find"], trainable=True)],
                    "tools": BASH_TOOL,
                    "metadata": {
                        "source": "v50_failed_state_after_recovery",
                        "instance_id": instance_id,
                        "copy_index": copy_index,
                        "target_format": "emptythink_all_assistant_tool_calls",
                        "target_command": target_spec["after_find"],
                    },
                }
            )
    return rows


def write_shards(rows: list[dict[str, Any]], output_dir: Path, shards: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = [
        (output_dir / f"shard-{idx:03d}.jsonl").open("w", encoding="utf-8")
        for idx in range(shards)
    ]
    cats: Counter[str] = Counter()
    try:
        for idx, row in enumerate(rows):
            target = next(
                (
                    message
                    for message in reversed(row.get("messages") or [])
                    if message.get("role") == "assistant"
                    and message.get("trainable") is not False
                    and message.get("loss") is not False
                ),
                None,
            )
            if target:
                cats[target_category(command_from_assistant(target))] += 1
            handles[idx % shards].write(json_dumps(row) + "\n")
    finally:
        for handle in handles:
            handle.close()
    for path in output_dir.glob("shard-*.jsonl"):
        if path.stat().st_size == 0:
            path.unlink()
    return {"rows": len(rows), "target_categories": dict(cats)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--traj-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-base-rows", type=int, default=14000)
    parser.add_argument("--exact-copies", type=int, default=600)
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--seed", type=int, default=51051)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = load_base_rows(args.base_root, args.max_base_rows, args.seed)
    exact_rows = failed_prefix_rows(args.traj_root, args.exact_copies)
    rows.extend(exact_rows)
    random.Random(args.seed).shuffle(rows)

    summary = write_shards(rows, args.output_root / "current_failed_state_v51", args.shards)
    manifest = {
        "output_root": str(args.output_root),
        "base_root": str(args.base_root),
        "traj_root": str(args.traj_root),
        "max_base_rows": args.max_base_rows,
        "exact_copies": args.exact_copies,
        "exact_rows": len(exact_rows),
        "sources": {"current_failed_state_v51": summary},
        "clean_generalization": False,
        "selection": (
            "v47 target-task public/exact recovery rows plus exact v50 failed-prefix "
            "corrections for current Apache find-src and Redis /tmp/redis bad-path loops."
        ),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
