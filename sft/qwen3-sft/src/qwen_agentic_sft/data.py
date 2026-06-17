#!/usr/bin/env python3
"""Raw dataset discovery, normalization, and smoke-set construction.

The training recipe consumes the raw local datasets directly. This module keeps
the schema handling in one place so smoke tests and training use the same
normalization path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional until setup is run.
    pa = None
    pq = None


RAW_ROOT = Path("/wbl-fast/usrs/ee/code-swe-data/data/code-swe-terminal-agentic-sft")
EVENT_LOG_TYPES = {"session", "message", "model_change", "thinking_level_change"}
THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def text_from_content(value: Any) -> str:
    value = parse_jsonish(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces: list[str] = []
        saw_structured_part = False
        for part in value:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                part_type = str(part.get("type", "")).lower()
                if part_type in ("thinking", "reasoning", "toolcall", "tool_call"):
                    saw_structured_part = True
                    continue
                for key in ("text", "value", "content"):
                    if key in part:
                        saw_structured_part = True
                        pieces.append(text_from_content(part[key]))
                        break
                else:
                    output = part.get("output")
                    if isinstance(output, dict) and "value" in output:
                        saw_structured_part = True
                        pieces.append(str(output["value"]))
        if pieces:
            return "\n".join(piece for piece in pieces if piece)
        if saw_structured_part:
            return ""
        return json_dumps(value)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return text_from_content(value[key])
        output = value.get("output")
        if isinstance(output, dict) and "value" in output:
            return str(output["value"])
        return json_dumps(value)
    return str(value)


def thinking_from_content(value: Any) -> str | None:
    value = parse_jsonish(value)
    if not isinstance(value, list):
        return None
    pieces: list[str] = []
    for part in value:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type", "")).lower()
        if part_type not in ("thinking", "reasoning"):
            continue
        for key in ("thinking", "reasoning", "text", "content"):
            if key in part and part[key] not in (None, ""):
                pieces.append(text_from_content(part[key]))
                break
    reasoning = "\n".join(piece for piece in pieces if piece.strip()).strip()
    return reasoning or None


def ensure_json_value(value: Any) -> Any | None:
    value = parse_jsonish(value)
    if value in (None, "", [], {}):
        return None
    return value


def merge_thinking(reasoning: Any, content: Any) -> str:
    content_text = text_from_content(content)
    if reasoning is None and content_text.lstrip().startswith("<think>"):
        return content_text
    reasoning_text = text_from_content(reasoning).strip()
    if content_text:
        return f"{THINK_OPEN}{reasoning_text}{THINK_CLOSE}{content_text}"
    return f"{THINK_OPEN}{reasoning_text}{THINK_CLOSE}".rstrip("\n")


def normalize_role(role: Any) -> str | None:
    if role is None:
        return None
    role_s = str(role).strip().lower()
    role_map = {
        "assistant": "assistant",
        "ai": "assistant",
        "bot": "assistant",
        "gpt": "assistant",
        "model": "assistant",
        "human": "user",
        "user": "user",
        "system": "system",
        "developer": "system",
        "tool": "tool",
        "function": "tool",
        "toolresult": "tool",
        "tool_result": "tool",
        "environment": "tool",
        "observation": "tool",
    }
    return role_map.get(role_s)


def message_role(msg: dict[str, Any]) -> str | None:
    return normalize_role(msg.get("role", msg.get("from", msg.get("speaker"))))


def message_content(msg: dict[str, Any]) -> Any:
    for key in ("content", "value", "text", "system_prompt"):
        if key in msg:
            return msg[key]
    return ""


def normalize_tool_call(call: Any) -> dict[str, Any] | None:
    call = parse_jsonish(call)
    if not isinstance(call, dict):
        return None
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = function.get("name", call.get("name", ""))
    arguments = parse_jsonish(function.get("arguments", {}))
    if not name:
        return None
    return {"function": {"name": str(name), "arguments": arguments}}


def is_valid_tool_call(call: Any) -> bool:
    call = parse_jsonish(call)
    if not isinstance(call, dict):
        return False
    function = call.get("function")
    if not isinstance(function, dict):
        return False
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return False
    arguments = parse_jsonish(function.get("arguments", {}))
    return isinstance(arguments, (dict, list))


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    tool_calls = parse_jsonish(tool_calls)
    if not tool_calls:
        return []
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if not isinstance(tool_calls, list):
        return []
    return [call for call in (normalize_tool_call(item) for item in tool_calls) if call is not None]


def tool_calls_from_content(content: Any) -> list[dict[str, Any]]:
    content = parse_jsonish(content)
    if not isinstance(content, list):
        return []
    calls: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type", "")).lower()
        if part_type not in ("toolcall", "tool_call"):
            continue
        call = normalize_tool_call(
            {
                "name": part.get("name", part.get("toolName", "")),
                "arguments": part.get("arguments", {}),
            }
        )
        if call is not None:
            calls.append(call)
    return calls


def normalize_message(msg: Any) -> dict[str, Any] | None:
    if not isinstance(msg, dict):
        return None
    role = message_role(msg)
    if role is None:
        return None

    if role == "assistant":
        reasoning = None
        for key in ("reasoning_content", "reasoning", "thinking", "thought"):
            if key in msg and msg[key] not in (None, ""):
                reasoning = msg[key]
                break
        if reasoning is None:
            reasoning = thinking_from_content(message_content(msg))
        out: dict[str, Any] = {
            "role": "assistant",
            "content": merge_thinking(reasoning, message_content(msg)),
        }
        if msg.get("trainable") is False:
            out["trainable"] = False
        if msg.get("loss") is False:
            out["loss"] = False
        calls = normalize_tool_calls(msg.get("tool_calls"))
        if not calls:
            calls = normalize_tool_calls(msg.get("function_calls"))
        if not calls:
            calls = normalize_tool_calls(msg.get("function_call"))
        if not calls:
            calls = tool_calls_from_content(message_content(msg))
        if calls:
            out["tool_calls"] = calls
        return out

    return {"role": role, "content": text_from_content(message_content(msg))}


def strip_empty_system_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    while messages and messages[0].get("role") == "system" and not messages[0].get("content", "").strip():
        messages = messages[1:]
    return messages


def normalize_tools(value: Any) -> Any | None:
    value = ensure_json_value(value)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value


def assistant_has_target(msg: dict[str, Any]) -> bool:
    if msg.get("role") != "assistant":
        return False
    if msg.get("tool_calls"):
        return True
    content = str(msg.get("content") or "").strip()
    empty_think = f"{THINK_OPEN}{THINK_CLOSE}".strip()
    return bool(content and content != empty_think)


def has_assistant_target(messages: list[dict[str, Any]]) -> bool:
    return any(assistant_has_target(msg) for msg in messages)


def assistant_has_reasoning(msg: dict[str, Any]) -> bool:
    if msg.get("role") != "assistant":
        return False
    for key in ("reasoning_content", "reasoning", "thinking", "thought"):
        value = msg.get(key)
        if value not in (None, "", [], {}) and text_from_content(value).strip():
            return True
    content = str(msg.get("content") or "")
    start = content.find("<think>")
    end = content.find("</think>", start + len("<think>"))
    if start == -1 or end == -1:
        return False
    reasoning = content[start + len("<think>") : end]
    return bool(reasoning.strip())


def assistant_has_valid_tool_calls(msg: dict[str, Any]) -> bool:
    if msg.get("role") != "assistant":
        return False
    return any(is_valid_tool_call(call) for call in msg.get("tool_calls") or [])


def first_value(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def parse_messages(value: Any) -> list[Any] | None:
    value = parse_jsonish(value)
    return value if isinstance(value, list) else None


def normalize_prompt_response(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = first_value(row, ("question", "instruction", "prompt", "problem", "task"))
    response = first_value(row, ("answer", "response", "completion", "output", "solution"))
    if prompt is None or response is None:
        return None
    input_text = first_value(row, ("input", "context"))
    if input_text:
        prompt = f"{text_from_content(prompt)}\n\n{text_from_content(input_text)}"
    messages = [
        {"role": "user", "content": text_from_content(prompt)},
        {"role": "assistant", "content": merge_thinking(None, response)},
    ]
    if not has_assistant_target(messages):
        return None
    out: dict[str, Any] = {"messages": messages}
    preserve_example_metadata(row, out)
    return out


def preserve_example_metadata(row: dict[str, Any], out: dict[str, Any]) -> None:
    """Keep compact row-level fields needed by loss policy and audits."""

    for key in (
        "passed",
        "pass",
        "resolved",
        "source_outcome",
        "metadata",
        "uuid",
        "task_id",
        "teacher",
        "language",
        "difficulty",
        "source",
        "source_note",
        "model_patch",
        "model_patch_bytes",
        "patch_bytes",
        "original_model_patch_bytes",
    ):
        if key in row:
            out[key] = row[key]


def normalize_row(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None

    raw_messages = parse_messages(row.get("messages"))
    if raw_messages is None:
        raw_messages = parse_messages(row.get("conversations"))
    if raw_messages is None:
        raw_messages = parse_messages(row.get("conversation"))
    if raw_messages is None:
        raw_messages = parse_messages(row.get("trajectory"))
    if raw_messages is None:
        return normalize_prompt_response(row)

    messages: list[dict[str, Any]] = []
    for raw_msg in raw_messages:
        msg = normalize_message(raw_msg)
        if msg is None:
            # Generated trajectory rows can include bookkeeping records such as
            # {"role": "exit", ...}; they are not part of the model-visible
            # chat history and cannot be rendered by chat templates.
            continue
        messages.append(msg)

    messages = strip_empty_system_prefix(messages)
    if not has_assistant_target(messages):
        return None
    out: dict[str, Any] = {"messages": messages}
    tools = normalize_tools(row.get("tools"))
    if tools is not None:
        out["tools"] = tools
    preserve_example_metadata(row, out)
    return out


def normalize_event_log_jsonl(
    path: Path,
    stats: dict[str, int],
    max_rows: int,
) -> tuple[dict[str, Any] | None, bool]:
    if path.suffix.lower() != ".jsonl":
        return None, False

    messages: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        first_row: dict[str, Any] | None = None
        first_idx = 0
        for first_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                return None, False
            if not isinstance(parsed, dict) or parsed.get("type") not in EVENT_LOG_TYPES:
                return None, False
            first_row = parsed
            break
        if first_row is None:
            return None, False

        def consume(row: dict[str, Any]) -> None:
            stats["rows_in"] += 1
            if row.get("type") != "message" or not isinstance(row.get("message"), dict):
                return
            msg = normalize_message(row["message"])
            if msg is not None:
                messages.append(msg)

        consume(first_row)
        for idx, line in enumerate(handle, start=first_idx + 1):
            if max_rows and idx >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            if isinstance(row, dict):
                consume(row)

    messages = strip_empty_system_prefix(messages)
    if not has_assistant_target(messages):
        return None, True
    stats["rows_out"] += 1
    return {"messages": messages}, True


def discover_raw_files(raw_root: Path, skip_roots: Iterable[Path] = ()) -> list[Path]:
    raw_root = raw_root.resolve()
    skip_resolved = [root.resolve() for root in skip_roots]
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(raw_root):
        current = Path(dirpath).resolve()
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname != ".cache"
            and not any(is_under((current / dirname).resolve(), root) for root in skip_resolved)
        ]
        if any(is_under(current, root) for root in skip_resolved):
            continue
        for filename in filenames:
            if filename.endswith((".jsonl", ".parquet")):
                files.append(current / filename)
    files.sort(key=lambda path: str(path.relative_to(raw_root)))
    return files


def iter_jsonl_rows(path: Path, max_rows: int = 0) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_rows and idx >= max_rows:
                return
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def iter_parquet_rows(path: Path, batch_size: int = 128, max_rows: int = 0) -> Iterator[dict[str, Any]]:
    if pq is None:
        raise RuntimeError("pyarrow is required for parquet input")
    seen = 0
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            if max_rows and seen >= max_rows:
                return
            seen += 1
            if isinstance(row, dict):
                yield row


def iter_normalized_examples_from_files(
    files: Iterable[Path],
    *,
    max_rows_per_file: int = 0,
    parquet_batch_size: int = 128,
    max_examples: int = 0,
) -> Iterator[dict[str, Any]]:
    stats = {"rows_in": 0, "rows_out": 0, "rows_dropped": 0, "bad_json": 0}
    emitted = 0
    for path in files:
        if path.suffix == ".jsonl":
            normalized_event, handled_as_event_log = normalize_event_log_jsonl(
                path, stats, max_rows_per_file
            )
            if handled_as_event_log:
                if normalized_event is not None:
                    yield normalized_event
                    emitted += 1
                if max_examples and emitted >= max_examples:
                    return
                continue
            rows = iter_jsonl_rows(path, max_rows=max_rows_per_file)
        elif path.suffix == ".parquet":
            rows = iter_parquet_rows(path, batch_size=parquet_batch_size, max_rows=max_rows_per_file)
        else:
            continue

        for row in rows:
            stats["rows_in"] += 1
            normalized = normalize_row(row)
            if normalized is None:
                stats["rows_dropped"] += 1
                continue
            stats["rows_out"] += 1
            yield normalized
            emitted += 1
            if max_examples and emitted >= max_examples:
                return


def iter_normalized_examples(
    raw_root: Path,
    *,
    skip_roots: Iterable[Path] = (),
    max_rows_per_file: int = 0,
    parquet_batch_size: int = 128,
    max_examples: int = 0,
    shuffle_files: bool = False,
    seed: int = 33333,
) -> Iterator[dict[str, Any]]:
    files = discover_raw_files(raw_root, skip_roots)
    if shuffle_files:
        rng = random.Random(seed)
        rng.shuffle(files)
    yield from iter_normalized_examples_from_files(
        files,
        max_rows_per_file=max_rows_per_file,
        parquet_batch_size=parquet_batch_size,
        max_examples=max_examples,
    )


def build_smoke_raw_dataset(
    *,
    raw_root: Path,
    output_root: Path,
    rows_per_dataset: int,
    max_event_log_lines: int,
    overwrite: bool,
) -> dict[str, int]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to rebuild it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    copied_by_dataset: dict[str, int] = {}
    for dataset_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        copied = 0
        target_dataset = output_root / dataset_dir.name
        for path in discover_raw_files(dataset_dir):
            if copied >= rows_per_dataset:
                break
            rel = path.relative_to(dataset_dir)
            out_path = target_dataset / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".jsonl":
                lines: list[str] = []
                is_event_log = False
                with path.open("r", encoding="utf-8") as handle:
                    for line_idx, line in enumerate(handle):
                        stripped = line.strip()
                        if line_idx == 0 and stripped:
                            try:
                                first = json.loads(stripped)
                            except json.JSONDecodeError:
                                first = None
                            is_event_log = isinstance(first, dict) and first.get("type") in EVENT_LOG_TYPES
                        line_limit = max_event_log_lines if is_event_log else rows_per_dataset - copied
                        if line_idx >= line_limit:
                            break
                        if stripped:
                            lines.append(line)
                if not lines:
                    continue
                out_path.write_text("".join(lines), encoding="utf-8")
                copied += 1 if is_event_log else min(rows_per_dataset - copied, len(lines))
            elif path.suffix == ".parquet" and pq is not None and pa is not None:
                rows = list(iter_parquet_rows(path, max_rows=rows_per_dataset - copied))
                if not rows:
                    continue
                pq.write_table(pa.Table.from_pylist(rows), out_path)
                copied += len(rows)
        copied_by_dataset[dataset_dir.name] = copied
        print(f"[smoke] {dataset_dir.name}: copied approximately {copied} rows", flush=True)
    return copied_by_dataset


def summarize_raw_root(raw_root: Path) -> dict[str, Any]:
    datasets = sorted(path for path in raw_root.iterdir() if path.is_dir())
    files = discover_raw_files(raw_root)
    return {
        "raw_root": str(raw_root),
        "datasets": len(datasets),
        "files": len(files),
        "dataset_names": [path.name for path in datasets],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("build-smoke-raw")
    smoke.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--rows-per-dataset", type=int, default=3)
    smoke.add_argument("--max-event-log-lines", type=int, default=400)
    smoke.add_argument("--overwrite", action="store_true")

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build-smoke-raw":
        result = build_smoke_raw_dataset(
            raw_root=args.raw_root,
            output_root=args.output_root,
            rows_per_dataset=args.rows_per_dataset,
            max_event_log_lines=args.max_event_log_lines,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "summarize":
        print(json.dumps(summarize_raw_root(args.raw_root), indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
