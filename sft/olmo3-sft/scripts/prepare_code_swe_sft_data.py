#!/usr/bin/env python3
"""Normalize, length-filter, and tokenize the code-SWE terminal-agentic SFT data.

The final output is OLMo-core numpy SFT data:

    <output_root>/shard_00000/token_ids_part_*.npy
    <output_root>/shard_00000/labels_mask_part_*.npy
    <output_root>/shard_00000/token_ids_part_*.csv.gz

OLMo-core performs fixed-length sequence packing from those token/document
arrays at training time.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - caught in real runs.
    pq = None


RAW_ROOT = Path("/wbl-fast/usrs/ee/code-swe-data/data/code-swe-terminal-agentic-sft")
SKIP_ROOT = RAW_ROOT / "AlienKevin__SWE-ZERO-12M-trajectories"
OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/tokenized/"
    "code-swe-terminal-agentic-sft-olmo3-65k-smallest-first"
)
TOKENIZER = Path("/wbl-fast/usrs/mk/data/tokenizers/Olmo-3-7B-Think-SFT")
OPEN_INSTRUCT_DIR = Path("/wbl-fast/usrs/mk/open-instruct")
OPEN_INSTRUCT_PYTHON = OPEN_INSTRUCT_DIR / ".venv/bin/python"
CONVERT_SCRIPT = OPEN_INSTRUCT_DIR / "scripts/data/convert_sft_data_for_olmocore.py"
FILTER_SCRIPT = Path("/wbl-fast/usrs/mk/OLMo-core/scripts_mk/filter_by_token_length.py")

THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n"
EVENT_LOG_TYPES = {"session", "message", "model_change", "thinking_level_change"}


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def text_from_content(value: Any) -> str:
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
                if "text" in part:
                    saw_structured_part = True
                    pieces.append(str(part["text"]))
                elif "value" in part:
                    saw_structured_part = True
                    pieces.append(str(part["value"]))
                elif "content" in part:
                    saw_structured_part = True
                    pieces.append(text_from_content(part["content"]))
                elif isinstance(part.get("output"), dict) and "value" in part["output"]:
                    saw_structured_part = True
                    pieces.append(str(part["output"]["value"]))
        if pieces:
            return "\n".join(p for p in pieces if p)
        if saw_structured_part:
            return ""
        return json_dumps(value)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return text_from_content(value[key])
        if isinstance(value.get("output"), dict) and "value" in value["output"]:
            return str(value["output"]["value"])
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


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def ensure_json_string(value: Any) -> str | None:
    value = parse_jsonish(value)
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str):
        return value
    return json_dumps(value)


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
        "tool": "environment",
        "function": "environment",
        "toolresult": "environment",
        "tool_result": "environment",
        "environment": "environment",
        "observation": "environment",
    }
    return role_map.get(role_s)


def serialize_tool_calls(tool_calls: Any) -> str | None:
    tool_calls = parse_jsonish(tool_calls)
    if not tool_calls:
        return None
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if not isinstance(tool_calls, list):
        return ensure_json_string(tool_calls)
    out = []
    for call in tool_calls:
        if not isinstance(call, dict):
            out.append(call)
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        args = parse_jsonish(fn.get("arguments", {}))
        out.append({"name": fn.get("name", call.get("name", "")), "arguments": args})
    return json_dumps(out)


def serialize_content_tool_calls(content: Any) -> str | None:
    content = parse_jsonish(content)
    if not isinstance(content, list):
        return None
    calls: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type", "")).lower()
        if part_type not in ("toolcall", "tool_call"):
            continue
        calls.append(
            {
                "name": part.get("name", part.get("toolName", "")),
                "arguments": parse_jsonish(part.get("arguments", {})),
            }
        )
    return json_dumps(calls) if calls else None


def message_role(msg: dict[str, Any]) -> str | None:
    return normalize_role(msg.get("role", msg.get("from", msg.get("speaker"))))


def message_content(msg: dict[str, Any]) -> Any:
    for key in ("content", "value", "text", "system_prompt"):
        if key in msg:
            return msg[key]
    return ""


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
        function_calls = ensure_json_string(msg.get("function_calls"))
        if function_calls is None:
            function_calls = serialize_tool_calls(msg.get("tool_calls"))
        if function_calls is None:
            function_calls = serialize_content_tool_calls(message_content(msg))
        if function_calls is None:
            function_calls = ensure_json_string(msg.get("function_call"))
        if function_calls:
            out["function_calls"] = function_calls
        return out

    if role == "environment":
        return {"role": "environment", "content": text_from_content(message_content(msg))}

    out = {"role": role, "content": text_from_content(message_content(msg))}
    functions = ensure_json_string(msg.get("functions"))
    if functions:
        out["functions"] = functions
    return out


def strip_empty_system_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    while messages and messages[0].get("role") == "system" and not messages[0].get("content", "").strip():
        messages = messages[1:]
    return messages


def inject_functions(messages: list[dict[str, Any]], functions_json: str | None) -> list[dict[str, Any]]:
    if not functions_json or any("functions" in m for m in messages):
        return messages
    out: list[dict[str, Any]] = []
    injected = False
    for msg in messages:
        if not injected and msg.get("role") in ("system", "user"):
            msg = dict(msg)
            msg["functions"] = functions_json
            injected = True
        out.append(msg)
    if not injected:
        out.insert(0, {"role": "system", "content": "", "functions": functions_json})
    return out


def assistant_has_target(msg: dict[str, Any]) -> bool:
    if msg.get("role") != "assistant":
        return False
    if msg.get("function_calls"):
        return True
    content = str(msg.get("content") or "").strip()
    empty_think = f"{THINK_OPEN}{THINK_CLOSE}".strip()
    return bool(content and content != empty_think)


def has_assistant_target(messages: list[dict[str, Any]]) -> bool:
    return any(assistant_has_target(msg) for msg in messages)


def first_value(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def parse_messages(value: Any) -> list[Any] | None:
    value = parse_jsonish(value)
    if isinstance(value, list):
        return value
    return None


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
    return {"messages": messages} if has_assistant_target(messages) else None


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
            return None
        messages.append(msg)

    messages = strip_empty_system_prefix(messages)
    messages = inject_functions(messages, ensure_json_string(row.get("tools")))
    if not has_assistant_target(messages):
        return None
    return {"messages": messages}


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
            if not isinstance(row, dict):
                continue
            consume(row)

    messages = strip_empty_system_prefix(messages)
    if not has_assistant_target(messages):
        return None, True
    return {"messages": messages}, True


def iter_jsonl(path: Path, stats: dict[str, int], max_rows: int) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_rows and idx >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1


def iter_parquet(
    path: Path,
    stats: dict[str, int],
    batch_size: int,
    max_rows: int,
) -> Iterable[dict[str, Any]]:
    if pq is None:
        raise RuntimeError("pyarrow is required for parquet input")
    seen = 0
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            if max_rows and seen >= max_rows:
                return
            seen += 1
            yield row


def iter_rows(
    path: Path,
    stats: dict[str, int],
    batch_size: int,
    max_rows: int,
) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from iter_jsonl(path, stats, max_rows)
    elif suffix == ".parquet":
        yield from iter_parquet(path, stats, batch_size, max_rows)
    else:
        return


def discover_files(raw_root: Path, skip_roots: list[Path]) -> list[tuple[Path, int]]:
    files: list[tuple[Path, int]] = []
    skip_roots = [root.resolve() for root in skip_roots]
    for dirpath, dirnames, filenames in os.walk(raw_root):
        current = Path(dirpath).resolve()
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname != ".cache"
            and not any(is_under((current / dirname).resolve(), skip_root) for skip_root in skip_roots)
        ]
        if any(is_under(current, skip_root) for skip_root in skip_roots):
            continue
        for filename in filenames:
            if not filename.endswith((".jsonl", ".parquet")):
                continue
            path = current / filename
            files.append((path, path.stat().st_size))
    files.sort(key=lambda item: str(item[0]))
    return files


def converted_instance_count(tokenized_dir: Path) -> int | None:
    stats_path = tokenized_dir / "dataset_statistics.json"
    if not stats_path.exists():
        return None
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        return int(stats["overall_statistics"]["total_instances"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def completed_source_paths(tokenized_root: Path) -> set[Path]:
    """Return raw source files whose shard has a completed tokenized output."""
    manifest = tokenized_root / "manifest.tsv"
    if not manifest.exists():
        return set()

    shard_to_paths: dict[int, list[Path]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            shard_s, _size_s, path_s = line.rstrip("\n").split("\t", 2)
            shard_to_paths.setdefault(int(shard_s), []).append(Path(path_s).resolve())

    done: set[Path] = set()
    for shard, paths in shard_to_paths.items():
        shard_name = f"shard_{shard:05d}"
        tokenized_dir = tokenized_root / shard_name
        if not (tokenized_dir / "_SUCCESS").exists():
            continue
        filtered_jsonl = tokenized_root / ".filtered" / shard_name / "data/train.jsonl"
        expected_instances = count_lines(filtered_jsonl) if filtered_jsonl.exists() else None
        actual_instances = converted_instance_count(tokenized_dir)
        if (
            expected_instances is not None
            and actual_instances is not None
            and actual_instances != expected_instances
        ):
            print(
                f"[manifest] not excluding invalid completed shard {shard_name} from "
                f"{tokenized_root}: converted {actual_instances:,} rows, expected "
                f"{expected_instances:,}",
                flush=True,
            )
            continue
        done.update(paths)
    return done


def top_level_dataset(raw_root: Path, path: Path) -> str:
    try:
        return path.relative_to(raw_root).parts[0]
    except (ValueError, IndexError):
        return path.parent.name


def make_buckets(
    files: list[tuple[Path, int]],
    raw_root: Path,
    num_shards: int,
    manifest_mode: str,
) -> list[list[tuple[Path, int]]]:
    if manifest_mode == "balanced":
        if num_shards <= 0:
            raise ValueError("--num-shards must be > 0 for --manifest-mode balanced")
        buckets: list[list[tuple[Path, int]]] = [[] for _ in range(num_shards)]
        bucket_bytes = [0 for _ in range(num_shards)]
        for path, size in sorted(files, key=lambda item: item[1], reverse=True):
            shard = min(range(num_shards), key=lambda idx: bucket_bytes[idx])
            buckets[shard].append((path, size))
            bucket_bytes[shard] += size
        return buckets

    if manifest_mode == "file_size_asc":
        return [[item] for item in sorted(files, key=lambda item: (item[1], str(item[0])))]

    if manifest_mode == "dataset_size_asc":
        grouped: dict[str, list[tuple[Path, int]]] = {}
        for path, size in files:
            grouped.setdefault(top_level_dataset(raw_root, path), []).append((path, size))
        groups = sorted(
            grouped.items(),
            key=lambda item: (sum(size for _path, size in item[1]), item[0]),
        )
        return [
            sorted(items, key=lambda item: (item[1], str(item[0])))
            for _name, items in groups
        ]

    raise ValueError(f"unknown manifest mode: {manifest_mode}")


def build_manifest(
    raw_root: Path,
    output_root: Path,
    skip_roots: list[Path],
    num_shards: int,
    limit_files: int,
    manifest_mode: str,
    exclude_tokenized_roots: list[Path],
) -> dict[int, list[tuple[Path, int]]]:
    files = discover_files(raw_root, skip_roots)
    excluded: set[Path] = set()
    for tokenized_root in exclude_tokenized_roots:
        excluded.update(completed_source_paths(tokenized_root))
    if excluded:
        before = len(files)
        files = [(path, size) for path, size in files if path.resolve() not in excluded]
        print(
            f"[manifest] excluded {before - len(files)} already-tokenized source files",
            flush=True,
        )
    if limit_files:
        files = files[:limit_files]

    buckets = make_buckets(files, raw_root, num_shards, manifest_mode)
    bucket_bytes = [sum(size for _path, size in items) for items in buckets]
    num_shards = len(buckets)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.tsv"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for shard, items in enumerate(buckets):
            for path, size in items:
                handle.write(f"{shard}\t{size}\t{path}\n")

    summary = {
        "raw_root": str(raw_root),
        "skip_roots": [str(root) for root in skip_roots],
        "exclude_tokenized_roots": [str(root) for root in exclude_tokenized_roots],
        "excluded_already_tokenized_files": len(excluded),
        "manifest_mode": manifest_mode,
        "num_shards": num_shards,
        "num_files": len(files),
        "total_bytes": sum(size for _, size in files),
        "shards": [
            {"shard": idx, "num_files": len(items), "bytes": bucket_bytes[idx]}
            for idx, items in enumerate(buckets)
        ],
    }
    (output_root / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[manifest] wrote {manifest_path} with {len(files)} files, "
        f"{sum(size for _, size in files):,} bytes across {num_shards} shards "
        f"({manifest_mode})",
        flush=True,
    )
    return {idx: buckets[idx] for idx in range(num_shards)}


def load_manifest(path: Path, num_shards: int) -> dict[int, list[tuple[Path, int]]]:
    shards: dict[int, list[tuple[Path, int]]] = (
        {idx: [] for idx in range(num_shards)} if num_shards > 0 else {}
    )
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            shard_s, size_s, path_s = line.rstrip("\n").split("\t", 2)
            shard = int(shard_s)
            if num_shards > 0 and shard not in shards:
                raise ValueError(
                    f"manifest has shard {shard}, but --num-shards is {num_shards}; "
                    "rerun with --rebuild-manifest"
                )
            shards.setdefault(shard, [])
            shards[shard].append((Path(path_s), int(size_s)))
    return shards


def get_manifest(args: argparse.Namespace) -> dict[int, list[tuple[Path, int]]]:
    manifest_path = args.output_root / "manifest.tsv"
    if manifest_path.exists() and not args.rebuild_manifest:
        print(f"[manifest] using existing {manifest_path}", flush=True)
        return load_manifest(manifest_path, args.num_shards)
    return build_manifest(
        raw_root=args.raw_root,
        output_root=args.output_root,
        skip_roots=args.skip_root,
        num_shards=args.num_shards,
        limit_files=args.limit_files,
        manifest_mode=args.manifest_mode,
        exclude_tokenized_roots=args.exclude_tokenized_root,
    )


def write_dataset_readme(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    readme = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---
"""
    (dataset_root / "README.md").write_text(readme, encoding="utf-8")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
    return count


def normalize_shard(
    shard: int,
    files: list[tuple[Path, int]],
    normalized_jsonl: Path,
    args: argparse.Namespace,
) -> dict[str, int]:
    success = normalized_jsonl.parent.parent / "_SUCCESS"
    stats_path = normalized_jsonl.parent.parent / "stats.json"
    if success.exists() and not args.force_normalize:
        print(f"[shard {shard:05d}] normalize already complete", flush=True)
        return json.loads(stats_path.read_text(encoding="utf-8"))

    normalized_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_dataset_readme(normalized_jsonl.parent.parent)
    tmp_path = normalized_jsonl.with_suffix(".jsonl.tmp")
    stats = {
        "files": len(files),
        "rows_in": 0,
        "rows_out": 0,
        "rows_dropped": 0,
        "bad_json": 0,
    }
    t0 = time.time()
    last_log = t0
    with tmp_path.open("w", encoding="utf-8") as out:
        for file_idx, (path, _size) in enumerate(files, start=1):
            event_normalized, handled_as_event_log = normalize_event_log_jsonl(
                path, stats, args.max_rows_per_file
            )
            if handled_as_event_log:
                if event_normalized is None:
                    stats["rows_dropped"] += 1
                else:
                    out.write(json_dumps(event_normalized) + "\n")
                    stats["rows_out"] += 1
            else:
                for row in iter_rows(path, stats, args.parquet_batch_size, args.max_rows_per_file):
                    stats["rows_in"] += 1
                    normalized = normalize_row(row)
                    if normalized is None:
                        stats["rows_dropped"] += 1
                    else:
                        out.write(json_dumps(normalized) + "\n")
                        stats["rows_out"] += 1
                    now = time.time()
                    if now - last_log >= 30:
                        rate = stats["rows_in"] / max(1e-9, now - t0)
                        print(
                            f"[shard {shard:05d}] normalize files={file_idx}/{len(files)} "
                            f"in={stats['rows_in']:,} out={stats['rows_out']:,} "
                            f"drop={stats['rows_dropped']:,} rate={rate:,.0f} rows/s",
                            flush=True,
                        )
                        last_log = now
            now = time.time()
            if now - last_log >= 30:
                rate = stats["rows_in"] / max(1e-9, now - t0)
                print(
                    f"[shard {shard:05d}] normalize files={file_idx}/{len(files)} "
                    f"in={stats['rows_in']:,} out={stats['rows_out']:,} "
                    f"drop={stats['rows_dropped']:,} rate={rate:,.0f} rows/s",
                    flush=True,
                )
                last_log = now
    os.replace(tmp_path, normalized_jsonl)
    stats["elapsed_sec"] = int(time.time() - t0)
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    success.write_text("ok\n", encoding="utf-8")
    print(
        f"[shard {shard:05d}] normalized {stats['rows_out']:,}/{stats['rows_in']:,} rows",
        flush=True,
    )
    return stats


def run_command(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def filter_shard(
    shard: int,
    normalized_jsonl: Path,
    filtered_jsonl: Path,
    args: argparse.Namespace,
) -> int:
    success = filtered_jsonl.parent.parent / "_SUCCESS"
    if success.exists() and not args.force_filter:
        print(f"[shard {shard:05d}] filter already complete", flush=True)
        return count_lines(filtered_jsonl)

    filtered_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_dataset_readme(filtered_jsonl.parent.parent)
    cmd = [
        str(args.open_instruct_python),
        str(args.filter_script),
        "--in",
        str(normalized_jsonl),
        "--out",
        str(filtered_jsonl),
        "--tokenizer",
        str(args.tokenizer),
        "--max-seq-length",
        str(args.sequence_length),
        "--workers",
        str(args.filter_workers),
        "--chunksize",
        str(args.filter_chunksize),
    ]
    run_command(cmd)
    kept = count_lines(filtered_jsonl)
    success.write_text("ok\n", encoding="utf-8")
    print(f"[shard {shard:05d}] filter kept {kept:,} rows", flush=True)
    return kept


def convert_shard(shard: int, filtered_dir: Path, tokenized_dir: Path, args: argparse.Namespace) -> None:
    filtered_jsonl = filtered_dir / "data/train.jsonl"
    expected_instances = count_lines(filtered_jsonl)
    success = tokenized_dir / "_SUCCESS"
    cache_dir = args.output_root / ".oi-cache" / f"shard_{shard:05d}"
    if success.exists() and not args.force_convert:
        actual_instances = converted_instance_count(tokenized_dir)
        if actual_instances == expected_instances:
            print(f"[shard {shard:05d}] convert already complete", flush=True)
            return
        print(
            f"[shard {shard:05d}] invalid converted output: converted "
            f"{actual_instances if actual_instances is not None else 'unknown'} rows, "
            f"expected {expected_instances:,}; repairing",
            flush=True,
        )
        shutil.rmtree(tokenized_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
    elif tokenized_dir.exists() and any(tokenized_dir.iterdir()) and not args.force_convert:
        print(
            f"[shard {shard:05d}] removing incomplete converted output before restart",
            flush=True,
        )
        shutil.rmtree(tokenized_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    if args.force_convert:
        shutil.rmtree(tokenized_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(args.open_instruct_dir)
        if not existing_pythonpath
        else f"{args.open_instruct_dir}:{existing_pythonpath}"
    )
    env.setdefault("HF_DATASETS_CACHE", str(cache_dir / "hf_datasets"))
    env.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    env.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    cmd = [
        str(args.open_instruct_python),
        str(args.convert_script),
        "--dataset_mixer_list",
        str(filtered_jsonl),
        "1.0",
        "--tokenizer_name_or_path",
        str(args.tokenizer),
        "--output_dir",
        str(tokenized_dir),
        "--max_seq_length",
        str(args.sequence_length),
        "--chat_template_name",
        "olmo",
        "--dataset_local_cache_dir",
        str(cache_dir),
        "--resume",
        "--visualize",
        "False",
    ]
    if args.convert_num_examples:
        cmd.extend(["--num_examples", str(args.convert_num_examples)])
    run_command(cmd, cwd=args.open_instruct_dir, env=env)
    token_parts = list(tokenized_dir.glob("token_ids_part_*.npy"))
    mask_parts = list(tokenized_dir.glob("labels_mask_part_*.npy"))
    if not token_parts or not mask_parts:
        raise RuntimeError(f"conversion produced no token/mask parts in {tokenized_dir}")
    actual_instances = converted_instance_count(tokenized_dir)
    if actual_instances != expected_instances:
        raise RuntimeError(
            f"conversion instance mismatch for shard {shard:05d}: converted "
            f"{actual_instances if actual_instances is not None else 'unknown'} rows, "
            f"expected {expected_instances:,}"
        )
    success.write_text("ok\n", encoding="utf-8")
    print(
        f"[shard {shard:05d}] convert wrote {len(token_parts)} token parts and "
        f"{len(mask_parts)} mask parts",
        flush=True,
    )


def process_shard(shard: int, files: list[tuple[Path, int]], args: argparse.Namespace) -> None:
    shard_name = f"shard_{shard:05d}"
    normalized_dir = args.output_root / ".normalized" / shard_name
    filtered_dir = args.output_root / ".filtered" / shard_name
    tokenized_dir = args.output_root / shard_name
    normalized_jsonl = normalized_dir / "data/train.jsonl"
    filtered_jsonl = filtered_dir / "data/train.jsonl"

    if not files:
        print(f"[shard {shard:05d}] no files assigned; skipping", flush=True)
        return

    stats = normalize_shard(shard, files, normalized_jsonl, args)
    if args.stop_after == "normalize":
        return
    if stats["rows_out"] == 0:
        print(f"[shard {shard:05d}] no normalized rows; skipping filter/convert", flush=True)
        return

    kept = filter_shard(shard, normalized_jsonl, filtered_jsonl, args)
    if args.stop_after == "filter":
        return
    if kept == 0:
        print(f"[shard {shard:05d}] no filtered rows; skipping convert", flush=True)
        return

    convert_shard(shard, filtered_dir, tokenized_dir, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--skip-root", type=Path, action="append", default=[SKIP_ROOT])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--sequence-length", type=int, default=65536)
    parser.add_argument("--num-shards", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument(
        "--manifest-mode",
        choices=["dataset_size_asc", "file_size_asc", "balanced"],
        default="dataset_size_asc",
        help="How to assign source files to tokenized shards.",
    )
    parser.add_argument(
        "--exclude-tokenized-root",
        type=Path,
        action="append",
        default=[],
        help="Existing tokenized root whose completed shards should not be retokenized.",
    )
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--parquet-batch-size", type=int, default=128)
    parser.add_argument("--filter-workers", type=int, default=16)
    parser.add_argument("--filter-chunksize", type=int, default=64)
    parser.add_argument("--convert-num-examples", type=int, default=0)
    parser.add_argument(
        "--stop-after",
        choices=["manifest", "normalize", "filter", "convert"],
        default="convert",
    )
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--force-normalize", action="store_true")
    parser.add_argument("--force-filter", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--open-instruct-dir", type=Path, default=OPEN_INSTRUCT_DIR)
    parser.add_argument("--open-instruct-python", type=Path, default=OPEN_INSTRUCT_PYTHON)
    parser.add_argument("--convert-script", type=Path, default=CONVERT_SCRIPT)
    parser.add_argument("--filter-script", type=Path, default=FILTER_SCRIPT)
    args = parser.parse_args()

    args.raw_root = args.raw_root.resolve()
    args.skip_root = [path.resolve() for path in args.skip_root]
    args.output_root = args.output_root.resolve()
    args.tokenizer = args.tokenizer.resolve()
    args.exclude_tokenized_root = [path.resolve() for path in args.exclude_tokenized_root]
    args.open_instruct_dir = Path(os.path.abspath(args.open_instruct_dir))
    args.open_instruct_python = Path(os.path.abspath(args.open_instruct_python))
    args.convert_script = Path(os.path.abspath(args.convert_script))
    args.filter_script = Path(os.path.abspath(args.filter_script))

    if args.shard_index is not None and args.num_shards > 0 and not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, --num-shards)")
    return args


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    shards = get_manifest(args)
    if args.stop_after == "manifest":
        return 0

    if args.shard_index is not None:
        if args.shard_index not in shards:
            raise ValueError(f"--shard-index {args.shard_index} is not present in the manifest")
        shard_indices = [args.shard_index]
    else:
        shard_indices = sorted(shards)
    for shard in shard_indices:
        process_shard(shard, shards[shard], args)
    print("[done] data preparation complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
