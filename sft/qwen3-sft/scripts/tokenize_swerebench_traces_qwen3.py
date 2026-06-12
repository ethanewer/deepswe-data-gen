#!/usr/bin/env python3
"""Offline Qwen3 tokenization for SWE-rebench trace datasets.

This builder intentionally does not pack examples. Each output row corresponds
to one source row and contains full-length ``input_ids``, ``labels``,
``attention_mask``, and token-count metadata.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
QWEN_SFT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QWEN_SFT_ROOT / "src"))

from qwen_agentic_sft import data as qwen_data  # noqa: E402
from qwen_agentic_sft.online_packed_dataset import apply_assistant_loss_policy  # noqa: E402


IGNORE_INDEX = -100
DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_INPUT_ROOT = (
    REPO_ROOT
    / ".cache/hf-datasets/eewer__swerebench-traces-highquality-2x-duplicate-reasoning-90pct"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / ".cache/hf-datasets/"
    "eewer__swerebench-traces-highquality-2x-duplicate-reasoning-90pct-qwen3-4b-thinking-tokenized"
)
DEFAULT_CHAT_TEMPLATE = REPO_ROOT / "eval/chat_templates/qwen3_thinking_acc.jinja2"
BUILDER_RELATIVE_PATH = "sft/qwen3-sft/scripts/tokenize_swerebench_traces_qwen3.py"

WORKER_TOKENIZER: Any | None = None
WORKER_CHAT_TEMPLATE = ""
WORKER_MODEL = DEFAULT_MODEL
WORKER_LOCAL_FILES_ONLY = True


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


def git_commit() -> str:
    return git_output(["git", "rev-parse", "HEAD"])


def git_is_dirty() -> bool:
    return bool(git_output(["git", "status", "--short"]))


def iter_zstd_jsonl_lines(path: Path) -> Iterator[str]:
    with path.open("rb") as handle:
        reader = zstd.ZstdDecompressor().stream_reader(handle)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        for line in text:
            line = line.strip()
            if line:
                yield line


def load_total_rows(input_root: Path) -> int:
    info_path = input_root / "dataset_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        rows = info.get("rows")
        if isinstance(rows, int):
            return rows
    index_path = input_root / "metadata" / "index.jsonl"
    if index_path.exists():
        with index_path.open("rb") as handle:
            return sum(1 for _ in handle)
    return 0


def top_level_reasoning_by_index(row: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    reasoning = row.get("reasoning")
    if not isinstance(reasoning, list):
        return out
    for item in reasoning:
        if not isinstance(item, dict):
            continue
        index = item.get("message_index")
        content = item.get("content")
        if isinstance(index, int) and isinstance(content, str) and content.strip():
            out[index] = content.strip()
    return out


def enrich_top_level_reasoning(row: dict[str, Any]) -> dict[str, Any]:
    """Copy generated top-level reasoning onto raw assistant messages if needed."""
    top_reasoning = top_level_reasoning_by_index(row)
    if not top_reasoning:
        return row
    raw_messages = qwen_data.parse_messages(row.get("messages"))
    if raw_messages is None:
        return row
    copied = dict(row)
    copied_messages: list[Any] = []
    changed = False
    for index, raw_msg in enumerate(raw_messages):
        if not isinstance(raw_msg, dict):
            copied_messages.append(raw_msg)
            continue
        msg = dict(raw_msg)
        if msg.get("role") == "assistant" and not qwen_data.assistant_has_reasoning(msg):
            reasoning = top_reasoning.get(index)
            if reasoning:
                msg["reasoning_content"] = reasoning
                changed = True
        copied_messages.append(msg)
    if changed:
        copied["messages"] = copied_messages
    return copied


def normalize_row_for_tokenization(row: dict[str, Any]) -> tuple[dict[str, Any] | None, Counter[str]]:
    """Normalize and mask one row using the same SFT path as online training."""

    stats: Counter[str] = Counter()
    raw_messages = qwen_data.parse_messages(row.get("messages"))
    if raw_messages is None:
        stats["dropped_no_messages"] += 1
        return None, stats
    stats["skipped_bookkeeping_messages"] += sum(
        1 for raw_msg in raw_messages if isinstance(raw_msg, dict) and qwen_data.message_role(raw_msg) is None
    )

    example = qwen_data.normalize_row(enrich_top_level_reasoning(row))
    if example is None:
        stats["dropped_normalize_row_none"] += 1
        return None, stats

    assistant_before: list[tuple[bool, bool, bool, bool]] = []
    for message in example.get("messages", []):
        if message.get("role") != "assistant":
            continue
        has_reasoning = qwen_data.assistant_has_reasoning(message)
        has_valid_tool_calls = qwen_data.assistant_has_valid_tool_calls(message)
        source_masked = message.get("trainable") is False or message.get("loss") is False
        assistant_before.append((has_reasoning, has_valid_tool_calls, source_masked, message.get("loss") is False))

    apply_assistant_loss_policy(
        example,
        require_assistant_reasoning_for_loss=True,
        require_assistant_tool_calls_for_loss=True,
    )

    for message, (has_reasoning, has_valid_tool_calls, source_masked, was_loss_false) in zip(
        [m for m in example.get("messages", []) if m.get("role") == "assistant"],
        assistant_before,
    ):
        stats["assistant_turns"] += 1
        if has_reasoning:
            stats["assistant_with_reasoning"] += 1
        if not has_reasoning:
            stats["masked_no_reasoning_assistant_turns"] += 1
        if not has_valid_tool_calls:
            stats["masked_bad_tool_call_assistant_turns"] += 1
            stats["masked_missing_or_invalid_tool_call_assistant_turns"] += 1
        if source_masked or was_loss_false:
            stats["masked_source_loss_false_assistant_turns"] += 1
        if message.get("loss") is False:
            stats["masked_assistant_turns"] += 1

    return example, stats


def render_chat_with_spans(example: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Render with the checked-in Qwen template semantics and assistant spans."""

    messages = example["messages"]
    tools = example.get("tools")
    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    cursor = 0

    def append(text: str) -> None:
        nonlocal cursor
        parts.append(text)
        cursor += len(text)

    if tools:
        append("<|im_start|>system\n")
        if messages and messages[0].get("role") == "system":
            append(str(messages[0].get("content") or "") + "\n\n")
        append(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        for tool in tools:
            append("\n")
            append(json_dumps(tool))
        append(
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
            'within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, '
            '"arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
        )
    elif messages and messages[0].get("role") == "system":
        append("<|im_start|>system\n" + str(messages[0].get("content") or "") + "<|im_end|>\n")

    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content") if isinstance(message.get("content"), str) else ""
        if role == "user" or (role == "system" and index != 0):
            append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            continue

        if role == "assistant":
            span_start = cursor
            reasoning_content = message.get("reasoning_content")
            reasoning_content = reasoning_content if isinstance(reasoning_content, str) else ""
            if not reasoning_content and "</think>" in content:
                reasoning_content = content.split("</think>", 1)[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                content = content.split("</think>", 1)[-1].lstrip("\n")
            if reasoning_content:
                append(
                    "<|im_start|>assistant\n<think>\n"
                    + reasoning_content.strip("\n")
                    + "\n</think>\n\n"
                    + content.lstrip("\n")
                )
            else:
                append("<|im_start|>assistant\n" + content)
            tool_calls = message.get("tool_calls")
            if tool_calls:
                for call_index, tool_call in enumerate(tool_calls):
                    if (call_index == 0 and content) or call_index != 0:
                        append("\n")
                    function = tool_call.get("function") if isinstance(tool_call, dict) else None
                    if isinstance(function, dict):
                        tool_call = function
                    name = tool_call.get("name", "") if isinstance(tool_call, dict) else ""
                    arguments = tool_call.get("arguments", {}) if isinstance(tool_call, dict) else {}
                    append('<tool_call>\n{"name": "')
                    append(str(name))
                    append('", "arguments": ')
                    append(arguments if isinstance(arguments, str) else json_dumps(arguments))
                    append("}\n</tool_call>")
            append("<|im_end|>\n")
            spans.append(
                {
                    "message_index": index,
                    "start_char": span_start,
                    "end_char": cursor,
                    "trainable": message.get("trainable") is not False and message.get("loss") is not False,
                }
            )
            continue

        if role == "tool":
            previous_role = messages[index - 1].get("role") if index else None
            next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
            if index == 0 or previous_role != "tool":
                append("<|im_start|>user")
            append("\n<tool_response>\n")
            append(content)
            append("\n</tool_response>")
            if index == len(messages) - 1 or next_role != "tool":
                append("<|im_end|>\n")

    return "".join(parts), spans


def tokenizer_render(tokenizer: Any, example: dict[str, Any]) -> str:
    kwargs: dict[str, Any] = {
        "conversation": example["messages"],
        "tokenize": False,
        "add_generation_prompt": False,
        "chat_template": WORKER_CHAT_TEMPLATE,
    }
    if example.get("tools") is not None:
        kwargs["tools"] = example["tools"]
    return tokenizer.apply_chat_template(**kwargs)


def init_worker(model: str, chat_template: str, local_files_only: bool) -> None:
    global WORKER_TOKENIZER, WORKER_CHAT_TEMPLATE, WORKER_MODEL, WORKER_LOCAL_FILES_ONLY
    WORKER_MODEL = model
    WORKER_CHAT_TEMPLATE = chat_template
    WORKER_LOCAL_FILES_ONLY = local_files_only
    WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def process_task(task: tuple[str, str, int, bool]) -> tuple[dict[str, Any] | None, dict[str, int], str | None]:
    line, source_shard, source_row_index, verify_template = task
    tokenizer = WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("worker tokenizer was not initialized")

    try:
        row = json.loads(line)
        if not isinstance(row, dict):
            return None, {"rows_dropped": 1, "dropped_bad_json": 1}, "row is not an object"

        example, stats = normalize_row_for_tokenization(row)
        if example is None:
            out_stats = Counter(stats)
            out_stats["rows_dropped"] += 1
            return None, dict(out_stats), None

        rendered, spans = render_chat_with_spans(example)
        if verify_template:
            expected = tokenizer_render(tokenizer, example)
            if rendered != expected:
                mismatch = next(
                    (idx for idx, (left, right) in enumerate(zip(rendered, expected)) if left != right),
                    min(len(rendered), len(expected)),
                )
                raise ValueError(
                    "manual render does not match tokenizer chat template at char "
                    f"{mismatch}: manual={rendered[mismatch:mismatch+120]!r} "
                    f"template={expected[mismatch:mismatch+120]!r}"
                )

        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = [int(token_id) for token_id in encoded["input_ids"]]
        labels = [IGNORE_INDEX] * len(input_ids)
        label_spans: list[dict[str, int]] = []
        offsets = encoded["offset_mapping"]
        for span in spans:
            if not span["trainable"]:
                continue
            token_indexes = [
                idx
                for idx, (start, end) in enumerate(offsets)
                if end > start and start >= span["start_char"] and end <= span["end_char"]
            ]
            if not token_indexes:
                continue
            first = token_indexes[0]
            last = token_indexes[-1] + 1
            for idx in range(first, last):
                labels[idx] = input_ids[idx]
            label_spans.append(
                {
                    "message_index": int(span["message_index"]),
                    "start_token": int(first),
                    "end_token": int(last),
                }
            )

        num_label_tokens = sum(1 for label in labels if label != IGNORE_INDEX)
        source_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metadata = {
            **source_metadata,
            "num_tokens": len(input_ids),
            "num_label_tokens": num_label_tokens,
            "source_shard": source_shard,
            "source_row_index": source_row_index,
            "tokenizer": WORKER_MODEL,
            "chat_template": "eval/chat_templates/qwen3_thinking_acc.jinja2",
            "tokenization": "qwen3_unpacked_per_row",
            "ignore_index": IGNORE_INDEX,
            "masked_no_reasoning_assistant_turns": int(stats["masked_no_reasoning_assistant_turns"]),
            "masked_bad_tool_call_assistant_turns": int(stats["masked_bad_tool_call_assistant_turns"]),
            "masked_missing_or_invalid_tool_call_assistant_turns": int(
                stats["masked_missing_or_invalid_tool_call_assistant_turns"]
            ),
            "masked_source_loss_false_assistant_turns": int(stats["masked_source_loss_false_assistant_turns"]),
            "masked_assistant_turns": int(stats["masked_assistant_turns"]),
            "assistant_turns": int(stats["assistant_turns"]),
            "assistant_with_reasoning": int(stats["assistant_with_reasoning"]),
            "skipped_bookkeeping_messages": int(stats["skipped_bookkeeping_messages"]),
            "builder": BUILDER_RELATIVE_PATH,
        }
        tokenized = {
            "uuid": str(row.get("uuid", "")),
            "task_id": str(row.get("task_id", source_metadata.get("task_id", ""))),
            "teacher": str(row.get("teacher", source_metadata.get("teacher", ""))),
            "passed": bool(row.get("passed", source_metadata.get("passed", False))),
            "source_shard": source_shard,
            "source_row_index": int(source_row_index),
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "num_tokens": int(len(input_ids)),
            "num_label_tokens": int(num_label_tokens),
            "label_spans_json": json_dumps(label_spans),
            "metadata_json": json_dumps(metadata),
        }
        out_stats = Counter(stats)
        out_stats["rows_tokenized"] += 1
        out_stats["tokens"] += len(input_ids)
        out_stats["label_tokens"] += num_label_tokens
        if num_label_tokens <= 0:
            out_stats["zero_label_rows"] += 1
        return tokenized, dict(out_stats), None
    except Exception as exc:
        return (
            None,
            {"rows_dropped": 1, "dropped_exceptions": 1},
            f"{source_shard}:{source_row_index}: {type(exc).__name__}: {exc}",
        )


def iter_tasks(
    shards: list[Path],
    *,
    limit: int,
    verify_template_rows: int,
) -> Iterator[tuple[str, str, int, bool]]:
    emitted = 0
    for shard in shards:
        for shard_row_index, line in enumerate(iter_zstd_jsonl_lines(shard)):
            if limit and emitted >= limit:
                return
            verify = emitted < verify_template_rows
            yield line, shard.name, shard_row_index, verify
            emitted += 1


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "uuid": pa.array([row["uuid"] for row in rows], type=pa.string()),
            "task_id": pa.array([row["task_id"] for row in rows], type=pa.string()),
            "teacher": pa.array([row["teacher"] for row in rows], type=pa.string()),
            "passed": pa.array([row["passed"] for row in rows], type=pa.bool_()),
            "source_shard": pa.array([row["source_shard"] for row in rows], type=pa.string()),
            "source_row_index": pa.array([row["source_row_index"] for row in rows], type=pa.int32()),
            "input_ids": pa.array([row["input_ids"] for row in rows], type=pa.list_(pa.int32())),
            "labels": pa.array([row["labels"] for row in rows], type=pa.list_(pa.int32())),
            "attention_mask": pa.array([row["attention_mask"] for row in rows], type=pa.list_(pa.int8())),
            "num_tokens": pa.array([row["num_tokens"] for row in rows], type=pa.int32()),
            "num_label_tokens": pa.array([row["num_label_tokens"] for row in rows], type=pa.int32()),
            "label_spans_json": pa.array([row["label_spans_json"] for row in rows], type=pa.string()),
            "metadata_json": pa.array([row["metadata_json"] for row in rows], type=pa.string()),
        }
    )
    pq.write_table(table, path, compression="zstd", compression_level=6)


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


def log_progress(stats: Counter[str], total_rows: int, started: float, workers: int, note: str = "") -> None:
    elapsed = max(time.time() - started, 1e-9)
    rows_done = int(stats["rows_done"])
    rows_written_or_pending = int(stats["rows_written"] + stats["pending_rows"])
    avg_tokens = (stats["tokens"] / rows_written_or_pending) if rows_written_or_pending else 0.0
    estimated_total_tokens = int(avg_tokens * total_rows) if total_rows else 0
    rows_per_sec = rows_done / elapsed
    tokens_per_sec = stats["tokens"] / elapsed
    remaining_rows = max(total_rows - rows_done, 0) if total_rows else 0
    eta = remaining_rows / rows_per_sec if rows_per_sec > 0 and total_rows else float("nan")
    pct = (100.0 * rows_done / total_rows) if total_rows else 0.0
    print(
        "progress "
        f"rows={rows_done:,}/{total_rows:,} ({pct:.2f}%) "
        f"written={int(stats['rows_written']):,} pending={int(stats['pending_rows']):,} "
        f"dropped={int(stats['rows_dropped']):,} workers={workers} "
        f"tokens={int(stats['tokens']):,} est_total_tokens={estimated_total_tokens:,} "
        f"avg_tokens_per_row={avg_tokens:,.1f} label_tokens={int(stats['label_tokens']):,} "
        f"rows_per_sec={rows_per_sec:.2f} tokens_per_sec={tokens_per_sec:,.0f} "
        f"eta={format_duration(eta)} elapsed={format_duration(elapsed)}"
        + (f" {note}" if note else ""),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--rows-per-output-shard", type=int, default=64)
    parser.add_argument("--progress-every-rows", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0, help="0 means os.cpu_count().")
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--multiprocessing-context", choices=["fork", "spawn", "forkserver"], default="fork")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--drop-zero-label-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-template-rows", type=int, default=16)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def clear_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        for path in sorted(output_dir.glob("**/*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()


def write_metadata(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "tokenization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# SWE-rebench Traces Qwen3 Tokenized\n\n"
        f"Source dataset: `{summary['source_dataset']}`\n\n"
        f"Tokenizer: `{summary['model']}`\n\n"
        f"Chat template: `{summary['chat_template']}`\n\n"
        f"Builder: `{summary['builder']}` at git commit `{summary['git_commit']}`"
        + (" (dirty worktree when run)" if summary.get("git_dirty") else "")
        + "\n\n"
        "This dataset is tokenized but not packed. Rows contain `input_ids`, `labels`, "
        "`attention_mask`, `num_tokens`, and `num_label_tokens`. Labels use `-100` for "
        "system/user/tool tokens and for assistant turns masked because they lack "
        "reasoning or contain malformed tool calls.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite to rebuild")
    if args.overwrite:
        clear_output_dir(output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    chat_template = args.chat_template.read_text(encoding="utf-8")
    shards = sorted((input_root / "data").glob("train-*.jsonl.zst"))
    if not shards:
        raise FileNotFoundError(f"no train-*.jsonl.zst shards under {input_root / 'data'}")
    total_rows = args.limit or load_total_rows(input_root)
    workers = args.num_workers or (os.cpu_count() or 1)
    workers = max(1, workers)

    stats: Counter[str] = Counter()
    pending: list[dict[str, Any]] = []
    out_index = 0
    started = time.time()
    commit = git_commit()
    dirty = git_is_dirty()
    log_progress(stats, total_rows, started, workers, "starting")

    context = mp.get_context(args.multiprocessing_context)
    with context.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(args.model, chat_template, args.local_files_only),
    ) as pool:
        results = pool.imap_unordered(
            process_task,
            iter_tasks(shards, limit=args.limit, verify_template_rows=args.verify_template_rows),
            chunksize=max(1, args.chunksize),
        )
        for tokenized, row_stats, error in results:
            stats["rows_done"] += 1
            stats.update(row_stats)
            if error:
                print(f"warning dropped row error={error}", flush=True)
            if tokenized is not None and (tokenized["num_label_tokens"] > 0 or not args.drop_zero_label_rows):
                pending.append(tokenized)
                stats["pending_rows"] = len(pending)
            elif tokenized is not None:
                stats["rows_dropped"] += 1
                stats["rows_dropped_zero_labels"] += 1

            if len(pending) >= args.rows_per_output_shard:
                out_path = data_dir / f"train-{out_index:05d}.parquet"
                write_parquet(pending, out_path)
                stats["rows_written"] += len(pending)
                pending.clear()
                stats["pending_rows"] = 0
                out_index += 1

            if stats["rows_done"] % args.progress_every_rows == 0:
                log_progress(stats, total_rows, started, workers)

    if pending:
        out_path = data_dir / f"train-{out_index:05d}.parquet"
        write_parquet(pending, out_path)
        stats["rows_written"] += len(pending)
        pending.clear()
        stats["pending_rows"] = 0
        out_index += 1

    elapsed = time.time() - started
    summary = {
        "source_dataset": str(input_root),
        "output_dir": str(output_dir),
        "model": args.model,
        "chat_template": str(args.chat_template),
        "builder": BUILDER_RELATIVE_PATH,
        "git_commit": commit,
        "git_dirty": dirty,
        "format": "parquet",
        "packed": False,
        "num_workers": workers,
        "ignore_index": IGNORE_INDEX,
        "rows_seen": int(stats["rows_done"]),
        "rows_written": int(stats["rows_written"]),
        "rows_dropped": int(stats["rows_dropped"]),
        "parquet_shards": out_index,
        "tokens": int(stats["tokens"]),
        "label_tokens": int(stats["label_tokens"]),
        "avg_tokens_per_written_row": (
            float(stats["tokens"]) / float(stats["rows_written"]) if stats["rows_written"] else 0.0
        ),
        "masked_no_reasoning_assistant_turns": int(stats["masked_no_reasoning_assistant_turns"]),
        "masked_bad_tool_call_assistant_turns": int(stats["masked_bad_tool_call_assistant_turns"]),
        "masked_missing_or_invalid_tool_call_assistant_turns": int(
            stats["masked_missing_or_invalid_tool_call_assistant_turns"]
        ),
        "masked_source_loss_false_assistant_turns": int(stats["masked_source_loss_false_assistant_turns"]),
        "masked_assistant_turns": int(stats["masked_assistant_turns"]),
        "skipped_bookkeeping_messages": int(stats["skipped_bookkeeping_messages"]),
        "assistant_turns": int(stats["assistant_turns"]),
        "assistant_with_reasoning": int(stats["assistant_with_reasoning"]),
        "elapsed_sec": elapsed,
    }
    write_metadata(output_dir, summary)
    log_progress(stats, total_rows, started, workers, "complete")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
