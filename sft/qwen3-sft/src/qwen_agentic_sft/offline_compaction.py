#!/usr/bin/env python3
"""Offline synthetic compaction for long OpenAI-style agent traces."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from .data import (
    has_assistant_target,
    iter_jsonl_rows,
    iter_normalized_examples,
    json_dumps,
    normalize_row,
)
from .online_packed_dataset import DEFAULT_CHAT_TEMPLATE, DEFAULT_MODEL, encode_text, load_chat_template, render_chat


DEFAULT_COMPACTION_PROMPT = """We need to compact the conversation so far.

Write a concise state summary for continuing the task. Preserve the original task, repository or environment state, important files, commands already run, tool results that affect the next step, decisions made, and remaining work. Do not invent details."""

SUMMARY_MARKER = "[synthetic-compaction-summary]"


@dataclass
class CompactionConfig:
    max_sequence_length: int
    boundary_tokens: int
    include_compaction: bool
    summary_token_budget: int = 1536
    compaction_prompt: str = DEFAULT_COMPACTION_PROMPT
    min_body_messages_per_chunk: int = 1
    max_split_shrink_steps: int = 64

    def __post_init__(self) -> None:
        if self.max_sequence_length <= 0:
            raise ValueError("--max-sequence-length must be positive")
        if self.boundary_tokens <= 0:
            raise ValueError("--boundary-tokens must be positive")
        if self.boundary_tokens > self.max_sequence_length:
            raise ValueError("--boundary-tokens cannot exceed --max-sequence-length")
        if self.summary_token_budget <= 0:
            raise ValueError("--summary-token-budget must be positive")
        if self.min_body_messages_per_chunk <= 0:
            raise ValueError("--min-body-messages-per-chunk must be positive")


@dataclass
class CompactionStats:
    rows_in: int = 0
    rows_out: int = 0
    rows_unchanged: int = 0
    rows_compacted: int = 0
    rows_without_target_skipped: int = 0
    messages_truncated: int = 0
    max_output_tokens: int = 0
    output_token_histogram: dict[str, int] = field(default_factory=dict)

    def add_output_length(self, tokens: int) -> None:
        self.max_output_tokens = max(self.max_output_tokens, int(tokens))
        bucket = f"{(tokens // 1024) * 1024}-{((tokens // 1024) + 1) * 1024 - 1}"
        self.output_token_histogram[bucket] = self.output_token_histogram.get(bucket, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_unchanged": self.rows_unchanged,
            "rows_compacted": self.rows_compacted,
            "rows_without_target_skipped": self.rows_without_target_skipped,
            "messages_truncated": self.messages_truncated,
            "max_output_tokens": self.max_output_tokens,
            "output_token_histogram": dict(sorted(self.output_token_histogram.items())),
        }


class ChatMeasurer:
    def __init__(self, tokenizer: Any, *, chat_template: str):
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    def render(self, messages: list[dict[str, Any]], tools: Any = None) -> str:
        return render_chat(self.tokenizer, messages, tools, self.chat_template)

    def count_chat(self, messages: list[dict[str, Any]], tools: Any = None) -> int:
        if not messages:
            return 0
        return len(encode_text(self.tokenizer, self.render(messages, tools)))

    def count_text(self, text: str) -> int:
        return len(encode_text(self.tokenizer, text))

    def clip_text(self, text: str, max_tokens: int, *, keep: str = "head") -> str:
        if max_tokens <= 0 or not text:
            return ""
        ids = encode_text(self.tokenizer, text)
        if len(ids) <= max_tokens:
            return text
        clipped = ids[:max_tokens] if keep == "head" else ids[-max_tokens:]
        try:
            return self.tokenizer.decode(clipped, skip_special_tokens=False)
        except TypeError:
            return self.tokenizer.decode(clipped)


def leading_system_messages(messages: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system: list[dict[str, Any]] = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        system.append(dict(messages[idx]))
        idx += 1
    return system, [dict(msg) for msg in messages[idx:]]


def make_summary_context_message(summary: str) -> dict[str, Any]:
    return {"role": "system", "content": summary}


def make_compaction_prompt_message(prompt: str) -> dict[str, Any]:
    return {"role": "user", "content": prompt}


def make_summary_assistant_message(summary: str) -> dict[str, Any]:
    return {"role": "assistant", "content": summary}


def is_synthetic_summary_message(message: dict[str, Any]) -> bool:
    return message.get("role") == "system" and str(message.get("content", "")).startswith(SUMMARY_MARKER)


def format_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    formatted: list[str] = []
    for call in tool_calls if isinstance(tool_calls, list) else [tool_calls]:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        name = function.get("name", "")
        arguments = function.get("arguments", {})
        formatted.append(f"{name}({json_dumps(arguments)})" if name else json_dumps(call))
    return "; ".join(piece for piece in formatted if piece)


def message_excerpt(message: dict[str, Any], measurer: ChatMeasurer, *, token_budget: int) -> str:
    role = str(message.get("role", "unknown"))
    content = str(message.get("content") or "")
    keep = "tail" if role == "tool" else "head"
    content = measurer.clip_text(content.strip(), token_budget, keep=keep).strip()
    calls = format_tool_calls(message.get("tool_calls"))
    if calls:
        call_budget = max(32, token_budget // 3)
        calls = measurer.clip_text(calls, call_budget, keep="head").strip()
        content = f"{content}\nTool calls: {calls}".strip()
    if not content:
        content = "(empty)"
    return f"- {role}: {content}"


def build_extractive_summary(
    messages: list[dict[str, Any]],
    measurer: ChatMeasurer,
    *,
    max_tokens: int,
) -> str:
    def fit_summary(text: str) -> str:
        text = text.strip()
        if measurer.count_text(text) <= max_tokens:
            return text
        marker_tokens = measurer.count_text(SUMMARY_MARKER + "\n")
        remaining = max(1, max_tokens - marker_tokens)
        body_text = text
        if body_text.startswith(SUMMARY_MARKER):
            body_text = body_text[len(SUMMARY_MARKER) :].lstrip()
        return f"{SUMMARY_MARKER}\n{measurer.clip_text(body_text, remaining, keep='tail').strip()}".strip()

    body = [msg for msg in messages if msg.get("role") != "system" or not is_synthetic_summary_message(msg)]
    first_user = next((msg for msg in body if msg.get("role") == "user" and str(msg.get("content", "")).strip()), None)
    prior_summary = next((msg for msg in messages if is_synthetic_summary_message(msg)), None)

    header = (
        f"{SUMMARY_MARKER}\n"
        "This is a deterministic extractive summary of the conversation before compaction."
    )
    sections: list[str] = [header]
    if prior_summary is not None:
        prior = str(prior_summary.get("content", "")).replace(SUMMARY_MARKER, "").strip()
        prior = measurer.clip_text(prior, max(64, max_tokens // 5), keep="tail").strip()
        if prior:
            sections.append(f"Prior compacted context:\n{prior}")
    if first_user is not None:
        task = measurer.clip_text(str(first_user.get("content") or "").strip(), max(96, max_tokens // 5), keep="head")
        if task.strip():
            sections.append(f"Original task:\n{task.strip()}")

    role_counts: dict[str, int] = {}
    for msg in body:
        role = str(msg.get("role", "unknown"))
        role_counts[role] = role_counts.get(role, 0) + 1
    if role_counts:
        counts = ", ".join(f"{role}={count}" for role, count in sorted(role_counts.items()))
        sections.append(f"Messages before compaction: {counts}.")

    base = "\n\n".join(sections)
    if measurer.count_text(base) >= max_tokens:
        return fit_summary(base)

    recent: list[str] = []
    for msg in reversed(body):
        excerpt = message_excerpt(msg, measurer, token_budget=max(64, min(384, max_tokens // 4)))
        candidate_recent = [excerpt, *recent]
        candidate = f"{base}\n\nRecent transcript before compaction:\n" + "\n".join(candidate_recent)
        if measurer.count_text(candidate) > max_tokens:
            if recent:
                break
            excerpt = message_excerpt(msg, measurer, token_budget=max(24, max_tokens // 6))
            candidate = f"{base}\n\nRecent transcript before compaction:\n{excerpt}"
            if measurer.count_text(candidate) <= max_tokens:
                recent = [excerpt]
            break
        recent = candidate_recent

    summary = base
    if recent:
        summary = f"{base}\n\nRecent transcript before compaction:\n" + "\n".join(recent)
    if measurer.count_text(summary) > max_tokens:
        summary = fit_summary(summary)
    return fit_summary(summary)


def output_row(
    messages: list[dict[str, Any]],
    tools: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"messages": messages}
    if tools is not None:
        row["tools"] = tools
    row["metadata"] = metadata
    return row


def find_split_index(
    system_prefix: list[dict[str, Any]],
    summary_context: dict[str, Any] | None,
    body: list[dict[str, Any]],
    tools: Any,
    measurer: ChatMeasurer,
    config: CompactionConfig,
) -> int:
    max_index = len(body) - 1
    if max_index <= 0:
        return 1

    fixed = system_prefix + ([summary_context] if summary_context is not None else [])
    low = config.min_body_messages_per_chunk
    high = max_index
    best = 0
    while low <= high:
        mid = (low + high) // 2
        tokens = measurer.count_chat(fixed + body[:mid], tools)
        if tokens <= config.boundary_tokens and tokens <= config.max_sequence_length:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return max(best, 1)


def truncate_last_message_to_fit(
    messages: list[dict[str, Any]],
    tools: Any,
    measurer: ChatMeasurer,
    max_sequence_length: int,
) -> tuple[list[dict[str, Any]], bool]:
    if measurer.count_chat(messages, tools) <= max_sequence_length:
        return messages, False
    if not messages:
        return messages, False

    truncated = [dict(msg) for msg in messages]
    idx = -1
    largest = -1
    for candidate_idx, message in enumerate(truncated):
        content = str(message.get("content") or "")
        if not content:
            continue
        token_count = len(encode_text(measurer.tokenizer, content))
        if token_count > largest:
            largest = token_count
            idx = candidate_idx
    if idx < 0:
        return truncated, False

    original = str(truncated[idx].get("content") or "")
    low = 0
    high = min(len(encode_text(measurer.tokenizer, original)), max_sequence_length)
    best = ""
    keep = "tail" if truncated[idx].get("role") == "tool" else "head"
    marker = "\n\n[truncated at synthetic token boundary]"
    while low <= high:
        mid = (low + high) // 2
        clipped = measurer.clip_text(original, mid, keep=keep).rstrip()
        candidate_msg = dict(truncated[idx])
        candidate_msg["content"] = f"{clipped}{marker}" if clipped else marker.strip()
        candidate = truncated[:idx] + [candidate_msg] + truncated[idx + 1 :]
        if measurer.count_chat(candidate, tools) <= max_sequence_length:
            best = candidate_msg["content"]
            low = mid + 1
        else:
            high = mid - 1
    if not best:
        truncated = truncated[:idx] + truncated[idx + 1 :]
        return truncate_last_message_to_fit(truncated, tools, measurer, max_sequence_length)
    truncated[idx]["content"] = best
    return truncated, True


def emit_if_trainable(
    rows: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tools: Any,
    metadata: dict[str, Any],
    measurer: ChatMeasurer,
    stats: CompactionStats,
) -> None:
    if not has_assistant_target(messages):
        stats.rows_without_target_skipped += 1
        return
    tokens = measurer.count_chat(messages, tools)
    if tokens > stats.max_output_tokens:
        stats.max_output_tokens = tokens
    stats.add_output_length(tokens)
    rows.append(output_row(messages, tools, metadata))
    stats.rows_out += 1


def compact_example(
    example: dict[str, Any],
    measurer: ChatMeasurer,
    config: CompactionConfig,
    stats: CompactionStats | None = None,
    *,
    source_index: int | None = None,
) -> list[dict[str, Any]]:
    local_stats = stats if stats is not None else CompactionStats()
    messages = [dict(msg) for msg in example.get("messages", [])]
    tools = example.get("tools")
    if not messages:
        return []

    full_tokens = measurer.count_chat(messages, tools)
    if full_tokens <= config.max_sequence_length:
        local_stats.rows_unchanged += 1
        local_stats.add_output_length(full_tokens)
        local_stats.rows_out += 1
        return [
            output_row(
                messages,
                tools,
                {
                    "compaction": "unchanged",
                    "source_index": source_index,
                    "source_tokens": full_tokens,
                    "chunk_index": 0,
                },
            )
        ]

    rows: list[dict[str, Any]] = []
    system_prefix, body = leading_system_messages(messages)
    summary_context: dict[str, Any] | None = None
    chunk_index = 0
    split_count = 0

    while body:
        active = system_prefix + ([summary_context] if summary_context is not None else []) + body
        active_tokens = measurer.count_chat(active, tools)
        if active_tokens <= config.max_sequence_length:
            emit_if_trainable(
                rows,
                active,
                tools,
                {
                    "compaction": "final",
                    "source_index": source_index,
                    "source_tokens": full_tokens,
                    "chunk_index": chunk_index,
                    "token_count": active_tokens,
                },
                measurer,
                local_stats,
            )
            break

        split_idx = find_split_index(system_prefix, summary_context, body, tools, measurer, config)
        split_idx = min(max(split_idx, 1), len(body))
        if split_idx >= len(body):
            split_idx = max(1, len(body) - 1)

        prior_messages = system_prefix + ([summary_context] if summary_context is not None else []) + body[:split_idx]
        summary = build_extractive_summary(
            prior_messages,
            measurer,
            max_tokens=config.summary_token_budget,
        )

        if config.include_compaction:
            emitted_messages = prior_messages + [
                make_compaction_prompt_message(config.compaction_prompt),
                make_summary_assistant_message(summary),
            ]
        else:
            emitted_messages = prior_messages

        shrinks = 0
        while measurer.count_chat(emitted_messages, tools) > config.max_sequence_length and split_idx > 1:
            split_idx -= 1
            shrinks += 1
            prior_messages = system_prefix + ([summary_context] if summary_context is not None else []) + body[:split_idx]
            summary = build_extractive_summary(
                prior_messages,
                measurer,
                max_tokens=config.summary_token_budget,
            )
            emitted_messages = (
                prior_messages
                + [make_compaction_prompt_message(config.compaction_prompt), make_summary_assistant_message(summary)]
                if config.include_compaction
                else prior_messages
            )
            if shrinks >= config.max_split_shrink_steps:
                break

        emitted_messages, was_truncated = truncate_last_message_to_fit(
            emitted_messages,
            tools,
            measurer,
            config.max_sequence_length,
        )
        if was_truncated:
            local_stats.messages_truncated += 1
            summary = build_extractive_summary(
                emitted_messages,
                measurer,
                max_tokens=config.summary_token_budget,
            )

        emit_if_trainable(
            rows,
            emitted_messages,
            tools,
            {
                "compaction": "included" if config.include_compaction else "excluded",
                "source_index": source_index,
                "source_tokens": full_tokens,
                "chunk_index": chunk_index,
                "split_index": split_idx,
                "token_count": measurer.count_chat(emitted_messages, tools),
            },
            measurer,
            local_stats,
        )
        local_stats.rows_compacted += 1
        split_count += 1
        chunk_index += 1
        summary_context = make_summary_context_message(summary)
        body = body[split_idx:]

    if split_count == 0:
        return rows
    return rows


def iter_hf_examples(
    dataset_name: str,
    *,
    split: str,
    streaming: bool,
    max_examples: int,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    for idx, row in enumerate(dataset):
        if max_examples and idx >= max_examples:
            break
        normalized = normalize_row(row)
        if normalized is not None:
            yield normalized


def iter_hf_jsonl_examples(
    dataset_name: str,
    *,
    filename: str,
    max_examples: int,
) -> Iterator[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename=filename))
    emitted = 0
    for row in iter_jsonl_rows(path):
        normalized = normalize_row(row)
        if normalized is None:
            continue
        yield normalized
        emitted += 1
        if max_examples and emitted >= max_examples:
            return


def iter_input_examples(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    if args.dataset:
        if args.hf_jsonl_file:
            yield from iter_hf_jsonl_examples(
                args.dataset,
                filename=args.hf_jsonl_file,
                max_examples=args.max_examples,
            )
            return
        yield from iter_hf_examples(
            args.dataset,
            split=args.split,
            streaming=args.streaming,
            max_examples=args.max_examples,
        )
        return
    if args.input_root:
        yield from iter_normalized_examples(
            args.input_root,
            max_rows_per_file=args.max_rows_per_file,
            parquet_batch_size=args.parquet_batch_size,
            max_examples=args.max_examples,
            shuffle_files=args.shuffle_files,
            seed=args.seed,
        )
        return
    raise ValueError("pass either --dataset or --input-root")


def load_tokenizer(model: str, *, local_files_only: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def write_dataset(args: argparse.Namespace) -> int:
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite to rebuild it")
        shutil.rmtree(args.output_root)
    source_dir = args.output_root / args.source_name
    source_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model, local_files_only=args.local_files_only)
    chat_template = load_chat_template(args.chat_template)
    measurer = ChatMeasurer(tokenizer, chat_template=chat_template)
    config = CompactionConfig(
        max_sequence_length=args.max_sequence_length,
        boundary_tokens=args.boundary_tokens,
        include_compaction=args.mode == "included",
        summary_token_budget=args.summary_token_budget,
        compaction_prompt=args.compaction_prompt,
    )

    stats = CompactionStats()
    out_path = source_dir / "data.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for source_index, example in enumerate(iter_input_examples(args)):
            stats.rows_in += 1
            for row in compact_example(
                example,
                measurer,
                config,
                stats,
                source_index=source_index,
            ):
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        **stats.as_dict(),
        "dataset": args.dataset,
        "input_root": str(args.input_root) if args.input_root else None,
        "output_root": str(args.output_root),
        "output_jsonl": str(out_path),
        "model": args.model,
        "chat_template": str(args.chat_template),
        "mode": args.mode,
        "max_sequence_length": args.max_sequence_length,
        "boundary_tokens": args.boundary_tokens,
        "summary_token_budget": args.summary_token_budget,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def inspect_boundaries(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.model, local_files_only=args.local_files_only)
    measurer = ChatMeasurer(tokenizer, chat_template=load_chat_template(args.chat_template))
    result: list[dict[str, Any]] = []
    for idx, example in enumerate(iter_input_examples(args)):
        if idx >= args.inspect_examples:
            break
        messages = example["messages"]
        system_prefix, body = leading_system_messages(messages)
        prefix_counts: list[dict[str, int]] = []
        for end in range(1, len(body) + 1):
            tokens = measurer.count_chat(system_prefix + body[:end], example.get("tools"))
            if tokens >= args.boundary_tokens or end == len(body):
                prefix_counts.append({"message_index": end, "tokens": tokens})
                if len(prefix_counts) >= args.inspect_boundaries:
                    break
        result.append(
            {
                "source_index": idx,
                "messages": len(messages),
                "full_tokens": measurer.count_chat(messages, example.get("tools")),
                "boundary_probe": prefix_counts,
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        source = subparser.add_mutually_exclusive_group(required=True)
        source.add_argument("--dataset", help="Hugging Face dataset name, e.g. eewer/agent-traces-openai-style-all.")
        source.add_argument("--input-root", type=Path, help="Local raw root containing JSONL/parquet files.")
        subparser.add_argument("--split", default="train")
        subparser.add_argument(
            "--hf-jsonl-file",
            default="",
            help="Read a JSONL file from the HF dataset repo directly, bypassing datasets Arrow schema inference.",
        )
        subparser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
        subparser.add_argument("--max-examples", type=int, default=0)
        subparser.add_argument("--max-rows-per-file", type=int, default=0)
        subparser.add_argument("--parquet-batch-size", type=int, default=128)
        subparser.add_argument("--shuffle-files", action="store_true")
        subparser.add_argument("--seed", type=int, default=33333)
        subparser.add_argument("--model", default=DEFAULT_MODEL)
        subparser.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
        subparser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
        subparser.add_argument("--max-sequence-length", type=int, required=True)
        subparser.add_argument("--boundary-tokens", type=int, required=True)

    build = subparsers.add_parser("build")
    add_common(build)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--source-name", default="synthetic_compaction")
    build.add_argument("--mode", choices=["included", "excluded"], default="included")
    build.add_argument("--summary-token-budget", type=int, default=1536)
    build.add_argument("--compaction-prompt", default=DEFAULT_COMPACTION_PROMPT)
    build.add_argument("--overwrite", action="store_true")

    inspect = subparsers.add_parser("inspect-boundaries")
    add_common(inspect)
    inspect.add_argument("--inspect-examples", type=int, default=3)
    inspect.add_argument("--inspect-boundaries", type=int, default=4)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return write_dataset(args)
    if args.command == "inspect-boundaries":
        return inspect_boundaries(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
