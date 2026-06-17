#!/usr/bin/env python3
"""Audit compacted trajectory outputs without filtering them.

The audit is intentionally metadata-only. It reads compaction output
directories, joins optional source quality signals, and writes row-level flags
that downstream SFT builders can use for filtering or weighting.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TOKENIZER = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_CHAT_TEMPLATE = Path(
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen-mimo-clean-harness/"
    "eval/chat_templates/qwen3_thinking_acc.jinja2"
)
BANNED_PROMPT_RE = re.compile(
    r"("
    r"create a new initial mini-swe task prompt|"
    r"compacted successful trajectory|"
    r"discarded[- ]prefix|"
    r"retained suffix|"
    r"cut turn index|"
    r"replacement user message|"
    r"replacement prompt|"
    r"original row id:|"
    r"compaction variant:|"
    r"full discarded prefix|"
    r"mini_swe_task_prompt|"
    r"json repair"
    r")",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compaction-output", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-quality-signals", type=Path, action="append", default=[])
    parser.add_argument("--aligned-quality-signals", type=Path, action="append", default=[])
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER)
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--qwen-token-audit", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token-threshold", type=int, default=65_000)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def load_quality(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in iter_jsonl(path):
            uuid = str(row.get("uuid") or "")
            if uuid:
                rows[uuid] = row
    return rows


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "passed", "submitted"}
    return False


def message_role(message: dict[str, Any]) -> str:
    return str(message.get("role") or "").lower()


def text_from_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text_from_content(item) for item in value)
    if isinstance(value, dict):
        return text_from_content(value.get("text", value.get("content", "")))
    return str(value)


def has_submit_marker(messages: list[Any]) -> bool:
    marker = "complete_task_and_submit_final_output"
    for message in messages:
        if not isinstance(message, dict) or message_role(message) != "assistant":
            continue
        if marker in text_from_content(message.get("content")).lower():
            return True
        if marker in json.dumps(message.get("tool_calls", ""), ensure_ascii=False, default=str).lower():
            return True
    return False


def count_reasoning(messages: list[Any]) -> tuple[int, int]:
    assistant = 0
    with_reasoning = 0
    for message in messages:
        if not isinstance(message, dict) or message_role(message) != "assistant":
            continue
        assistant += 1
        provider = message.get("provider_specific_fields")
        candidates = [
            message.get("reasoning"),
            message.get("reasoning_content"),
            message.get("thinking"),
            provider.get("reasoning") if isinstance(provider, dict) else None,
            provider.get("reasoning_content") if isinstance(provider, dict) else None,
        ]
        if any(text_from_content(candidate).strip() for candidate in candidates):
            with_reasoning += 1
    return assistant, with_reasoning


def load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if args.chat_template and args.chat_template.exists():
        tokenizer.chat_template = args.chat_template.read_text(encoding="utf-8")
    return tokenizer


def qwen_tokens(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_quality = load_quality(args.raw_quality_signals)
    aligned_quality = load_quality(args.aligned_quality_signals)
    tokenizer = load_tokenizer(args) if args.qwen_token_audit else None

    indexes: list[dict[str, Any]] = []
    records: list[tuple[Path, dict[str, Any]]] = []
    for output in args.compaction_output:
        for row in iter_jsonl(output / "compaction_index.jsonl"):
            row["_compaction_output"] = str(output)
            indexes.append(row)
        for row in iter_jsonl(output / "compaction_records.jsonl"):
            row["_compaction_output"] = str(output)
            records.append((output, row))

    status_counts = Counter(str(row.get("status") or "missing") for row in indexes)
    source_ids = Counter(str(row.get("original_row_id") or "") for row in indexes if row.get("original_row_id"))
    compacted_ids: Counter[str] = Counter()
    task_patch = Counter()
    row_flags: list[dict[str, Any]] = []

    for output, wrapper in records:
        raw_record = wrapper.get("raw_record") if isinstance(wrapper.get("raw_record"), dict) else {}
        metadata = raw_record.get("metadata") if isinstance(raw_record.get("metadata"), dict) else {}
        messages = raw_record.get("messages") if isinstance(raw_record.get("messages"), list) else []
        prompt = str(raw_record.get("prompt") or metadata.get("prompt") or "")
        response = raw_record.get("compaction_model_response")
        response = response if isinstance(response, dict) else {}
        compaction_model_reasoning = str(
            raw_record.get("compaction_model_reasoning")
            or metadata.get("compaction_model_reasoning")
            or response.get("compaction_model_reasoning")
            or ""
        ).strip()
        assistant_count, assistant_with_reasoning = count_reasoning(messages)
        reasoning_fraction = (assistant_with_reasoning / assistant_count) if assistant_count else 0.0
        uuid = str(raw_record.get("uuid") or "")
        compacted_ids[uuid] += 1
        task_id = str(raw_record.get("task_id") or metadata.get("task_id") or "")
        patch_hash = str(metadata.get("model_patch_sha256") or raw_record.get("model_patch_sha256") or "")
        if task_id and patch_hash:
            task_patch[(task_id, patch_hash)] += 1

        original_id = str(metadata.get("compaction_original_row_id") or "")
        raw_signal = raw_quality.get(original_id, {})
        aligned_signal = aligned_quality.get(original_id, {})
        token_error = ""
        token_count = None
        if tokenizer is not None:
            try:
                token_count = qwen_tokens(tokenizer, messages)
            except Exception as exc:  # noqa: BLE001 - keep audit row instead of failing the batch
                token_error = f"{type(exc).__name__}: {exc}"

        row_flags.append(
            {
                "uuid": uuid,
                "task_id": task_id,
                "language": str(metadata.get("language") or ""),
                "request_id": str(wrapper.get("request_id") or metadata.get("compaction_request_id") or ""),
                "compaction_output": str(output),
                "compaction_original_row_id": original_id,
                "prompt_chars": len(prompt),
                "prompt_has_compaction_marker": bool(BANNED_PROMPT_RE.search(prompt)),
                "has_compaction_model_reasoning": bool(compaction_model_reasoning),
                "compaction_model_reasoning_chars": len(compaction_model_reasoning),
                "assistant_message_count": assistant_count,
                "assistant_messages_with_reasoning": assistant_with_reasoning,
                "percent_messages_with_reasoning": reasoning_fraction,
                "reasoning_under_90pct": reasoning_fraction < 0.9,
                "has_submit_marker": has_submit_marker(messages),
                "execution_validated_after_compaction": bool_from_any(
                    metadata.get("execution_validated_after_compaction")
                ),
                "model_patch_bytes": int(metadata.get("model_patch_bytes") or 0),
                "model_patch_sha256": patch_hash,
                "source_strict_basic_quality": raw_signal.get("strict_basic_quality"),
                "source_failure_class": raw_signal.get("failure_class"),
                "source_last_submit_quality": aligned_signal.get("last_submit_quality"),
                "source_trainable_submit_patch_available_strict": aligned_signal.get(
                    "trainable_submit_patch_available_strict"
                ),
                "source_good_submission_behavior_available": aligned_signal.get(
                    "good_submission_behavior_available"
                ),
                "source_test_evidence": aligned_signal.get("test_evidence"),
                "qwen3_thinking_tokens": token_count,
                "under_65k_qwen3_thinking": (token_count < args.token_threshold) if token_count is not None else None,
                "qwen_token_error": token_error,
            }
        )

    duplicate_compacted_uuids = {key: value for key, value in compacted_ids.items() if key and value > 1}
    duplicate_task_patch = {
        f"{task_id}::{patch_hash}": value for (task_id, patch_hash), value in task_patch.items() if value > 1
    }
    duplicate_source_ids = {key: value for key, value in source_ids.items() if key and value > 1}

    by_language = Counter(row["language"] for row in row_flags)
    prompt_leaks = sum(1 for row in row_flags if row["prompt_has_compaction_marker"])
    missing_compaction_reasoning = sum(1 for row in row_flags if not row["has_compaction_model_reasoning"])
    reasoning_bad = sum(1 for row in row_flags if row["reasoning_under_90pct"])
    missing_submit = sum(1 for row in row_flags if not row["has_submit_marker"])
    tokenized = [row for row in row_flags if row["qwen3_thinking_tokens"] is not None]
    under_65k = sum(1 for row in tokenized if row["under_65k_qwen3_thinking"])

    with (args.output_dir / "compaction_row_quality.jsonl").open("w", encoding="utf-8") as handle:
        for row in row_flags:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "compaction_outputs": [str(path) for path in args.compaction_output],
        "index_rows": len(indexes),
        "record_rows": len(row_flags),
        "status_counts": dict(sorted(status_counts.items())),
        "by_language": dict(sorted(by_language.items())),
        "prompt_leak_rows": prompt_leaks,
        "missing_compaction_model_reasoning_rows": missing_compaction_reasoning,
        "reasoning_under_90pct_rows": reasoning_bad,
        "missing_submit_marker_rows": missing_submit,
        "duplicate_compacted_uuids": duplicate_compacted_uuids,
        "duplicate_task_patch_rows": duplicate_task_patch,
        "duplicate_source_index_ids": duplicate_source_ids,
        "qwen_token_audit": bool(args.qwen_token_audit),
        "qwen_tokenized_rows": len(tokenized),
        "under_65k_qwen3_thinking_rows": under_65k,
        "under_65k_qwen3_thinking_fraction": (under_65k / len(tokenized)) if tokenized else None,
    }
    (args.output_dir / "compaction_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
