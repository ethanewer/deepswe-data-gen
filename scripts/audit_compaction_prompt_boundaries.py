#!/usr/bin/env python3
"""Audit prompt naturalness and retained-suffix boundaries for compacted rows."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROCESS_MARKER_RE = re.compile(
    r"("
    r"create a new initial mini-swe task prompt|compacted successful trajectory|"
    r"discarded[- ]prefix|retained suffix|cut turn|replacement prompt|replacement user message|"
    r"compaction_model_reasoning|mini_swe_task_prompt|current_repo_state|actions_already_taken|"
    r"uncertainty_notes|full discarded prefix|included discarded prefix|trajectory editing|"
    r"this compaction pipeline|this json conversion|json repair|original row id:"
    r")",
    re.IGNORECASE,
)
STATE_CONTEXT_RE = re.compile(
    r"("
    r"current(?:ly)?|investigat|examined|found|identified|file|test|failure|error|"
    r"command|diff|patch|implementation|behavior|module|function|class|method"
    r")",
    re.IGNORECASE,
)
TASK_INTENT_RE = re.compile(
    r"(fix|implement|update|change|add|remove|ensure|make|support|handle|debug|resolve|issue|bug|failing)",
    re.IGNORECASE,
)
CONTINUATION_MARKER_RE = re.compile(
    r"("
    r"\bcontinue\b|\balready\b|previous attempt|progress so far|what has already been done|"
    r"current repository state|already modified|already applied|fixes already applied|"
    r"has been (?:addressed|updated|changed|modified|implemented|added|removed|fixed|partially)|"
    r"have been (?:addressed|updated|changed|modified|implemented|added|removed|fixed|partially)|"
    r"was (?:addressed|updated|changed|modified|implemented|added|removed|fixed)|"
    r"were (?:addressed|updated|changed|modified|implemented|added|removed|fixed)|"
    r"implementation includes|changes? (?:made|introduced|include)|remaining (?:work|task|issue|step)|"
    r"next step|finish(?:es|ing)? (?:the )?(?:remaining )?(?:work|task)|"
    r"just verify|only need(?:s)? to verify|generate (?:the )?patch|submit (?:the )?(?:final )?(?:answer|output)|"
    r"required interface|interface notes|recommended workflow|command execution rules|finish by issuing|"
    r"complete_task_and_submit_final_output|##\s*boundaries"
    r")",
    re.IGNORECASE,
)
PATH_CONTRADICTION_RE = re.compile(
    r"(/testbed\s+(?:does not|doesn't)\s+exist|repo(?:sitory)?\s+(?:is\s+)?(?:located|checked out)\s+at\s+/(?!testbed\b))",
    re.IGNORECASE,
)
IMPERATIVE_TEST_EDIT_RE = re.compile(
    r"("
    r"(?<!non-)(?:add|create|modify|update|change|write)\s+(?:a\s+|new\s+|the\s+)?(?:unit\s+|integration\s+)?tests?\b|"
    r"(?:add|create|modify|update|change|write).{0,80}\btests?/"
    r")",
    re.IGNORECASE | re.DOTALL,
)
DOMAIN_COMPACTION_OK_RE = re.compile(r"\b(compaction|compact|compactor)\b", re.IGNORECASE)


def prompt_requests_test_edits(prompt: str) -> bool:
    test_edit_match = IMPERATIVE_TEST_EDIT_RE.search(prompt)
    if not test_edit_match:
        return False
    negative_test_boundary = re.search(
        r"do not (?:add|create|modify|update|change|write) (?:a |new |the )?(?:unit |integration )?tests?|"
        r"\bnon-test (?:source )?files?\b",
        prompt,
        re.IGNORECASE,
    )
    return negative_test_boundary is None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-sample-limit", type=int, default=5000)
    parser.add_argument("--source-load-limit", type=int, default=0)
    return parser.parse_args()


def iter_zst_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with subprocess.Popen(
        ["zstdcat", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row
        proc.wait()
        if proc.returncode:
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            raise RuntimeError(f"zstdcat failed for {path}: {stderr}")


def iter_dataset_rows(dataset: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    for path in sorted((dataset / "data").glob("*.jsonl.zst")):
        for row in iter_zst_jsonl(path):
            yield path.name, row


def text_from_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(text_from_content(item.get("text", item.get("content", item))))
            else:
                parts.append(text_from_content(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value:
                return text_from_content(value[key])
    return str(value)


def role(message: dict[str, Any]) -> str:
    return str(message.get("role") or "").lower()


def messages_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if isinstance(messages, list):
        return [item for item in messages if isinstance(item, dict)]
    return []


def first_user_index(messages: list[dict[str, Any]]) -> int | None:
    for index, message in enumerate(messages):
        if role(message) == "user":
            return index
    return None


def first_assistant_after(messages: list[dict[str, Any]], start: int) -> tuple[int | None, dict[str, Any] | None]:
    for index in range(start + 1, len(messages)):
        message = messages[index]
        if role(message) == "assistant":
            return index, message
    return None, None


def command_from_assistant(message: dict[str, Any] | None) -> str:
    if not isinstance(message, dict):
        return ""
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                    if isinstance(parsed, dict) and isinstance(parsed.get("command"), str):
                        return parsed["command"]
                except json.JSONDecodeError:
                    return arguments
            if isinstance(arguments, dict) and isinstance(arguments.get("command"), str):
                return arguments["command"]
    content = text_from_content(message.get("content"))
    marker = "complete_task_and_submit_final_output"
    if marker in content:
        return marker
    return content.strip()


def command_category(command: str) -> str:
    stripped = command.strip()
    lower = stripped.lower()
    if not lower:
        return "no_command"
    if "complete_task_and_submit_final_output" in lower:
        return "submit"
    if re.search(r"(^|\s)(pytest|tox|npm test|pnpm test|yarn test|go test|cargo test|mvn test|gradle test|make test|ctest|rspec|phpunit)\b", lower):
        return "test"
    if re.search(r"(^|\s)(apply_patch|sed -i|perl -pi|python[0-9.]*\s+- <<|python[0-9.]*\s+-c|cat\s+>|tee\s+|printf\s+.*>)", lower):
        return "edit"
    if re.search(r"(^|\s)git\s+(diff|status|show)\b", lower):
        return "diff_status"
    if re.search(r"(^|\s)(rg|grep|find|ls|pwd|sed -n|cat|head|tail|awk|tree)\b", lower):
        return "inspect"
    if re.search(r"(^|\s)(pip install|npm install|pnpm install|yarn install|go mod|cargo fetch|composer install)\b", lower):
        return "setup"
    return "other"


def source_messages(path_text: str, load_cache: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not path_text:
        return []
    if path_text in load_cache:
        return load_cache[path_text]
    path = Path(path_text)
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        load_cache[path_text] = []
        return []
    messages = []
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        messages = [item for item in data["messages"] if isinstance(item, dict)]
    load_cache[path_text] = messages
    return messages


def prompt_flags(prompt: str, first_category: str) -> list[str]:
    flags: list[str] = []
    normalized = prompt.strip()
    if not normalized:
        return ["empty_prompt"]
    if PROCESS_MARKER_RE.search(normalized):
        flags.append("process_marker")
    if CONTINUATION_MARKER_RE.search(normalized):
        flags.append("continuation_marker")
    if PATH_CONTRADICTION_RE.search(normalized):
        flags.append("path_contradiction")
    if prompt_requests_test_edits(normalized):
        flags.append("requests_test_edits")
    if len(normalized) < 180:
        flags.append("too_short_for_stateful_compaction")
    if len(normalized) > 9000:
        flags.append("very_long_prompt")
    if not TASK_INTENT_RE.search(normalized):
        flags.append("missing_clear_task_intent")
    if first_category in {"edit", "test", "diff_status", "submit"} and not STATE_CONTEXT_RE.search(normalized):
        flags.append("insufficient_state_for_non_inspect_start")
    if first_category == "submit":
        flags.append("starts_with_submit")
    if first_category == "diff_status":
        flags.append("starts_with_diff_status")
    return flags


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_out = args.output_dir / "compaction_prompt_boundary_audit.jsonl"
    sample_out = args.output_dir / "flagged_compaction_prompt_boundary_samples.jsonl"
    summary_out = args.output_dir / "compaction_prompt_boundary_audit_summary.json"

    source_cache: dict[str, list[dict[str, Any]]] = {}
    compact_rows: list[tuple[str, dict[str, Any]]] = []
    base_first_categories: Counter[str] = Counter()
    base_rows_seen = 0

    for shard, row in iter_dataset_rows(args.dataset):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        is_compacted = (
            metadata.get("row_source") == "compaction_prefix_v2"
            or row.get("compaction_model_response") is not None
            or metadata.get("source") == "deepswe-compaction"
        )
        if is_compacted:
            compact_rows.append((shard, row))
            continue
        if base_rows_seen < args.base_sample_limit:
            messages = messages_of(row)
            user_index = first_user_index(messages)
            assistant_index, assistant = first_assistant_after(messages, user_index if user_index is not None else -1)
            if assistant_index is not None:
                base_first_categories[command_category(command_from_assistant(assistant))] += 1
                base_rows_seen += 1

    first_categories: Counter[str] = Counter()
    flags_counter: Counter[str] = Counter()
    by_language: Counter[str] = Counter()
    problematic: list[dict[str, Any]] = []
    flagged_rows = 0
    rows_written = 0
    source_loads = 0

    with row_out.open("w", encoding="utf-8") as out:
        for shard, row in compact_rows:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            messages = messages_of(row)
            user_index = first_user_index(messages)
            assistant_index, assistant = first_assistant_after(messages, user_index if user_index is not None else -1)
            command = command_from_assistant(assistant)
            category = command_category(command)
            first_categories[category] += 1
            language = str(metadata.get("language") or "").lower() or "unknown"
            by_language[language] += 1
            prompt = str(row.get("prompt") or metadata.get("prompt") or "")
            if not prompt and user_index is not None:
                prompt = text_from_content(messages[user_index].get("content"))
            flags = prompt_flags(prompt, category)

            cut_turn_index = metadata.get("cut_turn_index")
            source_path = str(metadata.get("source_trajectory_path") or metadata.get("trajectory_path") or "")
            source_boundary: dict[str, Any] = {}
            if isinstance(cut_turn_index, int) and source_path and (args.source_load_limit <= 0 or source_loads < args.source_load_limit):
                before_cache_size = len(source_cache)
                original_messages = source_messages(source_path, source_cache)
                if len(source_cache) > before_cache_size:
                    source_loads += 1
                previous_message = original_messages[cut_turn_index - 1] if 0 < cut_turn_index <= len(original_messages) else None
                next_message = original_messages[cut_turn_index] if 0 <= cut_turn_index < len(original_messages) else None
                source_boundary = {
                    "source_previous_role": role(previous_message) if isinstance(previous_message, dict) else "",
                    "source_next_role": role(next_message) if isinstance(next_message, dict) else "",
                    "source_next_category": command_category(command_from_assistant(next_message if isinstance(next_message, dict) else None)),
                    "source_next_matches_compacted_first": (
                        command_from_assistant(next_message if isinstance(next_message, dict) else None).strip()[:500]
                        == command.strip()[:500]
                    ),
                }
                if source_boundary["source_next_role"] != "assistant":
                    flags.append("source_boundary_not_assistant")
                if not source_boundary["source_next_matches_compacted_first"]:
                    flags.append("source_boundary_mismatch")

            for flag in flags:
                flags_counter[flag] += 1
            if flags:
                flagged_rows += 1

            audit_row = {
                "uuid": row.get("uuid"),
                "task_id": row.get("task_id") or metadata.get("task_id"),
                "language": language,
                "shard": shard,
                "request_id": metadata.get("compaction_request_id"),
                "compaction_original_row_id": metadata.get("compaction_original_row_id"),
                "prompt_chars": len(prompt),
                "prompt_preview": prompt[:500],
                "first_assistant_index": assistant_index,
                "first_command_category": category,
                "first_command_preview": command[:500],
                "cut_turn_index": cut_turn_index,
                "retained_message_count": metadata.get("retained_message_count"),
                "original_estimated_token_length": metadata.get("original_estimated_token_length"),
                "compacted_estimated_token_length": metadata.get("compacted_estimated_token_length"),
                "flags": sorted(set(flags)),
                **source_boundary,
            }
            out.write(json.dumps(audit_row, ensure_ascii=False, sort_keys=True) + "\n")
            rows_written += 1
            if flags and len(problematic) < 300:
                problematic.append(audit_row)

    with sample_out.open("w", encoding="utf-8") as out:
        for row in problematic:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "dataset": str(args.dataset),
        "compacted_rows": rows_written,
        "base_rows_sampled": base_rows_seen,
        "base_first_command_categories": dict(sorted(base_first_categories.items())),
        "compacted_first_command_categories": dict(sorted(first_categories.items())),
        "compacted_rows_by_language": dict(sorted(by_language.items())),
        "flag_counts": dict(sorted(flags_counter.items())),
        "flagged_rows": flagged_rows,
        "flag_instances": sum(flags_counter.values()),
        "row_audit_path": str(row_out),
        "flagged_sample_path": str(sample_out),
        "source_trajectories_loaded": source_loads,
    }
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
