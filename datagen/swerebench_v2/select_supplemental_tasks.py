#!/usr/bin/env python3
"""Select supplemental SWE-rebench V2 tasks for reasoning data generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "nebius/SWE-rebench-V2"
SPLIT = "train"
LANGUAGES = {"c", "cpp", "go", "java", "js", "php", "python", "ruby", "rust", "ts"}
DIFFICULTIES = {"easy", "medium"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-unique", type=int, default=45_000)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--high-quality-ids",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "high_quality_conf_ge_0.95_instance_ids.txt",
    )
    parser.add_argument(
        "--metadata-index",
        type=Path,
        action="append",
        default=[],
        help="Dataset metadata/index JSONL files to use as local reasoning-covered task IDs.",
    )
    parser.add_argument(
        "--include-gate-failures-if-needed",
        action="store_true",
        help="After strict quality-gate rows, include lower-quality easy/medium rows if still below target.",
    )
    parser.add_argument(
        "--instruction-style-policy",
        choices=("deepswe", "original", "alternate"),
        default="alternate",
    )
    parser.add_argument("--assigned-model", default="local qwen3.6-35b-a3b-fp8")
    return parser.parse_args()


def load_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "passed"}
    return False


def local_reasoning_ids(index_paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in index_paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = row.get("task_id") or row.get("instance_id")
                if not task_id:
                    continue
                assistant_count = int(row.get("assistant_message_count") or 0)
                percent = float(row.get("percent_messages_with_reasoning") or 0.0)
                has_all = boolish(row.get("has_all_assistant_reasoning"))
                if assistant_count > 0 and (has_all or percent >= 1.0):
                    ids.add(str(task_id))
    return ids


def gate_failures(metadata: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if metadata.get("code") != "A":
        failures.append("code")
    if metadata.get("intent_completeness") != "complete":
        failures.append("intent")
    if metadata.get("test_alignment_issues"):
        failures.append("test_alignment")
    if any((metadata.get("detected_issues") or {}).values()):
        failures.append("detected_issues")
    return failures


def instruction_style(policy: str, index: int) -> str:
    if policy == "alternate":
        return "deepswe" if index % 2 else "original"
    return policy


def main() -> None:
    args = parse_args()
    high_quality_ids = load_ids(args.high_quality_ids)
    covered_ids = local_reasoning_ids(args.metadata_index)
    excluded_ids = high_quality_ids | covered_ids
    needed = max(0, args.target_unique - len(covered_ids))

    strict_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    for row in dataset:
        instance_id = row["instance_id"]
        language = row["language"]
        metadata = row["meta"]["llm_metadata"]
        difficulty = metadata.get("difficulty")
        if instance_id in excluded_ids:
            continue
        if language not in LANGUAGES or difficulty not in DIFFICULTIES:
            continue
        failures = gate_failures(metadata)
        record = {
            "instance_id": instance_id,
            "repo": row["repo"],
            "language": language,
            "difficulty": difficulty,
            "confidence": metadata.get("confidence"),
            "assigned_model": args.assigned_model,
            "instruction_style": "",
            "outside_original_high_quality_set": "true",
            "quality_gate_pass": "true" if not failures else "false",
            "quality_gate_failures": ",".join(failures),
            "quality_tier": "strict_confidence_relaxed" if not failures else "fallback_gate_relaxed",
            "annotation_code": metadata.get("code", ""),
            "intent_completeness": metadata.get("intent_completeness", ""),
            "test_alignment_issue_count": len(metadata.get("test_alignment_issues") or []),
            "detected_issue_count": sum(1 for value in (metadata.get("detected_issues") or {}).values() if value),
            "selection_policy": (
                "strict_gates_confidence_relaxed"
                if not failures
                else "fallback_after_strict_gates_exhausted"
            ),
        }
        if failures:
            fallback_rows.append(record)
        else:
            strict_rows.append(record)

    def sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
        confidence = record["confidence"]
        confidence_value = float(confidence) if confidence is not None else -1.0
        return (
            -confidence_value,
            record["difficulty"] != "easy",
            record["language"],
            record["instance_id"],
        )

    strict_rows.sort(key=sort_key)
    fallback_rows.sort(
        key=lambda record: (
            int(record["detected_issue_count"]) + int(record["test_alignment_issue_count"]),
            record["quality_gate_failures"],
            *sort_key(record),
        )
    )

    selected = strict_rows[:needed]
    if args.include_gate_failures_if_needed and len(selected) < needed:
        selected.extend(fallback_rows[: needed - len(selected)])

    for index, record in enumerate(selected):
        record["selection_rank"] = index
        record["instruction_style"] = instruction_style(args.instruction_style_policy, index)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "selection_rank",
        "instance_id",
        "repo",
        "language",
        "difficulty",
        "confidence",
        "assigned_model",
        "instruction_style",
        "outside_original_high_quality_set",
        "quality_gate_pass",
        "quality_gate_failures",
        "quality_tier",
        "annotation_code",
        "intent_completeness",
        "test_alignment_issue_count",
        "detected_issue_count",
        "selection_policy",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "target_unique": args.target_unique,
        "local_reasoning_unique_count": len(covered_ids),
        "high_quality_excluded_count": len(high_quality_ids),
        "needed_to_target": needed,
        "strict_candidate_count": len(strict_rows),
        "fallback_candidate_count": len(fallback_rows),
        "selected_count": len(selected),
        "target_reachable_with_selected": len(selected) >= needed,
        "target_reachable_with_all_easy_medium_allowed": len(strict_rows) + len(fallback_rows) >= needed,
        "selected_by_difficulty": Counter(record["difficulty"] for record in selected),
        "selected_by_language": Counter(record["language"] for record in selected),
        "selected_by_quality_tier": Counter(record["quality_tier"] for record in selected),
    }
    serializable = {
        key: (dict(value) if isinstance(value, Counter) else value)
        for key, value in summary.items()
    }
    args.summary_json.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(serializable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
