#!/usr/bin/env python3
"""Select a first targeted SWE-rebench V2 MiMo wave.

This intentionally keeps the policy simple and auditable: use existing
high-quality SWE-rebench V2 metadata, exclude tasks already represented by
strict successful traces, and prioritize the languages called out in the
dataset limitation notes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BASE = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swerebench-traces-raw-source-plus-all-local-generated-plus-other-sources-exact-20260616-0745"
)
DEFAULT_TASKS_CSV = Path(__file__).resolve().parents[2] / "datagen" / "swerebench_v2" / "data" / "high_quality_conf_ge_0.95_tasks.csv"
TARGETS = {
    "c": 120,
    "cpp": 120,
    "java": 120,
    "rust": 120,
    "php": 100,
    "go": 80,
    "js": 60,
    "ts": 60,
}
FIELDS = [
    "selection_rank",
    "instance_id",
    "repo",
    "language",
    "difficulty",
    "confidence",
    "assigned_model",
    "instruction_style",
    "outside_original_high_quality_set",
    "num_modified_files",
    "num_modified_lines",
    "fail_to_pass_count",
    "pass_to_pass_count",
    "selection_policy",
    "existing_raw_count",
    "existing_strict_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-raw-dataset", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--tasks-csv", type=Path, default=DEFAULT_TASKS_CSV)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--languages", default=",".join(TARGETS))
    parser.add_argument("--per-language", default="")
    parser.add_argument("--difficulty", action="append", default=[])
    parser.add_argument("--include-existing-strict", action="store_true")
    parser.add_argument("--assigned-model", default="xiaomi/mimo-v2.5")
    parser.add_argument("--instruction-style", default="original")
    return parser.parse_args()


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "passed", "submitted"}
    return False


def int_value(row: dict[str, Any], key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def float_value(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def iter_index_rows(base: Path) -> list[Path]:
    metadata = base / "metadata"
    preferred = [
        "parent_index.jsonl",
        "appended_full_index.jsonl",
        "appended_index.jsonl",
        "other_sources_exact_index.jsonl",
    ]
    return [metadata / name for name in preferred if (metadata / name).exists()]


def strict_success(row: dict[str, Any]) -> bool:
    reasoning = float_value(row, "percent_messages_with_reasoning")
    return (
        boolish(row.get("passed"))
        and float_value(row, "reward") > 0
        and str(row.get("agent_exit_status") or "") == "Submitted"
        and int_value(row, "model_patch_bytes") > 0
        and int_value(row, "api_calls") > 0
        and reasoning >= 0.9
    )


def load_existing_counts(base: Path) -> tuple[Counter[str], Counter[str], Counter[str]]:
    raw_counts: Counter[str] = Counter()
    strict_counts: Counter[str] = Counter()
    strict_by_language: Counter[str] = Counter()
    for path in iter_index_rows(base):
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = str(row.get("task_id") or row.get("instance_id") or "")
                if not task_id:
                    continue
                raw_counts[task_id] += 1
                if strict_success(row):
                    strict_counts[task_id] += 1
                    strict_by_language[str(row.get("language") or "")] += 1
    return raw_counts, strict_counts, strict_by_language


def parse_targets(languages: list[str], override: str) -> dict[str, int]:
    targets = {language: TARGETS.get(language, 50) for language in languages}
    if not override:
        return targets
    for part in override.split(","):
        if not part.strip():
            continue
        language, value = part.split("=", 1)
        targets[language.strip()] = int(value)
    return targets


def sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        row.get("difficulty") != "easy",
        int_value(row, "num_modified_files"),
        int_value(row, "num_modified_lines"),
        int_value(row, "fail_to_pass_count"),
        -float_value(row, "confidence"),
        row.get("repo", ""),
        row.get("instance_id", ""),
    )


def main() -> None:
    args = parse_args()
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    targets = parse_targets(languages, args.per_language)
    raw_counts, strict_counts, strict_by_language = load_existing_counts(args.base_raw_dataset)
    allowed_difficulties = set(args.difficulty or ["easy"])

    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.tasks_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row = {key: (value or "").strip() for key, value in row.items()}
            language = row.get("language", "")
            task_id = row.get("instance_id", "")
            if language not in targets or row.get("difficulty") not in allowed_difficulties:
                continue
            if not args.include_existing_strict and strict_counts[task_id] > 0:
                continue
            if int_value(row, "num_modified_files") > 2:
                continue
            if int_value(row, "num_modified_lines") > 80:
                continue
            candidates[language].append(row)

    selected: list[dict[str, Any]] = []
    for language in languages:
        rows = sorted(candidates.get(language, []), key=sort_key)
        for row in rows[: targets[language]]:
            selected.append(
                {
                    "selection_rank": len(selected),
                    "instance_id": row["instance_id"],
                    "repo": row.get("repo", ""),
                    "language": language,
                    "difficulty": row.get("difficulty", ""),
                    "confidence": row.get("confidence", ""),
                    "assigned_model": args.assigned_model,
                    "instruction_style": args.instruction_style,
                    "outside_original_high_quality_set": "false",
                    "num_modified_files": row.get("num_modified_files", ""),
                    "num_modified_lines": row.get("num_modified_lines", ""),
                    "fail_to_pass_count": row.get("fail_to_pass_count", ""),
                    "pass_to_pass_count": row.get("pass_to_pass_count", ""),
                    "selection_policy": "limitations_initial_easy_small_diff_no_existing_strict",
                    "existing_raw_count": raw_counts[row["instance_id"]],
                    "existing_strict_count": strict_counts[row["instance_id"]],
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "base_raw_dataset": str(args.base_raw_dataset),
        "tasks_csv": str(args.tasks_csv),
        "selected_count": len(selected),
        "targets": targets,
        "candidate_counts": {language: len(rows) for language, rows in sorted(candidates.items())},
        "selected_by_language": dict(Counter(row["language"] for row in selected)),
        "selected_by_difficulty": dict(Counter(row["difficulty"] for row in selected)),
        "existing_strict_by_language": dict(sorted(strict_by_language.items())),
        "filters": {
            "difficulties": sorted(allowed_difficulties),
            "max_modified_files": 2,
            "max_modified_lines": 80,
            "include_existing_strict": args.include_existing_strict,
        },
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
