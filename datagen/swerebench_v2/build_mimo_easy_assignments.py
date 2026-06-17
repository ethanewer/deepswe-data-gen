#!/usr/bin/env python3
"""Build easy-only MiMo v2.5 assignment CSVs for clean trajectory pilots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TASKS_CSV = Path(__file__).resolve().parent / "data" / "high_quality_conf_ge_0.95_tasks.csv"
MIMO_MODEL = "xiaomi/mimo-v2.5"
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
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-csv", type=Path, default=DEFAULT_TASKS_CSV)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument(
        "--instruction-style",
        default="mimo_clean_easy",
        help="Instruction-style label for downstream manifests. Use --config-file when running.",
    )
    return parser.parse_args(argv)


def int_value(row: dict[str, str], key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except ValueError:
        return 0


def float_value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except ValueError:
        return 0.0


def sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        int_value(row, "num_modified_files"),
        int_value(row, "num_modified_lines"),
        int_value(row, "fail_to_pass_count"),
        -float_value(row, "confidence"),
        row.get("language", ""),
        row.get("repo", ""),
        row.get("instance_id", ""),
    )


def load_easy_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    easy_rows = [row for row in rows if row.get("difficulty") == "easy"]
    easy_rows.sort(key=sort_key)
    return easy_rows


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    rows = load_easy_rows(args.tasks_csv)[: args.limit]
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        selected.append(
            {
                "selection_rank": index,
                "instance_id": row["instance_id"],
                "repo": row.get("repo", ""),
                "language": row.get("language", ""),
                "difficulty": "easy",
                "confidence": row.get("confidence", ""),
                "assigned_model": MIMO_MODEL,
                "instruction_style": args.instruction_style,
                "outside_original_high_quality_set": "false",
                "num_modified_files": row.get("num_modified_files", ""),
                "num_modified_lines": row.get("num_modified_lines", ""),
                "fail_to_pass_count": row.get("fail_to_pass_count", ""),
                "pass_to_pass_count": row.get("pass_to_pass_count", ""),
                "selection_policy": "easy_high_quality_smallest_change_first_mimo_v2_5",
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "source_tasks_csv": str(args.tasks_csv),
        "selected_count": len(selected),
        "difficulty": "easy",
        "assigned_model": MIMO_MODEL,
        "instruction_style": args.instruction_style,
        "selection_policy": "easy_high_quality_smallest_change_first_mimo_v2_5",
        "by_language": dict(Counter(row["language"] for row in selected)),
    }
    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
