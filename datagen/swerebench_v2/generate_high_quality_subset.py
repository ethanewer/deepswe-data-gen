#!/usr/bin/env python3
"""Generate the high-quality SWE-rebench V2 task subset.

Filters:
- language is in the union of the SWE-bench Multilingual predictive 30-task
  subset languages and the DeepSWE easiest 5-task subset languages
- difficulty is easy, medium, or hard
- annotation code is A
- intent completeness is complete
- no detected B issues
- no test alignment issues
- annotation confidence >= 0.95
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


LANGUAGES = ("c", "cpp", "go", "java", "js", "php", "python", "ruby", "rust", "ts")
DIFFICULTIES = ("easy", "medium", "hard")
MIN_CONFIDENCE = 0.95
DATASET_NAME = "nebius/SWE-rebench-V2"
SPLIT = "train"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"

FIELDS = [
    "instance_id",
    "repo",
    "language",
    "difficulty",
    "confidence",
    "created_at",
    "base_commit",
    "image_name",
    "license",
    "num_modified_files",
    "num_modified_lines",
    "fail_to_pass_count",
    "pass_to_pass_count",
    "pr_categories",
    "pr_labels",
    "pr_url",
]


def is_high_quality(row: dict) -> bool:
    llm_metadata = row["meta"]["llm_metadata"]

    if row["language"] not in LANGUAGES:
        return False
    if llm_metadata.get("difficulty") not in DIFFICULTIES:
        return False
    if llm_metadata.get("code") != "A":
        return False
    if llm_metadata.get("intent_completeness") != "complete":
        return False
    if llm_metadata.get("test_alignment_issues"):
        return False
    if any((llm_metadata.get("detected_issues") or {}).values()):
        return False

    confidence = llm_metadata.get("confidence")
    return confidence is not None and confidence >= MIN_CONFIDENCE


def export_row(row: dict) -> dict:
    llm_metadata = row["meta"]["llm_metadata"]
    meta = row["meta"]

    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "language": row["language"],
        "difficulty": llm_metadata["difficulty"],
        "confidence": llm_metadata["confidence"],
        "created_at": row["created_at"],
        "base_commit": row["base_commit"],
        "image_name": row["image_name"],
        "license": row["license"],
        "num_modified_files": meta["num_modified_files"],
        "num_modified_lines": meta["num_modified_lines"],
        "fail_to_pass_count": len(row["FAIL_TO_PASS"] or []),
        "pass_to_pass_count": len(row["PASS_TO_PASS"] or []),
        "pr_categories": "|".join(llm_metadata.get("pr_categories") or []),
        "pr_labels": "|".join(meta.get("pr_labels") or []),
        "pr_url": meta.get("pr_url") or "",
    }


def build_summary(rows: list[dict]) -> dict:
    return {
        "source_dataset": DATASET_NAME,
        "split": SPLIT,
        "filters": {
            "language": list(LANGUAGES),
            "difficulty": list(DIFFICULTIES),
            "meta.llm_metadata.code": "A",
            "meta.llm_metadata.intent_completeness": "complete",
            "meta.llm_metadata.detected_issues": "all false",
            "meta.llm_metadata.test_alignment_issues": "empty",
            "meta.llm_metadata.confidence": f">= {MIN_CONFIDENCE}",
        },
        "total": len(rows),
        "by_language": dict(Counter(row["language"] for row in rows)),
        "by_difficulty": dict(Counter(row["difficulty"] for row in rows)),
        "matrix": {
            language: {
                difficulty: sum(
                    1
                    for row in rows
                    if row["language"] == language and row["difficulty"] == difficulty
                )
                for difficulty in DIFFICULTIES
            }
            for language in LANGUAGES
        },
        "files": [
            "high_quality_conf_ge_0.95_tasks.csv",
            "high_quality_conf_ge_0.95_instance_ids.txt",
            "high_quality_conf_ge_0.95_easy_instance_ids.txt",
            "high_quality_conf_ge_0.95_medium_instance_ids.txt",
            "high_quality_conf_ge_0.95_hard_instance_ids.txt",
        ],
    }


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "high_quality_conf_ge_0.95_tasks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ids_path = output_dir / "high_quality_conf_ge_0.95_instance_ids.txt"
    with ids_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['instance_id']}\n")

    for difficulty in DIFFICULTIES:
        ids_by_difficulty_path = (
            output_dir / f"high_quality_conf_ge_0.95_{difficulty}_instance_ids.txt"
        )
        with ids_by_difficulty_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if row["difficulty"] == difficulty:
                    handle.write(f"{row['instance_id']}\n")

    summary_path = output_dir / "high_quality_conf_ge_0.95_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(build_summary(rows), handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the high-quality SWE-rebench V2 subset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the CSV, instance-id list, and summary JSON are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(DATASET_NAME, split=SPLIT)

    rows = [export_row(row) for row in dataset if is_high_quality(row)]
    rows.sort(key=lambda row: (row["language"], row["difficulty"], row["repo"], row["instance_id"]))

    write_outputs(rows, args.output_dir)
    print(f"Wrote {len(rows)} tasks to {args.output_dir}")


if __name__ == "__main__":
    main()
