#!/usr/bin/env python3
"""Select a small SWE-bench Multilingual diagnostic subset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TASKS_CSV = Path(
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/benchmarks/"
    "swebench_multilingual/predictive_30_tasks.csv"
)


def score(row: dict[str, str]) -> tuple[float, ...]:
    solved = sum(
        int(row[name])
        for name in (
            "resolved_gpt-5-mini",
            "resolved_claude-opus-4.7",
            "resolved_deepseek-v4-flash",
            "resolved_glm-5",
            "resolved_minimax-m2.5",
        )
    )
    return (
        solved,
        float(row["full_model_solve_rate"]),
        float(row["all_available_solve_rate"]),
        -float(row["patch_changed_lines"]),
        -float(row["fail_to_pass_count"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-csv", type=Path, default=TASKS_CSV)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    with args.tasks_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    strict_rows = [
        row
        for row in rows
        if row["resolved_gpt-5-mini"] == "1"
        and row["resolved_glm-5"] == "1"
        and float(row["full_model_solve_rate"]) >= 0.5
    ]
    relaxed_rows = [
        row
        for row in rows
        if float(row["full_model_solve_rate"]) >= 0.5
        and (
            row["resolved_glm-5"] == "1"
            or row["resolved_deepseek-v4-flash"] == "1"
            or row["resolved_minimax-m2.5"] == "1"
        )
    ]
    strict_rows.sort(key=score, reverse=True)
    relaxed_rows.sort(key=score, reverse=True)

    selected: list[dict[str, str]] = []
    seen_languages: set[str] = set()
    for candidate_rows in (strict_rows, relaxed_rows):
        for row in candidate_rows:
            if row["language_group"] in seen_languages:
                continue
            selected.append(row)
            seen_languages.add(row["language_group"])
            if len(selected) >= args.count:
                break
        if len(selected) >= args.count:
            break
    for row in strict_rows + relaxed_rows:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= args.count:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in selected[: args.count]:
            f.write(row["instance_id"] + "\n")
    for row in selected[: args.count]:
        print(
            "\t".join(
                [
                    row["instance_id"],
                    row["language_group"],
                    row["repo"],
                    row["full_model_solve_rate"],
                    row["patch_changed_lines"],
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
