#!/usr/bin/env python3
"""Analyze SWE-rebench prompts against DeepSWE-style prompt criteria."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


MODULE_DIR = Path(__file__).resolve().parent
DATASET_NAME = "nebius/SWE-rebench-V2"
SPLIT = "train"
TASKS_CSV = MODULE_DIR / "data" / "high_quality_conf_ge_0.95_tasks.csv"
DEFAULT_CSV = MODULE_DIR / "data" / "prompt_analysis.csv"
DEFAULT_MD = MODULE_DIR / "data" / "prompt_style_analysis.md"


FIELDS = [
    "instance_id",
    "language",
    "difficulty",
    "prompt_chars",
    "interface_chars",
    "url_count",
    "code_fence_count",
    "checkbox_line_count",
    "file_path_hint_count",
    "signature_block_count",
    "needs_deepswe_prompt_change",
    "reasons",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prompt style.")
    parser.add_argument("--tasks-file", type=Path, default=TASKS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def load_task_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["instance_id"]: row for row in csv.DictReader(handle)}


def count_regex(pattern: str, text: str, flags: int = 0) -> int:
    return len(re.findall(pattern, text, flags))


def analyze_row(row: dict, compact_meta: dict[str, str]) -> dict[str, str | int | bool]:
    prompt = row["problem_statement"] or ""
    interface = row["interface"] or ""
    url_count = count_regex(r"https?://\S+", prompt)
    code_fence_count = count_regex(r"```", prompt)
    checkbox_line_count = count_regex(r"(?m)^\s*[-*]\s*\[[ xX]\]", prompt)
    file_path_hint_count = count_regex(r"\b[\w./-]+\.(?:py|ts|tsx|js|jsx|go)\b", prompt)
    signature_block_count = count_regex(
        r"(?im)^\s*(Method|Function|Class|Location|Inputs|Outputs):", interface
    )

    reasons = []
    if len(prompt) > 2500:
        reasons.append("long_prompt")
    if interface and interface.strip() != "No new interfaces are introduced.":
        reasons.append("interface_dump")
    if url_count:
        reasons.append("external_urls")
    if code_fence_count:
        reasons.append("code_fences")
    if checkbox_line_count:
        reasons.append("issue_template_boilerplate")
    if file_path_hint_count:
        reasons.append("file_path_hints")
    if signature_block_count:
        reasons.append("signature_blocks")

    return {
        "instance_id": row["instance_id"],
        "language": row["language"],
        "difficulty": compact_meta["difficulty"],
        "prompt_chars": len(prompt),
        "interface_chars": len(interface),
        "url_count": url_count,
        "code_fence_count": code_fence_count,
        "checkbox_line_count": checkbox_line_count,
        "file_path_hint_count": file_path_hint_count,
        "signature_block_count": signature_block_count,
        "needs_deepswe_prompt_change": bool(reasons),
        "reasons": "|".join(reasons),
    }


def write_csv(path: Path, rows: list[dict[str, str | int | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str | int | bool]]) -> None:
    total = len(rows)
    needs_change = sum(1 for row in rows if row["needs_deepswe_prompt_change"])
    by_reason = Counter(
        reason
        for row in rows
        for reason in str(row["reasons"]).split("|")
        if reason
    )
    by_difficulty = Counter(
        row["difficulty"] for row in rows if row["needs_deepswe_prompt_change"]
    )
    by_language = Counter(row["language"] for row in rows if row["needs_deepswe_prompt_change"])

    prompt_lengths = sorted(int(row["prompt_chars"]) for row in rows)
    p50 = prompt_lengths[int((total - 1) * 0.50)]
    p90 = prompt_lengths[int((total - 1) * 0.90)]
    p95 = prompt_lengths[int((total - 1) * 0.95)]

    lines = [
        "# Prompt Style Analysis",
        "",
        "Basis: DeepSWE emphasizes short, natural, behavior-focused prompts, broad",
        "repository exploration, and behavioral verifiers. SWE-rebench prompts are",
        "derived from public issues/PRs, so many include issue-template boilerplate,",
        "external links, implementation hints, or generated interface blocks.",
        "",
        "## Conclusion",
        "",
        (
            f"Yes. {needs_change:,} of {total:,} high-quality confidence-filtered "
            "SWE-rebench prompts contain at least one signal that should be changed "
            "or reviewed before treating them as DeepSWE-style prompts."
        ),
        "",
        "The generated Harbor tasks therefore default to `instruction-style=deepswe`,",
        "which removes the generated interface section and keeps only a cleaned,",
        "natural task request. That improves prompt shape, but it does not make the",
        "tasks contamination-free or replace DeepSWE's hand-authored behavioral",
        "verifiers.",
        "",
        "## Prompt Length",
        "",
        f"- p50: {p50:,} characters",
        f"- p90: {p90:,} characters",
        f"- p95: {p95:,} characters",
        "",
        "## Change Signals",
        "",
    ]
    for reason, count in by_reason.most_common():
        lines.append(f"- {reason}: {count:,}")

    lines.extend(["", "## Needs Change By Difficulty", ""])
    for difficulty in ("easy", "medium", "hard"):
        lines.append(f"- {difficulty}: {by_difficulty[difficulty]:,}")

    lines.extend(["", "## Needs Change By Language", ""])
    for language in ("python", "ts", "go"):
        lines.append(f"- {language}: {by_language[language]:,}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    compact_by_id = load_task_metadata(args.tasks_file)
    dataset = load_dataset(DATASET_NAME, split=SPLIT)

    rows = []
    for row in dataset:
        compact_meta = compact_by_id.get(row["instance_id"])
        if compact_meta:
            rows.append(analyze_row(row, compact_meta))

    rows.sort(key=lambda row: (str(row["language"]), str(row["difficulty"]), str(row["instance_id"])))
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"Wrote {len(rows)} prompt analyses to {args.output_csv}")
    print(f"Wrote summary to {args.output_md}")


if __name__ == "__main__":
    main()
