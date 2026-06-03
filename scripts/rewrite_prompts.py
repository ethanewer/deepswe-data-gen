#!/usr/bin/env python3
"""Rewrite selected SWE-rebench prompts into DeepSWE-style task prompts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "nebius/SWE-rebench-V2"
SPLIT = "train"
TASKS_CSV = REPO_ROOT / "swerebench-v2" / "high_quality_conf_ge_0.95_tasks.csv"
PROMPT_ANALYSIS_CSV = REPO_ROOT / "swerebench-v2" / "prompt_analysis.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "swerebench-v2" / "rewritten-prompts"
DEFAULT_MODEL = "gpt-5.4-mini"
LANGUAGES = ("python", "ts", "go")
DIFFICULTIES = ("easy", "medium", "hard")


SYSTEM_PROMPT = """\
You rewrite software engineering benchmark prompts.

Goal: transform public issue/PR-style text into a concise DeepSWE-style task
prompt. DeepSWE-style prompts are short, natural, behavior-focused, and do not
tell the agent where the fix is or how to implement it.

Rules:
- Preserve every user-visible behavioral requirement.
- Remove issue template boilerplate, external URLs, PR/test references, and
  solution hints.
- Do not mention tests, hidden tests, patches, PRs, metadata, confidence,
  difficulty, files changed, or exact file paths.
- Do not invent requirements.
- Do not name specific source files unless the original requirement is an API
  addition whose public name is essential.
- Keep the prompt actionable for an agent exploring the repository.
- Prefer one to four short paragraphs or a short bullet list.

Return strict JSON with these keys:
- rewritten_prompt: string
- preserved_requirements: array of strings
- removed_noise: array of strings
- risk_notes: array of strings
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite selected prompts with an LLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tasks-file", type=Path, default=TASKS_CSV)
    parser.add_argument("--prompt-analysis-file", type=Path, default=PROMPT_ANALYSIS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--difficulty", choices=DIFFICULTIES)
    parser.add_argument("--language", choices=LANGUAGES)
    parser.add_argument(
        "--needs-change-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer prompts flagged by prompt_analysis.csv.",
    )
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def load_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["instance_id"]: row for row in csv.DictReader(handle)}


def selected_instance_ids(args: argparse.Namespace) -> list[str]:
    compact = load_csv_by_id(args.tasks_file)
    analysis = load_csv_by_id(args.prompt_analysis_file)
    requested = set(args.instance_id)
    selected = []

    for instance_id, row in compact.items():
        if requested and instance_id not in requested:
            continue
        if args.difficulty and row["difficulty"] != args.difficulty:
            continue
        if args.language and row["language"] != args.language:
            continue
        if args.needs_change_only:
            analysis_row = analysis.get(instance_id)
            if analysis_row and analysis_row["needs_deepswe_prompt_change"] != "True":
                continue
        selected.append(instance_id)
        if len(selected) >= args.limit:
            break
    return selected


def iter_rows_by_id(instance_ids: Iterable[str]) -> Iterable[dict]:
    wanted = set(instance_ids)
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    for row in dataset:
        if row["instance_id"] in wanted:
            yield row


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def user_prompt(row: dict) -> str:
    interface = normalize_text(row["interface"] or "")
    if interface == "No new interfaces are introduced.":
        interface = ""
    return "\n".join(
        [
            "Rewrite this task prompt.",
            "",
            f"Repository: {row['repo']}",
            f"Language: {row['language']}",
            "",
            "Original prompt:",
            normalize_text(row["problem_statement"] or ""),
            "",
            "Generated interface notes to use only if they describe required public behavior:",
            interface or "(none)",
        ]
    )


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data.get("rewritten_prompt"), str) or not data["rewritten_prompt"].strip():
        raise ValueError("missing rewritten_prompt")
    for key in ("preserved_requirements", "removed_noise", "risk_notes"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"missing list: {key}")
    data["rewritten_prompt"] = data["rewritten_prompt"].strip()
    return data


def rewrite_prompt(client: OpenAI, model: str, row: dict) -> tuple[dict, str]:
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_prompt(row),
        max_output_tokens=1200,
    )
    output_text = response.output_text
    return parse_json_response(output_text), output_text


def write_markdown(path: Path, record: dict) -> None:
    task = record["task"]
    rewrite = record["rewrite"]
    original = record["original"]
    lines = [
        f"# {task['instance_id']}",
        "",
        f"- repo: {task['repo']}",
        f"- language: {task['language']}",
        f"- difficulty: {task['difficulty']}",
        "",
        "## Rewritten Prompt",
        "",
        rewrite["rewritten_prompt"],
        "",
        "## Preserved Requirements",
        "",
    ]
    lines.extend(f"- {item}" for item in rewrite["preserved_requirements"])
    lines.extend(["", "## Removed Noise", ""])
    lines.extend(f"- {item}" for item in rewrite["removed_noise"])
    lines.extend(["", "## Risk Notes", ""])
    lines.extend(f"- {item}" for item in rewrite["risk_notes"])
    lines.extend(["", "## Original Prompt", "", original["problem_statement"]])
    if original["interface"]:
        lines.extend(["", "## Original Interface", "", original["interface"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    ids = selected_instance_ids(args)
    if not ids:
        raise SystemExit("No prompts matched the requested filters")

    compact = load_csv_by_id(args.tasks_file)
    id_order = {instance_id: index for index, instance_id in enumerate(ids)}
    rows = sorted(iter_rows_by_id(ids), key=lambda row: id_order[row["instance_id"]])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "rewrites.jsonl"
    client = OpenAI()

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            rewrite, raw_output = rewrite_prompt(client, args.model, row)
            task_meta = compact[row["instance_id"]]
            record = {
                "model": args.model,
                "task": task_meta,
                "rewrite": rewrite,
                "original": {
                    "problem_statement": normalize_text(row["problem_statement"] or ""),
                    "interface": normalize_text(row["interface"] or ""),
                },
                "raw_output": raw_output,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            markdown_path = args.output_dir / f"{row['instance_id']}.md"
            write_markdown(markdown_path, record)
            print(f"rewrote {row['instance_id']}")
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

    print(f"output: {jsonl_path}")


if __name__ == "__main__":
    main()
