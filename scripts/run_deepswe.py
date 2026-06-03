#!/usr/bin/env python3
"""Smoke-run DeepSWE-style task prompts through the OpenAI API.

This does not execute repository tests or apply patches. It verifies that the
task subset can be loaded and that the configured model can produce a concise
plan/patch sketch for selected tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS_FILE = REPO_ROOT / "swerebench-v2" / "high_quality_conf_ge_0.95_tasks.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs"
DEFAULT_MODEL = "gpt-5.4-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small OpenAI smoke test over selected DeepSWE tasks."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use.")
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=DEFAULT_TASKS_FILE,
        help="CSV task metadata file generated from SWE-rebench V2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSONL run outputs.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Maximum tasks to run.")
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium", "hard"),
        help="Optional difficulty filter.",
    )
    parser.add_argument(
        "--language",
        choices=("python", "ts", "go"),
        help="Optional language filter.",
    )
    return parser.parse_args()


def load_tasks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_tasks(
    tasks: Iterable[dict[str, str]],
    *,
    difficulty: str | None,
    language: str | None,
    limit: int,
) -> list[dict[str, str]]:
    selected = []
    for task in tasks:
        if difficulty and task["difficulty"] != difficulty:
            continue
        if language and task["language"] != language:
            continue
        selected.append(task)
        if len(selected) >= limit:
            break
    return selected


def build_prompt(task: dict[str, str]) -> str:
    return "\n".join(
        [
            "You are smoke-testing a DeepSWE data-generation pipeline.",
            "Given this SWE-rebench task metadata, return a concise JSON object",
            "with keys: instance_id, likely_area, first_steps, risk.",
            "",
            f"instance_id: {task['instance_id']}",
            f"repo: {task['repo']}",
            f"language: {task['language']}",
            f"difficulty: {task['difficulty']}",
            f"confidence: {task['confidence']}",
            f"num_modified_files: {task['num_modified_files']}",
            f"num_modified_lines: {task['num_modified_lines']}",
            f"fail_to_pass_count: {task['fail_to_pass_count']}",
            f"pass_to_pass_count: {task['pass_to_pass_count']}",
            f"pr_categories: {task['pr_categories']}",
        ]
    )


def call_model(client: OpenAI, *, model: str, task: dict[str, str]) -> str:
    response = client.responses.create(
        model=model,
        input=build_prompt(task),
        max_output_tokens=300,
    )
    return response.output_text


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    tasks = select_tasks(
        load_tasks(args.tasks_file),
        difficulty=args.difficulty,
        language=args.language,
        limit=args.limit,
    )
    if not tasks:
        raise SystemExit("No tasks matched the requested filters")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / f"deepswe_smoke_{timestamp}.jsonl"

    client = OpenAI()
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            output_text = call_model(client, model=args.model, task=task)
            record = {
                "model": args.model,
                "task": task,
                "output_text": output_text,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"wrote {task['instance_id']}")

    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
