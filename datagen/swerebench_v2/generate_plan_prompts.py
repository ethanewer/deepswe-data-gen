#!/usr/bin/env python3
"""Generate benchmark-aligned assisted prompts from SWE-rebench gold patches."""

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

from datagen.swerebench_v2.generate_harbor_tasks import clean_issue_prompt
from datagen.swerebench_v2.rewrite_prompts import (
    DATASET_NAME,
    DIFFICULTIES,
    LANGUAGES,
    SPLIT,
    TASKS_CSV,
    extract_edge_case_literals,
    extract_public_symbols,
    normalize_text,
)
from eval.paths import configure_ca_bundle


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = MODULE_DIR / "examples" / "planned-prompts"
DEFAULT_MODEL = "gpt-5.4-mini"


SYSTEM_PROMPT = """\
You write benchmark-aligned assisted prompts for software engineering agents.

You will receive a benchmark task and its gold patch. Use the patch only to
infer a small amount of high-level teacher guidance that would help an agent
solve very complex tasks during rollout. A separate SFT prompt will remove this
guidance, so the assisted rollout prompt should help the teacher avoid common
failure modes without becoming an answer key.

Rules:
- Keep the original task as the primary object. The extra guidance should feel
  like concise task clarification, not a replacement for exploration.
- Preserve the task's user-visible behavior and important public API names.
- Give strategic guidance about likely components, invariants, edge cases, and
  validation focus. Prefer subsystem descriptions over file names.
- Target common rollout failures: plausible but incomplete patches, narrow
  smoke-test validation, submitting after unresolved local failures, getting
  lost in large subsystems, and missing compatibility or cross-feature effects.
- Avoid a full sequential recipe. Do not tell the agent exactly every edit to
  make; leave room for normal benchmark-style investigation.
- Do not include exact code, pseudocode close enough to paste, diffs, imports,
  function bodies, literals copied only from the patch, line numbers, or exact
  source file paths.
- Do not reveal the gold patch, mention that a gold patch exists, or claim that
  hidden tests exist.
- Do not overfit to tests. Describe behavioral validation at a high level.
- Prefer 2 to 5 short guidance bullets, with validation notes only when useful.

Return strict JSON with these keys:
- plan_steps: array of strings
- preserved_requirements: array of strings
- validation_notes: array of strings
- leakage_risk_notes: array of strings
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-mode", choices=("responses", "chat"), default="responses")
    parser.add_argument("--max-output-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--extra-body-json", help="JSON object passed as provider-specific extra_body.")
    parser.add_argument(
        "--response-format-json",
        action="store_true",
        help="Request chat-completions JSON mode with response_format={type: json_object}.",
    )
    parser.add_argument("--tasks-file", type=Path, default=TASKS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--difficulty", choices=DIFFICULTIES)
    parser.add_argument("--language", choices=LANGUAGES)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--fail-on-leakage-warnings",
        action="store_true",
        help="Exit non-zero if a generated plan appears to leak exact patch content.",
    )
    return parser.parse_args()


def load_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["instance_id"]: row for row in csv.DictReader(handle)}


def selected_instance_ids(args: argparse.Namespace) -> list[str]:
    compact = load_csv_by_id(args.tasks_file)
    requested = set(args.instance_id)
    selected = []
    for instance_id, row in compact.items():
        if requested and instance_id not in requested:
            continue
        if args.difficulty and row["difficulty"] != args.difficulty:
            continue
        if args.language and row["language"] != args.language:
            continue
        selected.append(instance_id)
        if not requested and len(selected) >= args.limit:
            break
    return selected


def iter_rows_by_id(instance_ids: Iterable[str]) -> Iterable[dict]:
    wanted = set(instance_ids)
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    for row in dataset:
        if row["instance_id"] in wanted:
            yield row


def build_task_prompt(row: dict) -> str:
    return clean_issue_prompt(row["problem_statement"] or "", strip_urls=True)


def user_prompt(row: dict) -> str:
    interface = normalize_text(row["interface"] or "")
    if interface == "No new interfaces are introduced.":
        interface = ""
    public_symbols = extract_public_symbols(interface)
    public_symbols_text = ", ".join(public_symbols) if public_symbols else "(none)"
    return "\n".join(
        [
            "Create a benchmark-aligned assisted prompt for this task.",
            "",
            f"Repository: {row['repo']}",
            f"Language: {row['language']}",
            "",
            "Task prompt the agent will see:",
            build_task_prompt(row),
            "",
            "Public interface and compatibility notes:",
            interface or "(none)",
            "",
            "Likely public symbols to preserve in the assisted prompt:",
            public_symbols_text,
            "",
            "Gold patch for private planning only. Do not copy code, exact paths, or diff text:",
            normalize_text(row["patch"] or ""),
        ]
    )


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data.get("plan_steps"), list) or not data["plan_steps"]:
        raise ValueError("missing non-empty plan_steps")
    for key in ("preserved_requirements", "validation_notes", "leakage_risk_notes"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"missing list: {key}")
    data["plan_steps"] = [str(step).strip() for step in data["plan_steps"] if str(step).strip()]
    if not data["plan_steps"]:
        raise ValueError("empty plan_steps")
    return data


def generate_plan(
    client: OpenAI,
    model: str,
    row: dict,
    max_output_tokens: int,
    api_mode: str,
    temperature: float,
    extra_body: dict | None = None,
    response_format_json: bool = False,
) -> tuple[dict, str]:
    if api_mode == "chat":
        request_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(row)},
            ],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        if response_format_json:
            request_kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**request_kwargs)
        output_text = response.choices[0].message.content or ""
    else:
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=user_prompt(row),
            max_output_tokens=max_output_tokens,
        )
        output_text = response.output_text
    return parse_json_response(output_text), output_text


def normalize_snippet(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def patch_added_snippets(patch: str) -> list[str]:
    snippets = []
    for line in patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        value = line[1:].strip()
        if len(value) < 24:
            continue
        if value.startswith(("//", "#", "/*", "*")):
            continue
        if re.fullmatch(r"[{}\[\](),.;:=\s]+", value):
            continue
        snippets.append(value)
    return snippets


def quality_checked_text(plan: dict) -> str:
    return "\n".join(
        str(item)
        for key in ("plan_steps", "preserved_requirements", "validation_notes")
        for item in plan.get(key, [])
    )


def validate_plan_quality(row: dict, plan: dict) -> list[str]:
    text = quality_checked_text(plan)
    normalized = normalize_snippet(text)
    warnings = []

    if "```" in text:
        warnings.append("contains_code_fence")
    if re.search(r"(?m)^\s*(diff --git|[+-]{3}\s|@@\s)", text):
        warnings.append("contains_diff_marker")
    if re.search(r"\b[\w./-]+\.(?:c|cc|cpp|h|hpp|go|py|ts|tsx|js|jsx|java|rs|rb|php)\b", text):
        warnings.append("contains_exact_file_path")

    for snippet in patch_added_snippets(row["patch"] or ""):
        if normalize_snippet(snippet) in normalized:
            warnings.append(f"copies_patch_line:{snippet[:80]}")
            break

    for symbol in extract_public_symbols(normalize_text(row["interface"] or "")):
        if symbol.lower() not in normalized:
            parts = [part.lower() for part in symbol.split(".") if part]
            if not parts or not all(re.search(rf"\b{re.escape(part)}\b", normalized) for part in parts):
                warnings.append(f"missing_public_symbol:{symbol}")

    original = normalize_text(row["problem_statement"] or "")
    for literal in extract_edge_case_literals(original):
        if literal == "":
            if not re.search(r'\bempty string\b|""', text, re.I):
                warnings.append("missing_edge_literal:<empty string>")
            continue
        if literal.lower() not in normalized:
            warnings.append(f"missing_edge_literal:{literal}")

    return warnings


def sft_prompt(row: dict) -> str:
    return build_task_prompt(row).strip() + "\n"


def planned_prompt(row: dict, plan: dict) -> str:
    lines = [build_task_prompt(row).strip(), "", "Additional guidance:"]
    lines.extend(f"- {step}" for step in plan["plan_steps"])
    if plan.get("preserved_requirements"):
        lines.extend(["", "Compatibility requirements:"])
        lines.extend(f"- {requirement}" for requirement in plan["preserved_requirements"])
    if plan.get("validation_notes"):
        lines.extend(["", "Validation focus:"])
        lines.extend(f"- {note}" for note in plan["validation_notes"])
    return "\n".join(lines).strip() + "\n"


def add_prompt_variants(row: dict, plan: dict) -> None:
    rollout_prompt = planned_prompt(row, plan)
    no_hint_prompt = sft_prompt(row)
    plan["rollout_prompt"] = rollout_prompt
    plan["hinted_prompt"] = rollout_prompt
    plan["sft_prompt"] = no_hint_prompt
    # Backward-compatible name consumed by generate_harbor_tasks.
    plan["planned_prompt"] = rollout_prompt


def write_markdown(path: Path, record: dict) -> None:
    task = record["task"]
    plan = record["plan"]
    original = record["original"]
    lines = [
        f"# {task['instance_id']}",
        "",
        f"- repo: {task['repo']}",
        f"- language: {task['language']}",
        f"- difficulty: {task['difficulty']}",
        "",
        "## Planned Prompt",
        "",
        plan["planned_prompt"],
        "",
        "## SFT Prompt",
        "",
        plan.get("sft_prompt", ""),
        "",
        "## Plan Steps",
        "",
    ]
    lines.extend(f"- {item}" for item in plan["plan_steps"])
    lines.extend(["", "## Preserved Requirements", ""])
    lines.extend(f"- {item}" for item in plan["preserved_requirements"])
    lines.extend(["", "## Validation Notes", ""])
    lines.extend(f"- {item}" for item in plan["validation_notes"])
    lines.extend(["", "## Leakage Risk Notes", ""])
    lines.extend(f"- {item}" for item in plan["leakage_risk_notes"])
    if record["quality_warnings"]:
        lines.extend(["", "## Quality Warnings", ""])
        lines.extend(f"- {item}" for item in record["quality_warnings"])
    lines.extend(["", "## Original Prompt", "", original["problem_statement"]])
    if original["interface"]:
        lines.extend(["", "## Original Interface", "", original["interface"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")
    env = os.environ.copy()
    configure_ca_bundle(env)
    os.environ.update({key: env[key] for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE") if key in env})

    ids = selected_instance_ids(args)
    if not ids:
        raise SystemExit("No tasks matched the requested filters")

    compact = load_csv_by_id(args.tasks_file)
    id_order = {instance_id: index for index, instance_id in enumerate(ids)}
    rows = sorted(iter_rows_by_id(ids), key=lambda row: id_order[row["instance_id"]])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "plans.jsonl"
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None
    client_kwargs = {"api_key": api_key}
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    client = OpenAI(**client_kwargs)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            for attempt in range(1, args.max_retries + 1):
                try:
                    plan, raw_output = generate_plan(
                        client,
                        args.model,
                        row,
                        args.max_output_tokens,
                        args.api_mode,
                        args.temperature,
                        extra_body,
                        args.response_format_json,
                    )
                    break
                except Exception as exc:
                    if attempt == args.max_retries:
                        raise
                    print(
                        f"retry {row['instance_id']} attempt {attempt + 1}/{args.max_retries}: "
                        f"{type(exc).__name__}: {str(exc)[:300]}",
                        flush=True,
                    )
                    time.sleep(max(args.sleep_seconds, 1.0) * attempt)
            quality_warnings = validate_plan_quality(row, plan)
            add_prompt_variants(row, plan)
            task_meta = compact[row["instance_id"]]
            record = {
                "model": args.model,
                "task": task_meta,
                "plan": plan,
                "quality_warnings": quality_warnings,
                "original": {
                    "problem_statement": normalize_text(row["problem_statement"] or ""),
                    "interface": normalize_text(row["interface"] or ""),
                },
                "raw_output": raw_output,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            write_markdown(args.output_dir / f"{row['instance_id']}.md", record)
            print(f"planned {row['instance_id']}")
            for warning in quality_warnings:
                print(f"warning {row['instance_id']}: {warning}")
            if quality_warnings and args.fail_on_leakage_warnings:
                raise SystemExit(
                    f"quality warnings for {row['instance_id']}: {', '.join(quality_warnings)}"
                )
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

    print(f"output: {jsonl_path}")


if __name__ == "__main__":
    main()
