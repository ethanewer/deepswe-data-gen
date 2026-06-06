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

from eval.paths import configure_ca_bundle


MODULE_DIR = Path(__file__).resolve().parent
DATASET_NAME = "nebius/SWE-rebench-V2"
SPLIT = "train"
TASKS_CSV = MODULE_DIR / "data" / "high_quality_conf_ge_0.95_tasks.csv"
PROMPT_ANALYSIS_CSV = MODULE_DIR / "data" / "prompt_analysis.csv"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "examples" / "rewritten-prompts"
DEFAULT_MODEL = "gpt-5.4-mini"
LANGUAGES = ("c", "cpp", "go", "java", "js", "php", "python", "ruby", "rust", "ts")
DIFFICULTIES = ("easy", "medium", "hard")


SYSTEM_PROMPT = """\
You rewrite software engineering benchmark prompts.

Goal: transform public issue/PR-style text into a concise DeepSWE-style task
prompt. DeepSWE-style prompts are short, natural, behavior-focused, and do not
tell the agent where the fix is or how to implement it.

Rules:
- Preserve every user-visible behavioral requirement.
- Preserve public API names, class/function/method names, CLI flags, config keys,
  import/export compatibility, and exact input/output literals when they define
  the requested behavior.
- Preserve public interface compatibility details from generated interface
  notes when they describe callable signatures, constructor/default argument
  behavior, return shapes, exported struct fields, serialized/config keys, or
  importable/exported names. Summarize those contracts naturally; do not copy
  source locations.
- Remove issue template boilerplate, external URLs, PR/test references, and
  solution hints.
- Do not mention tests, hidden tests, patches, PRs, metadata, confidence,
  difficulty, files changed, or exact file paths.
- Do not invent requirements.
- Do not name specific source files unless the original requirement is an API
  addition whose public import path is essential.
- Do not erase a bug's concrete observable edge case just because the original
  issue also included an implementation hint. Keep the symptom and expected
  behavior; remove only the internal diagnosis.
- Write in the same natural language as the original task. Do not introduce
  characters from unrelated writing systems except when they are present in
  original code/API literals.
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
    parser.add_argument("--api-base", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=1200)
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
    parser.add_argument(
        "--fail-on-quality-warnings",
        action="store_true",
        help="Exit non-zero if a rewrite drops likely public symbols or exact edge-case literals.",
    )
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


def normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_public_symbols(interface: str) -> list[str]:
    """Return likely public API symbols that should survive a rewrite.

    The generated interface block is noisy, so this intentionally only reads
    declaration-like lines. Inputs, outputs, descriptions, and file locations
    are not treated as required prompt content.
    """
    if not interface or interface.strip() == "No new interfaces are introduced.":
        return []

    symbols = []
    for line in interface.splitlines():
        match = re.match(r"\s*(Method|Function|Class|CLI|Command|Option):\s*(.+)", line, re.I)
        if not match:
            continue
        declaration = match.group(2).strip()
        declaration = re.split(r"\s+[–-]\s+|\s+-\s+|\s+in\s+", declaration, maxsplit=1)[0]
        declaration = declaration.split("(", 1)[0]
        declaration = declaration.strip("` .")
        if not declaration:
            continue

        candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", declaration)
        for candidate in candidates:
            if candidate.lower() in {"method", "function", "class", "cli", "command", "option"}:
                continue
            if candidate not in symbols:
                symbols.append(candidate)
    return symbols


def extract_edge_case_literals(text: str) -> list[str]:
    """Return exact short literals that look like behavior-defining values."""
    if not text:
        return []

    fenced_ranges = markdown_fenced_ranges(text)
    literals = []
    quoted_literal_pattern = r'"([^"\n]{0,80})"|(?<!\w)\'([^\'\n]{0,80})\'(?!\w)'
    for match in re.finditer(quoted_literal_pattern, text):
        if in_ranges(match.start(), fenced_ranges):
            continue
        literal = match.group(1) if match.group(1) is not None else match.group(2)
        literal = literal.strip()
        if should_keep_literal(literal, text, match.start(), match.end()) and literal not in literals:
            literals.append(literal)

    if re.search(r"\bempty string\b", text, re.I) and "" not in literals:
        literals.append("")
    return literals


def markdown_fenced_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    fence_start = None
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            if fence_start is None:
                fence_start = offset
            else:
                ranges.append((fence_start, offset + len(line)))
                fence_start = None
        offset += len(line)
    if fence_start is not None:
        ranges.append((fence_start, len(text)))
    return ranges


def in_ranges(position: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in ranges)


def should_keep_literal(literal: str, text: str, start: int, end: int) -> bool:
    context = text[max(0, start - 120) : min(len(text), end + 120)]
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]

    if literal.startswith(("http://", "https://")):
        return False
    if len(literal) > 40:
        return False
    if re.fullmatch(r"X{2,}", literal):
        return False
    if literal.lower().startswith("screenshot "):
        return False
    if literal.endswith("="):
        return False
    if re.search(r"\.{2,}", literal):
        return False
    if re.fullmatch(r"\d+", literal):
        return False
    if re.search(r"\.(?:py|ts|tsx|js|jsx|go|md|json|yaml|yml|toml)(?::\d+)?$", literal):
        return False
    if re.fullmatch(r"v?\d+(?:\.\d+){1,3}", literal):
        return False
    if re.fullmatch(r"[A-Fa-f0-9]{12,}", literal):
        return False
    if not re.search(
        r"\b(actual|bug|currently|error|expect|expected|instead|must|returns?|should|throw|throws?|wrong)\b",
        context,
        re.I,
    ):
        return False
    if re.search(r"\b(example|e\\.g\\.|sample)\b", line, re.I) and not re.search(
        r"\b(expect|expected|must|should|return|returns?)\b", line, re.I
    ):
        return False
    return True


def rewrite_search_text(rewrite: dict) -> str:
    return "\n".join(
        [
            rewrite.get("rewritten_prompt", ""),
            "\n".join(rewrite.get("preserved_requirements", [])),
        ]
    )


def contains_literal(text: str, literal: str) -> bool:
    if re.fullmatch(r"[A-Za-z0-9_]+", literal):
        return re.search(rf"\b{re.escape(literal)}\b", text, re.I) is not None
    return normalize_for_search(literal) in normalize_for_search(text)


def contains_public_symbol(text: str, symbol: str) -> bool:
    normalized_text = normalize_for_search(text)
    normalized_symbol = normalize_for_search(symbol)
    if normalized_symbol in normalized_text:
        return True
    if "." in symbol:
        parts = [part for part in symbol.split(".") if part]
        if len(parts) == 2:
            qualifier, member = parts
            return re.search(rf"\b{re.escape(member)}\b", text, re.I) is not None and re.search(
                rf"\b{re.escape(qualifier)}\b", text, re.I
            ) is not None
    return False


def validate_rewrite_quality(row: dict, rewrite: dict) -> list[str]:
    """Warn when a rewrite likely lost behavior-critical prompt content."""
    rewritten = rewrite_search_text(rewrite)
    warnings = []

    for symbol in extract_public_symbols(normalize_text(row["interface"] or "")):
        if not contains_public_symbol(rewritten, symbol):
            warnings.append(f"missing_public_symbol:{symbol}")

    original = normalize_text(row["problem_statement"] or "")
    for literal in extract_edge_case_literals(original):
        if literal == "":
            if not re.search(r'\bempty string\b|""', rewritten, re.I):
                warnings.append("missing_edge_literal:<empty string>")
            continue
        if not contains_literal(rewritten, literal):
            warnings.append(f"missing_edge_literal:{literal}")

    return warnings


def user_prompt(row: dict) -> str:
    interface = normalize_text(row["interface"] or "")
    if interface == "No new interfaces are introduced.":
        interface = ""
    public_symbols = extract_public_symbols(interface)
    public_symbols_text = ", ".join(public_symbols) if public_symbols else "(none)"
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
            (
                "Generated public interface and compatibility notes. Preserve these when they "
                "describe callable signatures, constructor/default behavior, return shapes, "
                "exported fields, config/serialized keys, or import/export compatibility. "
                "Do not copy source locations."
            ),
            interface or "(none)",
            "",
            (
                "Likely required public symbols detected from those notes. Include these exact "
                "names in the rewritten prompt or preserved requirements unless the notes clearly "
                "make them private/internal:"
            ),
            public_symbols_text,
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


def rewrite_prompt(client: OpenAI, model: str, row: dict, max_output_tokens: int) -> tuple[dict, str]:
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_prompt(row),
        max_output_tokens=max_output_tokens,
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
        raise SystemExit("No prompts matched the requested filters")

    compact = load_csv_by_id(args.tasks_file)
    id_order = {instance_id: index for index, instance_id in enumerate(ids)}
    rows = sorted(iter_rows_by_id(ids), key=lambda row: id_order[row["instance_id"]])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "rewrites.jsonl"
    client_kwargs = {"api_key": api_key}
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    client = OpenAI(**client_kwargs)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            rewrite, raw_output = rewrite_prompt(client, args.model, row, args.max_output_tokens)
            quality_warnings = validate_rewrite_quality(row, rewrite)
            task_meta = compact[row["instance_id"]]
            record = {
                "model": args.model,
                "task": task_meta,
                "rewrite": rewrite,
                "quality_warnings": quality_warnings,
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
            for warning in quality_warnings:
                print(f"warning {row['instance_id']}: {warning}")
            if quality_warnings and args.fail_on_quality_warnings:
                raise SystemExit(
                    f"quality warnings for {row['instance_id']}: {', '.join(quality_warnings)}"
                )
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

    print(f"output: {jsonl_path}")


if __name__ == "__main__":
    main()
