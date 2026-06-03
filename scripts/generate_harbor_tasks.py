#!/usr/bin/env python3
"""Generate Pier/Harbor task directories from the SWE-rebench V2 subset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Iterable

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "nebius/SWE-rebench-V2"
SPLIT = "train"
LANGUAGES = ("python", "ts", "go")
DIFFICULTIES = ("easy", "medium", "hard")
MIN_CONFIDENCE = 0.95
TASKS_CSV = REPO_ROOT / "swerebench-v2" / "high_quality_conf_ge_0.95_tasks.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "swerebench-v2" / "harbor-tasks"
DEFAULT_REWRITES_JSONL = REPO_ROOT / "swerebench-v2" / "rewritten-prompts" / "rewrites.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Harbor-format task directories for Pier/mini-swe-agent."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks-file", type=Path, default=TASKS_CSV)
    parser.add_argument("--limit", type=int, help="Maximum task directories to write.")
    parser.add_argument("--difficulty", choices=DIFFICULTIES)
    parser.add_argument("--language", choices=LANGUAGES)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Specific instance_id to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--instruction-style",
        choices=("deepswe", "swe_rebench", "rewritten"),
        default="deepswe",
        help="Prompt style to write to instruction.md.",
    )
    parser.add_argument(
        "--rewrites-file",
        type=Path,
        default=DEFAULT_REWRITES_JSONL,
        help="JSONL file from scripts/rewrite_prompts.py.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before writing tasks.",
    )
    return parser.parse_args()


def is_high_quality(row: dict) -> bool:
    metadata = row["meta"]["llm_metadata"]
    confidence = metadata.get("confidence")
    return (
        row["language"] in LANGUAGES
        and metadata.get("difficulty") in DIFFICULTIES
        and metadata.get("code") == "A"
        and metadata.get("intent_completeness") == "complete"
        and not metadata.get("test_alignment_issues")
        and not any((metadata.get("detected_issues") or {}).values())
        and confidence is not None
        and confidence >= MIN_CONFIDENCE
    )


def load_selected_ids(
    path: Path,
    args: argparse.Namespace,
    rewritten_instance_ids: set[str] | None = None,
) -> list[str]:
    selected = []
    requested = set(args.instance_id)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if requested and row["instance_id"] not in requested:
                continue
            if rewritten_instance_ids is not None and row["instance_id"] not in rewritten_instance_ids:
                continue
            if args.difficulty and row["difficulty"] != args.difficulty:
                continue
            if args.language and row["language"] != args.language:
                continue
            selected.append(row["instance_id"])
            if args.limit and len(selected) >= args.limit:
                break
    return selected


def normalize_language(language: str) -> str:
    return {"ts": "typescript"}.get(language, language)


def repo_workdir(row: dict) -> str:
    return "/" + row["repo"].split("/")[-1]


def task_slug(instance_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", instance_id).strip("-").lower()


def toml_string(value: str) -> str:
    return json.dumps(value)


def clean_issue_prompt(text: str, *, strip_urls: bool = False) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^\s*[-*]\s*\[[ xX]\]\s+.*$", "", text)
    text = re.sub(r"(?m)^\s*\+\s?", "", text)
    if strip_urls:
        text = re.sub(r"\s*\(https?://[^)\s]+[^)]*\)", "", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r":\s*(?=\n|$)", "", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_rewrites(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rewrites = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rewrites[record["task"]["instance_id"]] = record["rewrite"]["rewritten_prompt"]
    return rewrites


def build_instruction(row: dict, style: str, rewrites: dict[str, str] | None = None) -> str:
    if style == "rewritten":
        rewritten_prompt = (rewrites or {}).get(row["instance_id"])
        if not rewritten_prompt:
            raise ValueError(f"missing rewritten prompt for {row['instance_id']}")
        return f"{rewritten_prompt.strip()}\n"

    problem = clean_issue_prompt(
        row["problem_statement"] or "",
        strip_urls=style == "deepswe",
    )
    interface = clean_issue_prompt(row["interface"] or "")

    if style == "swe_rebench":
        if interface and interface != "No new interfaces are introduced.":
            return f"{problem}\n\nInterface notes:\n{interface}\n"
        return f"{problem}\n"

    # DeepSWE prompts are natural, behavior-focused task requests. Avoid
    # metadata, test names, reference-patch hints, and generated interface dumps.
    return f"{problem}\n"


def build_task_toml(row: dict) -> str:
    metadata = row["meta"]["llm_metadata"]
    title = clean_issue_prompt(row["problem_statement"] or row["instance_id"]).splitlines()[0]
    title = title[:180]
    name = f"swerebench-v2/{row['instance_id']}"
    repo_url = f"https://github.com/{row['repo']}"

    return dedent(
        f"""
        schema_version = "1.1"
        artifacts = []

        [task]
        name = {toml_string(name)}
        description = ""
        authors = []
        keywords = []

        [metadata]
        task_id = {toml_string(row["instance_id"])}
        display_title = {toml_string(title)}
        display_description = ""
        original_title = {toml_string(title)}
        category = {toml_string("|".join(metadata.get("pr_categories") or []))}
        language = {toml_string(normalize_language(row["language"]))}
        repository_url = {toml_string(repo_url)}
        base_commit_hash = {toml_string(row["base_commit"])}
        swe_rebench_instance_id = {toml_string(row["instance_id"])}
        swe_rebench_difficulty = {toml_string(metadata["difficulty"])}
        swe_rebench_confidence = {metadata["confidence"]}

        [verifier]
        timeout_sec = 1800.0

        [verifier.env]

        [agent]
        timeout_sec = 5400.0

        [environment]
        build_timeout_sec = 1800.0
        docker_image = {toml_string(row["image_name"])}
        os = "linux"
        cpus = 2
        memory_mb = 8192
        storage_mb = 20480
        gpus = 0
        allow_internet = false
        mcp_servers = []
        workdir = {toml_string(repo_workdir(row))}

        [environment.env]

        [solution.env]
        """
    ).strip() + "\n"


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_test_sh(row: dict) -> str:
    test_cmd = row["install_config"]["test_cmd"]
    base_commit = row["base_commit"]
    workdir = repo_workdir(row)
    return dedent(
        f"""\
        #!/bin/bash

        set -uo pipefail

        fail() {{
            code="${{1:-1}}"
            mkdir -p /logs/verifier 2>/dev/null || true
            echo 0 > /logs/verifier/reward.txt 2>/dev/null || true
            exit "$code"
        }}

        log() {{
            echo "[verifier] $*"
        }}

        if [ -d {shell_quote(workdir)} ]; then
            cd {shell_quote(workdir)} || fail 6
        elif [ -d /testbed ]; then
            cd /testbed
        elif [ -d /app ]; then
            cd /app
        else
            log "ERROR: no repository workdir exists"
            fail 6
        fi

        PIER_MODEL_BASE_COMMIT={shell_quote(base_commit)}
        PIER_MODEL_PATCH_PATH="/logs/artifacts/model.patch"

        log "--- Step 0: Capturing model.patch artifact ---"
        mkdir -p "$(dirname "$PIER_MODEL_PATCH_PATH")" || fail 7
        git config --global --add safe.directory "$(pwd)" 2>/dev/null || true
        if ! git rev-parse --verify "${{PIER_MODEL_BASE_COMMIT}}^{{commit}}" >/dev/null 2>&1; then
            log "ERROR: Base commit $PIER_MODEL_BASE_COMMIT is not present"
            fail 7
        fi
        git reset --soft "$PIER_MODEL_BASE_COMMIT" || fail 7
        git add -A -- . || fail 7
        git diff --cached --binary > "$PIER_MODEL_PATCH_PATH" || fail 7
        git reset -q || fail 7

        log "--- Step 1: Resetting files touched by verifier patch ---"
        python3 - <<'PY' | while IFS= read -r f; do
        import re
        patch = open("/tests/test.patch", encoding="utf-8").read()
        files = set()
        for line in patch.splitlines():
            m = re.match(r'^diff --git "?a/(.+?)"? "?b/(.+?)"?$', line)
            if m:
                files.add(m.group(2))
        for path in sorted(files):
            print(path)
        PY
            if git checkout HEAD -- "$f" 2>/dev/null; then
                log "  Reset: $f"
            else
                git rm -r --cached --ignore-unmatch -- "$f" >/dev/null 2>&1 || true
                rm -rf "$f"
                log "  Removed or absent: $f"
            fi
        done

        log "--- Step 2: Applying verifier test.patch ---"
        git apply --whitespace=nowarn /tests/test.patch || fail 3

        log "--- Step 3: Running SWE-rebench test command ---"
        bash -lc {shell_quote(test_cmd)}
        RESULT=$?
        log "Verifier exit code: $RESULT"

        mkdir -p /logs/verifier || fail 5
        if [ "$RESULT" -eq 0 ]; then
            echo 1 > /logs/verifier/reward.txt
        else
            echo 0 > /logs/verifier/reward.txt
        fi
        exit "$RESULT"
        """
    )


def build_environment_dockerfile(row: dict) -> str:
    return f"FROM {row['image_name']}\nCMD [\"/bin/bash\"]\n"


def build_solve_sh() -> str:
    return dedent(
        """\
        #!/bin/bash

        set -euo pipefail

        git apply --whitespace=nowarn /solution/solution.patch
        """
    )


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def materialize_task(
    row: dict,
    output_dir: Path,
    instruction_style: str,
    rewrites: dict[str, str],
) -> Path:
    task_dir = output_dir / task_slug(row["instance_id"])
    if task_dir.exists():
        shutil.rmtree(task_dir)
    write_text(task_dir / "task.toml", build_task_toml(row))
    write_text(task_dir / "instruction.md", build_instruction(row, instruction_style, rewrites))
    write_text(task_dir / "environment" / "Dockerfile", build_environment_dockerfile(row))
    write_text(task_dir / "tests" / "test.patch", row["test_patch"] or "")
    write_text(task_dir / "tests" / "test.sh", build_test_sh(row), executable=True)
    write_text(task_dir / "solution" / "solution.patch", row["patch"] or "")
    write_text(task_dir / "solution" / "solve.sh", build_solve_sh(), executable=True)
    return task_dir


def iter_rows_by_id(instance_ids: Iterable[str]) -> Iterable[dict]:
    wanted = set(instance_ids)
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    for row in dataset:
        if row["instance_id"] in wanted and is_high_quality(row):
            yield row


def main() -> None:
    args = parse_args()
    rewrites = load_rewrites(args.rewrites_file)
    rewritten_instance_ids = set(rewrites) if args.instruction_style == "rewritten" else None
    selected_ids = load_selected_ids(args.tasks_file, args, rewritten_instance_ids)
    if not selected_ids:
        raise SystemExit("No tasks matched the requested filters")

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_id_order = {instance_id: index for index, instance_id in enumerate(selected_ids)}
    rows = sorted(iter_rows_by_id(selected_ids), key=lambda row: selected_id_order[row["instance_id"]])
    task_dirs = [
        materialize_task(row, args.output_dir, args.instruction_style, rewrites)
        for row in rows
    ]

    manifest = {
        "source_dataset": DATASET_NAME,
        "split": SPLIT,
        "instruction_style": args.instruction_style,
        "total": len(task_dirs),
        "tasks": [path.name for path in task_dirs],
    }
    write_text(args.output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(task_dirs)} Harbor task directories to {args.output_dir}")


if __name__ == "__main__":
    main()
