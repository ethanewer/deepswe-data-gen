#!/usr/bin/env python3
"""Build a Pyxis mini-swe-agent manifest from SWE-rebench assignments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tomllib
from pathlib import Path
from typing import Any


MODEL_SETTINGS: dict[str, dict[str, Any]] = {
    "deepseek-v4-flash": {
        "litellm_model": "openai/deepseek-v4-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_base": "https://api.deepseek.com",
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "deepseek-v4-pro": {
        "litellm_model": "openai/deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_base": "https://api.deepseek.com",
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "moonshotai/kimi-k2.6": {
        "litellm_model": "openrouter/moonshotai/kimi-k2.6",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_base": "",
        "extra_body": {"reasoning": {"effort": "none", "exclude": True}},
    },
    "xiaomi/mimo-v2.5-pro": {
        "litellm_model": "openrouter/xiaomi/mimo-v2.5-pro",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_base": "",
        "extra_body": {},
    },
}


def task_slug(instance_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", instance_id).strip("-").lower()


def safe_model(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", model).strip("-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-model", action="append", default=[])
    parser.add_argument("--include-difficulty", action="append", default=[])
    parser.add_argument("--include-language", action="append", default=[])
    parser.add_argument(
        "--instruction-style-override",
        default="",
        help="Override instruction_style metadata, useful when --tasks-dir points at another prompt style.",
    )
    parser.add_argument(
        "--rollout-count",
        type=int,
        default=1,
        help="Duplicate each selected task this many times with distinct workspaces.",
    )
    parser.add_argument(
        "--image-override",
        default="",
        help="Use this Pyxis image path/URI instead of the image in each task.toml.",
    )
    parser.add_argument("--skip-existing-result", action="store_true")
    return parser.parse_args()


def load_task_metadata(task_dir: Path) -> dict[str, Any]:
    path = task_dir / "task.toml"
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return {
            "image": data["environment"]["docker_image"],
            "workdir": data["environment"]["workdir"],
            "base_commit": data["metadata"]["base_commit_hash"],
        }
    except tomllib.TOMLDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")

        def field(name: str) -> str:
            match = re.search(rf'(?m)^{re.escape(name)}\s*=\s*"((?:\\.|[^"])*)"', text)
            if not match:
                raise
            return bytes(match.group(1), "utf-8").decode("unicode_escape", errors="replace")

        return {
            "image": field("docker_image"),
            "workdir": field("workdir"),
            "base_commit": field("base_commit_hash"),
        }


def main() -> None:
    args = parse_args()
    include_model = set(args.include_model)
    include_difficulty = set(args.include_difficulty)
    include_language = set(args.include_language)
    records: list[dict[str, Any]] = []

    with args.assignments_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row = {key: (value or "").strip() for key, value in row.items()}
            model = row["assigned_model"]
            if include_model and model not in include_model:
                continue
            if include_difficulty and row["difficulty"] not in include_difficulty:
                continue
            if include_language and row["language"] not in include_language:
                continue
            if model not in MODEL_SETTINGS:
                raise ValueError(f"unsupported model in assignments: {model}")

            task_dir = args.tasks_dir / task_slug(row["instance_id"])
            if not task_dir.exists():
                raise FileNotFoundError(f"missing task directory for {row['instance_id']}: {task_dir}")
            task_meta = load_task_metadata(task_dir)
            if args.image_override:
                task_meta["image"] = args.image_override
            settings = MODEL_SETTINGS[model]
            instruction_style = args.instruction_style_override or row["instruction_style"]
            for rollout_index in range(args.rollout_count):
                rollout_id = f"r{rollout_index:02d}"
                workspace = (
                    args.run_root
                    / "pyxis-traces"
                    / instruction_style
                    / safe_model(model)
                    / rollout_id
                    / row["instance_id"]
                )
                if args.skip_existing_result and (workspace / "result.json").exists():
                    continue

                records.append(
                    {
                    "index": len(records),
                    "rollout_id": rollout_id,
                    "instance_id": row["instance_id"],
                    "difficulty": row["difficulty"],
                    "language": row["language"],
                    "repo": row.get("repo", ""),
                    "instruction_style": instruction_style,
                    "model": model,
                    "litellm_model": settings["litellm_model"],
                    "api_key_env": settings["api_key_env"],
                    "api_base": settings["api_base"] or "-",
                    "extra_body_json": (
                        json.dumps(settings["extra_body"], separators=(",", ":"))
                        if settings["extra_body"]
                        else "-"
                    ),
                    "task_dir": str(task_dir.resolve()),
                    "workspace": str(workspace.resolve()),
                    **task_meta,
                    }
                )
                if args.limit and len(records) >= args.limit:
                    break
            if args.limit and len(records) >= args.limit:
                break

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    fields = [
        "index",
        "rollout_id",
        "instance_id",
        "task_dir",
        "workspace",
        "image",
        "model",
        "litellm_model",
        "api_key_env",
        "api_base",
        "extra_body_json",
        "difficulty",
        "language",
        "instruction_style",
        "repo",
    ]
    with args.output_tsv.open("w", encoding="utf-8") as handle:
        for record in records:
            values = []
            for field in fields:
                value = str(record.get(field, ""))
                value = value.replace("\t", " ").replace("\r", " ").replace("\n", " ")
                values.append(value)
            handle.write("\t".join(values) + "\n")

    summary = {
        "total": len(records),
        "by_model": {},
        "by_difficulty": {},
        "by_language": {},
    }
    for record in records:
        for key, field in (
            ("by_model", "model"),
            ("by_difficulty", "difficulty"),
            ("by_language", "language"),
        ):
            value = record[field]
            summary[key][value] = summary[key].get(value, 0) + 1
    (args.output_jsonl.parent / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
