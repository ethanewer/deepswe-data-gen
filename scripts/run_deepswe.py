#!/usr/bin/env python3
"""Run generated tasks with the same harness DeepSWE uses.

DeepSWE leaderboard runs use Pier with the model-agnostic mini-swe-agent
harness. This script deliberately shells out to that harness instead of calling
the OpenAI API directly.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS_DIR = REPO_ROOT / "swerebench-v2" / "harbor-tasks"
DEFAULT_JOBS_DIR = REPO_ROOT / "runs" / "pier-jobs"
DEFAULT_MODEL = "openai/gpt-5.4-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SWE-rebench-derived tasks through Pier/mini-swe-agent."
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help="Harbor task directory or dataset directory to pass to pier -p.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pier model name.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum tasks to run.")
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=DEFAULT_JOBS_DIR,
        help="Pier jobs output directory.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        help="Optional deterministic Pier sample seed.",
    )
    parser.add_argument(
        "--include-task-name",
        action="append",
        default=[],
        help="Task name or glob to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--disable-verification",
        action="store_true",
        help="Skip verifier execution after the agent run. Useful only for smoke tests.",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=1,
        help="Number of concurrent Pier trials.",
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Pier timeout multiplier.",
    )
    parser.add_argument(
        "--agent-kwarg",
        action="append",
        default=[],
        help="Additional Pier --agent-kwarg value, e.g. key=value.",
    )
    parser.add_argument(
        "--agent-env",
        action="append",
        default=[],
        help="Additional Pier --agent-env value, e.g. KEY=VALUE.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional .env file to pass to Pier.",
    )
    return parser.parse_args()


def pier_model_name(model: str) -> str:
    if "/" in model:
        return model
    return f"openai/{model}"


def ensure_prerequisites() -> None:
    if not shutil.which("pier"):
        raise SystemExit(
            "pier is not installed. Install it with: "
            "uv tool install git+https://github.com/datacurve-ai/pier"
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        "pier",
        "run",
        "-p",
        str(args.tasks_dir),
        "--agent",
        "mini-swe-agent",
        "--model",
        pier_model_name(args.model),
        "--jobs-dir",
        str(args.jobs_dir),
        "--n-tasks",
        str(args.limit),
        "--n-concurrent",
        str(args.n_concurrent),
        "--timeout-multiplier",
        str(args.timeout_multiplier),
        "--yes",
    ]
    if args.sample_seed is not None:
        command.extend(["--sample-seed", str(args.sample_seed)])
    for include in args.include_task_name:
        command.extend(["--include-task-name", include])
    for value in args.agent_kwarg:
        command.extend(["--agent-kwarg", value])
    for value in args.agent_env:
        command.extend(["--agent-env", value])
    if args.env_file:
        command.extend(["--env-file", str(args.env_file)])
    if args.disable_verification:
        command.append("--disable-verification")
    return command


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if not args.tasks_dir.exists():
        raise SystemExit(
            f"{args.tasks_dir} does not exist. Generate it with "
            "scripts/generate_harbor_tasks.py first."
        )
    ensure_prerequisites()

    command = build_command(args)
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
