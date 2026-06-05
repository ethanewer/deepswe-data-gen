#!/usr/bin/env python3
"""Run data generation, Harbor task generation, and a Pier smoke test."""

from __future__ import annotations

import argparse
import subprocess
import sys

from eval.paths import REPO_ROOT


HARBOR_TASKS_DIR = REPO_ROOT / "runs" / "swerebench-v2" / "harbor-tasks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data generation plus smoke test.")
    parser.add_argument("--model", default="openai/gpt-5.4-mini", help="Pier model to use.")
    parser.add_argument("--limit", type=int, default=1, help="Smoke-test task count.")
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium", "hard"),
        default="easy",
        help="Difficulty to materialize and smoke-test.",
    )
    parser.add_argument(
        "--language",
        choices=("python", "ts", "go"),
        help="Optional language to materialize and smoke-test.",
    )
    parser.add_argument(
        "--disable-verification",
        action="store_true",
        help="Pass through to Pier for fast harness smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subprocess.run(
        [sys.executable, "-m", "datagen.swerebench_v2.run_data_generation"],
        cwd=REPO_ROOT,
        check=True,
    )

    harbor_cmd = [
        sys.executable,
        "-m",
        "datagen.swerebench_v2.generate_harbor_tasks",
        "--output-dir",
        str(HARBOR_TASKS_DIR),
        "--clean",
        "--limit",
        str(args.limit),
        "--difficulty",
        args.difficulty,
    ]
    if args.language:
        harbor_cmd.extend(["--language", args.language])
    subprocess.run(harbor_cmd, cwd=REPO_ROOT, check=True)

    smoke_cmd = [
        sys.executable,
        "-m",
        "eval.benchmarks.deepswe.run",
        "--tasks-dir",
        str(HARBOR_TASKS_DIR),
        "--model",
        args.model,
        "--limit",
        str(args.limit),
    ]
    if args.disable_verification:
        smoke_cmd.append("--disable-verification")

    subprocess.run(smoke_cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
