#!/usr/bin/env python3
"""Run data generation and an OpenAI smoke test."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data generation plus smoke test.")
    parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model to use.")
    parser.add_argument("--limit", type=int, default=1, help="Smoke-test task count.")
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium", "hard"),
        default="easy",
        help="Smoke-test difficulty.",
    )
    parser.add_argument(
        "--language",
        choices=("python", "ts", "go"),
        help="Optional smoke-test language.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_data_generation.py")],
        cwd=REPO_ROOT,
        check=True,
    )

    smoke_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_deepswe.py"),
        "--model",
        args.model,
        "--limit",
        str(args.limit),
        "--difficulty",
        args.difficulty,
    ]
    if args.language:
        smoke_cmd.extend(["--language", args.language])

    subprocess.run(smoke_cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
