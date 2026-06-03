#!/usr/bin/env python3
"""Run the repository's data-generation jobs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate derived data files.")
    parser.add_argument(
        "--swerebench-output-dir",
        type=Path,
        default=REPO_ROOT / "swerebench-v2",
        help="Output directory for the SWE-rebench V2 high-quality subset.",
    )
    parser.add_argument(
        "--write-harbor-tasks",
        action="store_true",
        help="Also materialize Pier/Harbor task directories.",
    )
    parser.add_argument(
        "--harbor-output-dir",
        type=Path,
        default=REPO_ROOT / "swerebench-v2" / "harbor-tasks",
        help="Output directory for generated Harbor tasks.",
    )
    parser.add_argument("--harbor-limit", type=int, help="Maximum Harbor tasks to write.")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    parser.add_argument("--language", choices=("python", "ts", "go"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = REPO_ROOT / "swerebench-v2" / "generate_high_quality_subset.py"
    subprocess.run(
        [
            sys.executable,
            str(generator),
            "--output-dir",
            str(args.swerebench_output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    if args.write_harbor_tasks:
        harbor_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_harbor_tasks.py"),
            "--output-dir",
            str(args.harbor_output_dir),
            "--clean",
        ]
        if args.harbor_limit:
            harbor_cmd.extend(["--limit", str(args.harbor_limit)])
        if args.difficulty:
            harbor_cmd.extend(["--difficulty", args.difficulty])
        if args.language:
            harbor_cmd.extend(["--language", args.language])
        subprocess.run(harbor_cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
