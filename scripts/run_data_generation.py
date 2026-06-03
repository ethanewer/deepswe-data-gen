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


if __name__ == "__main__":
    main()
