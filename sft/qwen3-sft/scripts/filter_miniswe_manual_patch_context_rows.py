#!/usr/bin/env python3
"""Filter Mini-SWE-aligned JSONL views that contain manual patch artifacts.

The online trainer can mask assistant turns that hand-write patch.txt, but the
bad patch construction remains visible as context. For SWE-bench SFT, those
whole trajectories are risky enough to remove from mixed passrate views.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qwen_agentic_sft.online_packed_dataset import assistant_has_manual_patch_target, assistant_tool_command


def input_data_root(path: Path) -> Path:
    return path / "data" if (path / "data").is_dir() else path


def row_passed(row: dict[str, Any]) -> bool:
    source_outcome = row.get("source_outcome")
    if isinstance(source_outcome, dict) and "passed" in source_outcome:
        return bool(source_outcome.get("passed"))
    return bool(row.get("passed"))


def manual_patch_commands(row: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for message in row.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if assistant_has_manual_patch_target(message):
            commands.append(assistant_tool_command(message))
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--example-limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = input_data_root(args.input_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"input data root not found: {source_root}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)

    output_data = args.output_root / "data"
    output_data.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    per_file: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for source_file in sorted(source_root.glob("*.jsonl")):
        file_stats: Counter[str] = Counter()
        output_file = output_data / source_file.name
        with source_file.open("r", encoding="utf-8") as src, output_file.open("w", encoding="utf-8") as dst:
            for line_number, line in enumerate(src, start=1):
                text = line.rstrip("\n")
                if not text:
                    continue
                stats["rows_seen"] += 1
                file_stats["rows_seen"] += 1
                row = json.loads(text)
                commands = manual_patch_commands(row)
                if commands:
                    passed = row_passed(row)
                    stats["rows_dropped_manual_patch_context"] += 1
                    stats["dropped_passed" if passed else "dropped_nonpassing"] += 1
                    file_stats["rows_dropped_manual_patch_context"] += 1
                    if len(examples) < args.example_limit:
                        examples.append(
                            {
                                "source_file": str(source_file),
                                "line_number": line_number,
                                "task_id": row.get("task_id")
                                or (row.get("source_outcome") or {}).get("task_id"),
                                "uuid": row.get("uuid")
                                or (row.get("source_outcome") or {}).get("uuid"),
                                "passed": passed,
                                "manual_patch_commands": commands[:3],
                            }
                        )
                    continue
                dst.write(text + "\n")
                stats["rows_written"] += 1
                file_stats["rows_written"] += 1
                stats["written_passed" if row_passed(row) else "written_nonpassing"] += 1
        if output_file.stat().st_size == 0:
            output_file.unlink()
        per_file.append({"file": str(source_file), **dict(file_stats)})

    manifest = {
        "input_root": str(args.input_root),
        "input_data_root": str(source_root),
        "output_root": str(args.output_root),
        "filter": "drop whole Mini-SWE-aligned rows with assistant manual patch.txt construction",
        "stats": dict(sorted(stats.items())),
        "per_file": per_file,
        "examples": examples,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
