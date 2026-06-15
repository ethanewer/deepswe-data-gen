#!/usr/bin/env python3
"""Build a verify-weighted variant of the high-quality mini-SWE prefix rows.

The v28 corrective stage has many targets that write ``git diff`` to
``patch.txt`` and fewer targets that visibly inspect the patch before submit.
This script keeps all source rows and adds extra copies of the rows that teach
``cat patch.txt`` verification after a patch write. It optionally adds a small
number of submit-after-cat copies so the final transition is still represented.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "highquality-1x-duplicate-reasoning-90pct-30k-full-miniswe-aligned-passed-prefix-weighted-v2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "highquality-1x-duplicate-reasoning-90pct-30k-full-miniswe-aligned-passed-prefix-verifyweighted-v31"
)


def command_from_message(message: dict[str, Any]) -> str:
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return ""
    function = tool_calls[0].get("function") or {}
    args = function.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    if not isinstance(args, dict):
        return ""
    return str(args.get("command") or args.get("cmd") or "")


def target_command(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            return command_from_message(message)
    return ""


def selection(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    value = metadata.get("v21_selection") or {}
    return value if isinstance(value, dict) else {}


def mentions_patch_txt(command: str) -> bool:
    return "patch.txt" in command.lower()


def is_patch_cat_target(row: dict[str, Any]) -> bool:
    sel = selection(row)
    command = target_command(row).lower()
    target_category = str(sel.get("target_category") or "").lower()
    return (
        target_category == "cat"
        and "cat" in command
        and mentions_patch_txt(command)
    )


def is_submit_after_patch_cat(row: dict[str, Any]) -> bool:
    sel = selection(row)
    command = target_command(row).lower()
    previous = str(sel.get("previous_command") or "").lower()
    previous_category = str(sel.get("previous_category") or "").lower()
    return (
        "complete_task_and_submit_final_output" in command
        and previous_category == "cat"
        and mentions_patch_txt(previous)
    )


def with_augmented_metadata(row: dict[str, Any], *, reason: str, copy_index: int) -> dict[str, Any]:
    row = copy.deepcopy(row)
    metadata = row.setdefault("metadata", {})
    metadata["v31_verify_weighted"] = {
        "reason": reason,
        "copy_index": copy_index,
    }
    return row


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cat-extra-copies", type=int, default=20)
    parser.add_argument("--submit-extra-copies", type=int, default=0)
    args = parser.parse_args()

    input_jsonl = args.input_root / "data" / "data.jsonl"
    if not input_jsonl.is_file():
        raise FileNotFoundError(input_jsonl)

    output_data = args.output_root / "data"
    output_data.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_data / "data.jsonl"

    counters: Counter[str] = Counter()

    def generated_rows():
        for row in iter_rows(input_jsonl):
            counters["rows_in"] += 1
            counters["rows_out"] += 1
            yield row

            if is_patch_cat_target(row):
                counters["patch_cat_targets"] += 1
                for copy_index in range(args.cat_extra_copies):
                    counters["patch_cat_extra_rows"] += 1
                    counters["rows_out"] += 1
                    yield with_augmented_metadata(
                        row,
                        reason="patch_cat_verification_extra",
                        copy_index=copy_index,
                    )

            if is_submit_after_patch_cat(row):
                counters["submit_after_patch_cat_targets"] += 1
                for copy_index in range(args.submit_extra_copies):
                    counters["submit_after_patch_cat_extra_rows"] += 1
                    counters["rows_out"] += 1
                    yield with_augmented_metadata(
                        row,
                        reason="submit_after_patch_cat_extra",
                        copy_index=copy_index,
                    )

    write_jsonl(output_jsonl, generated_rows())

    input_manifest_path = args.input_root / "manifest.json"
    input_manifest = None
    if input_manifest_path.is_file():
        input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))

    manifest = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "cat_extra_copies": args.cat_extra_copies,
        "submit_extra_copies": args.submit_extra_copies,
        "stats": dict(counters),
        "selection": (
            "all v28 high-quality passed-prefix rows plus extra copies of "
            "patch.txt verification targets; submit-after-verification rows "
            "can be upweighted with --submit-extra-copies but are not duplicated by default"
        ),
        "input_manifest": input_manifest,
        "output": str(output_jsonl),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest["stats"], indent=2, sort_keys=True))
    print(output_jsonl)


if __name__ == "__main__":
    main()
