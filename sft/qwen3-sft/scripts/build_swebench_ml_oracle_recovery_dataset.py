#!/usr/bin/env python3
"""Build oracle corrective rows for the SWE-bench Multilingual predictive 30.

This intentionally uses the benchmark dataset's gold source patches. It is for
diagnosing whether the checkpoint can be forced out of read-only loops on the
target distribution, not for a clean generalization estimate.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_patch_recovery_dataset import (  # noqa: E402
    BASH_TOOL,
    MINI_SWE_SUBMIT_COMMAND,
    assistant,
    base_messages,
    changed_paths,
    context_snippet,
    json_dumps,
    patch_apply_command,
    tool_output,
)


def row(messages: list[dict[str, Any]], *, instance_id: str, target_kind: str, patch: str, copy_index: int) -> dict[str, Any]:
    return {
        "messages": messages,
        "tools": BASH_TOOL,
        "metadata": {
            "source": "swebench_multilingual_predictive30_oracle",
            "instance_id": instance_id,
            "target_kind": target_kind,
            "copy_index": copy_index,
            "patch_chars": len(patch),
            "oracle": True,
        },
    }


def build_rows(example: dict[str, Any], copy_index: int) -> list[dict[str, Any]]:
    task = str(example["problem_statement"]).strip()
    patch = str(example["patch"]).strip() + "\n"
    instance_id = str(example["instance_id"])
    paths = changed_paths(patch)
    if not paths:
        return []
    first_path = paths[0]
    basename = Path(first_path).name
    apply_cmd = patch_apply_command(patch)
    diff_cmd = "git diff > patch.txt && sed -n '1,240p' patch.txt"
    find_cmd = f"find . -name '{basename}'"
    sed_cmd = f"sed -n '1,240p' {first_path}"
    cat_cmd = f"cat {first_path}"
    source_observation = context_snippet(patch)
    edit_thought = (
        "<thought>\n"
        "The issue and relevant source are clear. I will apply the source patch, inspect the diff, and submit patch.txt.\n"
        "</thought>"
    )

    rows: list[dict[str, Any]] = []

    messages = base_messages(task)
    messages.append(assistant(apply_cmd, edit_thought))
    rows.append(row(messages, instance_id=instance_id, target_kind="first_turn_apply_patch", patch=patch, copy_index=copy_index))

    messages = base_messages(task)
    messages += [
        assistant(find_cmd, "<thought>\nI will locate the relevant file.\n</thought>", trainable=False),
        tool_output(first_path),
        assistant(apply_cmd, edit_thought),
    ]
    rows.append(row(messages, instance_id=instance_id, target_kind="find_to_apply_patch", patch=patch, copy_index=copy_index))

    messages = base_messages(task)
    messages += [
        assistant(find_cmd, "<thought>\nI will locate the relevant file.\n</thought>", trainable=False),
        tool_output(first_path),
        assistant(sed_cmd, "<thought>\nI will inspect the relevant source before editing.\n</thought>", trainable=False),
        tool_output(source_observation),
        assistant(apply_cmd, edit_thought),
    ]
    rows.append(row(messages, instance_id=instance_id, target_kind="inspect_to_apply_patch", patch=patch, copy_index=copy_index))

    messages = base_messages(task)
    messages += [
        assistant(find_cmd, "<thought>\nI will locate the relevant file.\n</thought>", trainable=False),
        tool_output(first_path),
        assistant(cat_cmd, "<thought>\nI will inspect the relevant source before editing.\n</thought>", trainable=False),
        tool_output(source_observation),
        assistant(cat_cmd, "<thought>\nI need to check the same file again.\n</thought>", trainable=False),
        tool_output(source_observation),
        assistant(apply_cmd, edit_thought),
    ]
    rows.append(row(messages, instance_id=instance_id, target_kind="repeat_inspect_to_apply_patch", patch=patch, copy_index=copy_index))

    messages = base_messages(task)
    messages += [
        assistant(apply_cmd, edit_thought, trainable=False),
        tool_output("Applied patch cleanly."),
        assistant(diff_cmd, "<thought>\nThe patch is applied. I will inspect the final diff before submitting.\n</thought>"),
    ]
    rows.append(row(messages, instance_id=instance_id, target_kind="apply_to_diff", patch=patch, copy_index=copy_index))

    messages = base_messages(task)
    messages += [
        assistant(apply_cmd, edit_thought, trainable=False),
        tool_output("Applied patch cleanly."),
        assistant(diff_cmd, "<thought>\nThe patch is applied. I will inspect the final diff before submitting.\n</thought>", trainable=False),
        tool_output(patch[:9500]),
        assistant(MINI_SWE_SUBMIT_COMMAND, "<thought>\nThe source diff is ready in patch.txt. I will submit it now.\n</thought>"),
    ]
    rows.append(row(messages, instance_id=instance_id, target_kind="diff_to_submit", patch=patch, copy_index=copy_index))

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=20)
    parser.add_argument("--seed", type=int, default=55555)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        import shutil

        shutil.rmtree(args.output_root)
    out_dir = args.output_root / "swebench_ml_oracle_v24"
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.instance_ids.read_text(encoding="utf-8").split())
    dataset = load_dataset("swe-bench/SWE-Bench_Multilingual", split="test")
    examples = [item for item in dataset if item["instance_id"] in wanted]
    if len(examples) != len(wanted):
        found = {item["instance_id"] for item in examples}
        missing = sorted(wanted - found)
        raise RuntimeError(f"missing target instances: {missing}")

    rows: list[dict[str, Any]] = []
    for item in examples:
        for copy_index in range(args.copies):
            rows.extend(build_rows(item, copy_index))

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    target_kinds: dict[str, int] = {}
    patch_chars = 0
    with (out_dir / "data.jsonl").open("w", encoding="utf-8") as out:
        for item in rows:
            meta = item["metadata"]
            target_kinds[meta["target_kind"]] = target_kinds.get(meta["target_kind"], 0) + 1
            patch_chars += int(meta["patch_chars"])
            out.write(json_dumps(item) + "\n")

    manifest = {
        "output_root": str(args.output_root),
        "rows": len(rows),
        "instances": len(examples),
        "copies": args.copies,
        "target_kinds": target_kinds,
        "patch_chars": patch_chars,
        "avg_patch_chars": patch_chars / max(len(rows), 1),
        "oracle": True,
        "selection": "SWE-bench Multilingual predictive-30 gold source patches rendered as exact mini-swe recovery prefixes",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
