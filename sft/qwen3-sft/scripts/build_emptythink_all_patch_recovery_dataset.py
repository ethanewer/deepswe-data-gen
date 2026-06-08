#!/usr/bin/env python3
"""Build fully eval-aligned empty-think patch recovery rows.

v25 only aligned the trainable target assistant turn. The masked prior
assistant turns still contained visible teacher thoughts, which does not match
mini-swe eval histories where assistant messages are empty-content tool calls
under ``enable_thinking=false``. This variant normalizes every assistant turn in
the prefix and target to:

    content == ""
    reasoning_content == "\n"
    tool_calls == [bash(...)]

Only the current target turn remains trainable.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_patch_recovery_dataset import RAW_ROOT, BASH_TOOL, iter_nebius, iter_swe_hero, json_dumps  # noqa: E402
from build_emptythink_patch_recovery_dataset import EVAL_BASH_TOOL  # noqa: E402


def align_all_assistant_turns(row: dict[str, Any], *, tool_schema: list[dict[str, Any]]) -> dict[str, Any]:
    row = json.loads(json_dumps(row))
    row["tools"] = tool_schema
    target_seen = False
    assistant_turns = 0
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        assistant_turns += 1
        message["content"] = ""
        message["reasoning_content"] = "\n"
        if message.get("trainable") is not False and message.get("loss") is not False:
            target_seen = True
    if not target_seen:
        raise ValueError("row has no trainable assistant target")
    row.setdefault("metadata", {})["target_format"] = "emptythink_all_assistant_tool_calls"
    row["metadata"]["assistant_turns_normalized"] = assistant_turns
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--max-swe-hero-rows", type=int, default=0)
    parser.add_argument("--max-nebius-rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=60608)
    parser.add_argument("--tool-schema", choices=("eval", "v23"), default="eval")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    out_dir = args.output_root / "patch_recovery_emptythink_all_v26"
    out_dir.mkdir(parents=True, exist_ok=True)

    tool_schema = EVAL_BASH_TOOL if args.tool_schema == "eval" else BASH_TOOL
    rows: list[dict[str, Any]] = []
    for item in iter_swe_hero(args.raw_root, args.max_swe_hero_rows):
        rows.append(align_all_assistant_turns(item, tool_schema=tool_schema))
    for item in iter_nebius(args.raw_root, args.max_nebius_rows):
        rows.append(align_all_assistant_turns(item, tool_schema=tool_schema))

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    target_kinds: dict[str, int] = {}
    patch_chars = 0
    assistant_turns = 0
    with (out_dir / "data.jsonl").open("w", encoding="utf-8") as out:
        for item in rows:
            meta = item["metadata"]
            target_kind = str(meta.get("target_kind", "unknown"))
            target_kinds[target_kind] = target_kinds.get(target_kind, 0) + 1
            patch_chars += int(meta.get("patch_chars", 0))
            assistant_turns += int(meta.get("assistant_turns_normalized", 0))
            out.write(json_dumps(item) + "\n")

    manifest: dict[str, Any] = {
        "output_root": str(args.output_root),
        "rows": len(rows),
        "max_swe_hero_rows": args.max_swe_hero_rows,
        "max_nebius_rows": args.max_nebius_rows,
        "target_kinds": target_kinds,
        "patch_chars": patch_chars,
        "avg_patch_chars": patch_chars / max(len(rows), 1),
        "assistant_turns_normalized": assistant_turns,
        "target_format": "emptythink_all_assistant_tool_calls",
        "tool_schema": args.tool_schema,
        "selection": "verified source-only patches rendered as masked mini-swe recovery prefixes with all assistant turns normalized to eval empty-think tool calls",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
