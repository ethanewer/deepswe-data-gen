#!/usr/bin/env python3
"""Build a patch-recovery-heavy empty-think mix with a small first-turn anchor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_jsonl_files(root: Path) -> Iterable[Path]:
    yield from sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def stable_fraction(*parts: object) -> float:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def assistant_command(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls:
        return ""
    function = calls[0].get("function", {}) if isinstance(calls[0], dict) else {}
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    if not isinstance(args, dict):
        return ""
    return str(args.get("command") or "")


def target_index(messages: list[dict[str, Any]]) -> int | None:
    targets = [
        idx
        for idx, message in enumerate(messages)
        if message.get("role") == "assistant"
        and message.get("trainable") is not False
        and message.get("loss") is not False
    ]
    return targets[-1] if targets else None


def target_turn(messages: list[dict[str, Any]], index: int) -> int:
    return sum(1 for message in messages[: index + 1] if message.get("role") == "assistant") - 1


def align_emptythink(row: dict[str, Any]) -> dict[str, Any]:
    emitted = copy.deepcopy(row)
    for message in emitted.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        message["content"] = ""
        message["reasoning_content"] = "\n"
    emitted.setdefault("metadata", {})["target_format"] = "emptythink_all_assistant_tool_calls"
    return emitted


def command_category(command: str) -> str:
    text = command.strip().lower()
    if "complete_task_and_submit_final_output" in text:
        return "submit"
    if "git diff" in text:
        return "diff"
    if "patch.txt" in text or "git apply" in text or "apply_patch" in text:
        return "patch"
    if text.startswith("ls") or text == "pwd":
        return "nav"
    if any(marker in text for marker in ("find ", "grep", "rg ", "cat ", "sed ", "head ", "tail ")):
        return "inspect"
    return "other"


def write_shards(rows: list[dict[str, Any]], output_dir: Path, shards: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = [
        (output_dir / f"shard-{idx:03d}.jsonl").open("w", encoding="utf-8")
        for idx in range(shards)
    ]
    categories: Counter[str] = Counter()
    try:
        for idx, row in enumerate(rows):
            target = target_index(row.get("messages") or [])
            if target is not None:
                categories[command_category(assistant_command(row["messages"][target]))] += 1
            handles[idx % shards].write(json_dumps(row) + "\n")
    finally:
        for handle in handles:
            handle.close()
    for path in output_dir.glob("shard-*.jsonl"):
        if path.stat().st_size == 0:
            path.unlink()
    return {"rows": len(rows), "target_categories": dict(categories)}


def load_patch_rows(root: Path, copies: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    for path in iter_jsonl_files(root):
        for line_number, line in enumerate(path.open("r", encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            target = target_index(row.get("messages") or [])
            if target is None:
                continue
            command = assistant_command(row["messages"][target])
            if "/path/to/" in command.lower() or "placeholder" in command.lower():
                continue
            categories[command_category(command)] += copies
            for copy_index in range(copies):
                emitted = align_emptythink(row)
                emitted.setdefault("metadata", {})["v50_mix"] = {
                    "bucket": "patch_recovery",
                    "copy_index": copy_index,
                    "source_file": str(path),
                    "source_line": line_number,
                }
                rows.append(emitted)
    return rows, categories


def load_first_turn_anchor_rows(root: Path, max_rows: int, seed: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    categories: Counter[str] = Counter()
    for path in iter_jsonl_files(root):
        for line_number, line in enumerate(path.open("r", encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            target = target_index(messages)
            if target is None or target_turn(messages, target) != 0:
                continue
            command = assistant_command(messages[target])
            cat = command_category(command)
            if cat not in {"nav", "inspect", "other"}:
                continue
            if "/app/" in command or "/path/to/" in command.lower() or command.strip() == "bash":
                continue
            score = stable_fraction(seed, path, line_number, command)
            emitted = align_emptythink(row)
            emitted.setdefault("metadata", {})["v50_mix"] = {
                "bucket": "first_turn_anchor",
                "source_file": str(path),
                "source_line": line_number,
            }
            candidates.append((score, emitted))
    candidates.sort(key=lambda item: item[0])
    rows = [row for _, row in candidates[:max_rows]]
    for row in rows:
        target = target_index(row.get("messages") or [])
        if target is not None:
            categories[command_category(assistant_command(row["messages"][target]))] += 1
    return rows, categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-copies", type=int, default=2)
    parser.add_argument("--max-anchor-rows", type=int, default=12000)
    parser.add_argument("--shards-per-source", type=int, default=96)
    parser.add_argument("--seed", type=int, default=50050)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    patch_rows, patch_categories = load_patch_rows(args.patch_root, args.patch_copies)
    anchor_rows, anchor_categories = load_first_turn_anchor_rows(args.anchor_root, args.max_anchor_rows, args.seed)

    rng = random.Random(args.seed)
    mixed_rows = patch_rows + anchor_rows
    rng.shuffle(mixed_rows)

    mixed_summary = write_shards(
        mixed_rows,
        args.output_root / "mixed_patch_anchor_emptythink_v50",
        args.shards_per_source,
    )

    manifest = {
        "output_root": str(args.output_root),
        "patch_root": str(args.patch_root),
        "anchor_root": str(args.anchor_root),
        "patch_copies": args.patch_copies,
        "max_anchor_rows": args.max_anchor_rows,
        "seed": args.seed,
        "sources": {"mixed_patch_anchor_emptythink_v50": mixed_summary},
        "input_target_categories": {
            "patch_recovery": dict(patch_categories),
            "first_turn_anchor": dict(anchor_categories),
        },
        "rows": mixed_summary["rows"],
        "selection": (
            "Mask-fixed v26 empty-thinking apply/diff/submit recovery rows duplicated "
            "for short-run weight, plus a smaller empty-thinking v17 first-turn tool anchor."
        ),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
