#!/usr/bin/env python3
"""Build a weighted prefix dataset that favors post-observation repair actions.

The input is an already-normalized prefix-target root where exactly one
assistant turn is trainable per row. The output keeps the same row format, but
duplicates high-value edit/diff/test/submit targets and down-samples repetitive
inspection targets. Rows are shuffled before writing so online file-shard
packing sees the intended mix during short training runs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


INSPECT_CATS = {"cat", "search", "view"}
EDIT_CATS = {"edit_shell", "edit_python"}
HIGH_VALUE_CATS = EDIT_CATS | {"diff", "test", "submit"}
MINI_SWE_SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_jsonl_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.jsonl"))


def source_roots(root: Path) -> list[Path]:
    dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if dirs:
        return dirs
    if iter_jsonl_files(root):
        return [root]
    return []


def command_from_assistant(message: dict[str, Any]) -> str:
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


def set_assistant_command(message: dict[str, Any], command: str) -> None:
    calls = message.get("tool_calls") or []
    if not calls or not isinstance(calls[0], dict):
        return
    function = calls[0].setdefault("function", {})
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            function["arguments"] = json.dumps({"command": command}, ensure_ascii=False)
            return
        if not isinstance(parsed, dict):
            parsed = {}
        parsed["command"] = command
        function["arguments"] = json.dumps(parsed, ensure_ascii=False)
        return
    if not isinstance(args, dict):
        args = {}
        function["arguments"] = args
    args["command"] = command


def normalize_target_command(command: str) -> str:
    if "complete_task_and_submit_final_output" in command.lower():
        return MINI_SWE_SUBMIT_COMMAND
    return command


def target_message_index(messages: list[dict[str, Any]]) -> int | None:
    targets = [
        idx
        for idx, message in enumerate(messages)
        if message.get("role") == "assistant"
        and message.get("trainable") is not False
        and message.get("loss") is not False
    ]
    return targets[-1] if targets else None


def previous_assistant_command(messages: list[dict[str, Any]], target_idx: int) -> str:
    for idx in range(target_idx - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return command_from_assistant(messages[idx])
    return ""


def command_category(command: str) -> str:
    text = command.strip().lower()
    if not text:
        return "none"
    if "complete_task_and_submit_final_output" in text:
        return "submit"
    if "git diff" in text or text.startswith("diff "):
        return "diff"
    if "python" in text and ("write_text" in text or ".replace(" in text or "insert" in text):
        return "edit_python"
    edit_markers = (
        "apply_patch",
        "cat >",
        "tee ",
        "perl -",
        "sed -i",
        "python3 - <<",
        "python - <<",
    )
    if any(marker in text for marker in edit_markers):
        return "edit_shell"
    if text.startswith("cat ") or " cat " in text:
        return "cat"
    if "grep" in text or "rg " in text or text.startswith("find "):
        return "search"
    if text.startswith("sed ") or " sed " in text or text.startswith("head ") or text.startswith("tail "):
        return "view"
    test_markers = (
        "pytest",
        "mvn test",
        "gradle test",
        "npm test",
        "go test",
        "cargo test",
        "phpunit",
        "unittest",
    )
    if any(marker in text for marker in test_markers):
        return "test"
    if text.startswith("ls") or text == "pwd" or " pwd" in text:
        return "nav"
    return "other"


def stable_fraction(*parts: object) -> float:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def bad_target_command(command: str) -> bool:
    text = command.lower()
    if text.strip() == MINI_SWE_SUBMIT_COMMAND.lower():
        return False
    if "/path/to/" in text or "placeholder" in text:
        return True
    if "patch.txt" in text:
        if re.search(r"\b(echo|printf)\b[^;&|]*>>\s*patch\.txt", text):
            return True
        if re.search(r"\b(cat|find)\b[^;&|]*>>\s*patch\.txt", text):
            return True
        if "diff -u /dev/null" in text:
            return True
        patch_write_markers = ("cat >", "tee ", "echo ", "printf ")
        writes_patch = any(marker in text for marker in patch_write_markers)
        manual_diff_markers = ("diff --git", "--- /dev/null", "+++ /dev/null", "new file mode", "index 0000000")
        if writes_patch and any(marker in text for marker in manual_diff_markers):
            return True
        if writes_patch and "git diff" not in text:
            return True
    return False


def row_weight(
    *,
    target_command: str,
    previous_command: str,
    target_turn: int,
    source: str,
    row_number: int,
    seed: int,
) -> tuple[int, str]:
    if bad_target_command(target_command):
        return 0, "drop_bad_target"

    target_cat = command_category(target_command)
    prev_cat = command_category(previous_command)

    if target_turn == 0:
        return 1, "first_turn"

    if target_command.strip() and target_command.strip() == previous_command.strip():
        return 0, "drop_same_command"

    if target_cat in EDIT_CATS:
        if prev_cat in INSPECT_CATS:
            return 8, "inspect_to_edit"
        return 5, "edit"
    if target_cat == "submit":
        if prev_cat in INSPECT_CATS | EDIT_CATS | {"diff", "test"}:
            return 6, "postwork_to_submit"
        return 3, "submit"
    if target_cat in {"diff", "test"}:
        return 5, target_cat

    if prev_cat in EDIT_CATS and target_cat in INSPECT_CATS | {"diff", "test", "submit"}:
        return 3, "after_edit_verify"

    frac = stable_fraction(seed, source, row_number, target_command, previous_command)
    if prev_cat in INSPECT_CATS and target_cat in INSPECT_CATS:
        return (1, "sample_inspect_to_inspect") if frac < 0.10 else (0, "drop_inspect_to_inspect")
    if target_cat in INSPECT_CATS:
        return (1, "sample_inspect") if frac < 0.25 else (0, "drop_inspect")
    if target_cat in {"nav", "other"}:
        return (1, f"sample_{target_cat}") if frac < 0.25 else (0, f"drop_{target_cat}")

    return 1, "keep_other"


def annotate_row(
    row: dict[str, Any],
    *,
    target_idx: int,
    target_command: str,
    raw_target_command: str,
    previous_command: str,
    target_cat: str,
    prev_cat: str,
    copy_index: int,
    weight: int,
    reason: str,
) -> dict[str, Any]:
    emitted = copy.deepcopy(row)
    messages = emitted.get("messages") or []
    if 0 <= target_idx < len(messages):
        set_assistant_command(messages[target_idx], target_command)
    metadata = emitted.setdefault("metadata", {})
    metadata["v21_selection"] = {
        "previous_command": previous_command,
        "target_command": target_command,
        "previous_category": prev_cat,
        "target_category": target_cat,
        "copy_index": copy_index,
        "weight": weight,
        "reason": reason,
    }
    if raw_target_command != target_command:
        metadata["v21_selection"]["raw_target_command"] = raw_target_command
    return emitted


def build_source(source: Path, output_source: Path, seed: int) -> dict[str, Any]:
    output_source.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stats = Counter()
    transition_stats = Counter()
    target_stats = Counter()
    source_rows_in = 0

    for path in iter_jsonl_files(source):
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                source_rows_in += 1
                row = json.loads(line)
                messages = row.get("messages") or []
                target_idx = target_message_index(messages)
                if target_idx is None:
                    stats["drop_missing_target"] += 1
                    continue

                raw_target_command = command_from_assistant(messages[target_idx])
                target_command = normalize_target_command(raw_target_command)
                previous_command = previous_assistant_command(messages, target_idx)
                target_turn = sum(1 for message in messages[: target_idx + 1] if message.get("role") == "assistant") - 1
                source_name = (
                    row.get("source")
                    or ((row.get("metadata") or {}).get("v17_selection") or {}).get("source")
                    or source.name
                )
                weight, reason = row_weight(
                    target_command=target_command,
                    previous_command=previous_command,
                    target_turn=target_turn,
                    source=str(source_name),
                    row_number=line_number,
                    seed=seed,
                )
                stats[reason] += 1
                if weight <= 0:
                    continue

                target_cat = command_category(target_command)
                prev_cat = command_category(previous_command)
                target_stats[target_cat] += weight
                transition_stats[f"{prev_cat}->{target_cat}"] += weight
                for copy_index in range(weight):
                    rows.append(
                        annotate_row(
                            row,
                            target_idx=target_idx,
                            target_command=target_command,
                            raw_target_command=raw_target_command,
                            previous_command=previous_command,
                            target_cat=target_cat,
                            prev_cat=prev_cat,
                            copy_index=copy_index,
                            weight=weight,
                            reason=reason,
                        )
                    )

    rng = random.Random(seed)
    rng.shuffle(rows)
    out_path = output_source / "data.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json_dumps(row) + "\n")

    return {
        "name": source.name,
        "rows_in": source_rows_in,
        "rows_out": len(rows),
        "stats": dict(sorted(stats.items())),
        "target_categories": dict(target_stats.most_common()),
        "transitions": dict(transition_stats.most_common(30)),
        "output": str(out_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=33333)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        build_source(source, args.output_root / source.name.replace("_v17", "_v21"), args.seed)
        for source in source_roots(args.input_root)
    ]
    input_manifest = args.input_root / "manifest.json"
    manifest: dict[str, Any] = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "rows_in": sum(item["rows_in"] for item in summaries),
        "rows_out": sum(item["rows_out"] for item in summaries),
        "sources": summaries,
        "selection": (
            "v17 prefix rows reweighted toward edit/diff/test/submit targets after observations; "
            "inspection-to-inspection and nav/other rows downsampled"
        ),
    }
    if input_manifest.exists():
        manifest["input_manifest"] = json.loads(input_manifest.read_text(encoding="utf-8"))
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
