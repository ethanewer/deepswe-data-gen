#!/usr/bin/env python3
"""Build a patch-finalization-heavy prefix SFT dataset.

This streams an existing prefix-target JSONL root and deterministically
resamples rows toward patch recovery and finalization targets. It is intended
for large JSONL roots where loading all rows into memory is unsafe. The online
packer's indexed row shuffle handles training-time randomization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_jsonl_files(root: Path) -> Iterable[Path]:
    return sorted(root.rglob("*.jsonl"))


def stable_fraction(seed: int, *parts: object) -> float:
    payload = "\0".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def copy_count(weight: float, *, seed: int, row_key: str) -> int:
    if weight <= 0:
        return 0
    copies = int(math.floor(weight))
    frac = weight - copies
    if frac and stable_fraction(seed, row_key, "fractional-copy") < frac:
        copies += 1
    return copies


def command_from_assistant(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls or not isinstance(calls[0], dict):
        return ""
    function = calls[0].get("function") or {}
    args = function.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    if not isinstance(args, dict):
        return ""
    return str(args.get("command") or args.get("cmd") or "")


def target_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if (
            message.get("role") == "assistant"
            and message.get("trainable") is not False
            and message.get("loss") is not False
        ):
            return message
    return None


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
    if any(marker in text for marker in ("apply_patch", "cat >", "tee ", "perl -", "sed -i", "python3 - <<", "python - <<")):
        return "edit_shell"
    if any(marker in text for marker in ("pytest", "mvn test", "gradle test", "npm test", "go test", "cargo test", "phpunit", "unittest")):
        return "test"
    if text.startswith("cat ") or " cat " in text:
        return "cat"
    if "grep" in text or "rg " in text or text.startswith("find "):
        return "search"
    if text.startswith("sed ") or " sed " in text or text.startswith("head ") or text.startswith("tail "):
        return "view"
    if text.startswith("ls") or text == "pwd" or " pwd" in text:
        return "nav"
    return "other"


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    value = metadata.get("v44_language_balance") or metadata.get("v21_selection") or {}
    return value if isinstance(value, dict) else {}


def row_weight(row: dict[str, Any]) -> tuple[float, str, str, str, str]:
    messages = row.get("messages") or []
    target = target_message(messages)
    command = command_from_assistant(target or {})
    meta = row_metadata(row)
    target_cat = str(meta.get("target_category") or command_category(command)).lower()
    previous_cat = str(meta.get("previous_category") or "unknown").lower()
    previous_command = str(meta.get("previous_command") or "")
    text = command.lower()
    prev_text = previous_command.lower()

    if target_cat == "submit":
        if previous_cat in {"cat", "diff", "test"} or "patch.txt" in prev_text:
            return 6.0, "submit_after_patch_or_verify", target_cat, previous_cat, command
        return 3.0, "submit_other", target_cat, previous_cat, command

    if target_cat == "cat" and "patch.txt" in text:
        return 6.0, "patch_txt_verify", target_cat, previous_cat, command

    if target_cat == "diff":
        if "patch.txt" in text or "git diff" in text:
            return 2.5, "git_diff_or_patch_write", target_cat, previous_cat, command
        return 1.0, "diff_other", target_cat, previous_cat, command

    if target_cat == "test":
        return 1.25, "test_keep", target_cat, previous_cat, command

    if target_cat == "edit_python":
        return 0.9, "edit_python_keep", target_cat, previous_cat, command

    if target_cat == "edit_shell":
        if "sed -i" in text:
            return 0.35, "downweight_sed_edit", target_cat, previous_cat, command
        if any(marker in text for marker in ("apply_patch", "python", "perl -")):
            return 0.85, "edit_shell_structured", target_cat, previous_cat, command
        return 0.55, "edit_shell_other", target_cat, previous_cat, command

    if target_cat in {"cat", "view", "search"}:
        return 0.20, "inspect_sample", target_cat, previous_cat, command

    if target_cat == "nav":
        return 0.03, "nav_sample", target_cat, previous_cat, command

    if target_cat in {"other", "none"}:
        return 0.12, "other_sample", target_cat, previous_cat, command

    return 0.25, "fallback_sample", target_cat, previous_cat, command


def annotated(row: dict[str, Any], *, reason: str, copy_index: int, copies: int, weight: float) -> dict[str, Any]:
    emitted = copy.deepcopy(row)
    metadata = emitted.setdefault("metadata", {})
    metadata["v50_patch_finalization"] = {
        "reason": reason,
        "copy_index": copy_index,
        "copies": copies,
        "weight": weight,
    }
    return emitted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=61650)
    parser.add_argument("--max-input-rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)

    output_data = args.output_root / "data"
    output_data.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_data / "data.jsonl"

    stats: Counter[str] = Counter()
    source_summaries = []
    with output_jsonl.open("w", encoding="utf-8") as out:
        for path in iter_jsonl_files(args.input_root):
            rel = path.relative_to(args.input_root).as_posix()
            rows_in = rows_out = 0
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if args.max_input_rows and stats["rows_in"] >= args.max_input_rows:
                        break
                    if not line.strip():
                        continue
                    rows_in += 1
                    stats["rows_in"] += 1
                    row = json.loads(line)
                    weight, reason, target_cat, prev_cat, command = row_weight(row)
                    row_key = f"{rel}:{line_number}:{reason}:{target_cat}:{command}"
                    copies = copy_count(weight, seed=args.seed, row_key=row_key)
                    stats[f"reason:{reason}"] += 1
                    stats[f"target_category_in:{target_cat}"] += 1
                    stats[f"transition_in:{prev_cat}->{target_cat}"] += 1
                    if copies <= 0:
                        stats["rows_dropped_by_sampling"] += 1
                        continue
                    for copy_index in range(copies):
                        out.write(json_dumps(annotated(row, reason=reason, copy_index=copy_index, copies=copies, weight=weight)))
                        out.write("\n")
                        rows_out += 1
                        stats["rows_out"] += 1
                        stats[f"target_category_out:{target_cat}"] += 1
                        stats[f"transition_out:{prev_cat}->{target_cat}"] += 1
            source_summaries.append({"input": str(path), "rows_in": rows_in, "rows_out": rows_out})
            if args.max_input_rows and stats["rows_in"] >= args.max_input_rows:
                break

    manifest = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "max_input_rows": args.max_input_rows,
        "output": str(output_jsonl),
        "selection": "streaming resample of v44 language-balanced prefix rows toward patch verification, git diff, and final submit targets",
        "stats": dict(sorted(stats.items())),
        "sources": source_summaries,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
