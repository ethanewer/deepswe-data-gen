#!/usr/bin/env python3
"""Build the v37 edit-anchor mix without duplicating long prefixes in memory.

The mix is intentionally small and targeted:

* v1 prefix rows provide the only observed checkpoint behavior that submitted a
  non-empty patch under mini-swe-agent.
* v17 prefix rows anchor anti-repeat first-turn behavior.
* v23 patch-recovery rows emphasize inspect->edit, edit->diff, and diff->submit
  transitions from verified source patches.

Rows are selected by deterministic hash after a counting pass, then written
directly into output shards. No tokenization is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


INSPECT_CATS = {"cat", "search", "view", "nav"}
EDIT_CATS = {"edit_shell", "edit_python", "edit_patch"}


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_jsonl_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.jsonl"))


def stable_fraction(*parts: object) -> float:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


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


def set_target_content(row: dict[str, Any], bucket: str, target_cat: str) -> None:
    messages = row.get("messages") or []
    target_idx = target_message_index(messages)
    if target_idx is None:
        return
    if bucket in {"inspect_to_edit", "edit"} or target_cat in EDIT_CATS:
        thought = "I have enough context to edit the source file now."
    elif target_cat == "diff":
        thought = "The change is applied. I will inspect the final diff."
    elif target_cat == "test":
        thought = "I will run a focused verification command."
    elif target_cat == "submit":
        thought = "The source diff is ready. I will submit patch.txt."
    else:
        thought = "I will inspect the repository with a focused shell command."
    messages[target_idx]["content"] = f"<thought>\n{thought}\n</thought>"
    messages[target_idx].pop("reasoning_content", None)


def bad_target_command(command: str) -> bool:
    text = command.strip().lower()
    if not text:
        return False
    if "git apply patch.txt" in text:
        return False
    edit_markers = ("apply_patch", "cat >", "tee ", "perl -", "sed -i", "python3 - <<", "python - <<", "git apply")
    if not any(marker in text for marker in edit_markers):
        return False
    bad_markers = (
        "/tmp/",
        "mkdir -p /tmp",
        "cat > test",
        "cat > ./test",
        "cat > tests/",
        "cat > __tests__/",
        "cat > spec/",
        ".sh <<",
        "reproduce",
        "debug_",
        "check_",
        "script_",
        "scratch_",
        "patch.txt",
    )
    return any(marker in text for marker in bad_markers)


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
    if "git apply" in text or "apply_patch" in text:
        return "edit_patch"
    if "python" in text and ("write_text" in text or ".replace(" in text or "insert" in text):
        return "edit_python"
    if any(marker in text for marker in ("cat >", "tee ", "perl -", "sed -i", "python3 - <<", "python - <<")):
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


def row_bucket(row: dict[str, Any]) -> tuple[str, str, str, int]:
    messages = row.get("messages") or []
    target_idx = target_message_index(messages)
    if target_idx is None:
        return "drop", "none", "none", -1
    target = command_from_assistant(messages[target_idx])
    if bad_target_command(target):
        return "drop_bad_target", command_category(target), "none", -1
    previous = previous_assistant_command(messages, target_idx)
    target_cat = command_category(target)
    prev_cat = command_category(previous)
    target_turn = sum(1 for message in messages[: target_idx + 1] if message.get("role") == "assistant") - 1
    if target_turn == 0:
        bucket = "first_turn"
    elif target.strip() and previous.strip() and target.strip() == previous.strip():
        bucket = "repeat_same"
    elif prev_cat in INSPECT_CATS and target_cat in EDIT_CATS:
        bucket = "inspect_to_edit"
    elif target_cat in EDIT_CATS:
        bucket = "edit"
    elif target_cat in {"diff", "test", "submit"}:
        bucket = target_cat
    elif prev_cat in EDIT_CATS and target_cat in INSPECT_CATS | {"diff", "test", "submit"}:
        bucket = "after_edit"
    elif target_cat in INSPECT_CATS:
        bucket = "inspect"
    else:
        bucket = "other"
    return bucket, target_cat, prev_cat, target_turn


def iter_rows(root: Path) -> Iterable[tuple[Path, int, str, dict[str, Any]]]:
    for path in iter_jsonl_files(root):
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                yield path, line_number, line, json.loads(line)


def count_buckets(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, _, _, row in iter_rows(root):
        bucket, _, _, _ = row_bucket(row)
        counts[bucket] += 1
    return counts


def selection_probability(bucket: str, count: int, caps: dict[str, int]) -> float:
    cap = caps.get(bucket, 0)
    if cap <= 0 or count <= 0:
        return 0.0
    return min(1.0, cap / count)


class ShardedWriter:
    def __init__(self, output_dir: Path, shards: int) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.paths = [output_dir / f"shard-{idx:03d}.jsonl" for idx in range(shards)]
        self.handles = [path.open("w", encoding="utf-8") for path in self.paths]
        self.rows = 0

    def write(self, row: dict[str, Any]) -> None:
        self.handles[self.rows % len(self.handles)].write(json_dumps(row) + "\n")
        self.rows += 1

    def close(self) -> int:
        for handle in self.handles:
            handle.close()
        nonempty = 0
        for path in self.paths:
            if path.stat().st_size == 0:
                path.unlink()
            else:
                nonempty += 1
        return nonempty


def select_source(
    *,
    input_root: Path,
    output_root: Path,
    source_name: str,
    caps: dict[str, int],
    seed: int,
    shards: int,
    tag: str,
    compact_target_content: bool,
) -> dict[str, Any]:
    counts = count_buckets(input_root)
    selected_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    prev_counts: Counter[str] = Counter()
    writer = ShardedWriter(output_root / source_name, shards)
    try:
        for path, line_number, _, row in iter_rows(input_root):
            bucket, target_cat, prev_cat, _ = row_bucket(row)
            prob = selection_probability(bucket, counts[bucket], caps)
            if stable_fraction(seed, source_name, path, line_number, bucket) >= prob:
                continue
            if compact_target_content:
                set_target_content(row, bucket, target_cat)
            metadata = row.setdefault("metadata", {})
            metadata[f"{tag}_selection"] = {
                "source_name": source_name,
                "source_file": str(path),
                "source_line": line_number,
                "bucket": bucket,
                "target_category": target_cat,
                "previous_category": prev_cat,
                "selection_probability": prob,
            }
            writer.write(row)
            selected_counts[bucket] += 1
            target_counts[target_cat] += 1
            prev_counts[prev_cat] += 1
    finally:
        nonempty = writer.close()
    return {
        "name": source_name,
        "input_root": str(input_root),
        "rows_in_by_bucket": dict(counts),
        "rows_out": sum(selected_counts.values()),
        "rows_out_by_bucket": dict(selected_counts),
        "target_categories": dict(target_counts),
        "previous_categories": dict(prev_counts),
        "output_files": nonempty,
        "caps": caps,
    }


def parse_caps(value: str) -> dict[str, int]:
    caps: dict[str, int] = {}
    for item in value.split(","):
        if not item:
            continue
        key, raw = item.split("=", 1)
        caps[key.strip()] = int(raw)
    return caps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-prefix-root", type=Path, required=True)
    parser.add_argument("--v17-root", type=Path, required=True)
    parser.add_argument("--v23-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=37037)
    parser.add_argument("--tag", default="v37")
    parser.add_argument("--shards-per-source", type=int, default=32)
    parser.add_argument("--compact-target-content", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--v1-caps",
        default="inspect_to_edit=14000,edit=6000,diff=2000,test=5000,submit=9000,after_edit=3000,first_turn=2000,inspect=2000,other=2000",
    )
    parser.add_argument(
        "--v17-caps",
        default="first_turn=8000,inspect=5000,inspect_to_edit=3000,edit=2000,diff=200,test=200,submit=1000,after_edit=1000,other=2000",
    )
    parser.add_argument(
        "--v23-caps",
        default="inspect_to_edit=12000,edit=12000,diff=6000,submit=6000,after_edit=4000,first_turn=0,inspect=0,other=0",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    sources = [
        select_source(
            input_root=args.v1_prefix_root,
            output_root=args.output_root,
            source_name=f"v1_edit_submit_prefix_anchor_{args.tag}",
            caps=parse_caps(args.v1_caps),
            seed=args.seed,
            shards=args.shards_per_source,
            tag=args.tag,
            compact_target_content=args.compact_target_content,
        ),
        select_source(
            input_root=args.v17_root,
            output_root=args.output_root,
            source_name=f"v17_antirepeat_anchor_{args.tag}",
            caps=parse_caps(args.v17_caps),
            seed=args.seed + 1,
            shards=args.shards_per_source,
            tag=args.tag,
            compact_target_content=args.compact_target_content,
        ),
        select_source(
            input_root=args.v23_root,
            output_root=args.output_root,
            source_name=f"v23_patch_recovery_{args.tag}",
            caps=parse_caps(args.v23_caps),
            seed=args.seed + 2,
            shards=args.shards_per_source,
            tag=args.tag,
            compact_target_content=args.compact_target_content,
        ),
    ]
    manifest = {
        "output_root": str(args.output_root),
        "seed": args.seed,
        "tag": args.tag,
        "compact_target_content": args.compact_target_content,
        "selection": (
            f"{args.tag} targeted mix: v1 prefix edit/submit behavior anchor, v17 anti-repeat "
            "first-turn anchor, and v23 verified patch recovery transitions"
        ),
        "rows_out": sum(item["rows_out"] for item in sources),
        "sources": sources,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
