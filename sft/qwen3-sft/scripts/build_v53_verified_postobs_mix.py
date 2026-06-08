#!/usr/bin/env python3
"""Build v53: verified v30 post-observation targets plus v17 anchor.

The output is raw chat JSONL. Tokenization and packing remain online during
training. The builder samples by byte budget instead of loading full sources
into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_V30_WEIGHTED = Path("qwen3-sft/data/swebench_ml_v53_v30_verified_weighted_prefix64_raw")
DEFAULT_V17_ANCHOR = Path("qwen3-sft/data/swebench_ml_v17_antirepeat_visible_prefix_raw_sharded")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_jsonl_files(root: Path) -> Iterable[Path]:
    return sorted(root.rglob("*.jsonl"))


def stable_fraction(seed: int, *parts: object) -> float:
    payload = "\0".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
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


def sample_root(
    *,
    input_root: Path,
    output_source: Path,
    seed: int,
    target_bytes: int,
    keep_prob_cap: float,
    source_tag: str,
) -> dict[str, Any]:
    output_source.mkdir(parents=True, exist_ok=True)
    files = list(iter_jsonl_files(input_root))
    rng = random.Random(seed)
    rng.shuffle(files)

    source_bytes = sum(path.stat().st_size for path in files)
    keep_prob = min(keep_prob_cap, target_bytes / max(source_bytes, 1))
    rows_in = rows_out = bytes_out = 0
    category_counts: Counter[str] = Counter()
    out_path = output_source / "data.jsonl"

    with out_path.open("w", encoding="utf-8") as out:
        for path in files:
            rel = path.relative_to(input_root).as_posix()
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    rows_in += 1
                    if bytes_out >= target_bytes:
                        break
                    if stable_fraction(seed, rel, line_number) > keep_prob:
                        continue
                    row = json.loads(line)
                    messages = row.get("messages") or []
                    target = target_message(messages)
                    category = command_category(command_from_assistant(target or {}))
                    metadata = row.setdefault("metadata", {})
                    metadata["v53_mix"] = {
                        "source_tag": source_tag,
                        "source_file": rel,
                        "source_line": line_number,
                        "target_category": category,
                    }
                    payload = json_dumps(row) + "\n"
                    out.write(payload)
                    rows_out += 1
                    bytes_out += len(payload.encode("utf-8"))
                    category_counts[category] += 1
            if bytes_out >= target_bytes:
                break

    return {
        "source_tag": source_tag,
        "input_root": str(input_root),
        "output": str(out_path),
        "source_files": len(files),
        "source_bytes": source_bytes,
        "target_bytes": target_bytes,
        "keep_prob": keep_prob,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "bytes_out": bytes_out,
        "rough_tokens": bytes_out // 4,
        "target_categories": dict(category_counts.most_common()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v30-weighted-root", type=Path, default=DEFAULT_V30_WEIGHTED)
    parser.add_argument("--v17-anchor-root", type=Path, default=DEFAULT_V17_ANCHOR)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--v30-target-mb", type=int, default=660)
    parser.add_argument("--v17-target-mb", type=int, default=180)
    parser.add_argument("--seed", type=int, default=6060853)
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
        sample_root(
            input_root=args.v30_weighted_root,
            output_source=args.output_root / "v30_verified_weighted_postobs_v53",
            seed=args.seed,
            target_bytes=args.v30_target_mb * 1024 * 1024,
            keep_prob_cap=1.0,
            source_tag="v30_verified_weighted_postobs",
        ),
        sample_root(
            input_root=args.v17_anchor_root,
            output_source=args.output_root / "v17_antirepeat_anchor_v53",
            seed=args.seed + 17,
            target_bytes=args.v17_target_mb * 1024 * 1024,
            keep_prob_cap=1.0,
            source_tag="v17_antirepeat_anchor",
        ),
    ]

    manifest = {
        "output_root": str(args.output_root),
        "seed": args.seed,
        "selection": "compact verified v30 weighted post-observation prefix targets plus v17 parser/anti-repeat anchor",
        "sources": summaries,
        "total_rows": sum(item["rows_out"] for item in summaries),
        "total_bytes": sum(item["bytes_out"] for item in summaries),
        "rough_token_estimate_from_bytes_div4": sum(item["bytes_out"] for item in summaries) // 4,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
