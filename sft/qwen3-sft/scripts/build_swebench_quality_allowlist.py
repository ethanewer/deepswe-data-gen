#!/usr/bin/env python3
"""Build a UUID allowlist from SWE trace metadata indexes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def is_trueish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "passed", "pass"}
    return False


def is_positive(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        try:
            return float(value.strip()) > 0
        except ValueError:
            return is_trueish(value)
    return False


def as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def source_name(path: Path) -> str:
    return path.name.removesuffix(".jsonl")


def passes(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if args.require_passed and not is_trueish(row.get("passed")):
        return False, "not_passed"
    if args.require_positive_reward and not is_positive(row.get("reward")):
        return False, "nonpositive_reward"
    if args.require_submitted and row.get("agent_exit_status") != "Submitted":
        return False, "not_submitted"
    if args.require_nonempty_patch and as_float(row.get("model_patch_bytes"), 0.0) <= 0:
        return False, "empty_patch"
    if as_float(row.get("percent_messages_with_reasoning"), -1.0) < args.min_reasoning_coverage:
        return False, "low_reasoning_coverage"
    if args.require_positive_api_calls and as_float(row.get("api_calls"), 0.0) <= 0:
        return False, "no_api_calls"
    if not row.get("uuid"):
        return False, "missing_uuid"
    return True, "kept"


def build(args: argparse.Namespace) -> dict[str, Any]:
    uuids: list[str] = []
    line_numbers: list[int] = []
    seen_uuids: set[str] = set()
    rows_seen = 0
    rows_kept = 0
    drop_reasons: Counter[str] = Counter()
    kept_by_language: Counter[str] = Counter()
    kept_by_source_index: Counter[str] = Counter()

    for index_path in args.index_jsonl:
        index_name = source_name(index_path)
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows_seen += 1
                row = json.loads(line)
                keep, reason = passes(row, args)
                if not keep:
                    drop_reasons[reason] += 1
                    continue
                uuid = str(row["uuid"])
                if uuid in seen_uuids:
                    drop_reasons["duplicate_uuid"] += 1
                    continue
                seen_uuids.add(uuid)
                uuids.append(uuid)
                line_numbers.append(int(row["line_number"]))
                rows_kept += 1
                kept_by_language[str(row.get("language") or "unknown")] += 1
                kept_by_source_index[index_name] += 1

    args.output_uuid_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_uuid_file.write_text("\n".join(uuids) + ("\n" if uuids else ""), encoding="utf-8")
    if args.output_line_number_file:
        args.output_line_number_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_line_number_file.write_text(
            "\n".join(str(line_number) for line_number in line_numbers) + ("\n" if line_numbers else ""),
            encoding="utf-8",
        )

    manifest = {
        "index_jsonl": [str(path) for path in args.index_jsonl],
        "output_uuid_file": str(args.output_uuid_file),
        "output_line_number_file": str(args.output_line_number_file) if args.output_line_number_file else None,
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "kept_by_language": dict(sorted(kept_by_language.items())),
        "kept_by_source_index": dict(sorted(kept_by_source_index.items())),
        "filters": {
            "require_passed": args.require_passed,
            "require_positive_reward": args.require_positive_reward,
            "require_submitted": args.require_submitted,
            "require_nonempty_patch": args.require_nonempty_patch,
            "min_reasoning_coverage": args.min_reasoning_coverage,
            "require_positive_api_calls": args.require_positive_api_calls,
        },
    }
    if args.output_manifest:
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-uuid-file", type=Path, required=True)
    parser.add_argument("--output-line-number-file", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--min-reasoning-coverage", type=float, default=0.90)
    parser.add_argument("--allow-failed", dest="require_passed", action="store_false")
    parser.add_argument("--allow-nonpositive-reward", dest="require_positive_reward", action="store_false")
    parser.add_argument("--allow-nonsubmitted", dest="require_submitted", action="store_false")
    parser.add_argument("--allow-empty-patch", dest="require_nonempty_patch", action="store_false")
    parser.add_argument("--allow-zero-api-calls", dest="require_positive_api_calls", action="store_false")
    parser.set_defaults(
        require_passed=True,
        require_positive_reward=True,
        require_submitted=True,
        require_nonempty_patch=True,
        require_positive_api_calls=True,
    )
    return parser.parse_args()


def main() -> int:
    manifest = build(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
