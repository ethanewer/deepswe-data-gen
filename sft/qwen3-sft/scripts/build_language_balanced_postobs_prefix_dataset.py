#!/usr/bin/env python3
"""Build a language-balanced post-observation prefix SFT dataset.

This starts from prefix-target rows produced by ``build_prefix_target_dataset.py``
and applies the same action weighting as
``build_weighted_postobs_prefix_dataset.py``. It then multiplies the final row
count by source language so short iterative runs are less dominated by Python,
Go, TypeScript, and JavaScript when targeting SWE-bench Multilingual.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from build_weighted_postobs_prefix_dataset import (
    command_category,
    command_from_assistant,
    iter_jsonl_files,
    json_dumps,
    normalize_target_command,
    previous_assistant_command,
    row_weight,
    set_assistant_command,
    source_roots,
    stable_fraction,
    target_message_index,
)


DEFAULT_LANGUAGE_WEIGHTS = {
    "c": 3.0,
    "cpp": 3.0,
    "java": 3.0,
    "php": 3.0,
    "rust": 3.0,
    "go": 1.5,
    "js": 1.0,
    "ts": 1.0,
    "python": 0.7,
}


def parse_language_weights(items: list[str]) -> dict[str, float]:
    weights = dict(DEFAULT_LANGUAGE_WEIGHTS)
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --language-weight {item!r}; expected lang=weight")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        weight = float(value)
        if not key:
            raise ValueError(f"invalid empty language in --language-weight {item!r}")
        if weight < 0:
            raise ValueError(f"negative weight for {key}: {weight}")
        weights[key] = weight
    return weights


def load_uuid_metadata(index_paths: list[Path]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for path in index_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                uuid = str(row.get("uuid") or "")
                if not uuid:
                    continue
                metadata[uuid] = {
                    "language": str(row.get("language") or "unknown").lower(),
                    "task_id": str(row.get("task_id") or ""),
                    "line_number": str(row.get("line_number") or ""),
                    "index_source": path.name.removesuffix(".jsonl"),
                }
    return metadata


def load_source_line_uuids(aligned_root: Path) -> dict[tuple[str, int], str]:
    source_line_uuid: dict[tuple[str, int], str] = {}
    for path in iter_jsonl_files(aligned_root):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                source_outcome = row.get("source_outcome") or {}
                uuid = str(source_outcome.get("uuid") or "")
                if uuid:
                    source_line_uuid[(str(path), line_number)] = uuid
    return source_line_uuid


def language_for_row(
    row: dict[str, Any],
    *,
    source_line_uuid: dict[tuple[str, int], str],
    uuid_metadata: dict[str, dict[str, str]],
) -> tuple[str, str, str, str]:
    metadata = row.get("metadata") or {}
    source_file = str(metadata.get("source_file") or "")
    try:
        source_line = int(metadata.get("source_line") or 0)
    except (TypeError, ValueError):
        source_line = 0
    uuid = source_line_uuid.get((source_file, source_line), "")
    source_metadata = uuid_metadata.get(uuid, {})
    return (
        str(source_metadata.get("language") or "unknown").lower(),
        str(source_metadata.get("task_id") or ""),
        uuid,
        str(source_metadata.get("index_source") or "unknown"),
    )


def scaled_copy_count(
    *,
    base_weight: int,
    language_scale: float,
    seed: int,
    source: str,
    row_number: int,
    language: str,
    target_command: str,
) -> int:
    if base_weight <= 0 or language_scale <= 0:
        return 0
    expected = base_weight * language_scale
    copies = int(math.floor(expected))
    frac = expected - copies
    if frac and stable_fraction(seed, source, row_number, language, target_command, "lang-scale") < frac:
        copies += 1
    return copies


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
    copies: int,
    base_weight: int,
    language_scale: float,
    reason: str,
    language: str,
    task_id: str,
    uuid: str,
    index_source: str,
) -> dict[str, Any]:
    emitted = copy.deepcopy(row)
    messages = emitted.get("messages") or []
    if 0 <= target_idx < len(messages):
        set_assistant_command(messages[target_idx], target_command)
    metadata = emitted.setdefault("metadata", {})
    metadata["v44_language_balance"] = {
        "previous_command": previous_command,
        "target_command": target_command,
        "previous_category": prev_cat,
        "target_category": target_cat,
        "copy_index": copy_index,
        "copies": copies,
        "base_action_weight": base_weight,
        "language_scale": language_scale,
        "reason": reason,
        "language": language,
        "task_id": task_id,
        "uuid": uuid,
        "index_source": index_source,
    }
    if raw_target_command != target_command:
        metadata["v44_language_balance"]["raw_target_command"] = raw_target_command
    return emitted


def build_source(
    source: Path,
    output_source: Path,
    *,
    seed: int,
    language_weights: dict[str, float],
    source_line_uuid: dict[tuple[str, int], str],
    uuid_metadata: dict[str, dict[str, str]],
) -> dict[str, Any]:
    output_source.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stats = Counter()
    language_stats = Counter()
    action_language_stats = Counter()
    target_stats = Counter()
    transition_stats = Counter()
    index_source_stats = Counter()
    source_rows_in = 0

    for path in iter_jsonl_files(source):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
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
                target_turn = sum(
                    1 for message in messages[: target_idx + 1] if message.get("role") == "assistant"
                ) - 1
                source_name = (
                    row.get("source")
                    or ((row.get("metadata") or {}).get("v17_selection") or {}).get("source")
                    or source.name
                )
                base_weight, reason = row_weight(
                    target_command=target_command,
                    previous_command=previous_command,
                    target_turn=target_turn,
                    source=str(source_name),
                    row_number=line_number,
                    seed=seed,
                )
                stats[reason] += 1
                if base_weight <= 0:
                    continue

                language, task_id, uuid, index_source = language_for_row(
                    row,
                    source_line_uuid=source_line_uuid,
                    uuid_metadata=uuid_metadata,
                )
                language_scale = language_weights.get(language, 1.0)
                copies = scaled_copy_count(
                    base_weight=base_weight,
                    language_scale=language_scale,
                    seed=seed,
                    source=str(source_name),
                    row_number=line_number,
                    language=language,
                    target_command=target_command,
                )
                if copies <= 0:
                    stats["drop_language_scaled_to_zero"] += 1
                    continue

                target_cat = command_category(target_command)
                prev_cat = command_category(previous_command)
                language_stats[language] += copies
                action_language_stats[f"{language}:{target_cat}"] += copies
                target_stats[target_cat] += copies
                transition_stats[f"{prev_cat}->{target_cat}"] += copies
                index_source_stats[index_source] += copies
                for copy_index in range(copies):
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
                            copies=copies,
                            base_weight=base_weight,
                            language_scale=language_scale,
                            reason=reason,
                            language=language,
                            task_id=task_id,
                            uuid=uuid,
                            index_source=index_source,
                        )
                    )

    rng = random.Random(seed)
    rng.shuffle(rows)
    output_path = output_source / "data.jsonl"
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json_dumps(row) + "\n")

    return {
        "name": source.name,
        "rows_in": source_rows_in,
        "rows_out": len(rows),
        "stats": dict(sorted(stats.items())),
        "language_rows": dict(language_stats.most_common()),
        "target_categories": dict(target_stats.most_common()),
        "action_language_rows": dict(action_language_stats.most_common(80)),
        "transitions": dict(transition_stats.most_common(30)),
        "index_source_rows": dict(index_source_stats.most_common()),
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--metadata-index-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--language-weight", action="append", default=[])
    parser.add_argument("--seed", type=int, default=61644)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    language_weights = parse_language_weights(args.language_weight)
    uuid_metadata = load_uuid_metadata(args.metadata_index_jsonl)
    source_line_uuid = load_source_line_uuids(args.aligned_root)
    summaries = [
        build_source(
            source,
            args.output_root / source.name,
            seed=args.seed,
            language_weights=language_weights,
            source_line_uuid=source_line_uuid,
            uuid_metadata=uuid_metadata,
        )
        for source in source_roots(args.input_root)
    ]
    manifest = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "aligned_root": str(args.aligned_root),
        "metadata_index_jsonl": [str(path) for path in args.metadata_index_jsonl],
        "seed": args.seed,
        "language_weights": language_weights,
        "rows_in": sum(item["rows_in"] for item in summaries),
        "rows_out": sum(item["rows_out"] for item in summaries),
        "sources": summaries,
        "selection": (
            "post-observation edit/diff/test/submit action weighting plus "
            "deterministic language balancing for SWE-bench Multilingual"
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
