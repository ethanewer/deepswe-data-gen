#!/usr/bin/env python3
"""Select original-prompt OpenRouter tasks outside generated/queued tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "nebius/SWE-rebench-V2"
SPLIT = "train"
LANGUAGES = ("c", "cpp", "go", "java", "js", "php", "python", "ruby", "rust", "ts")
MODELS = ("xiaomi/mimo-v2.5", "xiaomi/mimo-v2.5-pro", "deepseek/deepseek-v4-flash")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tasks", type=int, default=20_000)
    parser.add_argument("--pilot-medium-per-model", type=int, default=100)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--high-quality-ids",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "high_quality_conf_ge_0.95_instance_ids.txt",
    )
    parser.add_argument("--exclude-task-ids-file", action="append", type=Path, default=[])
    parser.add_argument("--metadata-index", action="append", type=Path, default=[])
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2"),
    )
    parser.add_argument("--seed", default="openrouter-original-wave-20260611")
    return parser.parse_args()


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()}


def ids_from_index(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    if path.suffix == ".sqlite":
        with sqlite3.connect(str(path)) as conn:
            for (task_id,) in conn.execute("select distinct task_id from traces where task_id != ''"):
                ids.add(str(task_id))
        return ids
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            task_id = row.get("task_id") or row.get("instance_id") or metadata.get("task_id") or metadata.get("instance_id")
            if task_id:
                ids.add(str(task_id))
    return ids


def ids_from_manifest(path: Path) -> set[str]:
    ids: set[str] = set()
    if path.suffix == ".tsv":
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 3 and fields[2] and fields[2] != "instance_id":
                    ids.add(fields[2])
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = row.get("task_id") or row.get("instance_id")
                if task_id:
                    ids.add(str(task_id))
    return ids


def manifest_ids(runs_root: Path) -> tuple[set[str], int]:
    ids: set[str] = set()
    files = 0
    if not runs_root.exists():
        return ids, files
    for manifest_dir in runs_root.glob("datagen-*/manifest"):
        if not manifest_dir.exists():
            continue
        for path in manifest_dir.iterdir():
            if path.suffix not in {".tsv", ".jsonl"}:
                continue
            before = len(ids)
            ids.update(ids_from_manifest(path))
            if len(ids) > before:
                files += 1
    return ids, files


def gate_failures(metadata: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if metadata.get("code") != "A":
        failures.append("code")
    if metadata.get("intent_completeness") != "complete":
        failures.append("intent")
    if metadata.get("test_alignment_issues"):
        failures.append("test_alignment")
    detected = metadata.get("detected_issues") or {}
    if isinstance(detected, dict) and any(detected.values()):
        failures.append("detected_issues")
    return failures


def patch_file_count(patch: str) -> int:
    return len(re.findall(r"(?m)^diff --git ", patch or ""))


def stable_float(seed: str, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def candidate_record(row: dict[str, Any], seed: str) -> dict[str, Any] | None:
    language = row.get("language")
    metadata = row.get("meta", {}).get("llm_metadata", {})
    difficulty = metadata.get("difficulty")
    if language not in LANGUAGES or difficulty not in {"easy", "medium"}:
        return None
    failures = gate_failures(metadata)
    if failures:
        return None
    confidence = metadata.get("confidence")
    confidence_value = float(confidence) if confidence is not None else -1.0
    patch = row.get("patch") or ""
    test_patch = row.get("test_patch") or ""
    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "language": language,
        "difficulty": difficulty,
        "confidence": confidence,
        "confidence_value": confidence_value,
        "gold_patch_bytes": len(patch.encode("utf-8")),
        "gold_patch_lines": patch.count("\n"),
        "gold_num_files_changed": patch_file_count(patch),
        "test_patch_bytes": len(test_patch.encode("utf-8")),
        "test_patch_lines": test_patch.count("\n"),
        "prompt_chars": len(row.get("problem_statement") or ""),
        "outside_original_high_quality_set": "true",
        "quality_gate_pass": "true",
        "quality_gate_failures": "",
        "quality_tier": "strict_gates_confidence_relaxed",
        "annotation_code": metadata.get("code", ""),
        "intent_completeness": metadata.get("intent_completeness", ""),
        "random_key": stable_float(seed, row["instance_id"]),
    }


def model_balanced_assign(
    rows: list[dict[str, Any]],
    models: tuple[str, ...],
    *,
    start_rank: int,
    split: str,
) -> list[dict[str, Any]]:
    by_bin: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bin[(row["difficulty"], row["language"])].append(row)
    ordered: list[dict[str, Any]] = []
    for key in sorted(by_bin):
        by_bin[key].sort(
            key=lambda row: (
                -row["confidence_value"],
                row["gold_num_files_changed"],
                row["gold_patch_bytes"],
                row["prompt_chars"],
                row["random_key"],
            )
        )
        ordered.extend(by_bin[key])

    assignments: list[dict[str, Any]] = []
    counts = Counter()
    for i, row in enumerate(ordered):
        model = models[i % len(models)]
        counts[model] += 1
        record = {k: v for k, v in row.items() if k not in {"confidence_value", "random_key"}}
        record.update(
            {
                "selection_rank": start_rank + len(assignments),
                "assigned_model": model,
                "instruction_style": "original",
                "benchmark_profile": "swebench-multilingual",
                "comparison_split": split,
                "eligible_for_controlled_comparison": "true" if split == "pilot_medium_100_per_model" else "false",
                "uses_updated_alignment": "true",
                "reason_excluded_from_comparison": "" if split == "pilot_medium_100_per_model" else "main_generation_not_controlled_pilot",
            }
        )
        assignments.append(record)
    return assignments


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "selection_rank",
        "comparison_split",
        "eligible_for_controlled_comparison",
        "uses_updated_alignment",
        "reason_excluded_from_comparison",
        "instance_id",
        "repo",
        "language",
        "difficulty",
        "confidence",
        "assigned_model",
        "instruction_style",
        "benchmark_profile",
        "outside_original_high_quality_set",
        "quality_gate_pass",
        "quality_gate_failures",
        "quality_tier",
        "annotation_code",
        "intent_completeness",
        "gold_patch_bytes",
        "gold_patch_lines",
        "gold_num_files_changed",
        "test_patch_bytes",
        "test_patch_lines",
        "prompt_chars",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    models = tuple(args.model) if args.model else MODELS
    high_quality_ids = load_ids(args.high_quality_ids)
    explicit_ids = set().union(*(load_ids(path) for path in args.exclude_task_ids_file)) if args.exclude_task_ids_file else set()
    index_ids = set().union(*(ids_from_index(path) for path in args.metadata_index)) if args.metadata_index else set()
    queued_ids, manifest_file_count = manifest_ids(args.runs_root)
    excluded = high_quality_ids | explicit_ids | index_ids | queued_ids

    candidates: list[dict[str, Any]] = []
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    for row in dataset:
        if row["instance_id"] in excluded:
            continue
        record = candidate_record(row, args.seed)
        if record is not None:
            candidates.append(record)

    candidates.sort(
        key=lambda row: (
            row["difficulty"] != "medium",
            -row["confidence_value"],
            row["language"],
            row["gold_num_files_changed"],
            row["gold_patch_bytes"],
            row["random_key"],
        )
    )
    medium = [row for row in candidates if row["difficulty"] == "medium"]
    pilot_needed = args.pilot_medium_per_model * len(models)
    if len(medium) < pilot_needed:
        raise SystemExit(f"not enough medium candidates for pilot: need {pilot_needed}, have {len(medium)}")
    pilot_rows = medium[:pilot_needed]
    pilot_ids = {row["instance_id"] for row in pilot_rows}
    remaining = [row for row in candidates if row["instance_id"] not in pilot_ids]
    main_needed = max(0, args.target_tasks - len(pilot_rows))
    main_rows = remaining[:main_needed]

    pilot_assignments = model_balanced_assign(
        pilot_rows,
        models,
        start_rank=0,
        split="pilot_medium_100_per_model",
    )
    main_assignments = model_balanced_assign(
        main_rows,
        models,
        start_rank=len(pilot_assignments),
        split="main_generation",
    )
    assignments = pilot_assignments + main_assignments

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "assignments_all.csv", assignments)
    write_csv(args.output_dir / "assignments_pilot_medium.csv", pilot_assignments)
    write_csv(args.output_dir / "assignments_main.csv", main_assignments)
    for model in models:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model).strip("-")
        write_csv(args.output_dir / f"assignments_{safe}.csv", [row for row in assignments if row["assigned_model"] == model])

    summary = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "target_tasks": args.target_tasks,
        "selected_total": len(assignments),
        "selected_unique_tasks": len({row["instance_id"] for row in assignments}),
        "instruction_style": "original",
        "benchmark_profile": "swebench-multilingual",
        "models": list(models),
        "pilot_medium_per_model": args.pilot_medium_per_model,
        "candidate_count": len(candidates),
        "candidate_medium_count": len(medium),
        "excluded_counts": {
            "high_quality_ids": len(high_quality_ids),
            "explicit_task_ids": len(explicit_ids),
            "metadata_index_ids": len(index_ids),
            "manifest_or_queued_ids": len(queued_ids),
            "manifest_files_scanned_count": manifest_file_count,
            "union_excluded_ids": len(excluded),
        },
        "by_model": dict(Counter(row["assigned_model"] for row in assignments)),
        "by_difficulty": dict(Counter(row["difficulty"] for row in assignments)),
        "by_language": dict(Counter(row["language"] for row in assignments)),
        "pilot_by_model": dict(Counter(row["assigned_model"] for row in pilot_assignments)),
        "pilot_by_language": dict(Counter(row["language"] for row in pilot_assignments)),
        "main_by_model": dict(Counter(row["assigned_model"] for row in main_assignments)),
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if len(assignments) != len({row["instance_id"] for row in assignments}):
        raise SystemExit("selected assignments are not task-disjoint")
    if len(assignments) < min(args.target_tasks, len(candidates)):
        raise SystemExit(f"selected {len(assignments)} rows but target was {args.target_tasks}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
