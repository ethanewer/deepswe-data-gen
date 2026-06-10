#!/usr/bin/env python3
"""Prepare balanced OpenRouter comparison task sets for SWE-rebench V2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "nebius/SWE-rebench-V2"
SPLIT = "train"
LANGUAGES = ("c", "cpp", "go", "java", "js", "php", "python", "ruby", "rust", "ts")
DIFFICULTIES = ("easy", "medium")
MODELS = (
    "inclusionai/ring-2.6-1t",
    "xiaomi/mimo-v2.5",
    "deepseek/deepseek-v4-flash",
)
INSTRUCTION_STYLES = ("original", "deepswe")
CELLS = tuple((model, style) for model in MODELS for style in INSTRUCTION_STYLES)
BENCHMARK_PROFILE_BY_STYLE = {
    "original": "swebench-multilingual",
    "swe_rebench": "swebench-multilingual",
    "deepswe": "deepswe",
    "rewritten": "deepswe",
    "planned": "deepswe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size-per-model-profile", type=int, default=60)
    parser.add_argument("--pilot-size-per-model-profile", type=int, default=2)
    parser.add_argument(
        "--high-quality-ids",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "high_quality_conf_ge_0.95_instance_ids.txt",
    )
    parser.add_argument(
        "--metadata-index",
        type=Path,
        action="append",
        default=[],
        help="Dataset metadata/index JSONL files whose task IDs must be excluded.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2"),
        help="Run root. One-level */manifest files are scanned, never trace trees.",
    )
    parser.add_argument("--seed", default="openrouter-comparison-updated-22fa515")
    return parser.parse_args()


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()}


def covered_ids_from_index(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                task_id = (
                    row.get("task_id")
                    or row.get("instance_id")
                    or metadata.get("task_id")
                    or metadata.get("instance_id")
                )
                if task_id:
                    ids.add(str(task_id))
    return ids


def ids_from_manifest_file(path: Path) -> set[str]:
    ids: set[str] = set()
    if path.suffix == ".tsv":
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 3 and fields[2] and fields[2] != "instance_id":
                    ids.add(fields[2])
        return ids
    if path.suffix == ".jsonl":
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
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            return ids
        rows = data if isinstance(data, list) else data.get("rows") if isinstance(data, dict) else None
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    task_id = row.get("task_id") or row.get("instance_id")
                    if task_id:
                        ids.add(str(task_id))
    return ids


def manifest_ids(runs_root: Path) -> tuple[set[str], list[str]]:
    ids: set[str] = set()
    files: list[str] = []
    if not runs_root.exists():
        return ids, files
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        manifest_dir = run_dir / "manifest"
        if not manifest_dir.exists():
            continue
        for path in sorted(manifest_dir.iterdir()):
            if path.suffix not in {".tsv", ".jsonl", ".json"}:
                continue
            before = len(ids)
            ids.update(ids_from_manifest_file(path))
            if len(ids) > before:
                files.append(str(path))
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
    if language not in LANGUAGES or difficulty not in DIFFICULTIES:
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
        "quality_gate_pass": "true",
        "quality_gate_failures": "",
        "quality_tier": "strict_gates_confidence_relaxed",
        "annotation_code": metadata.get("code", ""),
        "intent_completeness": metadata.get("intent_completeness", ""),
        "outside_original_high_quality_set": "true",
        "random_key": stable_float(seed, row["instance_id"]),
    }


def target_by_difficulty(total: int, available: dict[tuple[str, str], int]) -> dict[str, int]:
    desired = {"easy": total // 2, "medium": total - (total // 2)}
    caps = {
        difficulty: sum(count for (candidate_difficulty, _), count in available.items() if candidate_difficulty == difficulty)
        for difficulty in DIFFICULTIES
    }
    for difficulty in DIFFICULTIES:
        if desired[difficulty] > caps[difficulty]:
            shortage = desired[difficulty] - caps[difficulty]
            desired[difficulty] = caps[difficulty]
            other = "medium" if difficulty == "easy" else "easy"
            desired[other] = min(caps[other], desired[other] + shortage)
    return desired


def allocate_bins(total: int, candidates_by_bin: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[tuple[str, str], int]:
    available_blocks = {key: len(rows) // len(CELLS) for key, rows in candidates_by_bin.items()}
    targets = target_by_difficulty(total, available_blocks)
    quotas: dict[tuple[str, str], int] = {}
    for difficulty in DIFFICULTIES:
        bins = [(key, count) for key, count in available_blocks.items() if key[0] == difficulty and count > 0]
        diff_total = targets[difficulty]
        capacity = sum(count for _, count in bins)
        if not bins or diff_total <= 0 or capacity <= 0:
            continue
        raw = [(key, diff_total * count / capacity) for key, count in bins]
        assigned = 0
        for key, value in raw:
            quota = min(available_blocks[key], int(math.floor(value)))
            quotas[key] = quota
            assigned += quota
        for key, _ in sorted(raw, key=lambda item: (-(item[1] - math.floor(item[1])), item[0])):
            if assigned >= diff_total:
                break
            if quotas.get(key, 0) < available_blocks[key]:
                quotas[key] = quotas.get(key, 0) + 1
                assigned += 1
    return quotas


def select_assignments(
    candidates: list[dict[str, Any]],
    size_per_model_profile: int,
    pilot_per_model_profile: int,
) -> list[dict[str, Any]]:
    by_bin: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_bin[(record["difficulty"], record["language"])].append(record)
    for rows in by_bin.values():
        rows.sort(
            key=lambda row: (
                -row["confidence_value"],
                row["gold_num_files_changed"],
                row["gold_patch_bytes"],
                row["prompt_chars"],
                row["random_key"],
                row["instance_id"],
            )
        )

    quotas = allocate_bins(size_per_model_profile, by_bin)
    assignments: list[dict[str, Any]] = []
    block_index = 0
    for key in sorted(quotas):
        rows = by_bin[key]
        quota = quotas[key]
        for i in range(quota):
            block = rows[i * len(CELLS) : (i + 1) * len(CELLS)]
            if len(block) < len(CELLS):
                continue
            rotation = block_index % len(CELLS)
            ordered_cells = CELLS[rotation:] + CELLS[:rotation]
            for (model, style), task in zip(ordered_cells, block):
                record = {k: v for k, v in task.items() if k not in {"confidence_value", "random_key"}}
                record.update(
                    {
                        "assigned_model": model,
                        "instruction_style": style,
                        "benchmark_profile": BENCHMARK_PROFILE_BY_STYLE[style],
                        "comparison_block_id": block_index,
                        "comparison_set": f"{model}|{style}",
                        "comparison_pilot": "true" if block_index < pilot_per_model_profile else "false",
                        "eligible_for_controlled_comparison": "true",
                        "uses_updated_alignment": "true",
                        "reason_excluded_from_comparison": "",
                        "selection_rank": len(assignments),
                    }
                )
                assignments.append(record)
            block_index += 1
    return assignments


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "selection_rank",
        "comparison_block_id",
        "comparison_set",
        "comparison_pilot",
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


def summarize(rows: list[dict[str, Any]], excluded_counts: dict[str, int], manifest_files: list[str]) -> dict[str, Any]:
    by_model = Counter(row["assigned_model"] for row in rows)
    by_profile = Counter(row["benchmark_profile"] for row in rows)
    by_model_profile = Counter((row["assigned_model"], row["instruction_style"]) for row in rows)
    by_cell_diff = Counter((row["assigned_model"], row["instruction_style"], row["difficulty"]) for row in rows)
    by_cell_lang = Counter((row["assigned_model"], row["instruction_style"], row["language"]) for row in rows)
    means: dict[str, dict[str, float]] = {}
    for model, style in CELLS:
        cell = f"{model}|{style}"
        cell_rows = [row for row in rows if row["assigned_model"] == model and row["instruction_style"] == style]
        means[cell] = {
            "count": len(cell_rows),
            "gold_patch_bytes_mean": (
                sum(int(row["gold_patch_bytes"]) for row in cell_rows) / len(cell_rows) if cell_rows else 0.0
            ),
            "gold_num_files_changed_mean": (
                sum(int(row["gold_num_files_changed"]) for row in cell_rows) / len(cell_rows) if cell_rows else 0.0
            ),
            "prompt_chars_mean": (
                sum(int(row["prompt_chars"]) for row in cell_rows) / len(cell_rows) if cell_rows else 0.0
            ),
            "confidence_mean": (
                sum(float(row["confidence"] or 0.0) for row in cell_rows) / len(cell_rows) if cell_rows else 0.0
            ),
        }
    return {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "models": list(MODELS),
        "instruction_styles": list(INSTRUCTION_STYLES),
        "total_rows": len(rows),
        "unique_tasks": len({row["instance_id"] for row in rows}),
        "rows_per_model": dict(by_model),
        "rows_per_benchmark_profile": dict(by_profile),
        "rows_per_model_instruction_style": {f"{model}|{style}": count for (model, style), count in by_model_profile.items()},
        "by_cell_difficulty": {f"{model}|{style}|{difficulty}": count for (model, style, difficulty), count in by_cell_diff.items()},
        "by_cell_language": {f"{model}|{style}|{language}": count for (model, style, language), count in by_cell_lang.items()},
        "cell_feature_means": means,
        "excluded_counts": excluded_counts,
        "manifest_files_scanned_count": len(manifest_files),
        "manifest_files_scanned_sample": manifest_files[:30],
        "selection_notes": (
            "Strict-gate easy/medium SWE-rebench V2 tasks outside the original high-quality ID set. "
            "Existing dataset indexes and one-level run manifests are excluded. Each model has exactly "
            "60 original/SWE-bench-Multilingual rows and 60 DeepSWE rows when enough candidates exist. "
            "Difficulty/language quotas are identical for each model/style cell; adjacent high-confidence "
            "patch-size blocks are rotated across cells."
        ),
    }


def main() -> None:
    args = parse_args()
    high_quality_ids = load_ids(args.high_quality_ids)
    index_ids = covered_ids_from_index(args.metadata_index)
    manifest_excluded_ids, manifest_files = manifest_ids(args.runs_root)
    excluded = high_quality_ids | index_ids | manifest_excluded_ids

    candidates: list[dict[str, Any]] = []
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    for row in dataset:
        if row["instance_id"] in excluded:
            continue
        record = candidate_record(row, args.seed)
        if record is not None:
            candidates.append(record)

    assignments = select_assignments(
        candidates,
        args.sample_size_per_model_profile,
        args.pilot_size_per_model_profile,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "assignments_all.csv", assignments)
    write_csv(args.output_dir / "assignments_pilot.csv", [row for row in assignments if row["comparison_pilot"] == "true"])
    write_csv(args.output_dir / "assignments_main.csv", [row for row in assignments if row["comparison_pilot"] != "true"])
    for style in INSTRUCTION_STYLES:
        style_rows = [row for row in assignments if row["instruction_style"] == style]
        write_csv(args.output_dir / f"assignments_{style}.csv", style_rows)
        write_csv(args.output_dir / f"assignments_{style}_pilot.csv", [row for row in style_rows if row["comparison_pilot"] == "true"])
        write_csv(args.output_dir / f"assignments_{style}_main.csv", [row for row in style_rows if row["comparison_pilot"] != "true"])

    excluded_counts = {
        "high_quality_ids": len(high_quality_ids),
        "metadata_index_ids": len(index_ids),
        "manifest_ids": len(manifest_excluded_ids),
        "union_excluded_ids": len(excluded),
        "strict_candidate_count": len(candidates),
    }
    summary = summarize(assignments, excluded_counts, manifest_files)
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    expected = len(MODELS) * len(INSTRUCTION_STYLES) * args.sample_size_per_model_profile
    if len(assignments) != expected or len({row["instance_id"] for row in assignments}) != expected:
        raise SystemExit(f"expected {expected} disjoint assignments, selected {len(assignments)}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
