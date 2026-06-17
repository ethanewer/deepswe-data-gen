#!/usr/bin/env python3
"""Build UUID allowlists for v5 compaction-aware mixed SWE-rebench SFT views.

The v5 raw source appends repaired compacted trajectories and ships UUID
sidecars that identify which older original/compacted rows they supersede. This
builder uses those sidecars directly:

- include recommended v5 repaired compactions as passed rows;
- skip superseded originals and earlier compacted variants;
- exclude retained-but-not-recommended v5 rows;
- select additional strict passed rows and ranked high-quality non-passing rows
  from the remaining raw source.

The output UUID files are intended for build_swe260612_miniswe_raw.py
--allow-uuid-file.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


DEFAULT_INPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swerebench-traces-raw-source-targeted-limitations-compaction-prompt-firstturn-"
    "repaired-v5-1000plus-20260617"
)
DEFAULT_SIDECAR_DIR = (
    DEFAULT_INPUT_ROOT
    / "metadata"
    / "compaction_prompt_firstturn_repaired_v5_1000plus_20260617"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617/"
    "swerebench-v5-compaction-mixed-allowlists"
)

LANG_ALIASES = {
    "c": "c",
    "c++": "cpp",
    "cpp": "cpp",
    "go": "go",
    "java": "java",
    "javascript": "js",
    "js": "js",
    "php": "php",
    "python": "python",
    "rust": "rust",
    "typescript": "ts",
    "ts": "ts",
}


@dataclass(frozen=True)
class RowRecord:
    uuid: str
    task_id: str
    language: str
    passed: bool
    source_group: str
    metadata: dict[str, Any] = field(compare=False)
    quality_score: float = 0.0
    quality_notes: tuple[str, ...] = field(default_factory=tuple)


def load_text_set(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            value = text.split()[0]
            if not value:
                raise ValueError(f"empty value in {path}:{line_number}")
            values.add(value)
    return values


def normalize_language(value: Any) -> str:
    key = str(value or "").strip().lower()
    return LANG_ALIASES.get(key, key or "unknown")


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


def to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def to_int(value: Any, default: int = 0) -> int:
    return int(to_float(value, float(default)))


def metadata_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata.get(key)
    return row.get(key, default)


def iter_jsonl_zst(path: Path) -> Iterator[dict[str, Any]]:
    process = subprocess.Popen(
        ["zstd", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    try:
        for line_number, line in enumerate(process.stdout, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            if isinstance(row, dict):
                yield row
    finally:
        process.stdout.close()
        stderr = process.stderr.read().strip() if process.stderr is not None else ""
        return_code = process.wait()
        if process.stderr is not None:
            process.stderr.close()
        if return_code != 0:
            raise RuntimeError(f"zstd failed while reading {path}: {stderr}")


def discover_raw_shards(input_root: Path) -> list[Path]:
    data_root = input_root / "data" if (input_root / "data").is_dir() else input_root
    shards = sorted(data_root.glob("*.jsonl.zst"))
    if not shards:
        raise FileNotFoundError(f"no .jsonl.zst shards found under {data_root}")
    return shards


def patch_score(patch_bytes: int) -> tuple[float, str]:
    if patch_bytes <= 0:
        return -100.0, "empty_patch_rejected"
    if patch_bytes < 80:
        return -1.5, "tiny_patch"
    if patch_bytes < 200:
        return 0.4, "small_patch"
    if patch_bytes <= 20_000:
        return 2.0, "normal_patch"
    if patch_bytes <= 100_000:
        return 0.4, "large_patch"
    return -100.0, "huge_patch_rejected"


def trajectory_score(trajectory_bytes: int) -> tuple[float, str]:
    if trajectory_bytes <= 0:
        return 0.0, "unknown_trajectory_size"
    if trajectory_bytes <= 2_500_000:
        return 1.0, "normal_trajectory"
    if trajectory_bytes <= 5_000_000:
        return 0.1, "large_trajectory"
    return -100.0, "huge_trajectory_rejected"


def quality_score(row: dict[str, Any], source_group: str) -> tuple[float, tuple[str, ...]]:
    notes: list[str] = [source_group]
    score = 0.0
    if source_group == "recommended_v5_compaction":
        score += 50.0
    elif source_group == "strict_passed_remaining":
        score += 30.0
    elif source_group == "failed_candidate_remaining":
        score += 10.0

    reasoning = to_float(metadata_value(row, "percent_messages_with_reasoning"), 0.0)
    score += 10.0 * min(max(reasoning, 0.0), 1.0)
    if reasoning >= 0.99:
        score += 1.0
        notes.append("near_full_reasoning")
    elif reasoning < 0.9:
        score -= 20.0
        notes.append("low_reasoning_penalty")

    api_calls = to_int(metadata_value(row, "api_calls"))
    score += min(api_calls, 80) / 20.0
    if 4 <= api_calls <= 120:
        score += 1.0
        notes.append("healthy_api_calls")
    elif api_calls < 4:
        score -= 4.0
        notes.append("low_api_calls_penalty")
    elif api_calls > 200:
        score -= 1.0
        notes.append("very_high_api_calls_penalty")

    patch_delta, patch_note = patch_score(to_int(metadata_value(row, "model_patch_bytes")))
    score += patch_delta
    notes.append(patch_note)

    trajectory_delta, trajectory_note = trajectory_score(
        to_int(metadata_value(row, "trajectory_bytes", metadata_value(row, "trajectory_chars")))
    )
    score += trajectory_delta
    notes.append(trajectory_note)

    difficulty = str(metadata_value(row, "difficulty") or "").lower()
    if difficulty == "easy":
        score += 0.7
        notes.append("easy_task")
    elif difficulty == "medium":
        score += 0.3
        notes.append("medium_task")
    elif difficulty == "hard":
        score -= 0.2
        notes.append("hard_task")

    teacher = str(row.get("teacher") or metadata_value(row, "teacher") or "").lower()
    if "mimo-v2.5-pro" in teacher or "ring" in teacher:
        score += 0.5
        notes.append("strong_teacher")
    elif "mimo-v2.5" in teacher:
        score += 0.4
        notes.append("mimo_teacher")
    elif "deepseek" in teacher:
        score += 0.2
        notes.append("deepseek_teacher")

    return score, tuple(notes)


def hard_quality_ok(row: dict[str, Any], *, require_passed: bool) -> bool:
    passed = is_trueish(row.get("passed", metadata_value(row, "passed")))
    reward = is_positive(row.get("reward", metadata_value(row, "reward")))
    if require_passed and not (passed and reward):
        return False
    if not require_passed and passed:
        return False
    if str(metadata_value(row, "agent_exit_status") or "") != "Submitted":
        return False
    if to_int(metadata_value(row, "model_patch_bytes")) <= 0:
        return False
    if to_int(metadata_value(row, "model_patch_bytes")) > 100_000:
        return False
    if to_float(metadata_value(row, "percent_messages_with_reasoning"), 0.0) < 0.9:
        return False
    if to_int(metadata_value(row, "api_calls")) <= 0:
        return False
    if to_int(metadata_value(row, "assistant_message_count")) <= 0:
        return False
    trajectory_bytes = to_int(metadata_value(row, "trajectory_bytes", metadata_value(row, "trajectory_chars")))
    if trajectory_bytes > 5_000_000:
        return False
    return True


def record_from_row(row: dict[str, Any], source_group: str, passed: bool) -> RowRecord:
    score, notes = quality_score(row, source_group)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    uuid = str(row.get("uuid") or metadata.get("uuid") or "")
    if not uuid:
        raise ValueError("row is missing uuid")
    return RowRecord(
        uuid=uuid,
        task_id=str(row.get("task_id") or metadata.get("task_id") or ""),
        language=normalize_language(metadata.get("language") or row.get("language")),
        passed=passed,
        source_group=source_group,
        metadata=row,
        quality_score=score,
        quality_notes=notes,
    )


def scan_records(args: argparse.Namespace) -> tuple[dict[str, RowRecord], list[RowRecord], dict[str, Any]]:
    recommended = load_text_set(args.sidecar_dir / "recommended_prompt_firstturn_v5_repaired_uuids.txt")
    not_recommended = load_text_set(args.sidecar_dir / "not_recommended_prompt_firstturn_v5_repaired_uuids.txt")
    skip_uuids: set[str] = set()
    skip_sources = [
        "source_raw_compacted_uuids_to_skip_if_training_prompt_firstturn_v5_repaired.txt",
        "source_firstturn_repaired_uuids_to_skip_if_training_prompt_firstturn_v5_repaired.txt",
        "original_row_ids_to_skip_if_training_prompt_firstturn_v5_repaired_compactions.txt",
        "v4_uuids_superseded_by_v5.txt",
    ]
    for name in skip_sources:
        path = args.sidecar_dir / name
        if path.exists():
            skip_uuids |= load_text_set(path)

    pass_records: dict[str, RowRecord] = {}
    failed_records: list[RowRecord] = []
    stats: Counter[str] = Counter()
    by_source: Counter[str] = Counter()

    for shard in discover_raw_shards(args.input_root):
        for row in iter_jsonl_zst(shard):
            stats["rows_seen"] += 1
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            uuid = str(row.get("uuid") or metadata.get("uuid") or "")
            source = str(metadata.get("row_source") or row.get("row_source") or "unknown")
            by_source[source] += 1
            if not uuid:
                stats["skip_missing_uuid"] += 1
                continue
            if uuid in recommended:
                if not is_trueish(row.get("passed", metadata.get("passed"))) or not is_positive(
                    row.get("reward", metadata.get("reward"))
                ):
                    stats["skip_recommended_v5_bad_outcome"] += 1
                    continue
                pass_records[uuid] = record_from_row(row, "recommended_v5_compaction", True)
                stats["include_recommended_v5_compaction"] += 1
                continue
            if uuid in not_recommended:
                stats["skip_not_recommended_v5"] += 1
                continue
            if uuid in skip_uuids:
                stats["skip_superseded_uuid"] += 1
                continue
            original_row_id = str(metadata.get("compaction_original_row_id") or "")
            if original_row_id and original_row_id in skip_uuids:
                stats["skip_superseded_original_id_field"] += 1
                continue

            if hard_quality_ok(row, require_passed=True):
                pass_records[uuid] = record_from_row(row, "strict_passed_remaining", True)
                stats["include_strict_passed_remaining"] += 1
            elif hard_quality_ok(row, require_passed=False):
                failed_records.append(record_from_row(row, "failed_candidate_remaining", False))
                stats["candidate_failed_remaining"] += 1

    stats.update({f"source:{key}": value for key, value in by_source.items()})
    stats["recommended_v5_sidecar_rows"] = len(recommended)
    stats["not_recommended_v5_sidecar_rows"] = len(not_recommended)
    stats["skip_uuid_sidecar_rows"] = len(skip_uuids)
    return pass_records, failed_records, dict(sorted(stats.items()))


def sort_records(records: list[RowRecord]) -> list[RowRecord]:
    return sorted(records, key=lambda record: (-record.quality_score, record.language, record.task_id, record.uuid))


def allocate_language_targets(
    pass_counts_by_language: Counter[str],
    available_failed_by_language: Counter[str],
    target_failed_count: int,
) -> dict[str, int]:
    total_pass = sum(pass_counts_by_language.values())
    targets: dict[str, int] = {}
    fractional: list[tuple[float, str]] = []
    assigned = 0
    for language in sorted(set(pass_counts_by_language) | set(available_failed_by_language)):
        exact = target_failed_count * pass_counts_by_language.get(language, 0) / max(total_pass, 1)
        base = min(available_failed_by_language.get(language, 0), int(math.floor(exact)))
        targets[language] = base
        assigned += base
        fractional.append((exact - math.floor(exact), language))
    for _, language in sorted(fractional, reverse=True):
        if assigned >= target_failed_count:
            break
        if targets[language] < available_failed_by_language.get(language, 0):
            targets[language] += 1
            assigned += 1
    return targets


def select_failed_records(
    failed_records: list[RowRecord],
    pass_records: dict[str, RowRecord],
    target_passrate: float,
    max_failed_per_task: int,
) -> list[RowRecord]:
    pass_count = len(pass_records)
    target_failed_count = max(0, int(round(pass_count * (1.0 - target_passrate) / target_passrate)))
    ranked = sort_records(failed_records)
    pass_counts_by_language = Counter(record.language for record in pass_records.values())
    available_failed_by_language = Counter(record.language for record in ranked)
    language_targets = allocate_language_targets(
        pass_counts_by_language,
        available_failed_by_language,
        target_failed_count,
    )
    by_language: dict[str, list[RowRecord]] = defaultdict(list)
    for record in ranked:
        by_language[record.language].append(record)

    selected: list[RowRecord] = []
    selected_uuids: set[str] = set()
    selected_task_counts: Counter[str] = Counter()

    def try_add(record: RowRecord, task_cap: int) -> bool:
        if record.uuid in selected_uuids:
            return False
        if selected_task_counts[record.task_id] >= task_cap:
            return False
        selected.append(record)
        selected_uuids.add(record.uuid)
        selected_task_counts[record.task_id] += 1
        return True

    for task_cap in range(1, max_failed_per_task + 1):
        for language in sorted(language_targets):
            current = sum(1 for record in selected if record.language == language)
            needed = language_targets[language] - current
            if needed <= 0:
                continue
            for record in by_language.get(language, []):
                if needed <= 0:
                    break
                if try_add(record, task_cap):
                    needed -= 1
        if len(selected) >= target_failed_count:
            break

    for task_cap in range(1, max_failed_per_task + 1):
        for record in ranked:
            if len(selected) >= target_failed_count:
                break
            try_add(record, task_cap)
        if len(selected) >= target_failed_count:
            break
    return selected[:target_failed_count]


def record_report(record: RowRecord) -> dict[str, Any]:
    row = record.metadata
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "uuid": record.uuid,
        "task_id": record.task_id,
        "language": record.language,
        "passed": record.passed,
        "source_group": record.source_group,
        "row_source": metadata.get("row_source") or row.get("row_source"),
        "quality_score": round(record.quality_score, 6),
        "quality_notes": list(record.quality_notes),
        "api_calls": metadata_value(row, "api_calls"),
        "assistant_reasoning_fraction": metadata_value(row, "percent_messages_with_reasoning"),
        "model_patch_bytes": metadata_value(row, "model_patch_bytes"),
        "trajectory_bytes": metadata_value(row, "trajectory_bytes", metadata_value(row, "trajectory_chars")),
        "difficulty": metadata_value(row, "difficulty"),
        "teacher": row.get("teacher") or metadata.get("teacher"),
        "quality_flags": metadata.get("quality_flags"),
        "compaction_original_row_id": metadata.get("compaction_original_row_id"),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def write_view(
    output_root: Path,
    label: str,
    target_passrate: float,
    pass_records: dict[str, RowRecord],
    failed_records: list[RowRecord],
    max_failed_per_task: int,
) -> dict[str, Any]:
    selected_failed = select_failed_records(failed_records, pass_records, target_passrate, max_failed_per_task)
    selected = list(pass_records.values()) + selected_failed
    selected_uuids = sorted(record.uuid for record in selected)
    uuid_path = output_root / f"{label}_uuids.txt"
    uuid_path.write_text("\n".join(selected_uuids) + "\n", encoding="utf-8")

    selected_failed_ranked = sort_records(selected_failed)
    write_jsonl(output_root / f"{label}_selected_failed_ranked.jsonl", [record_report(r) for r in selected_failed_ranked])

    pass_count = len(pass_records)
    failed_count = len(selected_failed)
    passrate = pass_count / max(pass_count + failed_count, 1)
    summary = {
        "label": label,
        "target_passrate": target_passrate,
        "actual_passrate": passrate,
        "uuid_file": str(uuid_path),
        "rows_total": pass_count + failed_count,
        "rows_passed": pass_count,
        "rows_failed_selected": failed_count,
        "failed_target_count": int(round(pass_count * (1.0 - target_passrate) / target_passrate)),
        "failed_max_per_task": max_failed_per_task,
        "passed_by_language": dict(sorted(Counter(record.language for record in pass_records.values()).items())),
        "passed_by_source_group": dict(sorted(Counter(record.source_group for record in pass_records.values()).items())),
        "failed_by_language": dict(sorted(Counter(record.language for record in selected_failed).items())),
        "failed_unique_tasks": len({record.task_id for record in selected_failed}),
        "failed_min_quality_score": min((record.quality_score for record in selected_failed), default=None),
        "failed_mean_quality_score": (
            sum(record.quality_score for record in selected_failed) / failed_count if failed_count else None
        ),
        "top_failed_examples": [record_report(record) for record in selected_failed_ranked[:20]],
    }
    (output_root / f"{label}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--sidecar-dir", type=Path, default=DEFAULT_SIDECAR_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-passrate", type=float, action="append", default=[])
    parser.add_argument("--max-failed-per-task", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_passrates = args.target_passrate or [0.5, 0.25]
    if any(value <= 0.0 or value >= 1.0 for value in target_passrates):
        raise ValueError("--target-passrate values must be between 0 and 1")
    if args.max_failed_per_task < 1:
        raise ValueError("--max-failed-per-task must be >= 1")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    pass_records, failed_records, scan_stats = scan_records(args)
    write_jsonl(args.output_root / "all_pass_records.jsonl", [record_report(r) for r in sort_records(list(pass_records.values()))])
    write_jsonl(args.output_root / "all_failed_candidates_ranked.jsonl", [record_report(r) for r in sort_records(failed_records)])

    summaries = {}
    for target_passrate in target_passrates:
        label = f"mixed{int(round(target_passrate * 100)):02d}"
        summaries[label] = write_view(
            args.output_root,
            label,
            target_passrate,
            pass_records,
            failed_records,
            args.max_failed_per_task,
        )

    manifest = {
        "input_root": str(args.input_root),
        "sidecar_dir": str(args.sidecar_dir),
        "output_root": str(args.output_root),
        "max_failed_per_task": args.max_failed_per_task,
        "pass_records": len(pass_records),
        "failed_candidate_records": len(failed_records),
        "scan_stats": scan_stats,
        "selection_policy": {
            "start_checkpoint": "Qwen/Qwen3-4B-Thinking-2507",
            "recommended_v5": "included as passed rows",
            "superseded_sources": "skipped by UUID sidecars before remaining-row selection",
            "non_recommended_v5": "excluded",
            "failed_rows": (
                "submitted, non-empty-patch, >=90% reasoning, positive API calls, "
                "<=100k patch bytes, <=5M trajectory bytes, ranked by quality score"
            ),
            "intended_training": "no prefix expansion; whole trajectories; turn masking; pass-last optional",
        },
        "views": summaries,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
