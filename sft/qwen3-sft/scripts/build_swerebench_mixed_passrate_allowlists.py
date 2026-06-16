#!/usr/bin/env python3
"""Build ranked raw-row allowlists for mixed pass/fail SWE-rebench SFT views.

The output is a set of global 0-based raw-source line numbers suitable for
build_swe260612_miniswe_raw.py --allow-line-number-file.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PASS_LINES = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260616/"
    "swerebench-raw2030-targeted-limitations-allowlists/"
    "strict_passed_base_plus_new_line_numbers.txt"
)
DEFAULT_BASE_QUALITY = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260616/"
    "trace-quality-metadata-audit-20260616/raw_index_row_quality_signals.jsonl"
)
DEFAULT_NEW_QUALITY = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swerebench-traces-raw-source-targeted-limitations-compaction-full-20260616-2030/"
    "metadata/strict_quality_index.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260616/"
    "swerebench-raw2030-targeted-limitations-mixed-allowlists"
)

LANG_ORDER = ("c", "cpp", "go", "java", "js", "php", "python", "rust", "ts")
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
    line_number: int
    task_id: str
    language: str
    passed: bool
    source_group: str
    metadata: dict[str, Any] = field(compare=False)
    quality_score: float = 0.0
    quality_notes: tuple[str, ...] = field(default_factory=tuple)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc


def normalize_language(value: Any) -> str:
    key = str(value or "").strip().lower()
    if key not in LANG_ALIASES:
        return key or "unknown"
    return LANG_ALIASES[key]


def is_trueish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "passed", "pass"}
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


def load_line_numbers(path: Path) -> set[int]:
    out: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for file_line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("{"):
                row = json.loads(text)
                value = row.get("line_number")
            else:
                value = text.split()[0]
            try:
                line_number = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid line number in {path}:{file_line_number}") from exc
            if line_number < 0:
                raise ValueError(f"negative line number in {path}:{file_line_number}")
            out.add(line_number)
    return out


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
    return -3.0, "huge_patch_penalty"


def trajectory_score(trajectory_bytes: int) -> tuple[float, str]:
    if trajectory_bytes <= 0:
        return 0.0, "unknown_trajectory_size"
    if trajectory_bytes <= 2_500_000:
        return 1.0, "normal_trajectory"
    if trajectory_bytes <= 5_000_000:
        return 0.1, "large_trajectory"
    return -3.0, "huge_trajectory_penalty"


def quality_score(row: dict[str, Any], source_group: str) -> tuple[float, tuple[str, ...]]:
    notes: list[str] = []
    score = 0.0

    if source_group == "new_outcome_only_failed":
        score += 30.0
        notes.append("new_raw2030_outcome_only_reject")
    elif source_group == "base_structural_failed":
        score += 10.0
        notes.append("base_structural_failed")

    reasoning = to_float(
        row.get("assistant_reasoning_fraction", row.get("percent_messages_with_reasoning")),
        0.0,
    )
    score += 10.0 * min(max(reasoning, 0.0), 1.0)
    if reasoning >= 0.99:
        score += 1.0
        notes.append("near_full_reasoning")
    elif reasoning < 0.9:
        score -= 10.0
        notes.append("low_reasoning_penalty")

    api_calls = to_int(row.get("api_calls"))
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

    patch_delta, patch_note = patch_score(to_int(row.get("model_patch_bytes")))
    score += patch_delta
    notes.append(patch_note)

    trajectory_delta, trajectory_note = trajectory_score(to_int(row.get("trajectory_bytes", row.get("trajectory_chars"))))
    score += trajectory_delta
    notes.append(trajectory_note)

    if row.get("huge_patch_flag"):
        score -= 100.0
        notes.append("huge_patch_rejected")
    if row.get("huge_trajectory_flag"):
        score -= 100.0
        notes.append("huge_trajectory_rejected")

    difficulty = str(row.get("difficulty") or "").lower()
    if difficulty == "easy":
        score += 0.7
        notes.append("easy_task")
    elif difficulty == "medium":
        score += 0.3
        notes.append("medium_task")
    elif difficulty == "hard":
        score -= 0.2
        notes.append("hard_task")

    teacher = str(row.get("teacher") or "").lower()
    if "mimo-v2.5-pro" in teacher or "ring" in teacher:
        score += 0.5
        notes.append("strong_teacher")
    elif "mimo-v2.5" in teacher:
        score += 0.4
        notes.append("mimo_teacher")
    elif "deepseek" in teacher:
        score += 0.2
        notes.append("deepseek_teacher")

    task_trace_count = to_int(row.get("task_trace_count_raw"))
    if task_trace_count > 4:
        score -= min(1.0, 0.1 * (task_trace_count - 4))
        notes.append("many_traces_for_task_penalty")
    if row.get("task_overrepresented_strict"):
        score -= 0.5
        notes.append("overrepresented_strict_task_penalty")

    return score, tuple(notes)


def old_failed_candidate(row: dict[str, Any], include_huge: bool) -> bool:
    if not row.get("candidate_failed_curriculum"):
        return False
    if row.get("failure_class") != "submitted_task_failure":
        return False
    if is_trueish(row.get("passed")):
        return False
    if not include_huge and (row.get("huge_patch_flag") or row.get("huge_trajectory_flag")):
        return False
    return True


def new_outcome_only_failed_candidate(row: dict[str, Any], include_huge: bool) -> bool:
    if row.get("strict_quality_passed"):
        return False
    reasons = set(row.get("strict_quality_reject_reasons") or [])
    if not reasons or not reasons.issubset({"not_passed", "reward_not_positive"}):
        return False
    if is_trueish(row.get("passed")):
        return False
    if str(row.get("agent_exit_status") or "") != "Submitted":
        return False
    if to_int(row.get("model_patch_bytes")) <= 0:
        return False
    if to_float(row.get("percent_messages_with_reasoning")) < 0.9:
        return False
    if to_int(row.get("api_calls")) <= 0:
        return False
    if to_int(row.get("assistant_message_count")) <= 0:
        return False
    if not include_huge and (row.get("huge_patch_flag") or row.get("huge_trajectory_flag")):
        return False
    return True


def record_from_row(row: dict[str, Any], source_group: str, passed: bool) -> RowRecord:
    score, notes = quality_score(row, source_group)
    return RowRecord(
        line_number=to_int(row.get("line_number")),
        task_id=str(row.get("task_id") or ""),
        language=normalize_language(row.get("language")),
        passed=passed,
        source_group=source_group,
        metadata=row,
        quality_score=score,
        quality_notes=notes,
    )


def load_records(
    pass_lines_path: Path,
    base_quality_path: Path,
    new_quality_path: Path,
    include_huge: bool,
) -> tuple[dict[int, RowRecord], list[RowRecord]]:
    pass_lines = load_line_numbers(pass_lines_path)
    records_by_line: dict[int, RowRecord] = {}
    failed: list[RowRecord] = []

    for row in iter_jsonl(base_quality_path):
        line_number = to_int(row.get("line_number"), -1)
        if line_number in pass_lines and row.get("strict_basic_quality"):
            records_by_line[line_number] = record_from_row(row, "strict_passed", True)
        if old_failed_candidate(row, include_huge):
            failed.append(record_from_row(row, "base_structural_failed", False))

    for row in iter_jsonl(new_quality_path):
        line_number = to_int(row.get("line_number"), -1)
        if line_number in pass_lines and row.get("strict_quality_passed"):
            records_by_line[line_number] = record_from_row(row, "strict_passed", True)
        if new_outcome_only_failed_candidate(row, include_huge):
            failed.append(record_from_row(row, "new_outcome_only_failed", False))

    missing_pass_lines = sorted(pass_lines - set(records_by_line))
    if missing_pass_lines:
        raise RuntimeError(
            f"{len(missing_pass_lines)} pass-line records missing from metadata; "
            f"first missing: {missing_pass_lines[:10]}"
        )

    failed = [record for record in failed if record.line_number not in pass_lines]
    return records_by_line, failed


def sort_records(records: list[RowRecord]) -> list[RowRecord]:
    return sorted(
        records,
        key=lambda record: (
            -record.quality_score,
            record.language,
            record.task_id,
            record.line_number,
        ),
    )


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
    pass_records: dict[int, RowRecord],
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
    selected_lines: set[int] = set()
    selected_task_counts: Counter[str] = Counter()

    def try_add(record: RowRecord, task_cap: int) -> bool:
        if record.line_number in selected_lines:
            return False
        if selected_task_counts[record.task_id] >= task_cap:
            return False
        selected.append(record)
        selected_lines.add(record.line_number)
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

    if len(selected) >= target_failed_count:
        return selected[:target_failed_count]

    # If some language targets cannot be met, fill the remaining slots by
    # global quality rank after all language quotas have had the same cap
    # relaxation opportunity.
    for task_cap in range(1, max_failed_per_task + 1):
        for record in ranked:
            if len(selected) >= target_failed_count:
                break
            try_add(record, task_cap)
        if len(selected) >= target_failed_count:
            break

    return selected[:target_failed_count]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def write_selected_view(
    output_root: Path,
    label: str,
    target_passrate: float,
    pass_records: dict[int, RowRecord],
    failed_records: list[RowRecord],
    max_failed_per_task: int,
) -> dict[str, Any]:
    selected_failed = select_failed_records(
        failed_records,
        pass_records,
        target_passrate,
        max_failed_per_task,
    )
    selected_lines = sorted(set(pass_records) | {record.line_number for record in selected_failed})

    line_path = output_root / f"{label}_line_numbers.txt"
    line_path.write_text("\n".join(str(line_number) for line_number in selected_lines) + "\n", encoding="utf-8")

    selected_failed_for_report = sort_records(selected_failed)
    selected_failed_rows = [
        {
            "line_number": record.line_number,
            "task_id": record.task_id,
            "language": record.language,
            "source_group": record.source_group,
            "quality_score": round(record.quality_score, 6),
            "quality_notes": list(record.quality_notes),
            "api_calls": record.metadata.get("api_calls"),
            "assistant_reasoning_fraction": record.metadata.get(
                "assistant_reasoning_fraction",
                record.metadata.get("percent_messages_with_reasoning"),
            ),
            "model_patch_bytes": record.metadata.get("model_patch_bytes"),
            "trajectory_bytes": record.metadata.get("trajectory_bytes", record.metadata.get("trajectory_chars")),
            "difficulty": record.metadata.get("difficulty"),
            "teacher": record.metadata.get("teacher"),
            "failure_class": record.metadata.get("failure_class"),
            "strict_quality_reject_reasons": record.metadata.get("strict_quality_reject_reasons"),
        }
        for record in selected_failed_for_report
    ]
    write_jsonl(output_root / f"{label}_selected_failed_ranked.jsonl", selected_failed_rows)

    pass_count = len(pass_records)
    failed_count = len(selected_failed)
    passrate = pass_count / max(pass_count + failed_count, 1)
    summary = {
        "label": label,
        "target_passrate": target_passrate,
        "actual_passrate": passrate,
        "line_number_file": str(line_path),
        "rows_total": pass_count + failed_count,
        "rows_passed": pass_count,
        "rows_failed_selected": failed_count,
        "failed_target_count": int(round(pass_count * (1.0 - target_passrate) / target_passrate)),
        "failed_max_per_task": max_failed_per_task,
        "passed_by_language": dict(sorted(Counter(record.language for record in pass_records.values()).items())),
        "failed_by_language": dict(sorted(Counter(record.language for record in selected_failed).items())),
        "failed_by_source_group": dict(sorted(Counter(record.source_group for record in selected_failed).items())),
        "failed_unique_tasks": len({record.task_id for record in selected_failed}),
        "failed_min_quality_score": min((record.quality_score for record in selected_failed), default=None),
        "failed_mean_quality_score": (
            sum(record.quality_score for record in selected_failed) / failed_count if failed_count else None
        ),
        "top_failed_examples": selected_failed_rows[:20],
    }
    (output_root / f"{label}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-line-numbers", type=Path, default=DEFAULT_PASS_LINES)
    parser.add_argument("--base-quality-jsonl", type=Path, default=DEFAULT_BASE_QUALITY)
    parser.add_argument("--new-quality-jsonl", type=Path, default=DEFAULT_NEW_QUALITY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--target-passrate",
        type=float,
        action="append",
        default=[],
        help="target passed-row fraction, for example 0.5 or 0.25; may be repeated",
    )
    parser.add_argument(
        "--max-failed-per-task",
        type=int,
        default=3,
        help="maximum selected failed traces per task after cap relaxation",
    )
    parser.add_argument(
        "--include-huge",
        action="store_true",
        help="include huge patch/trajectory failed candidates; disabled by default",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_passrates = args.target_passrate or [0.5, 0.25]
    if any(value <= 0.0 or value >= 1.0 for value in target_passrates):
        raise ValueError("--target-passrate values must be between 0 and 1")
    if args.max_failed_per_task < 1:
        raise ValueError("--max-failed-per-task must be >= 1")

    args.output_root.mkdir(parents=True, exist_ok=True)
    pass_records, failed_records = load_records(
        args.pass_line_numbers,
        args.base_quality_jsonl,
        args.new_quality_jsonl,
        args.include_huge,
    )
    failed_ranked = sort_records(failed_records)
    write_jsonl(
        args.output_root / "all_failed_candidates_ranked.jsonl",
        [
            {
                "line_number": record.line_number,
                "task_id": record.task_id,
                "language": record.language,
                "source_group": record.source_group,
                "quality_score": round(record.quality_score, 6),
                "quality_notes": list(record.quality_notes),
                "api_calls": record.metadata.get("api_calls"),
                "assistant_reasoning_fraction": record.metadata.get(
                    "assistant_reasoning_fraction",
                    record.metadata.get("percent_messages_with_reasoning"),
                ),
                "model_patch_bytes": record.metadata.get("model_patch_bytes"),
                "trajectory_bytes": record.metadata.get("trajectory_bytes", record.metadata.get("trajectory_chars")),
                "difficulty": record.metadata.get("difficulty"),
                "teacher": record.metadata.get("teacher"),
                "failure_class": record.metadata.get("failure_class"),
                "strict_quality_reject_reasons": record.metadata.get("strict_quality_reject_reasons"),
            }
            for record in failed_ranked
        ],
    )

    summaries = {}
    for target_passrate in target_passrates:
        label = f"mixed{int(round(target_passrate * 100)):02d}"
        summaries[label] = write_selected_view(
            args.output_root,
            label,
            target_passrate,
            pass_records,
            failed_records,
            args.max_failed_per_task,
        )

    manifest = {
        "pass_line_numbers": str(args.pass_line_numbers),
        "base_quality_jsonl": str(args.base_quality_jsonl),
        "new_quality_jsonl": str(args.new_quality_jsonl),
        "output_root": str(args.output_root),
        "include_huge": args.include_huge,
        "max_failed_per_task": args.max_failed_per_task,
        "strict_passed_rows": len(pass_records),
        "failed_candidate_rows": len(failed_records),
        "failed_candidate_by_language": dict(sorted(Counter(record.language for record in failed_records).items())),
        "failed_candidate_by_source_group": dict(sorted(Counter(record.source_group for record in failed_records).items())),
        "selection_policy": {
            "passed_rows": "all strict passed rows from raw2030 allowlist",
            "failed_rows": (
                "submitted, non-empty-patch, high-reasoning structural failures; "
                "new raw2030 outcome-only strict rejects receive first-tier score; "
                "huge patch/trajectory candidates are excluded unless --include-huge is set"
            ),
            "ranking": (
                "quality score uses strict/outcome-only source tier, reasoning coverage, API calls, "
                "patch size, trajectory size, difficulty, teacher, and task duplication penalties"
            ),
            "language_balance": (
                "failed-row targets are allocated from strict-passed language proportions, "
                "then remaining slots are filled by global quality rank"
            ),
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
