#!/usr/bin/env python3
"""Build the v75 strict-passed allowlist from the verification-enhanced dataset.

This script is intentionally explicit about the dataset identity:

    eewer/swerebench-traces-raw-source-verification-enhanced-20260617

It selects high-quality passing traces, treats verification-enhanced rows as
replacements for their source traces, applies the v5 compaction supersession
sidecars, and caps each task at a small number of rollouts.  The output UUID
allowlist is consumed by build_swe260612_miniswe_raw.py, which performs the
same strict mini-swe-agent transform used for the v54 recipe.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


DATASET_ID = "eewer/swerebench-traces-raw-source-verification-enhanced-20260617"
DEFAULT_LOCAL_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swerebench-traces-raw-source-verification-enhanced-20260617"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617/"
    "swerebench-verification-enhanced-v75-strictpassed-cap4-allowlist"
)
V5_SIDECAR_REL = Path("metadata/compaction_prompt_firstturn_repaired_v5_1000plus_20260617")

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
class Candidate:
    uuid: str
    task_id: str
    language: str
    source_group: str
    score: float
    notes: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


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


def normalize_language(value: Any) -> str:
    key = str(value or "").strip().lower()
    return LANG_ALIASES.get(key, key or "unknown")


def load_text_set(path: Path) -> set[str]:
    values: set[str] = set()
    if not path.exists():
        return values
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


def discover_source_root(local_root: Path, dataset_id: str) -> Path:
    if local_root.exists():
        return local_root.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"{local_root} does not exist and huggingface_hub is not installed; "
            f"cannot fetch dataset {dataset_id}"
        ) from exc
    return Path(snapshot_download(repo_id=dataset_id, repo_type="dataset")).resolve()


def discover_shards(source_root: Path) -> list[Path]:
    data_root = source_root / "data" if (source_root / "data").is_dir() else source_root
    shards = sorted(data_root.glob("*.jsonl.zst"))
    if not shards:
        raise FileNotFoundError(f"no .jsonl.zst shards under {data_root}")
    return shards


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


def hard_pass_quality_ok(row: dict[str, Any]) -> bool:
    if not (is_trueish(metadata_value(row, "passed")) and is_positive(metadata_value(row, "reward"))):
        return False
    if str(metadata_value(row, "agent_exit_status") or "") != "Submitted":
        return False
    patch_bytes = to_int(metadata_value(row, "model_patch_bytes"))
    if patch_bytes <= 0 or patch_bytes > 100_000:
        return False
    if to_float(metadata_value(row, "percent_messages_with_reasoning")) < 0.9:
        return False
    if to_int(metadata_value(row, "api_calls")) <= 0:
        return False
    if to_int(metadata_value(row, "assistant_message_count")) <= 0:
        return False
    trajectory_bytes = to_int(metadata_value(row, "trajectory_bytes", metadata_value(row, "trajectory_chars")))
    return trajectory_bytes <= 5_000_000


def source_uuid_for_verification_row(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in (
        "verification_source_uuid",
        "synthetic_patchtxt_verification_source_uuid",
        "synthetic_empty_submit_verification_stop_source_uuid",
        "verification_original_row_uuid",
        "original_uuid",
    ):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def load_verification_signals(source_root: Path) -> dict[str, dict[str, Any]]:
    path = source_root / "metadata" / "verification_enhanced_row_signals.jsonl"
    signals: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return signals
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            uuid = str(row.get("uuid") or "")
            if not uuid:
                raise ValueError(f"missing uuid in {path}:{line_number}")
            signals[uuid] = row
    return signals


def find_verification_replacements(shards: list[Path]) -> tuple[set[str], set[str], Counter[str]]:
    selected: set[str] = set()
    replaced_sources: set[str] = set()
    stats: Counter[str] = Counter()
    for shard in shards:
        for row in iter_jsonl_zst(shard):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            family = str(metadata.get("verification_modification_family") or "")
            if not family:
                continue
            stats[f"verification_rows_seen:{family}"] += 1
            uuid = str(row.get("uuid") or metadata.get("uuid") or "")
            if (
                uuid
                and metadata.get("verification_recommended_for_standard_sft")
                and not metadata.get("verification_should_not_be_counted_as_passed")
                and hard_pass_quality_ok(row)
            ):
                selected.add(uuid)
                source_uuid = source_uuid_for_verification_row(row)
                if source_uuid:
                    replaced_sources.add(source_uuid)
                stats[f"verification_rows_selected:{family}"] += 1
            else:
                stats[f"verification_rows_skipped:{family}"] += 1
    return selected, replaced_sources, stats


def load_v5_sidecars(source_root: Path) -> tuple[set[str], set[str], set[str]]:
    sidecar_dir = source_root / V5_SIDECAR_REL
    recommended = load_text_set(sidecar_dir / "recommended_prompt_firstturn_v5_repaired_uuids.txt")
    not_recommended = load_text_set(sidecar_dir / "not_recommended_prompt_firstturn_v5_repaired_uuids.txt")
    superseded: set[str] = set()
    for name in (
        "source_raw_compacted_uuids_to_skip_if_training_prompt_firstturn_v5_repaired.txt",
        "source_firstturn_repaired_uuids_to_skip_if_training_prompt_firstturn_v5_repaired.txt",
        "original_row_ids_to_skip_if_training_prompt_firstturn_v5_repaired_compactions.txt",
        "v4_uuids_superseded_by_v5.txt",
    ):
        superseded |= load_text_set(sidecar_dir / name)
    return recommended, not_recommended, superseded


def is_compaction_source(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    source = str(metadata.get("row_source") or row.get("row_source") or "")
    return source.startswith("compaction_") or bool(metadata.get("compaction_original_row_id"))


def score_candidate(row: dict[str, Any], source_group: str, signals: dict[str, Any]) -> tuple[float, tuple[str, ...]]:
    notes: list[str] = [source_group]
    score = {
        "recommended_v5_compaction": 60.0,
        "verification_standard_replacement": 55.0,
        "strict_passed_remaining": 35.0,
    }[source_group]

    reasoning = to_float(metadata_value(row, "percent_messages_with_reasoning"))
    score += 10.0 * min(max(reasoning, 0.0), 1.0)
    if reasoning >= 0.99:
        score += 1.0
        notes.append("near_full_reasoning")

    api_calls = to_int(metadata_value(row, "api_calls"))
    score += min(api_calls, 80) / 20.0
    if 4 <= api_calls <= 120:
        score += 1.0
        notes.append("healthy_api_calls")

    patch_bytes = to_int(metadata_value(row, "model_patch_bytes"))
    if 200 <= patch_bytes <= 20_000:
        score += 2.0
        notes.append("normal_patch")
    elif patch_bytes < 200:
        score -= 1.0
        notes.append("small_patch")

    if signals.get("natural_patch_verification"):
        score += 10.0
        notes.append("natural_patch_verification")
    if signals.get("natural_submit_command_cats_patch"):
        score += 4.0
        notes.append("submit_cats_patch")
    if signals.get("natural_visible_diff_evidence"):
        score += 2.0
        notes.append("visible_diff_evidence")
    if signals.get("missing_nonempty_patch_verification"):
        score -= 3.0
        notes.append("missing_nonempty_patch_verification")

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    family = str(metadata.get("verification_modification_family") or "")
    if family == "synthetic_patchtxt_verification_prototype":
        score += 8.0
        notes.append("synthetic_patchtxt_verification")

    teacher = str(metadata_value(row, "teacher") or "").lower()
    if "mimo-v2.5" in teacher:
        score += 0.4
        notes.append("mimo_teacher")
    elif "deepseek" in teacher:
        score += 0.2
        notes.append("deepseek_teacher")

    return score, tuple(notes)


def record_from_row(row: dict[str, Any], source_group: str, signals: dict[str, Any]) -> Candidate:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    uuid = str(row.get("uuid") or metadata.get("uuid") or "")
    task_id = str(metadata_value(row, "task_id") or "")
    if not uuid or not task_id:
        raise ValueError("candidate row missing uuid or task_id")
    score, notes = score_candidate(row, source_group, signals)
    language = normalize_language(metadata_value(row, "language", signals.get("language")))
    return Candidate(
        uuid=uuid,
        task_id=task_id,
        language=language,
        source_group=source_group,
        score=score,
        notes=notes,
        metadata=row,
    )


def scan_candidates(
    shards: list[Path],
    *,
    verification_selected: set[str],
    verification_replaced_sources: set[str],
    v5_recommended: set[str],
    v5_not_recommended: set[str],
    v5_superseded: set[str],
    verification_signals: dict[str, dict[str, Any]],
) -> tuple[list[Candidate], Counter[str]]:
    candidates: list[Candidate] = []
    stats: Counter[str] = Counter()
    for shard in shards:
        for row in iter_jsonl_zst(shard):
            stats["rows_seen"] += 1
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            uuid = str(row.get("uuid") or metadata.get("uuid") or "")
            if not uuid:
                stats["skip_missing_uuid"] += 1
                continue

            source_group = ""
            if uuid in verification_selected:
                source_group = "verification_standard_replacement"
            elif uuid in verification_replaced_sources:
                stats["skip_verification_replaced_source"] += 1
                continue
            elif uuid in v5_recommended:
                if hard_pass_quality_ok(row):
                    source_group = "recommended_v5_compaction"
                else:
                    stats["skip_v5_recommended_hard_quality"] += 1
                    continue
            elif (
                uuid in v5_not_recommended
                or uuid in v5_superseded
                or str(metadata.get("compaction_original_row_id") or "") in v5_superseded
            ):
                stats["skip_v5_superseded_or_not_recommended"] += 1
                continue
            elif is_compaction_source(row):
                stats["skip_non_recommended_compaction"] += 1
                continue
            elif hard_pass_quality_ok(row):
                source_group = "strict_passed_remaining"
            else:
                continue

            try:
                candidate = record_from_row(row, source_group, verification_signals.get(uuid, {}))
            except ValueError:
                stats["skip_candidate_missing_required_fields"] += 1
                continue
            candidates.append(candidate)
            stats[f"candidate:{source_group}"] += 1
    return candidates, stats


def candidate_sort_key(candidate: Candidate) -> tuple[float, str, str]:
    return (-candidate.score, candidate.task_id, candidate.uuid)


def select_with_task_cap(candidates: list[Candidate], max_per_task: int, target_count: int) -> list[Candidate]:
    by_task: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_task[candidate.task_id].append(candidate)
    for rows in by_task.values():
        rows.sort(key=candidate_sort_key)

    task_order = sorted(by_task, key=lambda task: candidate_sort_key(by_task[task][0]))
    selected: list[Candidate] = []
    for rollout_index in range(max_per_task):
        for task_id in task_order:
            rows = by_task[task_id]
            if rollout_index >= len(rows):
                continue
            selected.append(rows[rollout_index])
            if target_count > 0 and len(selected) >= target_count:
                return selected
    return selected


def candidate_report(candidate: Candidate, order_index: int) -> dict[str, Any]:
    row = candidate.metadata
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "order_index": order_index,
        "uuid": candidate.uuid,
        "task_id": candidate.task_id,
        "language": candidate.language,
        "source_group": candidate.source_group,
        "quality_score": round(candidate.score, 6),
        "quality_notes": list(candidate.notes),
        "row_source": metadata.get("row_source") or row.get("row_source"),
        "teacher": metadata_value(row, "teacher"),
        "difficulty": metadata_value(row, "difficulty"),
        "api_calls": metadata_value(row, "api_calls"),
        "model_patch_bytes": metadata_value(row, "model_patch_bytes"),
        "percent_messages_with_reasoning": metadata_value(row, "percent_messages_with_reasoning"),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-pass-per-task", type=int, default=4)
    parser.add_argument("--target-pass-traces", type=int, default=12_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_pass_per_task < 1:
        raise ValueError("--max-pass-per-task must be >= 1")
    if args.target_pass_traces < 0:
        raise ValueError("--target-pass-traces must be >= 0")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    source_root = discover_source_root(args.local_root, args.dataset_id)
    shards = discover_shards(source_root)
    verification_signals = load_verification_signals(source_root)
    verification_selected, verification_replaced_sources, verification_stats = find_verification_replacements(shards)
    v5_recommended, v5_not_recommended, v5_superseded = load_v5_sidecars(source_root)
    candidates, scan_stats = scan_candidates(
        shards,
        verification_selected=verification_selected,
        verification_replaced_sources=verification_replaced_sources,
        v5_recommended=v5_recommended,
        v5_not_recommended=v5_not_recommended,
        v5_superseded=v5_superseded,
        verification_signals=verification_signals,
    )
    selected = select_with_task_cap(candidates, args.max_pass_per_task, args.target_pass_traces)

    uuid_path = args.output_root / "selected_pass_uuids.txt"
    uuid_path.write_text("\n".join(candidate.uuid for candidate in selected) + "\n", encoding="utf-8")
    write_jsonl(
        args.output_root / "selected_pass_records_training_order.jsonl",
        [candidate_report(candidate, idx) for idx, candidate in enumerate(selected)],
    )
    write_jsonl(
        args.output_root / "all_pass_candidates_ranked.jsonl",
        [candidate_report(candidate, idx) for idx, candidate in enumerate(sorted(candidates, key=candidate_sort_key))],
    )

    selected_task_counts = Counter(candidate.task_id for candidate in selected)
    manifest = {
        "dataset_id": args.dataset_id,
        "source_root": str(source_root),
        "output_root": str(args.output_root),
        "selected_uuid_file": str(uuid_path),
        "selected_records_training_order": str(args.output_root / "selected_pass_records_training_order.jsonl"),
        "selection_policy": {
            "pass_only": True,
            "max_pass_per_task": args.max_pass_per_task,
            "target_pass_traces": args.target_pass_traces,
            "duplicate_order": "round_robin_by_task_quality_rank",
            "verification_replacements": (
                "metadata.verification_recommended_for_standard_sft rows replace their source UUIDs"
            ),
            "v5_compactions": "recommended v5 repaired compactions replace superseded originals",
            "quality_preference": "passing traces with patch verification before submit are scored higher",
        },
        "counts": {
            "raw_shards": len(shards),
            "candidates_before_cap": len(candidates),
            "selected_pass_traces": len(selected),
            "selected_unique_tasks": len(selected_task_counts),
            "selected_tasks_with_multiple_rollouts": sum(1 for value in selected_task_counts.values() if value > 1),
            "selected_max_rollouts_per_task": max(selected_task_counts.values(), default=0),
        },
        "selected_by_source_group": dict(sorted(Counter(candidate.source_group for candidate in selected).items())),
        "selected_by_language": dict(sorted(Counter(candidate.language for candidate in selected).items())),
        "scan_stats": dict(sorted((scan_stats + verification_stats).items())),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
