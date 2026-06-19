#!/usr/bin/env python3
"""Build a v75-expanded allowlist with high-quality non-passing traces.

The strict-passed v75 recipe only trains on clean successful trajectories. This
variant keeps that strict-passed majority, then adds a capped slice of submitted,
non-empty, patch-verified non-passing traces so the SFT mix includes realistic
struggle/recovery behavior.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_swerebench_verification_enhanced_strict_pass_allowlist import (
    DATASET_ID,
    DEFAULT_LOCAL_ROOT,
    Candidate,
    candidate_report,
    candidate_sort_key,
    compact_pass_quality_ok,
    discover_shards,
    discover_source_root,
    find_verification_replacements,
    hard_pass_quality_ok,
    is_compaction_source,
    is_trueish,
    lineage_ids_for_row,
    load_metadata_rows,
    load_v5_sidecars,
    load_verification_signals,
    merged_row_for_uuid,
    metadata_value,
    normalize_language,
    record_from_row,
    scan_candidates,
    select_with_task_cap,
    to_float,
    to_int,
    write_jsonl,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617/"
    "swerebench-verification-enhanced-v75-expanded-pass4-nonpass1-allowlist"
)


def nonpassing_quality_ok(row: dict[str, Any], signals: dict[str, Any]) -> bool:
    if is_trueish(metadata_value(row, "passed", signals.get("passed"))):
        return False
    patch_bytes = to_int(metadata_value(row, "model_patch_bytes", signals.get("patch_bytes")))
    if patch_bytes <= 0 or patch_bytes > 100_000:
        return False
    if signals.get("empty_patch"):
        return False
    if not signals.get("has_submit_command"):
        return False
    if signals.get("empty_patch_irreparable_candidate"):
        return False
    if to_float(metadata_value(row, "percent_messages_with_reasoning")) < 0.9:
        return False
    return bool(
        signals.get("natural_patch_verification")
        or signals.get("natural_patch_create")
        or signals.get("natural_visible_diff_evidence")
    )


def score_nonpassing_candidate(row: dict[str, Any], signals: dict[str, Any]) -> tuple[float, tuple[str, ...]]:
    notes = ["high_quality_nonpassing"]
    score = 20.0

    reasoning = to_float(metadata_value(row, "percent_messages_with_reasoning"))
    score += 10.0 * min(max(reasoning, 0.0), 1.0)
    if reasoning >= 0.99:
        score += 1.0
        notes.append("near_full_reasoning")

    api_calls = to_int(metadata_value(row, "api_calls"))
    score += min(api_calls, 120) / 20.0
    if 4 <= api_calls <= 120:
        score += 1.0
        notes.append("healthy_api_calls")

    patch_bytes = to_int(metadata_value(row, "model_patch_bytes", signals.get("patch_bytes")))
    if 200 <= patch_bytes <= 20_000:
        score += 2.0
        notes.append("normal_patch")
    elif patch_bytes < 200:
        score -= 1.0
        notes.append("small_patch")

    if signals.get("natural_patch_verification"):
        score += 10.0
        notes.append("natural_patch_verification")
    if signals.get("natural_patch_create"):
        score += 4.0
        notes.append("natural_patch_create")
    if signals.get("natural_visible_diff_evidence"):
        score += 2.0
        notes.append("visible_diff_evidence")
    if signals.get("natural_submit_command_cats_patch"):
        score += 2.0
        notes.append("submit_cats_patch")
    if signals.get("missing_nonempty_patch_verification"):
        score -= 3.0
        notes.append("missing_nonempty_patch_verification")

    return score, tuple(notes)


def nonpassing_record_from_row(row: dict[str, Any], signals: dict[str, Any]) -> Candidate:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    uuid = str(row.get("uuid") or metadata.get("uuid") or "")
    task_id = str(metadata_value(row, "task_id") or "")
    line_number = row.get("_global_line_number")
    if not uuid or not task_id or line_number is None:
        raise ValueError("candidate row missing uuid, task_id, or global line number")
    score, notes = score_nonpassing_candidate(row, signals)
    return Candidate(
        uuid=uuid,
        task_id=task_id,
        language=normalize_language(metadata_value(row, "language", signals.get("language"))),
        line_number=int(line_number),
        lineage_ids=lineage_ids_for_row(row, signals),
        source_group="high_quality_nonpassing",
        score=score,
        notes=notes,
        metadata=row,
    )


def scan_nonpassing_candidates(
    *,
    metadata_rows: dict[str, dict[str, Any]],
    verification_signals: dict[str, dict[str, Any]],
    excluded_lineage_ids: set[str],
) -> tuple[list[Candidate], Counter[str]]:
    candidates: list[Candidate] = []
    stats: Counter[str] = Counter()
    for uuid, signals in verification_signals.items():
        stats["nonpassing_rows_seen"] += 1
        row = merged_row_for_uuid(
            uuid,
            metadata_rows=metadata_rows,
            verification_signals=verification_signals,
        )
        if not nonpassing_quality_ok(row, signals):
            stats["skip_nonpassing_quality"] += 1
            continue
        lineage_ids = set(lineage_ids_for_row(row, signals))
        if lineage_ids & excluded_lineage_ids:
            stats["skip_nonpassing_pass_lineage_duplicate"] += 1
            continue
        try:
            candidate = nonpassing_record_from_row(row, signals)
        except ValueError:
            stats["skip_nonpassing_missing_required_fields"] += 1
            continue
        candidates.append(candidate)
    stats["candidate:high_quality_nonpassing"] = len(candidates)
    return candidates, stats


def select_nonpassing_with_task_cap(
    candidates: list[Candidate],
    *,
    max_per_task: int,
    target_count: int,
    used_lineage_ids: set[str],
) -> tuple[list[Candidate], Counter[str]]:
    by_task: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_task[candidate.task_id].append(candidate)
    for rows in by_task.values():
        rows.sort(key=candidate_sort_key)

    task_order = sorted(by_task, key=lambda task: candidate_sort_key(by_task[task][0]))
    selected: list[Candidate] = []
    stats: Counter[str] = Counter()
    for rollout_index in range(max_per_task):
        for task_id in task_order:
            rows = by_task[task_id]
            if rollout_index >= len(rows):
                continue
            candidate = rows[rollout_index]
            lineage_ids = set(candidate.lineage_ids)
            if lineage_ids & used_lineage_ids:
                stats["skipped_lineage_duplicate:high_quality_nonpassing"] += 1
                continue
            selected.append(candidate)
            used_lineage_ids.update(lineage_ids)
            if target_count > 0 and len(selected) >= target_count:
                return selected, stats
    return selected, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-pass-per-task", type=int, default=4)
    parser.add_argument("--target-pass-traces", type=int, default=12_000)
    parser.add_argument("--max-nonpassing-per-task", type=int, default=1)
    parser.add_argument("--target-nonpassing-traces", type=int, default=4_016)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_pass_per_task < 1 or args.max_nonpassing_per_task < 1:
        raise ValueError("per-task caps must be >= 1")
    if args.target_pass_traces < 0 or args.target_nonpassing_traces < 0:
        raise ValueError("target counts must be >= 0")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    source_root = discover_source_root(args.local_root, args.dataset_id)
    verification_signals = load_verification_signals(source_root)
    metadata_rows, metadata_stats = load_metadata_rows(source_root)
    verification_selected, verification_replaced_sources, verification_stats = find_verification_replacements(
        verification_signals
    )
    v5_recommended, v5_not_recommended, v5_superseded = load_v5_sidecars(source_root)
    pass_candidates, pass_scan_stats = scan_candidates(
        metadata_rows=metadata_rows,
        verification_selected=verification_selected,
        verification_replaced_sources=verification_replaced_sources,
        v5_recommended=v5_recommended,
        v5_not_recommended=v5_not_recommended,
        v5_superseded=v5_superseded,
        verification_signals=verification_signals,
    )
    selected_pass, pass_selection_stats = select_with_task_cap(
        pass_candidates,
        args.max_pass_per_task,
        args.target_pass_traces,
    )
    used_lineage_ids = {lineage_id for candidate in selected_pass for lineage_id in candidate.lineage_ids}
    nonpassing_candidates, nonpassing_scan_stats = scan_nonpassing_candidates(
        metadata_rows=metadata_rows,
        verification_signals=verification_signals,
        excluded_lineage_ids=used_lineage_ids,
    )
    selected_nonpassing, nonpassing_selection_stats = select_nonpassing_with_task_cap(
        nonpassing_candidates,
        max_per_task=args.max_nonpassing_per_task,
        target_count=args.target_nonpassing_traces,
        used_lineage_ids=used_lineage_ids,
    )
    selected = selected_pass + selected_nonpassing

    uuid_path = args.output_root / "selected_expanded_uuids.txt"
    line_number_path = args.output_root / "selected_expanded_line_numbers.txt"
    uuid_path.write_text("\n".join(candidate.uuid for candidate in selected) + "\n", encoding="utf-8")
    line_number_path.write_text(
        "\n".join(str(candidate.line_number) for candidate in selected) + "\n",
        encoding="utf-8",
    )
    write_jsonl(
        args.output_root / "selected_expanded_records_training_order.jsonl",
        [candidate_report(candidate, idx) for idx, candidate in enumerate(selected)],
    )
    write_jsonl(
        args.output_root / "all_nonpassing_candidates_ranked.jsonl",
        [
            candidate_report(candidate, idx)
            for idx, candidate in enumerate(sorted(nonpassing_candidates, key=candidate_sort_key))
        ],
    )

    selected_task_counts = Counter(candidate.task_id for candidate in selected)
    manifest = {
        "dataset_id": args.dataset_id,
        "source_root": str(source_root),
        "output_root": str(args.output_root),
        "selected_uuid_file": str(uuid_path),
        "selected_line_number_file": str(line_number_path),
        "selected_records_training_order": str(args.output_root / "selected_expanded_records_training_order.jsonl"),
        "selection_policy": {
            "pass_only": False,
            "max_pass_per_task": args.max_pass_per_task,
            "target_pass_traces": args.target_pass_traces,
            "max_nonpassing_per_task": args.max_nonpassing_per_task,
            "target_nonpassing_traces": args.target_nonpassing_traces,
            "ordering": "strict-passed rows first, then high-quality non-passing rows",
            "nonpassing_quality": (
                "not passed, submitted, non-empty patch <=100k, has natural patch "
                "verification/create/visible-diff evidence, >=90% assistant reasoning, "
                "and not empty-patch irreparable"
            ),
            "lineage_deduplication": (
                "non-passing rows may not share known full/compacted/repaired/source UUID lineage with selected rows"
            ),
        },
        "counts": {
            "raw_shards": len(discover_shards(source_root)),
            "metadata_signal_rows": len(verification_signals),
            "metadata_index_rows": len(metadata_rows),
            "pass_candidates_before_cap": len(pass_candidates),
            "nonpassing_candidates_before_cap": len(nonpassing_candidates),
            "selected_pass_traces": len(selected_pass),
            "selected_nonpassing_traces": len(selected_nonpassing),
            "selected_total_traces": len(selected),
            "selected_nonpassing_fraction": (
                round(len(selected_nonpassing) / len(selected), 6) if selected else 0.0
            ),
            "selected_unique_tasks": len(selected_task_counts),
            "selected_max_rollouts_per_task": max(selected_task_counts.values(), default=0),
        },
        "selected_by_source_group": dict(sorted(Counter(candidate.source_group for candidate in selected).items())),
        "selected_by_language": dict(sorted(Counter(candidate.language for candidate in selected).items())),
        "scan_stats": dict(
            sorted(
                (
                    pass_scan_stats
                    + verification_stats
                    + metadata_stats
                    + pass_selection_stats
                    + nonpassing_scan_stats
                    + nonpassing_selection_stats
                ).items()
            )
        ),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
