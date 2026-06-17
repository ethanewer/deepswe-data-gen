#!/usr/bin/env python3
"""Fast raw-source dataset builder.

Parent `data/` shards are linked/copied unchanged. Only rows that were not in
the parent main split are materialized into appended shards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_260609_reasoning_datasets as base  # noqa: E402
import build_raw_all_generated_dataset_20260616 as raw  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dataset", type=Path, default=raw.DEFAULT_PARENT)
    parser.add_argument("--runs-root", type=Path, default=raw.DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=1500)
    parser.add_argument("--include-run-root", action="append", type=Path, default=[])
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def remember_index_path(row: dict[str, Any], seen_uuids: set[str], seen_trajectories: set[str], seen_results: set[str]) -> None:
    uuid = raw.path_text(row.get("uuid"))
    if uuid:
        seen_uuids.add(uuid)
    trajectory = raw.normalize_abs(row.get("trajectory_path", ""))
    result = raw.normalize_abs(row.get("result_path", ""))
    if trajectory:
        seen_trajectories.add(trajectory)
    if result:
        seen_results.add(result)


def remember_record_path(row: dict[str, Any], seen_uuids: set[str], seen_trajectories: set[str], seen_results: set[str]) -> None:
    uuid = raw.path_text(row.get("uuid"))
    if uuid:
        seen_uuids.add(uuid)
    trajectory, result = raw.row_paths(row)
    if trajectory:
        seen_trajectories.add(trajectory)
    if result:
        seen_results.add(result)


def record_seen(row: dict[str, Any], seen_uuids: set[str], seen_trajectories: set[str], seen_results: set[str]) -> bool:
    uuid = raw.path_text(row.get("uuid"))
    trajectory, result = raw.row_paths(row)
    return bool(
        (uuid and uuid in seen_uuids)
        or (trajectory and trajectory in seen_trajectories)
        or (result and result in seen_results)
    )


def candidate_started(candidate: dict[str, Any]) -> tuple[bool, str, str]:
    workspace = raw.workspace_from_candidate(candidate)
    if workspace is None:
        return False, "", ""
    result_path = raw.normalize_abs(raw.path_text(candidate.get("result_path")) or workspace / "result.json")
    trajectory_path = raw.normalize_abs(
        raw.path_text(candidate.get("trajectory_path")) or workspace / "agent" / "mini-swe-agent.trajectory.json"
    )
    return Path(result_path).exists() or Path(trajectory_path).exists(), trajectory_path, result_path


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output dir: {args.output_dir}")
    (args.output_dir / "data").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    parent_shards = sorted((args.parent_dataset / "data").glob("*.jsonl.zst"))
    for shard in parent_shards:
        link_or_copy(shard, args.output_dir / "data" / shard.name)
    for name in ("dataset_info.json",):
        src = args.parent_dataset / name
        if src.exists():
            link_or_copy(src, args.output_dir / name)

    index_rows = load_jsonl(args.parent_dataset / "metadata" / "index.jsonl")
    full_rows = load_jsonl(args.parent_dataset / "metadata" / "full_index.jsonl") or list(index_rows)
    seen_uuids: set[str] = set()
    seen_trajectories: set[str] = set()
    seen_results: set[str] = set()
    for row in index_rows:
        remember_index_path(row, seen_uuids, seen_trajectories, seen_results)

    writer = raw.ShardedWriter(args.output_dir, args.shard_size)
    writer.shard_index = len(parent_shards)
    skipped = Counter()
    appended_by_source = Counter()
    already_present_by_source = Counter()
    candidate_counts: dict[str, int] = {}
    status_counts = Counter()
    appended_over65k = 0

    for path in sorted((args.parent_dataset / "data_over65k").glob("*.jsonl.zst")):
        for row in raw.open_jsonl_zst(path):
            split = "data_over65k"
            if record_seen(row, seen_uuids, seen_trajectories, seen_results):
                skipped["parent_over65k_already_in_main"] += 1
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            metadata.setdefault("raw_source_parent_dataset", str(args.parent_dataset))
            metadata.setdefault("raw_source_parent_split", split)
            row["metadata"] = metadata
            line_number = len(index_rows)
            writer.write(row)
            compact, full = raw.add_index_rows(row, line_number)
            index_rows.append(compact)
            full_rows.append(full)
            remember_record_path(row, seen_uuids, seen_trajectories, seen_results)
            appended_over65k += 1

    print(
        f"linked_parent_main_rows={len(index_rows) - appended_over65k} "
        f"appended_parent_over65k={appended_over65k} elapsed={time.time() - started:.1f}s",
        flush=True,
    )

    for run_root in raw.run_roots(args):
        candidates = raw.collect_generated_candidates(run_root)
        candidate_counts[str(run_root)] = len(candidates)
        for candidate in candidates:
            started_candidate, trajectory_path, result_path = candidate_started(candidate)
            if not started_candidate:
                skipped["generated_planned_not_started"] += 1
                continue
            if (trajectory_path and trajectory_path in seen_trajectories) or (result_path and result_path in seen_results):
                already_present_by_source[str(run_root)] += 1
                continue
            row, status = raw.build_generated_row(run_root, candidate)
            status_counts[status] += 1
            if row is None:
                skipped[f"generated_{status}"] += 1
                continue
            if record_seen(row, seen_uuids, seen_trajectories, seen_results):
                already_present_by_source[str(run_root)] += 1
                continue
            line_number = len(index_rows)
            writer.write(row)
            compact, full = raw.add_index_rows(row, line_number)
            index_rows.append(compact)
            full_rows.append(full)
            remember_record_path(row, seen_uuids, seen_trajectories, seen_results)
            appended_by_source[str(run_root)] += 1
        print(
            "scanned_run_root="
            f"{run_root} candidates={len(candidates)} appended={appended_by_source[str(run_root)]} "
            f"already_present={already_present_by_source[str(run_root)]} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    writer.flush()
    raw.write_index_files(args.output_dir, index_rows, full_rows)
    summary = raw.summarize_index(index_rows)
    new_rows = appended_over65k + sum(appended_by_source.values())
    manifest = {
        "name": args.output_dir.name,
        "created_at_unix": time.time(),
        "parent_dataset": str(args.parent_dataset),
        "parent_main_shards_linked_or_copied": len(parent_shards),
        "parent_main_rows_inherited": len(index_rows) - new_rows,
        "parent_over65k_rows_appended_to_main": appended_over65k,
        "new_generated_rows_appended": sum(appended_by_source.values()),
        "total_rows": len(index_rows),
        "generated_candidate_counts": candidate_counts,
        "generated_rows_appended_by_run_root": dict(sorted(appended_by_source.items())),
        "generated_already_present_by_run_root": dict(sorted(already_present_by_source.items())),
        "generated_build_status_counts": dict(sorted(status_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "no_filters_applied_to_generated_rows": [
            "pass_reward",
            "qwen3_token_length",
            "assistant_reasoning_coverage",
            "model_patch_nonempty",
            "per_task_trace_cap",
            "api_calls_positive",
        ],
        "elapsed_sec": time.time() - started,
        **summary,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "build_summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = [
        f"# {args.output_dir.name}",
        "",
        "Raw-source JSONL+zstd dataset. Parent `data/` shards are preserved, parent `data_over65k/` rows are appended to the main split, and local generated traces are appended without filtering.",
        "",
        "Downstream refinement/training jobs should filter this dataset for passrate, token length, reasoning coverage, and task caps.",
        "",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
