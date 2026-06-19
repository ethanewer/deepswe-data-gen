#!/usr/bin/env python3
"""Append all local generated raw traces to an existing parent-shard dataset.

This builder avoids reading or rewriting the parent dataset. It links parent
data shards if needed, appends all generated rows in new shards, and writes
append-only indexes for the new rows.
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


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def next_shard_index(data_dir: Path) -> int:
    indices: list[int] = []
    for path in data_dir.glob("train-*.jsonl.zst"):
        try:
            indices.append(int(path.stem.split("-")[-1].split(".")[0]))
        except Exception:
            continue
    return max(indices) + 1 if indices else 0


def candidate_started(candidate: dict[str, Any]) -> tuple[bool, str, str]:
    workspace = raw.workspace_from_candidate(candidate)
    if workspace is None:
        return False, "", ""
    result_path = raw.normalize_abs(raw.path_text(candidate.get("result_path")) or workspace / "result.json")
    trajectory_path = raw.normalize_abs(
        raw.path_text(candidate.get("trajectory_path")) or workspace / "agent" / "mini-swe-agent.trajectory.json"
    )
    return Path(result_path).exists() or Path(trajectory_path).exists(), trajectory_path, result_path


def collect_manifest_candidates(run_root: Path) -> list[dict[str, Any]]:
    by_workspace: dict[str, dict[str, Any]] = {}

    def add(candidate: dict[str, Any]) -> None:
        workspace = raw.workspace_from_candidate(candidate)
        if workspace is None:
            return
        key = raw.normalize_abs(workspace)
        current = by_workspace.get(key, {})
        if candidate.get("_source_manifest_kind") == "result_index":
            merged = {**current, **candidate}
        else:
            merged = {**candidate, **current}
        merged["workspace"] = str(workspace)
        by_workspace[key] = merged

    result_index_paths = sorted((run_root / "manifest").glob("*result_index*.jsonl"))
    if result_index_paths:
        for path in result_index_paths:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(item, dict):
                        continue
                    item = dict(item)
                    item["_source_manifest"] = str(path)
                    item["_source_manifest_line"] = line_no
                    item["_source_manifest_kind"] = "result_index"
                    add(item)
        return [by_workspace[key] for key in sorted(by_workspace)]

    for candidate in raw.manifest_jsonl_candidates(run_root):
        add(candidate)
    for candidate in raw.manifest_tsv_candidates(run_root):
        add(candidate)
    return [by_workspace[key] for key in sorted(by_workspace)]


def append_index_row(
    row: dict[str, Any],
    appended_line: int,
    global_line: int,
    appended_index_rows: list[dict[str, Any]],
    appended_full_rows: list[dict[str, Any]],
) -> None:
    compact, full = raw.add_index_rows(row, global_line)
    compact["appended_line_number"] = appended_line
    full["appended_line_number"] = appended_line
    appended_index_rows.append(compact)
    appended_full_rows.append(full)


def main() -> None:
    args = parse_args()
    started = time.time()
    (args.output_dir / "data").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    parent_manifest = load_json(args.parent_dataset / "manifest.json")
    parent_summary = load_json(args.parent_dataset / "build_summary.json")
    parent_main_rows = int(
        parent_manifest.get("records")
        or parent_manifest.get("rows")
        or parent_manifest.get("total_rows")
        or parent_summary.get("records")
        or parent_summary.get("main_rows")
        or 0
    )
    parent_shards = sorted((args.parent_dataset / "data").glob("*.jsonl.zst"))
    for shard in parent_shards:
        link_or_copy(shard, args.output_dir / "data" / shard.name)
    for src_name, dst_name in (
        ("metadata/index.jsonl", "metadata/parent_index.jsonl"),
        ("metadata/full_index.jsonl", "metadata/parent_full_index.jsonl"),
        ("metadata/index.csv", "metadata/parent_index.csv"),
        ("metadata/index.sqlite", "metadata/parent_index.sqlite"),
        ("dataset_info.json", "dataset_info.json"),
    ):
        src = args.parent_dataset / src_name
        if src.exists():
            link_or_copy(src, args.output_dir / dst_name)

    writer = raw.ShardedWriter(args.output_dir, args.shard_size)
    writer.shard_index = next_shard_index(args.output_dir / "data")
    appended_index_rows: list[dict[str, Any]] = []
    appended_full_rows: list[dict[str, Any]] = []
    seen_generated_workspaces: set[str] = set()
    seen_generated_trajectories: set[str] = set()
    seen_generated_results: set[str] = set()
    appended_by_source = Counter()
    candidate_counts: dict[str, int] = {}
    skipped = Counter()
    status_counts = Counter()
    appended_over65k = 0

    def append_row(row: dict[str, Any], source: str) -> None:
        nonlocal appended_over65k
        appended_line = len(appended_index_rows)
        global_line = parent_main_rows + appended_line
        writer.write(row)
        append_index_row(row, appended_line, global_line, appended_index_rows, appended_full_rows)
        if source == "parent_data_over65k":
            appended_over65k += 1
        else:
            appended_by_source[source] += 1

    for path in sorted((args.parent_dataset / "data_over65k").glob("*.jsonl.zst")):
        for row in raw.open_jsonl_zst(path):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            metadata.setdefault("raw_source_parent_dataset", str(args.parent_dataset))
            metadata.setdefault("raw_source_parent_split", "data_over65k")
            row["metadata"] = metadata
            append_row(row, "parent_data_over65k")

    print(
        f"parent_main_rows={parent_main_rows} linked_parent_shards={len(parent_shards)} "
        f"appended_parent_over65k={appended_over65k} elapsed={time.time() - started:.1f}s",
        flush=True,
    )

    for run_root in raw.run_roots(args):
        candidates = collect_manifest_candidates(run_root)
        candidate_counts[str(run_root)] = len(candidates)
        for candidate in candidates:
            workspace = raw.workspace_from_candidate(candidate)
            if workspace is None:
                skipped["generated_missing_workspace"] += 1
                continue
            workspace_key = raw.normalize_abs(workspace)
            if workspace_key in seen_generated_workspaces:
                skipped["generated_duplicate_workspace"] += 1
                continue
            started_candidate, trajectory_path, result_path = candidate_started(candidate)
            if not started_candidate:
                skipped["generated_planned_not_started"] += 1
                continue
            if (trajectory_path and trajectory_path in seen_generated_trajectories) or (
                result_path and result_path in seen_generated_results
            ):
                skipped["generated_duplicate_trace_path"] += 1
                continue
            row, status = raw.build_generated_row(run_root, candidate)
            status_counts[status] += 1
            if row is None:
                skipped[f"generated_{status}"] += 1
                continue
            seen_generated_workspaces.add(workspace_key)
            row_trajectory, row_result = raw.row_paths(row)
            if row_trajectory:
                seen_generated_trajectories.add(row_trajectory)
            if row_result:
                seen_generated_results.add(row_result)
            append_row(row, str(run_root))
        print(
            f"scanned_run_root={run_root} candidates={len(candidates)} "
            f"appended={appended_by_source[str(run_root)]} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    writer.flush()
    metadata_dir = args.output_dir / "metadata"
    with (metadata_dir / "appended_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in appended_index_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (metadata_dir / "appended_full_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in appended_full_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    note = (
        "This raw-source dataset reuses the parent `data/` shards unchanged. "
        "`metadata/parent_index.*` points to the inherited parent rows, and "
        "`metadata/appended_index.jsonl` indexes rows appended after the parent "
        f"main split starting at global line {parent_main_rows}.\n"
    )
    (metadata_dir / "index_note.md").write_text(note, encoding="utf-8")

    appended_summary = raw.summarize_index(appended_index_rows)
    total_rows = parent_main_rows + len(appended_index_rows)
    manifest = {
        "name": args.output_dir.name,
        "created_at_unix": time.time(),
        "parent_dataset": str(args.parent_dataset),
        "parent_main_rows_inherited": parent_main_rows,
        "parent_main_shards_linked_or_copied": len(parent_shards),
        "parent_over65k_rows_appended_to_main": appended_over65k,
        "local_generated_rows_appended_raw_unfiltered": sum(appended_by_source.values()),
        "appended_rows_total": len(appended_index_rows),
        "total_rows": total_rows,
        "generated_candidate_counts": candidate_counts,
        "generated_rows_appended_by_run_root": dict(sorted(appended_by_source.items())),
        "generated_build_status_counts": dict(sorted(status_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "appended_summary": appended_summary,
        "parent_summary": {
            "records": parent_main_rows,
            "source_manifest_records": parent_manifest.get("records"),
            "source_manifest_total_rows": parent_manifest.get("total_rows"),
            "source_manifest_passing_rows": parent_manifest.get("passing_rows"),
        },
        "no_filters_applied_to_generated_rows": [
            "pass_reward",
            "qwen3_token_length",
            "assistant_reasoning_coverage",
            "model_patch_nonempty",
            "per_task_trace_cap",
            "api_calls_positive",
        ],
        "note": "Parent main rows are inherited as-is; all local generated rows are appended even if a filtered version was already present in the parent dataset.",
        "elapsed_sec": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "build_summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = [
        f"# {args.output_dir.name}",
        "",
        "Raw-source JSONL+zstd dataset. Parent main shards are inherited unchanged; parent over-65k rows and all local generated traces are appended to the main `data/` split without filtering.",
        "",
        "Use `metadata/parent_index.*` for inherited rows and `metadata/appended_index.jsonl` for rows appended after the parent main split.",
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
