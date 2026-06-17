#!/usr/bin/env python3
"""Build a local dataset that appends accepted other-source traces.

This intentionally does not retokenize. It copies the existing HF-staged
JSONL.zst shards unchanged, writes one additional shard for strictly audited
new traces, and regenerates lightweight metadata for the new dataset directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import zstandard as zstd

from build_260609_reasoning_datasets import (
    build_mimo_record,
    record_to_index,
    write_index_files,
)


DEFAULT_BASE = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swerebench-traces-highquality-2x-duplicate-reasoning-90pct"
)
DEFAULT_RUNS = Path(
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen-tasksource-research-20260615/runs"
)
DEFAULT_OUTPUT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swerebench-traces-highquality-2x-duplicate-reasoning-90pct-plus-other-sources-20260616"
)
DATASET_NAME = "swerebench-traces-highquality-2x-duplicate-reasoning-90pct-plus-other-sources-20260616"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-traces-per-task", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def audit_files(runs_root: Path) -> list[Path]:
    paths: list[Path] = []
    for run_dir in sorted(runs_root.glob("other-sources-datagen-*")):
        for path in sorted(run_dir.glob("strict*.json")):
            name = path.name
            if "partial" in name or "no_apply" in name:
                continue
            paths.append(path)
    return paths


def audit_rows(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    rows = data.get("rows") if isinstance(data, dict) else data
    return rows if isinstance(rows, list) else []


def accepted_audit_rows(paths: list[Path]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    seen_workspaces: set[str] = set()
    for path in paths:
        for row in audit_rows(path):
            if not isinstance(row, dict):
                continue
            workspace = row.get("workspace")
            if not workspace:
                continue
            workspace_path = Path(str(workspace))
            try:
                workspace_key = str(workspace_path.resolve())
            except OSError:
                workspace_key = str(workspace_path.absolute())
            if workspace_key in seen_workspaces:
                continue
            seen_workspaces.add(workspace_key)
            if not row.get("accepted"):
                continue
            if int(row.get("reward") or 0) != 1:
                continue
            if row.get("status") not in (None, "", "Submitted"):
                continue
            if row.get("patch_applies") is False:
                continue
            if row.get("clean_submission") is False:
                continue
            if float(row.get("assistant_reasoning_fraction") or 0.0) < 0.9:
                continue
            item = dict(row)
            item["audit_path"] = str(path)
            item["workspace_key"] = workspace_key
            accepted.append(item)
    return accepted


def manifest_index(runs_root: Path) -> dict[str, tuple[Path, Path, list[str]]]:
    index: dict[str, tuple[Path, Path, list[str]]] = {}
    for run_dir in sorted(runs_root.glob("other-sources-datagen-*")):
        manifest_dir = run_dir / "manifest"
        if not manifest_dir.exists():
            continue
        for path in sorted(manifest_dir.glob("*.tsv")):
            with path.open(encoding="utf-8", errors="replace") as handle:
                for raw in handle:
                    parts = raw.rstrip("\n").split("\t")
                    if len(parts) < 15:
                        continue
                    workspace = Path(parts[4])
                    if not workspace.is_absolute():
                        workspace = Path.cwd() / workspace
                    try:
                        key = str(workspace.resolve())
                    except OSError:
                        key = str(workspace.absolute())
                    index.setdefault(key, (run_dir, path, parts))
    return index


def read_source_index(base: Path) -> list[dict[str, Any]]:
    return iter_jsonl(base / "metadata" / "index.jsonl")


def count_source_tasks(index_rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in index_rows:
        task_id = str(row.get("task_id") or "")
        if task_id:
            counts[task_id] += 1
    return counts


def build_new_records(
    accepted_rows: list[dict[str, Any]],
    manifests: dict[str, tuple[Path, Path, list[str]]],
    source_task_counts: Counter[str],
    max_traces_per_task: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for audit_row in accepted_rows:
        manifest = manifests.get(str(audit_row["workspace_key"]))
        if manifest is None:
            rejected["missing_manifest"] += 1
            continue
        run_root, manifest_path, parts = manifest
        record = build_mimo_record(run_root, manifest_path, parts)
        if record is None:
            rejected["build_record_failed"] += 1
            continue
        if int(record.get("reward") or 0) != 1 or not record.get("passed"):
            rejected["not_passing_after_conversion"] += 1
            continue
        if float(record.get("percent_messages_with_reasoning") or 0.0) < 0.9:
            rejected["reasoning_under_90pct_after_conversion"] += 1
            continue
        patch = record.get("model_patch")
        if not isinstance(patch, str) or not patch.strip():
            rejected["empty_patch"] += 1
            continue
        task_id = str(record.get("task_id") or "")
        if not task_id:
            rejected["missing_task_id"] += 1
            continue
        metadata = dict(record.get("metadata") or {})
        metadata.update(
            {
                "dataset": DATASET_NAME,
                "source": "deepswe-data-gen-other-sources",
                "row_source": "other_source_strict_audit_trace",
                "strict_audit_path": audit_row.get("audit_path", ""),
                "strict_audit_accepted": True,
                "strict_audit_changed_paths": audit_row.get("changed_paths", []),
                "strict_audit_submission_bytes": audit_row.get("submission_bytes", 0),
                "combined_base_dataset": str(DEFAULT_BASE),
            }
        )
        record["metadata"] = metadata
        candidates.append(record)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[str(record.get("task_id") or "")].append(record)

    selected: list[dict[str, Any]] = []
    per_task_selected: Counter[str] = Counter()
    for task_id in sorted(grouped):
        remaining = max(0, max_traces_per_task - source_task_counts.get(task_id, 0))
        if remaining <= 0:
            rejected["task_cap_from_base_dataset"] += len(grouped[task_id])
            continue
        records = sorted(
            grouped[task_id],
            key=lambda row: (
                int((row.get("metadata") or {}).get("trajectory_bytes") or 0),
                str((row.get("metadata") or {}).get("workspace") or ""),
            ),
        )
        keep = records[:remaining]
        selected.extend(keep)
        per_task_selected[task_id] += len(keep)
        if len(records) > len(keep):
            rejected["task_cap_among_new_traces"] += len(records) - len(keep)

    selected.sort(
        key=lambda row: (
            str((row.get("metadata") or {}).get("language") or ""),
            str(row.get("task_id") or ""),
            int((row.get("metadata") or {}).get("trajectory_bytes") or 0),
        )
    )
    diagnostics = {
        "accepted_audit_rows": len(accepted_rows),
        "converted_candidates": len(candidates),
        "selected_records": len(selected),
        "selected_unique_tasks": len(per_task_selected),
        "rejected_after_audit_counts": dict(sorted(rejected.items())),
    }
    return selected, diagnostics


def copy_base_shards(base: Path, output: Path) -> list[dict[str, Any]]:
    source_shards = sorted((base / "data").glob("*.jsonl.zst"))
    copied: list[dict[str, Any]] = []
    (output / "data").mkdir(parents=True, exist_ok=True)
    for path in source_shards:
        dest = output / "data" / path.name
        shutil.copy2(path, dest)
        copied.append(
            {
                "path": str(dest.relative_to(output)),
                "compressed_bytes": dest.stat().st_size,
            }
        )
    return copied


def write_new_shard(output: Path, shard_index: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = output / "data" / f"train-{shard_index:05d}.jsonl.zst"
    cctx = zstd.ZstdCompressor(level=3, threads=0)
    uncompressed_bytes = 0
    with path.open("wb") as raw:
        with cctx.stream_writer(raw) as compressor:
            for row in rows:
                payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
                uncompressed_bytes += len(payload)
                compressor.write(payload)
    return {
        "path": str(path.relative_to(output)),
        "rows": len(rows),
        "compressed_bytes": path.stat().st_size,
        "uncompressed_bytes": uncompressed_bytes,
    }


def write_metadata(
    base: Path,
    output: Path,
    source_index_rows: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    copied_shards: list[dict[str, Any]],
    new_shard: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    index_rows = [dict(row) for row in source_index_rows]
    offset = len(index_rows)
    for idx, row in enumerate(new_records, start=offset):
        index_rows.append(record_to_index(row, idx))

    write_index_files(output, index_rows)

    # Preserve the richer source full index when present, appending the new rows
    # with the standard index schema.
    full_index_source = base / "metadata" / "full_index.jsonl"
    with (output / "metadata" / "full_index.jsonl").open("w", encoding="utf-8") as handle:
        if full_index_source.exists():
            with full_index_source.open(encoding="utf-8", errors="replace") as source:
                shutil.copyfileobj(source, handle)
        else:
            for row in source_index_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        for row in index_rows[offset:]:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    task_ids = sorted({str(row.get("task_id") or "") for row in index_rows if row.get("task_id")})
    (output / "metadata" / "task_ids.txt").write_text("\n".join(task_ids) + "\n", encoding="utf-8")
    (output / "metadata" / "exclude_task_ids_for_future_generation.txt").write_text(
        "\n".join(task_ids) + "\n", encoding="utf-8"
    )

    base_info = load_json(base / "dataset_info.json")
    source_shards_by_path = {
        str(shard.get("path")): shard for shard in base_info.get("shards", []) if isinstance(shard, dict)
    }
    shards: list[dict[str, Any]] = []
    for copied in copied_shards:
        original = source_shards_by_path.get(copied["path"], {})
        shard = dict(original)
        shard.update(copied)
        shards.append(shard)
    shards.append(new_shard)

    by_language = Counter(str(row.get("language") or "") for row in index_rows)
    by_passed = Counter(str(row.get("passed", False)).lower() for row in index_rows)
    by_teacher = Counter(str(row.get("teacher") or "") for row in index_rows)
    by_difficulty = Counter(str(row.get("difficulty") or "") for row in index_rows)
    by_style = Counter(str(row.get("instruction_style") or "") for row in index_rows)
    reasoning_mean = (
        sum(float(row.get("percent_messages_with_reasoning") or 0.0) for row in index_rows) / len(index_rows)
        if index_rows
        else 0.0
    )
    new_index_rows = index_rows[offset:]
    new_by_language = Counter(str(row.get("language") or "") for row in new_index_rows)

    manifest = {
        "name": DATASET_NAME,
        "created_at_unix": time.time(),
        "base_dataset": str(base),
        "base_dataset_rows": len(source_index_rows),
        "new_other_source_rows": len(new_records),
        "rows": len(index_rows),
        "unique_tasks": len(task_ids),
        "max_traces_per_task": 2,
        "reasoning_threshold": 0.9,
        "format": "jsonl.zst",
        "data_files": [shard["path"] for shard in shards],
        "index_jsonl": str(output / "metadata" / "index.jsonl"),
        "index_csv": str(output / "metadata" / "index.csv"),
        "index_sqlite": str(output / "metadata" / "index.sqlite"),
        "by_language": dict(sorted(by_language.items())),
        "new_by_language": dict(sorted(new_by_language.items())),
        "by_passed": dict(sorted(by_passed.items())),
        "by_teacher": dict(sorted(by_teacher.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "by_instruction_style": dict(sorted(by_style.items())),
        "mean_percent_messages_with_reasoning": reasoning_mean,
        "shards": shards,
        "diagnostics": diagnostics,
        "notes": [
            "Base shards were copied unchanged from the existing staged dataset.",
            "New rows come only from strict audit rows accepted as passing, clean submissions with nonempty patches.",
            "No qwen3 retokenization was performed for this combined dataset.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "dataset_info.json").write_text(
        json.dumps({**manifest, "compressed_bytes": sum(int(s.get("compressed_bytes") or 0) for s in shards)}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output / "build_summary.json").write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "output": str(output),
                "base": str(base),
                "base_rows": len(source_index_rows),
                "new_rows": len(new_records),
                "total_rows": len(index_rows),
                "new_by_language": dict(sorted(new_by_language.items())),
                "diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    readme = [
        f"# {DATASET_NAME}",
        "",
        "Combined local dataset built from the staged high-quality SWE-rebench traces plus strictly audited other-source traces.",
        "",
        "The existing dataset files were not modified. Base JSONL.zst shards were copied unchanged; new accepted traces are in the final shard.",
        "",
        "No qwen3 retokenization was performed for this build.",
        "",
        "```json",
        json.dumps(
            {
                "rows": len(index_rows),
                "base_rows": len(source_index_rows),
                "new_other_source_rows": len(new_records),
                "new_by_language": dict(sorted(new_by_language.items())),
                "diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    output = args.output.resolve()
    runs_root = args.runs_root.resolve()
    if not base.exists():
        raise SystemExit(f"Base dataset does not exist: {base}")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists; refusing to overwrite: {output}")
        shutil.rmtree(output)
    (output / "data").mkdir(parents=True)
    (output / "metadata").mkdir(parents=True)

    source_index_rows = read_source_index(base)
    source_task_counts = count_source_tasks(source_index_rows)
    audits = audit_files(runs_root)
    accepted_rows = accepted_audit_rows(audits)
    manifests = manifest_index(runs_root)
    new_records, diagnostics = build_new_records(
        accepted_rows,
        manifests,
        source_task_counts,
        max_traces_per_task=args.max_traces_per_task,
    )
    if not new_records:
        raise SystemExit("No new records selected; refusing to create an unchanged combined dataset")

    copied_shards = copy_base_shards(base, output)
    new_shard = write_new_shard(output, len(copied_shards), new_records)
    write_metadata(base, output, source_index_rows, new_records, copied_shards, new_shard, diagnostics)

    print(
        json.dumps(
            {
                "output": str(output),
                "base_rows": len(source_index_rows),
                "new_rows": len(new_records),
                "total_rows": len(source_index_rows) + len(new_records),
                "new_shard": new_shard,
                "diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
