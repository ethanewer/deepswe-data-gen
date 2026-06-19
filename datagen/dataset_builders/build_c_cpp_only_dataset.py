#!/usr/bin/env python3
"""Build a local C/C++-only dataset from strict-audited new traces."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import zstandard as zstd

from build_260609_reasoning_datasets import build_mimo_record, record_to_index, write_index_files


DEFAULT_RUNS = Path(
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen-tasksource-research-20260615/runs"
)
DEFAULT_OUTPUT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swesmith-c-cpp-clean-traces-20260616-checkpoint"
)
DATASET_NAME = "swesmith-c-cpp-clean-traces-20260616-checkpoint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def strict_audit_files(runs_root: Path) -> list[Path]:
    paths: list[Path] = []
    for run_dir in sorted(runs_root.glob("other-sources-datagen-20260616-swesmith-cpp-*")):
        for path in sorted(run_dir.glob("strict*.json")):
            if "partial" not in path.name and "no_apply" not in path.name:
                paths.append(path)
    return paths


def manifest_index(runs_root: Path) -> dict[str, tuple[Path, Path, list[str]]]:
    out: dict[str, tuple[Path, Path, list[str]]] = {}
    for run_dir in sorted(runs_root.glob("other-sources-datagen-20260616-swesmith-cpp-*")):
        manifest_dir = run_dir / "manifest"
        if not manifest_dir.exists():
            continue
        for path in sorted(manifest_dir.glob("*.tsv")):
            with path.open(encoding="utf-8", errors="replace") as handle:
                for raw in handle:
                    parts = raw.rstrip("\n").split("\t")
                    if len(parts) < 16:
                        continue
                    workspace = Path(parts[4])
                    try:
                        key = str(workspace.resolve())
                    except OSError:
                        key = str(workspace.absolute())
                    out.setdefault(key, (run_dir, path, parts))
    return out


def accepted_audit_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        data = load_json(path)
        for row in data.get("rows", []):
            if not isinstance(row, dict) or not row.get("accepted"):
                continue
            if str(row.get("language") or "").lower() not in {"c", "cpp"}:
                continue
            if int(row.get("reward") or 0) != 1:
                continue
            if row.get("patch_applies") is False or row.get("clean_submission") is False:
                continue
            if float(row.get("assistant_reasoning_fraction") or 0.0) < 0.9:
                continue
            workspace = Path(str(row.get("workspace") or ""))
            try:
                key = str(workspace.resolve())
            except OSError:
                key = str(workspace.absolute())
            if key in seen:
                continue
            seen.add(key)
            item = dict(row)
            item["workspace_key"] = key
            item["strict_audit_path"] = str(path)
            rows.append(item)
    return rows


def build_records(
    audit_rows: list[dict[str, Any]],
    manifests: dict[str, tuple[Path, Path, list[str]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen_tasks: set[str] = set()
    for audit_row in audit_rows:
        manifest = manifests.get(str(audit_row["workspace_key"]))
        if manifest is None:
            rejected["missing_manifest"] += 1
            continue
        run_root, manifest_path, parts = manifest
        record = build_mimo_record(run_root, manifest_path, parts)
        if record is None:
            rejected["build_record_failed"] += 1
            continue
        language = str((record.get("metadata") or {}).get("language") or "").lower()
        if language not in {"c", "cpp"}:
            rejected["not_c_cpp_after_conversion"] += 1
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
        if task_id in seen_tasks:
            rejected["duplicate_task"] += 1
            continue
        seen_tasks.add(task_id)
        metadata = dict(record.get("metadata") or {})
        metadata.update(
            {
                "dataset": DATASET_NAME,
                "source": "SWE-bench/SWE-smith-cpp",
                "row_source": "other_source_strict_audit_trace",
                "strict_audit_path": audit_row.get("strict_audit_path", ""),
                "strict_audit_accepted": True,
                "strict_audit_changed_paths": audit_row.get("changed_paths", []),
                "strict_audit_submission_bytes": audit_row.get("submission_bytes", 0),
            }
        )
        record["metadata"] = metadata
        records.append(record)
    records.sort(key=lambda row: str(row.get("task_id") or ""))
    return records, dict(sorted(rejected.items()))


def write_dataset(output: Path, records: list[dict[str, Any]], diagnostics: dict[str, Any]) -> None:
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(parents=True, exist_ok=True)
    shard = output / "data" / "train-00000.jsonl.zst"
    uncompressed_bytes = 0
    cctx = zstd.ZstdCompressor(level=3, threads=0)
    with shard.open("wb") as raw, cctx.stream_writer(raw) as compressor:
        for row in records:
            payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
            uncompressed_bytes += len(payload)
            compressor.write(payload)

    index_rows = [record_to_index(row, idx) for idx, row in enumerate(records)]
    write_index_files(output, index_rows)
    (output / "metadata" / "full_index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    task_ids = sorted({str(row.get("task_id") or "") for row in records if row.get("task_id")})
    (output / "metadata" / "task_ids.txt").write_text("\n".join(task_ids) + "\n", encoding="utf-8")
    (output / "metadata" / "exclude_task_ids_for_future_generation.txt").write_text(
        "\n".join(task_ids) + "\n", encoding="utf-8"
    )

    by_language = Counter(str((row.get("metadata") or {}).get("language") or "") for row in records)
    by_passed = Counter(str(row.get("passed", False)).lower() for row in records)
    by_teacher = Counter(str(row.get("teacher") or "") for row in records)
    by_repo = Counter(str((row.get("metadata") or {}).get("repo") or "") for row in records)
    summary = {
        "name": DATASET_NAME,
        "created_at_unix": time.time(),
        "format": "jsonl.zst",
        "rows": len(records),
        "unique_tasks": len(task_ids),
        "max_traces_per_task": 1,
        "reasoning_threshold": 0.9,
        "data_files": ["data/train-00000.jsonl.zst"],
        "by_language": dict(sorted(by_language.items())),
        "by_passed": dict(sorted(by_passed.items())),
        "by_teacher": dict(sorted(by_teacher.items())),
        "by_repo": dict(sorted(by_repo.items())),
        "shards": [
            {
                "path": "data/train-00000.jsonl.zst",
                "rows": len(records),
                "compressed_bytes": shard.stat().st_size,
                "uncompressed_bytes": uncompressed_bytes,
            }
        ],
        "diagnostics": diagnostics,
        "notes": [
            "C/C++-only checkpoint dataset built from strict-audited new other-source traces.",
            "No qwen3 retokenization was performed.",
            "Rejected rollouts remain in their original run directories and are not included here.",
        ],
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (output / "manifest.json").write_text(text, encoding="utf-8")
    (output / "dataset_info.json").write_text(
        json.dumps({**summary, "compressed_bytes": shard.stat().st_size}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "build_summary.json").write_text(text, encoding="utf-8")
    (output / "README.md").write_text(
        "# " + DATASET_NAME + "\n\n"
        "C/C++-only checkpoint dataset from strict-audited SWE-smith C++ traces.\n\n"
        "```json\n" + text + "```\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists; refusing to overwrite: {output}")
        shutil.rmtree(output)
    audits = strict_audit_files(args.runs_root)
    audit_rows = accepted_audit_rows(audits)
    manifests = manifest_index(args.runs_root)
    records, rejected = build_records(audit_rows, manifests)
    if not records:
        raise SystemExit("No records selected")
    diagnostics = {
        "strict_audit_files": [str(path) for path in audits],
        "accepted_audit_rows": len(audit_rows),
        "selected_records": len(records),
        "rejected_after_audit_counts": rejected,
    }
    write_dataset(output, records, diagnostics)
    print(json.dumps({"output": str(output), "rows": len(records), "diagnostics": diagnostics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
