#!/usr/bin/env python3
"""Build the v74 verification-enhanced mixed50 SFT view.

The existing mixed50 view is already filtered for the base v5/v62/v73 data
mixture. Verification-enhanced rows are modified versions of existing traces,
so this script treats them as replacements instead of simple append-only data:

* select only source-recommended verification rows;
* keep at most one modified row per source UUID, preferring recovery examples;
* filter matching original source UUIDs out of the inherited mixed50 view;
* append the selected modified rows as one extra shard.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


DEFAULT_BASE_VIEW = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617/"
    "swerebench-v5only-compaction-mixed50-cleanpatch-provenance-miniswe-aligned/data"
)
DEFAULT_SOURCE_DATASET = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swerebench-traces-raw-source-verification-enhanced-20260617"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260617/"
    "swerebench-verification-enhanced-v74-mixed50-cleanpatch-provenance-miniswe-aligned"
)


def iter_zstd_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    proc = subprocess.Popen(
        ["zstdcat", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if line.strip():
                yield json.loads(line)
    finally:
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"zstdcat failed for {path} with code {code}: {stderr}")


def row_is_recommended(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    return bool(
        metadata.get("verification_recommended_for_standard_sft")
        or metadata.get("verification_recommended_for_recovery_prefix_training")
    )


def row_source_uuid(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") or {}
    for key in (
        "verification_source_uuid",
        "synthetic_patchtxt_verification_source_uuid",
        "synthetic_empty_submit_verification_stop_source_uuid",
        "original_uuid",
        "verification_original_row_uuid",
    ):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def base_row_uuid(row: dict[str, Any]) -> str | None:
    source_outcome = row.get("source_outcome")
    if isinstance(source_outcome, dict) and source_outcome.get("uuid"):
        return str(source_outcome["uuid"])
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("uuid", "source_uuid", "original_uuid"):
            if metadata.get(key):
                return str(metadata[key])
    if row.get("uuid"):
        return str(row["uuid"])
    return None


def selection_priority(row: dict[str, Any]) -> tuple[int, int, str]:
    metadata = row.get("metadata") or {}
    family = metadata.get("verification_modification_family")
    priority_by_family = {
        "synthetic_empty_patch_one_turn_recovery": 0,
        "synthetic_patchtxt_verification_prototype": 1,
        "synthetic_empty_submit_verification_stop": 2,
    }
    family_priority = priority_by_family.get(str(family), 9)
    message_count = len(row.get("messages") or [])
    return family_priority, -message_count, str(row.get("uuid") or "")


def select_appended_rows(source_shard: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recommended_by_source: dict[str, list[dict[str, Any]]] = {}
    recommended_without_source: list[dict[str, Any]] = []
    skipped_counts: Counter[str] = Counter()
    skipped_total = 0

    for row in iter_zstd_jsonl(source_shard):
        metadata = row.get("metadata") or {}
        family = str(metadata.get("verification_modification_family") or "unknown")
        if not row_is_recommended(row):
            skipped_total += 1
            skipped_counts[family] += 1
            continue
        source_uuid = row_source_uuid(row)
        if source_uuid is None:
            recommended_without_source.append(row)
        else:
            recommended_by_source.setdefault(source_uuid, []).append(row)

    selected: list[dict[str, Any]] = []
    duplicate_source_counts: Counter[str] = Counter()
    duplicate_family_counts: Counter[str] = Counter()
    for source_uuid, rows in sorted(recommended_by_source.items()):
        rows = sorted(rows, key=selection_priority)
        selected.append(rows[0])
        if len(rows) > 1:
            duplicate_source_counts[source_uuid] = len(rows)
            for duplicate in rows[1:]:
                family = str((duplicate.get("metadata") or {}).get("verification_modification_family") or "unknown")
                duplicate_family_counts[family] += 1

    selected.extend(sorted(recommended_without_source, key=selection_priority))
    selected.sort(key=lambda row: (str((row.get("metadata") or {}).get("verification_modification_family")), str(row.get("uuid") or "")))
    metadata = {
        "recommended_rows_seen": sum(len(rows) for rows in recommended_by_source.values()) + len(recommended_without_source),
        "recommended_source_uuid_count": len(recommended_by_source),
        "recommended_rows_without_source_uuid": len(recommended_without_source),
        "duplicate_recommended_source_uuid_count": len(duplicate_source_counts),
        "duplicate_recommended_rows_dropped": sum(duplicate_source_counts.values()) - len(duplicate_source_counts),
        "duplicate_recommended_rows_dropped_by_family": dict(sorted(duplicate_family_counts.items())),
        "skipped_unrecommended_rows": skipped_total,
        "skipped_unrecommended_rows_by_family": dict(sorted(skipped_counts.items())),
    }
    return selected, metadata


def normalize_appended_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    metadata = dict(row.get("metadata") or {})
    if metadata.get("verification_should_not_be_counted_as_passed"):
        row["passed"] = False
        metadata["passed"] = False
        source_outcome = row.get("source_outcome")
        if isinstance(source_outcome, dict):
            source_outcome = dict(source_outcome)
            source_outcome["passed"] = False
            row["source_outcome"] = source_outcome
    metadata["sft_view"] = "v74_verification_enhanced_mixed50"
    row["metadata"] = metadata
    return row


def filter_base_shards(base_view: Path, data_dir: Path, replacement_source_uuids: set[str], *, force: bool) -> dict[str, Any]:
    written: list[str] = []
    removed_counts: Counter[str] = Counter()
    kept_rows = 0
    removed_rows = 0
    unmatched_replacement_uuids = set(replacement_source_uuids)
    for src in sorted(base_view.glob("*.jsonl")):
        dst = data_dir / src.name
        if dst.exists():
            if force:
                dst.unlink()
            else:
                raise FileExistsError(f"{dst} already exists; pass --force to replace output files")
        with src.open("r", encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
            for line in inp:
                row = json.loads(line)
                uuid = base_row_uuid(row)
                if uuid is not None and uuid in replacement_source_uuids:
                    removed_rows += 1
                    source_outcome = row.get("source_outcome")
                    task_id = source_outcome.get("task_id") if isinstance(source_outcome, dict) else row.get("task_id")
                    removed_counts[str(task_id or "unknown")] += 1
                    unmatched_replacement_uuids.discard(uuid)
                    continue
                out.write(line)
                kept_rows += 1
        written.append(src.name)
    return {
        "base_shards_written": len(written),
        "base_shard_names": written,
        "base_rows_kept": kept_rows,
        "base_rows_replaced_removed": removed_rows,
        "base_rows_replaced_removed_by_task": dict(sorted(removed_counts.items())),
        "replacement_source_uuids": len(replacement_source_uuids),
        "replacement_source_uuids_not_found_in_base_view": len(unmatched_replacement_uuids),
        "replacement_source_uuids_not_found_sample": sorted(unmatched_replacement_uuids)[:25],
    }


def build_view(args: argparse.Namespace) -> dict[str, Any]:
    base_view = args.base_view.resolve()
    source_dataset = args.source_dataset.resolve()
    output_root = args.output_root.resolve()
    data_dir = output_root / "data"

    if not base_view.is_dir():
        raise FileNotFoundError(base_view)
    source_shard = source_dataset / "data" / "train-00036.jsonl.zst"
    if not source_shard.is_file():
        raise FileNotFoundError(source_shard)
    if shutil.which("zstdcat") is None:
        raise RuntimeError("zstdcat is required to read the verification-enhanced shard")

    selected_rows, selection_metadata = select_appended_rows(source_shard)
    replacement_source_uuids = {source_uuid for row in selected_rows if (source_uuid := row_source_uuid(row)) is not None}

    output_root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    base_metadata = filter_base_shards(base_view, data_dir, replacement_source_uuids, force=args.force)

    appended_path = data_dir / f"shard-{base_metadata['base_shards_written']:03d}.jsonl"
    if appended_path.exists():
        if args.force:
            appended_path.unlink()
        else:
            raise FileExistsError(f"{appended_path} already exists; pass --force to replace it")

    selected_counts: Counter[str] = Counter()
    selected_total = 0
    with appended_path.open("w", encoding="utf-8") as out:
        for row in selected_rows:
            metadata = row.get("metadata") or {}
            family = str(metadata.get("verification_modification_family") or "unknown")
            row = normalize_appended_row(row)
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            selected_total += 1
            selected_counts[family] += 1

    manifest = {
        "view": "swerebench-verification-enhanced-v74-mixed50-cleanpatch-provenance-miniswe-aligned",
        "base_view": str(base_view),
        "source_dataset": str(source_dataset),
        "source_appended_shard": str(source_shard),
        "output_root": str(output_root),
        "train_raw_root": str(data_dir),
        "appended_shard_name": appended_path.name,
        "selected_appended_rows": selected_total,
        "selected_appended_rows_by_family": dict(sorted(selected_counts.items())),
        "selection_rule": (
            "metadata.verification_recommended_for_standard_sft or "
            "metadata.verification_recommended_for_recovery_prefix_training; "
            "then one row per verification source UUID, preferring one-turn recovery rows"
        ),
        "forced_nonpassed_when_should_not_count_as_passed": True,
        **base_metadata,
        **selection_metadata,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-view", type=Path, default=DEFAULT_BASE_VIEW)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = build_view(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
