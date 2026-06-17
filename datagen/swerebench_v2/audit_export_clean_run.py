#!/usr/bin/env python3
"""Audit a full datagen result index and save accepted and rejected traces."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from datagen.swerebench_v2 import audit_export_clean_trajectory as single

MANIFEST_FIELDS = [
    "selection_rank",
    "rollout_id",
    "instance_id",
    "task_dir",
    "workspace",
    "docker_image",
    "model",
    "litellm_model",
    "api_key_env",
    "api_base",
    "extra_body_json",
    "difficulty",
    "language",
    "instruction_style",
    "repo",
    "outside_original_high_quality_set",
]
TASK_METADATA_FIELDS = [
    "instance_id",
    "rollout_id",
    "repo",
    "language",
    "difficulty",
    "model",
    "teacher",
    "litellm_model",
    "instruction_style",
    "benchmark_profile",
    "docker_image",
    "base_commit",
    "task_dir",
    "outside_original_high_quality_set",
    "selection_rank",
    "result_path",
    "trajectory_path",
    "patch_path",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, default=single.DEFAULT_TARGET_CONFIG)
    parser.add_argument(
        "--manifest-tsv",
        action="append",
        type=Path,
        default=[],
        help="Optional generation manifest(s) to join into audit metadata by instance/rollout/model.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args(argv)


def safe_slug(*parts: str) -> str:
    text = "__".join(part for part in parts if part)
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text).strip("-")[:180] or "trace"


def read_index(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def read_manifest_metadata(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, str]]:
    metadata: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.rstrip("\n")
                if not line:
                    continue
                fields = line.split("\t")
                if len(fields) < 3:
                    continue
                record = {
                    field: fields[index] if index < len(fields) else ""
                    for index, field in enumerate(MANIFEST_FIELDS)
                }
                key = (
                    record.get("instance_id", ""),
                    record.get("rollout_id", ""),
                    record.get("model", ""),
                )
                metadata[key] = record
    return metadata


def coerce_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def manifest_record_for(
    manifest_metadata: dict[tuple[str, str, str], dict[str, str]],
    row: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, str]:
    instance_id = coerce_metadata_value(row.get("instance_id") or result.get("instance_id"))
    rollout_id = coerce_metadata_value(row.get("rollout_id") or result.get("rollout_id"))
    model = coerce_metadata_value(row.get("model") or result.get("model"))
    for key in (
        (instance_id, rollout_id, model),
        (instance_id, rollout_id, ""),
        (instance_id, "", model),
        (instance_id, "", ""),
    ):
        if key in manifest_metadata:
            return manifest_metadata[key]
    for (candidate_instance_id, candidate_rollout_id, candidate_model), record in manifest_metadata.items():
        if candidate_instance_id != instance_id:
            continue
        if rollout_id and candidate_rollout_id and candidate_rollout_id != rollout_id:
            continue
        if model and candidate_model and candidate_model != model:
            continue
        return record
    return {}


def task_metadata(
    row: dict[str, Any],
    result: dict[str, Any],
    manifest_record: dict[str, str],
) -> dict[str, str]:
    sources = (result, row, manifest_record)
    metadata: dict[str, str] = {}
    for field in TASK_METADATA_FIELDS:
        value = ""
        for source in sources:
            if field in source and source.get(field) not in (None, ""):
                value = coerce_metadata_value(source.get(field))
                break
        if field == "teacher" and not value:
            value = metadata.get("model", "")
        metadata[field] = value
    return metadata


def process_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    manifest_metadata: dict[tuple[str, str, str], dict[str, str]],
) -> dict[str, Any]:
    result_path = Path(row["result_path"])
    if result_path.exists():
        result = single.load_json(result_path)
    else:
        result = single.missing_result_placeholder(result_path, row)
    result.setdefault("result_path", str(result_path))
    manifest_record = manifest_record_for(manifest_metadata, row, result)
    metadata = task_metadata(row, result, manifest_record)
    workspace = single.workspace_from_result(result_path)
    trajectory_path, patch_path = single.infer_paths(
        argparse.Namespace(
            result_json=result_path,
            trajectory_json=single.host_path(row.get("trajectory_path"), workspace),
            patch_file=single.host_path(row.get("patch_path"), workspace),
        ),
        result,
    )
    trajectory, trajectory_missing = single.load_trajectory_or_placeholder(trajectory_path, result)
    workspace_patch_text = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.exists() else ""
    patch_text, patch_source = single.audit_patch_text(trajectory, workspace_patch_text)

    audit = single.audit(result, trajectory, patch_text, patch_path, trajectory_missing=trajectory_missing)
    slug = safe_slug(
        str(row.get("instance_id") or result.get("instance_id") or ""),
        str(row.get("rollout_id") or result.get("rollout_id") or ""),
        str(row.get("model") or result.get("model") or ""),
    )
    accepted_path = args.output_dir / "accepted" / f"{slug}.trajectory.json"
    accepted_raw_path = args.output_dir / "accepted_raw" / f"{slug}.trajectory.json"
    rejected_path = args.output_dir / "rejected" / f"{slug}.trajectory.json"
    accepted_patch_path = args.output_dir / "accepted_patches" / f"{slug}.patch"
    rejected_patch_path = args.output_dir / "rejected_patches" / f"{slug}.patch"
    audit_path = args.output_dir / "audits" / f"{slug}.audit.json"

    if audit["accepted"]:
        exported = single.export_trajectory(trajectory, result, args.target_config)
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        accepted_path.write_text(json.dumps(exported, indent=2) + "\n", encoding="utf-8")
        accepted_raw_path.parent.mkdir(parents=True, exist_ok=True)
        accepted_raw_path.write_text(json.dumps(trajectory, indent=2) + "\n", encoding="utf-8")
        single.write_patch_artifact(accepted_patch_path, patch_text)
        saved_trace_path = accepted_path
        saved_patch_path = accepted_patch_path
        saved_trace_kind = "benchmark_shaped_accepted"
    else:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_path.write_text(json.dumps(trajectory, indent=2) + "\n", encoding="utf-8")
        single.write_patch_artifact(rejected_patch_path, patch_text)
        saved_trace_path = rejected_path
        saved_patch_path = rejected_patch_path
        saved_trace_kind = "raw_rejected"

    audit.update(
        {
            **metadata,
            "task_metadata": metadata,
            "result_json": str(result_path),
            "trajectory_json": str(trajectory_path),
            "patch_file": str(patch_path),
            "workspace_patch_bytes": len(workspace_patch_text.encode("utf-8")),
            "audit_patch_source": patch_source,
            "trajectory_missing": trajectory_missing,
            "target_config": str(args.target_config),
            "saved_trace_path": str(saved_trace_path),
            "saved_patch_path": str(saved_patch_path),
            "saved_trace_kind": saved_trace_kind,
            "accepted_raw_trace_path": str(accepted_raw_path) if audit["accepted"] else "",
        }
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    args = parse_args()
    rows = read_index(args.result_index, args.limit)
    manifest_metadata = read_manifest_metadata(args.manifest_tsv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("accepted", "accepted_raw", "rejected", "accepted_patches", "rejected_patches", "audits"):
        shutil.rmtree(args.output_dir / name, ignore_errors=True)
    audits: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row in rows:
        try:
            audits.append(process_row(row, args, manifest_metadata))
        except Exception as exc:  # noqa: BLE001 - preserve batch progress and summarize
            error = {
                "instance_id": str(row.get("instance_id", "")),
                "result_path": str(row.get("result_path", "")),
                "type": type(exc).__name__,
                "message": str(exc),
            }
            errors.append(error)
            if args.fail_on_error:
                raise

    reject_reasons = Counter(
        reason
        for audit in audits
        if not audit.get("accepted")
        for reason in audit.get("reject_reasons", [])
    )
    summary = {
        "result_index": str(args.result_index),
        "output_dir": str(args.output_dir),
        "manifest_tsv": [str(path) for path in args.manifest_tsv],
        "total_rows": len(rows),
        "audited": len(audits),
        "accepted": sum(1 for audit in audits if audit.get("accepted")),
        "rejected": sum(1 for audit in audits if not audit.get("accepted")),
        "errors": len(errors),
        "reject_reasons": dict(reject_reasons),
        "by_outcome_category": dict(Counter(audit.get("outcome_category", "") for audit in audits)),
        "accepted_by_language": dict(
            Counter(audit.get("language", "") for audit in audits if audit.get("accepted"))
        ),
        "accepted_by_difficulty": dict(
            Counter(audit.get("difficulty", "") for audit in audits if audit.get("accepted"))
        ),
        "rejected_by_language": dict(
            Counter(audit.get("language", "") for audit in audits if not audit.get("accepted"))
        ),
        "rejected_by_difficulty": dict(
            Counter(audit.get("difficulty", "") for audit in audits if not audit.get("accepted"))
        ),
        "error_samples": errors[:20],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors and args.fail_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
