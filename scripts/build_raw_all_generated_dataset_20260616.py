#!/usr/bin/env python3
"""Build a raw-source dataset that includes all generated local traces.

This intentionally does not enforce passrate, token length, reasoning coverage,
or per-task caps. Downstream refinement jobs are responsible for filtering.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import zstandard as zstd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_260609_reasoning_datasets as base  # noqa: E402
from build_260611_3x_reasoning_dataset import provider_usage  # noqa: E402


DEFAULT_PARENT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swerebench-traces-highquality-2x-duplicate-reasoning-90pct-plus-other-sources-c-cpp-topup-"
    "plus-swesmith-plus-newgen-20260616-063854"
)
DEFAULT_RUNS_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen-mimo-clean-harness/runs/swerebench-v2"
)
DEFAULT_OUTPUT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swerebench-traces-raw-source-plus-all-local-generated-20260616"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dataset", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-size", type=int, default=1500)
    parser.add_argument("--include-run-root", action="append", type=Path, default=[])
    parser.add_argument("--limit-generated", type=int, default=0)
    return parser.parse_args()


def open_jsonl_zst(path: Path) -> Iterable[dict[str, Any]]:
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as raw:
        with dctx.stream_reader(raw) as reader:
            import io

            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if isinstance(data, dict):
                    yield data


def write_jsonl_zst(path: Path, rows: list[dict[str, Any]]) -> None:
    cctx = zstd.ZstdCompressor(level=3, threads=0)
    with path.open("wb") as raw:
        with cctx.stream_writer(raw) as writer:
            for row in rows:
                writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                writer.write(b"\n")


def json_load(path: Path) -> dict[str, Any]:
    return base.load_json(path)


def path_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text if text else ""


def normalize_abs(path: str | Path) -> str:
    text = path_text(path)
    if not text:
        return ""
    try:
        return str(Path(text).resolve())
    except Exception:
        return text


def iter_parent_rows(parent: Path) -> Iterable[tuple[dict[str, Any], str]]:
    for split_name in ("data", "data_over65k"):
        split_dir = parent / split_name
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.jsonl.zst")):
            for row in open_jsonl_zst(path):
                yield row, split_name


def row_paths(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    trajectory = base.first_nonempty(metadata.get("trajectory_path"), row.get("trajectory_path"))
    result = base.first_nonempty(metadata.get("result_path"), row.get("result_path"))
    return normalize_abs(trajectory), normalize_abs(result)


def stable_generated_uuid(
    trajectory_path: Path,
    result_path: Path,
    task_id: str,
    rollout_id: str,
    teacher: str,
    instruction_style: str,
) -> str:
    return base.stable_uuid(
        [
            normalize_abs(trajectory_path),
            normalize_abs(result_path),
            task_id,
            rollout_id,
            teacher,
            instruction_style,
        ]
    )


def run_roots(args: argparse.Namespace) -> list[Path]:
    if args.include_run_root:
        roots = args.include_run_root
    else:
        roots = sorted(path for path in args.runs_root.glob("mimo-*") if path.is_dir())
    skip_names = {"mimo-clean-easy-pilot-20260615"}
    # The pilot task-prep root has no completed result index in practice, but
    # keep it if it does contain trace outputs.
    return [path for path in roots if path.name not in skip_names or (path / "pyxis-traces").exists()]


def manifest_jsonl_candidates(run_root: Path) -> Iterable[dict[str, Any]]:
    manifest_dir = run_root / "manifest"
    if not manifest_dir.exists():
        return
    for path in sorted(manifest_dir.glob("*.jsonl")):
        kind = "result_index" if "result_index" in path.name else "manifest"
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                out = dict(row)
                out["_source_manifest"] = str(path)
                out["_source_manifest_line"] = line_no
                out["_source_manifest_kind"] = kind
                yield out


def manifest_tsv_candidates(run_root: Path) -> Iterable[dict[str, Any]]:
    manifest_dir = run_root / "manifest"
    if not manifest_dir.exists():
        return
    for path in sorted(manifest_dir.glob("*.manifest.tsv")):
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for line_no, parts in enumerate(reader, 1):
                if len(parts) < 5:
                    continue
                if parts[0] == "index" or parts[4] == "workspace":
                    continue
                workspace = parts[4]
                if "/pyxis-traces/" not in workspace:
                    continue
                row: dict[str, Any] = {
                    "index": parts[0],
                    "rollout_id": parts[1] if len(parts) > 1 else "",
                    "instance_id": parts[2] if len(parts) > 2 else "",
                    "workspace": workspace,
                    "model": parts[6] if len(parts) > 6 else "",
                    "litellm_model": parts[7] if len(parts) > 7 else "",
                    "difficulty": parts[11] if len(parts) > 11 else "",
                    "language": parts[12] if len(parts) > 12 else "",
                    "instruction_style": parts[13] if len(parts) > 13 else "",
                    "repo": parts[14] if len(parts) > 14 else "",
                    "_source_manifest": str(path),
                    "_source_manifest_line": line_no,
                    "_source_manifest_kind": "manifest",
                }
                yield row


def workspace_from_candidate(candidate: dict[str, Any]) -> Path | None:
    workspace = path_text(candidate.get("workspace"))
    if workspace:
        return Path(workspace)
    result_path = path_text(candidate.get("result_path"))
    if result_path:
        return Path(result_path).parent
    trajectory_path = path_text(candidate.get("trajectory_path"))
    if trajectory_path:
        return Path(trajectory_path).parent.parent
    return None


def candidate_from_workspace(workspace: Path) -> dict[str, Any]:
    return {
        "workspace": str(workspace),
        "result_path": str(workspace / "result.json"),
        "trajectory_path": str(workspace / "agent" / "mini-swe-agent.trajectory.json"),
        "_source_manifest_kind": "pyxis_scan",
    }


def scan_pyxis_candidates(run_root: Path) -> Iterable[dict[str, Any]]:
    pyxis = run_root / "pyxis-traces"
    if not pyxis.exists():
        return
    for dirpath, _dirnames, filenames in os.walk(pyxis):
        if "result.json" in filenames or "mini-swe-agent.trajectory.json" in filenames:
            path = Path(dirpath)
            if path.name == "agent":
                workspace = path.parent
            else:
                workspace = path
            yield candidate_from_workspace(workspace)


def collect_generated_candidates(run_root: Path) -> list[dict[str, Any]]:
    by_workspace: dict[str, dict[str, Any]] = {}

    def add(candidate: dict[str, Any]) -> None:
        workspace = workspace_from_candidate(candidate)
        if workspace is None:
            return
        key = normalize_abs(workspace)
        current = by_workspace.get(key, {})
        # Prefer result-index metadata, then preserve missing fields from manifests.
        if candidate.get("_source_manifest_kind") == "result_index":
            merged = {**current, **candidate}
        else:
            merged = {**candidate, **current}
        merged["workspace"] = str(workspace)
        by_workspace[key] = merged

    for candidate in manifest_jsonl_candidates(run_root):
        add(candidate)
    for candidate in manifest_tsv_candidates(run_root):
        add(candidate)
    for candidate in scan_pyxis_candidates(run_root):
        add(candidate)
    return [by_workspace[key] for key in sorted(by_workspace)]


def patch_candidates(workspace: Path, result: dict[str, Any], candidate: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for raw in (
        result.get("patch_path"),
        candidate.get("patch_path"),
        workspace / "logs" / "artifacts" / "model.patch",
        workspace / "model.patch",
        workspace / "patch.txt",
    ):
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_absolute() and not path.exists():
            try:
                path = workspace / path.relative_to("/workspace")
            except ValueError:
                pass
        paths.append(path)
    return paths


def first_existing_patch(workspace: Path, result: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, str]:
    for patch_path in patch_candidates(workspace, result, candidate):
        text = base.read_text_if_small(patch_path)
        if text:
            return text, str(patch_path)
    if isinstance(result.get("patch"), str) and result.get("patch"):
        return str(result.get("patch")), ""
    return "", path_text(candidate.get("patch_path"))


def infer_instruction_style(workspace: Path, result: dict[str, Any], candidate: dict[str, Any]) -> str:
    style = base.first_nonempty(result.get("instruction_style"), candidate.get("instruction_style"))
    if style:
        return style
    parts = workspace.parts
    if "pyxis-traces" in parts:
        index = parts.index("pyxis-traces")
        if index + 1 < len(parts):
            return parts[index + 1]
    return ""


def build_generated_row(run_root: Path, candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    workspace = workspace_from_candidate(candidate)
    if workspace is None:
        return None, "missing_workspace"
    result_path = Path(path_text(candidate.get("result_path")) or workspace / "result.json")
    trajectory_path = Path(path_text(candidate.get("trajectory_path")) or workspace / "agent" / "mini-swe-agent.trajectory.json")
    result = json_load(result_path) if result_path.exists() else {}
    # The result index and result.json already carry the task metadata needed
    # for this raw-source row. Some sidecar metadata.json files can be slow to
    # stat/read on FSx, so avoid making them part of the critical path.
    metadata_json: dict[str, Any] = {}
    trajectory = json_load(trajectory_path) if trajectory_path.exists() else {}
    raw_messages = trajectory.get("messages")
    if not isinstance(raw_messages, list):
        raw_messages = []
    reasoning = base.collect_message_reasoning(raw_messages)
    messages = base.normalize_messages(raw_messages)
    metrics = base.reasoning_metrics(messages, reasoning)
    info = trajectory.get("info") if isinstance(trajectory.get("info"), dict) else {}
    model_stats = info.get("model_stats") if isinstance(info.get("model_stats"), dict) else {}
    config = info.get("config") if isinstance(info.get("config"), dict) else {}
    model_config = config.get("model") if isinstance(config.get("model"), dict) else {}
    usage = provider_usage(raw_messages)
    agent_exception = result.get("agent_exception") if isinstance(result.get("agent_exception"), dict) else {}

    task_id = base.first_nonempty(result.get("instance_id"), metadata_json.get("instance_id"), candidate.get("instance_id"), workspace.name)
    rollout_id = base.first_nonempty(result.get("rollout_id"), metadata_json.get("rollout_id"), candidate.get("rollout_id"), workspace.parent.name)
    teacher = base.first_nonempty(result.get("model"), result.get("teacher"), metadata_json.get("model"), candidate.get("model"), workspace.parent.parent.name)
    litellm_model = base.first_nonempty(result.get("litellm_model"), metadata_json.get("litellm_model"), candidate.get("litellm_model"))
    difficulty = base.first_nonempty(result.get("difficulty"), metadata_json.get("difficulty"), candidate.get("difficulty"))
    language = base.first_nonempty(result.get("language"), metadata_json.get("language"), candidate.get("language"))
    instruction_style = infer_instruction_style(workspace, result, candidate)
    repo = base.first_nonempty(result.get("repo"), metadata_json.get("repo"), candidate.get("repo"))
    reward_raw = result.get("reward", candidate.get("reward", 0))
    try:
        reward_int = int(reward_raw or 0)
    except Exception:
        reward_int = 1 if base.bool_from_any(reward_raw) else 0
    passed = reward_int == 1 or base.bool_from_any(result.get("passed", candidate.get("passed", False)))
    patch_text, patch_path = first_existing_patch(workspace, result, candidate)
    model_patch_sha = base.sha256_text(patch_text) if patch_text else ""
    prompt = base.prompt_from_messages(messages)
    cost_usd = result.get("cost_usd", candidate.get("cost_usd", model_stats.get("instance_cost", usage["provider_cost_usd"])))
    api_calls = result.get("api_calls", candidate.get("api_calls", model_stats.get("api_calls", usage["provider_api_calls"])))
    has_result = result_path.exists()
    has_trajectory = trajectory_path.exists()
    trajectory_bytes = trajectory_path.stat().st_size if has_trajectory else 0

    metadata = {
        "dataset": "swerebench-traces-raw-source-plus-all-local-generated-20260616",
        "source": "deepswe-data-gen-mimo-clean-harness",
        "source_run_root": str(run_root),
        "source_manifest": candidate.get("_source_manifest"),
        "source_manifest_line": candidate.get("_source_manifest_line"),
        "source_manifest_kind": candidate.get("_source_manifest_kind"),
        "workspace": str(workspace),
        "trajectory_path": str(trajectory_path),
        "result_path": str(result_path),
        "patch_path": patch_path,
        "task_id": task_id,
        "instance_id": task_id,
        "rollout_id": rollout_id,
        "repo": repo or None,
        "difficulty": difficulty or None,
        "language": language or None,
        "instruction_style": instruction_style or None,
        "benchmark_profile": base.first_nonempty(result.get("benchmark_profile"), metadata_json.get("benchmark_profile"), candidate.get("benchmark_profile")) or None,
        "deepswe_prompt_augmentation": instruction_style == "deepswe",
        "teacher": teacher,
        "model": teacher,
        "litellm_model": litellm_model or None,
        "api_base": base.first_nonempty(result.get("api_base"), metadata_json.get("api_base"), candidate.get("api_base")) or None,
        "docker_image": base.first_nonempty(result.get("docker_image"), metadata_json.get("docker_image"), candidate.get("docker_image"), candidate.get("image")) or None,
        "outside_original_high_quality_set": base.bool_from_any(
            result.get("outside_original_high_quality_set", metadata_json.get("outside_original_high_quality_set", candidate.get("outside_original_high_quality_set", False)))
        ),
        "passed": passed,
        "reward": reward_int,
        "has_result_json": has_result,
        "result_json_missing": not has_result,
        "trajectory_saved": has_trajectory,
        "trajectory_missing": not has_trajectory,
        "trace_build_status": "ok" if has_trajectory and raw_messages else ("empty_messages" if has_trajectory else "missing_trajectory"),
        "raw_source_unfiltered": True,
        "raw_source_filters_not_applied": [
            "pass_reward",
            "qwen3_token_length",
            "assistant_reasoning_coverage",
            "model_patch_nonempty",
            "per_task_trace_cap",
            "api_calls_positive",
        ],
        "agent_exit_status": base.first_nonempty(result.get("agent_exit_status"), candidate.get("agent_exit_status"), info.get("exit_status")) or None,
        "agent_exception_type": base.first_nonempty(agent_exception.get("type"), result.get("agent_exception_type"), candidate.get("agent_exception_type")) or None,
        "api_calls": api_calls,
        "cost_usd": cost_usd,
        "provider_usage": usage,
        "message_count": len(messages),
        "assistant_message_count": metrics["assistant_message_count"],
        "assistant_messages_with_reasoning": metrics["assistant_messages_with_reasoning"],
        "assistant_messages_without_reasoning": metrics["assistant_messages_without_reasoning"],
        "percent_messages_with_reasoning": metrics["percent_messages_with_reasoning"],
        "has_any_reasoning": metrics["has_any_reasoning"],
        "has_all_assistant_reasoning": metrics["has_all_assistant_reasoning"],
        "reasoning_turns": len(reasoning),
        "reasoning_chars": sum(len(str(x.get("content", ""))) for x in reasoning if isinstance(x, dict)),
        "trajectory_chars": len(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
        "trajectory_bytes": trajectory_bytes,
        "trajectory_format": trajectory.get("trajectory_format", ""),
        "mini_swe_agent_version": info.get("mini_version", ""),
        "mini_swe_agent_git_sha": base.first_nonempty(result.get("mini_swe_agent_git_sha"), metadata_json.get("mini_swe_agent_git_sha"), candidate.get("mini_swe_agent_git_sha")) or None,
        "mini_swe_agent_config_file": base.first_nonempty(result.get("mini_swe_agent_config_file"), metadata_json.get("mini_swe_agent_config_file"), candidate.get("mini_swe_agent_config_file")) or None,
        "datagen_code_commit": base.first_nonempty(result.get("datagen_code_commit"), metadata_json.get("datagen_code_commit"), candidate.get("datagen_code_commit")) or None,
        "uses_updated_alignment": base.bool_from_any(result.get("uses_updated_alignment", metadata_json.get("uses_updated_alignment", candidate.get("uses_updated_alignment", False)))),
        "eligible_for_controlled_comparison": base.bool_from_any(result.get("eligible_for_controlled_comparison", candidate.get("eligible_for_controlled_comparison", False))),
        "reason_excluded_from_comparison": base.first_nonempty(result.get("reason_excluded_from_comparison"), candidate.get("reason_excluded_from_comparison")) or None,
        "model_patch_bytes": len(patch_text.encode("utf-8")) if patch_text else 0,
        "model_patch_sha256": model_patch_sha,
        "code_diff_sha256": model_patch_sha,
        "model_config": model_config,
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "prompt_sha256": base.sha256_text(prompt) if prompt else "",
        "row_source": "generated_raw_trace",
        "dataset_source_stage": "local_generated_raw_source_unfiltered",
    }
    record: dict[str, Any] = {
        "uuid": stable_generated_uuid(trajectory_path, result_path, task_id, rollout_id, teacher, instruction_style),
        "task_id": task_id,
        "teacher": teacher,
        "reward": reward_int,
        "passed": passed,
        "percent_messages_with_reasoning": metrics["percent_messages_with_reasoning"],
        "deepswe_prompt_augmentation": instruction_style == "deepswe",
        "prompt": prompt,
        "messages": messages,
        "tools": trajectory.get("tools") if isinstance(trajectory.get("tools"), list) else [base.BASH_TOOL],
        "reasoning": reasoning,
        "metadata": metadata,
    }
    if patch_text:
        record["model_patch"] = patch_text
    return record, "ok"


def add_index_rows(row: dict[str, Any], line_number: int) -> tuple[dict[str, Any], dict[str, Any]]:
    compact = base.record_to_index(row, line_number)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    full = {
        **compact,
        "has_result_json": metadata.get("has_result_json"),
        "trajectory_missing": metadata.get("trajectory_missing"),
        "trace_build_status": metadata.get("trace_build_status"),
        "raw_source_unfiltered": metadata.get("raw_source_unfiltered"),
        "dataset_source_stage": metadata.get("dataset_source_stage"),
        "source_manifest": metadata.get("source_manifest"),
        "source_manifest_kind": metadata.get("source_manifest_kind"),
        "uses_updated_alignment": metadata.get("uses_updated_alignment"),
        "provider_usage": metadata.get("provider_usage"),
    }
    return compact, full


def write_index_files(dataset_dir: Path, index_rows: list[dict[str, Any]], full_rows: list[dict[str, Any]]) -> None:
    metadata_dir = dataset_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    base.write_index_files(dataset_dir, index_rows)
    with (metadata_dir / "full_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in full_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    task_ids = sorted({row.get("task_id", "") for row in index_rows if row.get("task_id")})
    (metadata_dir / "task_ids.txt").write_text("\n".join(task_ids) + "\n", encoding="utf-8")
    conn = sqlite3.connect(metadata_dir / "full_index.sqlite")
    try:
        conn.execute(
            "CREATE TABLE traces ("
            "line_number TEXT, uuid TEXT, task_id TEXT, teacher TEXT, reward TEXT, passed TEXT, "
            "difficulty TEXT, language TEXT, instruction_style TEXT, rollout_id TEXT, "
            "source_run_root TEXT, trajectory_path TEXT, result_path TEXT, trace_build_status TEXT, "
            "raw_source_unfiltered TEXT)"
        )
        conn.executemany(
            "INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                [
                    str(row.get("line_number", "")),
                    str(row.get("uuid", "")),
                    str(row.get("task_id", "")),
                    str(row.get("teacher", "")),
                    str(row.get("reward", "")),
                    str(row.get("passed", "")),
                    str(row.get("difficulty", "")),
                    str(row.get("language", "")),
                    str(row.get("instruction_style", "")),
                    str(row.get("rollout_id", "")),
                    str(row.get("source_run_root", "")),
                    str(row.get("trajectory_path", "")),
                    str(row.get("result_path", "")),
                    str(row.get("trace_build_status", "")),
                    str(row.get("raw_source_unfiltered", "")),
                ]
                for row in full_rows
            ],
        )
        conn.execute("CREATE INDEX traces_task_id_idx ON traces(task_id)")
        conn.execute("CREATE INDEX traces_source_idx ON traces(source_run_root)")
        conn.execute("CREATE INDEX traces_language_idx ON traces(language)")
        conn.commit()
    finally:
        conn.close()


class ShardedWriter:
    def __init__(self, dataset_dir: Path, shard_size: int):
        self.dataset_dir = dataset_dir
        self.data_dir = dataset_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.buffer: list[dict[str, Any]] = []
        self.shard_index = 0

    def write(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.data_dir / f"train-{self.shard_index:05d}.jsonl.zst"
        write_jsonl_zst(path, self.buffer)
        self.buffer.clear()
        self.shard_index += 1


def summarize_index(index_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_language = Counter(row.get("language", "") or "unknown" for row in index_rows)
    by_difficulty = Counter(row.get("difficulty", "") or "unknown" for row in index_rows)
    by_passed = Counter(str(row.get("passed", False)).lower() for row in index_rows)
    by_style = Counter(row.get("instruction_style", "") or "unknown" for row in index_rows)
    by_source = Counter(row.get("source_run_root", "") or "parent_dataset" for row in index_rows)
    return {
        **base.summarize(index_rows),
        "by_language": dict(sorted(by_language.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "by_passed": dict(sorted(by_passed.items())),
        "by_instruction_style": dict(sorted(by_style.items())),
        "by_source_run_root": dict(sorted(by_source.items())),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output dir: {args.output_dir}")
    (args.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    writer = ShardedWriter(args.output_dir, args.shard_size)
    index_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()
    seen_trajectory_paths: set[str] = set()
    seen_result_paths: set[str] = set()
    skipped = Counter()
    generated_seen_workspaces: set[str] = set()
    generated_candidate_counts: dict[str, int] = {}
    generated_included_counts: Counter[str] = Counter()
    generated_status_counts: Counter[str] = Counter()
    generated_already_present = Counter()

    for row, parent_split in iter_parent_rows(args.parent_dataset):
        uuid = path_text(row.get("uuid"))
        trajectory_path, result_path = row_paths(row)
        key = uuid or trajectory_path or result_path
        if key and key in seen_uuids:
            skipped["parent_duplicate_key"] += 1
            continue
        if trajectory_path and trajectory_path in seen_trajectory_paths:
            skipped["parent_duplicate_trajectory_path"] += 1
            continue
        if result_path and result_path in seen_result_paths:
            skipped["parent_duplicate_result_path"] += 1
            continue
        if uuid:
            seen_uuids.add(uuid)
        if trajectory_path:
            seen_trajectory_paths.add(trajectory_path)
        if result_path:
            seen_result_paths.add(result_path)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metadata.setdefault("raw_source_parent_dataset", str(args.parent_dataset))
        metadata.setdefault("raw_source_parent_split", parent_split)
        row["metadata"] = metadata
        line_number = len(index_rows)
        writer.write(row)
        compact, full = add_index_rows(row, line_number)
        index_rows.append(compact)
        full_rows.append(full)

    print(f"loaded_parent_rows={len(index_rows)} elapsed={time.time() - started:.1f}s", flush=True)

    for run_root in run_roots(args):
        if not run_root.exists():
            skipped["missing_run_root"] += 1
            continue
        candidates = collect_generated_candidates(run_root)
        generated_candidate_counts[str(run_root)] = len(candidates)
        for candidate in candidates:
            if args.limit_generated and sum(generated_included_counts.values()) >= args.limit_generated:
                break
            workspace = workspace_from_candidate(candidate)
            if workspace is None:
                skipped["generated_missing_workspace"] += 1
                continue
            workspace_key = normalize_abs(workspace)
            if workspace_key in generated_seen_workspaces:
                skipped["generated_duplicate_workspace"] += 1
                continue
            generated_seen_workspaces.add(workspace_key)
            result_path = normalize_abs(path_text(candidate.get("result_path")) or workspace / "result.json")
            trajectory_path = normalize_abs(path_text(candidate.get("trajectory_path")) or workspace / "agent" / "mini-swe-agent.trajectory.json")
            if not Path(result_path).exists() and not Path(trajectory_path).exists():
                skipped["generated_planned_not_started"] += 1
                continue
            if trajectory_path and trajectory_path in seen_trajectory_paths:
                generated_already_present[str(run_root)] += 1
                continue
            if result_path and result_path in seen_result_paths:
                generated_already_present[str(run_root)] += 1
                continue
            row, status = build_generated_row(run_root, candidate)
            generated_status_counts[status] += 1
            if row is None:
                skipped[f"generated_{status}"] += 1
                continue
            uuid = path_text(row.get("uuid"))
            if uuid and uuid in seen_uuids:
                generated_already_present[str(run_root)] += 1
                continue
            trajectory_path, result_path = row_paths(row)
            if trajectory_path and trajectory_path in seen_trajectory_paths:
                generated_already_present[str(run_root)] += 1
                continue
            if result_path and result_path in seen_result_paths:
                generated_already_present[str(run_root)] += 1
                continue
            if uuid:
                seen_uuids.add(uuid)
            if trajectory_path:
                seen_trajectory_paths.add(trajectory_path)
            if result_path:
                seen_result_paths.add(result_path)
            line_number = len(index_rows)
            writer.write(row)
            compact, full = add_index_rows(row, line_number)
            index_rows.append(compact)
            full_rows.append(full)
            generated_included_counts[str(run_root)] += 1
        print(
            "scanned_run_root="
            f"{run_root} candidates={len(candidates)} included={generated_included_counts[str(run_root)]} "
            f"already_present={generated_already_present[str(run_root)]} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    writer.flush()
    write_index_files(args.output_dir, index_rows, full_rows)
    summary = summarize_index(index_rows)
    generated_rows = sum(generated_included_counts.values())
    parent_rows = len(index_rows) - generated_rows
    manifest = {
        "name": args.output_dir.name,
        "created_at_unix": time.time(),
        "parent_dataset": str(args.parent_dataset),
        "runs_root": str(args.runs_root),
        "rows": len(index_rows),
        "parent_rows_included": parent_rows,
        "new_generated_rows_appended": generated_rows,
        "generated_candidate_counts": generated_candidate_counts,
        "generated_rows_appended_by_run_root": dict(sorted(generated_included_counts.items())),
        "generated_already_present_by_run_root": dict(sorted(generated_already_present.items())),
        "generated_build_status_counts": dict(sorted(generated_status_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "elapsed_sec": time.time() - started,
        "no_filters_applied_to_generated_rows": [
            "pass_reward",
            "qwen3_token_length",
            "assistant_reasoning_coverage",
            "model_patch_nonempty",
            "per_task_trace_cap",
            "api_calls_positive",
        ],
        **summary,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "build_summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = [
        f"# {args.output_dir.name}",
        "",
        "Raw-source JSONL+zstd dataset. The previous delivered dataset is preserved, and local generated traces are appended without pass, token, reasoning, patch, api-call, or per-task-cap filtering.",
        "",
        "Downstream refinement/training jobs should filter this dataset for their target policy.",
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
