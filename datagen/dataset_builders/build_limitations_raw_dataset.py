#!/usr/bin/env python3
"""Build an append-only raw-source dataset with strict-quality metadata.

The raw data path is intentionally unfiltered: every collectable generated trace
is appended to a new shard. Strict pass/reward/submission/patch/reasoning/API
checks are recorded only in metadata so downstream builders can decide what to
keep without losing diagnostics from failed generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a noninteractive bash command in the task environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}

STRICT_REASONING_THRESHOLD = 0.9
METADATA_OUTPUTS = {
    "new_raw_index.jsonl",
    "strict_quality_index.jsonl",
    "rejected_index.jsonl",
    "compaction_index.jsonl",
    "wave_summaries.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Existing raw dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="New dataset root to create.")
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=[],
        help="Generated run root. May be supplied more than once.",
    )
    parser.add_argument(
        "--compaction-output",
        type=Path,
        action="append",
        default=[],
        help="Compaction output file or directory. May be supplied more than once.",
    )
    parser.add_argument(
        "--append-dataset",
        type=Path,
        action="append",
        default=[],
        help="Additional raw dataset root whose data shards should be included unchanged.",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        action="append",
        default=[],
        help="Direct result.json path to append. May be supplied more than once.",
    )
    parser.add_argument(
        "--trajectory-path",
        type=Path,
        action="append",
        default=[],
        help="Direct trajectory JSON path to append. May be supplied more than once.",
    )
    parser.add_argument(
        "--allow-fallback-walk",
        action="store_true",
        help="If a run root has no result index or manifest TSV rows, walk it for result.json files.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="How to materialize base data shards into the output.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove output before writing.")
    parser.add_argument("--dry-run", action="store_true", help="Plan and summarize without writing files.")
    parser.add_argument(
        "--new-shard-name",
        default="",
        help="Optional filename for the appended raw shard under output/data.",
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Optional manifest name. Defaults to the output directory name.",
    )
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def load_json_dict(path: Path) -> dict[str, Any]:
    try:
        data = load_json(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {
                    "_parse_error": str(exc),
                    "_source_path": str(path),
                    "_source_line_number": line_number,
                }
                continue
            if isinstance(row, dict):
                yield row


def iter_json_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        yield from iter_jsonl(path)
        return
    data = load_json(path)
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                yield row
        return
    if isinstance(data, dict):
        rows = data.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield row
        else:
            yield data


def count_jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "passed", "submitted"}
    return False


def int_from_any(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def float_from_any(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def stable_uuid(parts: Iterable[Any]) -> str:
    text = "\n".join(str(part) for part in parts if part is not None)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def resolve_maybe_container_path(path_text: str, workspace: Path | None) -> Path:
    path = Path(path_text)
    if path.exists() or workspace is None:
        return path
    if path.is_absolute():
        try:
            return workspace / path.relative_to("/workspace")
        except ValueError:
            return path
    return workspace / path


def read_text_if_small(path: Path, max_bytes: int = 4_000_000) -> str:
    try:
        if path.exists() and path.stat().st_size <= max_bytes:
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return ""


def text_from_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces: list[str] = []
        for part in value:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                for key in ("text", "content", "value", "thinking", "reasoning"):
                    if key in part:
                        pieces.append(text_from_content(part[key]))
                        break
        return "\n".join(piece for piece in pieces if piece)
    if isinstance(value, dict):
        for key in ("text", "content", "value", "thinking", "reasoning"):
            if key in value:
                return text_from_content(value[key])
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def message_reasoning_content(message: dict[str, Any]) -> list[str]:
    contents: list[str] = []
    candidates: list[Any] = []
    for key in ("reasoning_content", "reasoning", "reasoning_details", "thinking", "thought"):
        candidates.append(message.get(key))
    provider = message.get("provider_specific_fields")
    if isinstance(provider, dict):
        for key in ("reasoning_content", "reasoning", "reasoning_details", "thinking", "thought"):
            candidates.append(provider.get(key))
    extra = message.get("extra")
    response = extra.get("response") if isinstance(extra, dict) else None
    choices = response.get("choices") if isinstance(response, dict) else None
    if isinstance(choices, list):
        for choice in choices:
            choice_message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(choice_message, dict):
                for key in ("reasoning_content", "reasoning", "reasoning_details", "thinking", "thought"):
                    candidates.append(choice_message.get(key))
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type", "")).lower()
            if part_type in {"thinking", "reasoning"}:
                candidates.append(part.get("text", part.get("content", part.get("thinking", part.get("reasoning")))))
    for value in candidates:
        text = text_from_content(value).strip()
        if text:
            contents.append(text)
    return contents


def collect_message_reasoning(messages: list[Any]) -> list[dict[str, Any]]:
    reasoning: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        for content in message_reasoning_content(message):
            key = (index, content)
            if key in seen:
                continue
            seen.add(key)
            reasoning.append({"message_index": index, "path": f"messages[{index}]", "content": content})
    return reasoning


def prompt_from_messages(messages: list[Any]) -> str:
    for message in messages:
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "user":
            return text_from_content(message.get("content"))
    return ""


def reasoning_metrics(messages: list[Any], reasoning: list[dict[str, Any]]) -> dict[str, Any]:
    top_indices = {
        item.get("message_index")
        for item in reasoning
        if isinstance(item.get("message_index"), int) and str(item.get("content") or "").strip()
    }
    assistant_count = 0
    with_reasoning = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or str(message.get("role", "")).lower() != "assistant":
            continue
        assistant_count += 1
        if index in top_indices or any(message_reasoning_content(message)):
            with_reasoning += 1
    return {
        "assistant_message_count": assistant_count,
        "assistant_messages_with_reasoning": with_reasoning,
        "assistant_messages_without_reasoning": assistant_count - with_reasoning,
        "percent_messages_with_reasoning": (with_reasoning / assistant_count) if assistant_count else 0.0,
    }


def extract_submit_marker(messages: list[Any]) -> bool:
    marker = "complete_task_and_submit_final_output"
    for message in messages:
        if not isinstance(message, dict) or str(message.get("role", "")).lower() != "assistant":
            continue
        if marker in text_from_content(message.get("content")).lower():
            return True
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if marker in json.dumps(call, ensure_ascii=False, default=str).lower():
                    return True
    return False


def workspace_from_result_path(result_path: Path | None) -> Path | None:
    if result_path is None:
        return None
    return result_path.parent


def workspace_from_trajectory_path(trajectory_path: Path | None) -> Path | None:
    if trajectory_path is None:
        return None
    if trajectory_path.parent.name == "agent":
        return trajectory_path.parent.parent
    return trajectory_path.parent


def result_path_for_workspace(workspace: Path) -> Path:
    return workspace / "result.json"


def trajectory_path_for_workspace(workspace: Path) -> Path:
    return workspace / "agent" / "mini-swe-agent.trajectory.json"


def patch_text_from_sources(result: dict[str, Any], trajectory: dict[str, Any], workspace: Path | None) -> tuple[str, str]:
    patch_candidates: list[Path] = []
    patch_path_text = first_nonempty(result.get("patch_path"), result.get("model_patch_path"))
    if patch_path_text:
        patch_candidates.append(resolve_maybe_container_path(patch_path_text, workspace))
    if workspace is not None:
        patch_candidates.extend(
            [
                workspace / "model.patch",
                workspace / "logs" / "artifacts" / "model.patch",
                workspace / "patch.txt",
            ]
        )
    for path in patch_candidates:
        text = read_text_if_small(path)
        if text:
            return text, str(path)
    for key in ("model_patch", "patch", "submission"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value, patch_path_text
    info = trajectory.get("info") if isinstance(trajectory.get("info"), dict) else {}
    submission = info.get("submission")
    if isinstance(submission, str) and submission.strip():
        return submission, patch_path_text
    return "", patch_path_text


def build_record_from_paths(
    *,
    source_kind: str,
    source_run_root: Path | None,
    result_path: Path | None,
    trajectory_path: Path | None,
    seed_row: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    workspace = workspace_from_result_path(result_path) or workspace_from_trajectory_path(trajectory_path)
    if result_path is None and workspace is not None:
        result_path = result_path_for_workspace(workspace)
    if trajectory_path is None and workspace is not None:
        trajectory_path = trajectory_path_for_workspace(workspace)

    reject_base = {
        "source_kind": source_kind,
        "source_run_root": str(source_run_root) if source_run_root is not None else "",
        "result_path": str(result_path) if result_path is not None else "",
        "trajectory_path": str(trajectory_path) if trajectory_path is not None else "",
        "task_id": first_nonempty(seed_row.get("instance_id"), seed_row.get("task_id")),
        "rollout_id": first_nonempty(seed_row.get("rollout_id")),
        "teacher": first_nonempty(seed_row.get("model"), seed_row.get("teacher")),
        "collection_reject": True,
    }
    if trajectory_path is None:
        return None, {**reject_base, "reject_reasons": ["missing_trajectory_path"]}
    if not trajectory_path.exists() or trajectory_path.stat().st_size <= 2:
        return None, {**reject_base, "reject_reasons": ["trajectory_missing_or_empty"]}

    trajectory = load_json_dict(trajectory_path)
    raw_messages = trajectory.get("messages")
    if not isinstance(raw_messages, list):
        return None, {**reject_base, "reject_reasons": ["trajectory_missing_messages"]}

    result = dict(seed_row)
    if result_path is not None and result_path.exists():
        result.update(load_json_dict(result_path))
    info = trajectory.get("info") if isinstance(trajectory.get("info"), dict) else {}
    model_stats = info.get("model_stats") if isinstance(info.get("model_stats"), dict) else {}
    config = info.get("config") if isinstance(info.get("config"), dict) else {}
    model_config = config.get("model") if isinstance(config.get("model"), dict) else {}

    reasoning = collect_message_reasoning(raw_messages)
    metrics = reasoning_metrics(raw_messages, reasoning)
    prompt = prompt_from_messages(raw_messages)
    patch_text, patch_path_text = patch_text_from_sources(result, trajectory, workspace)
    reward = result.get("reward", 0)
    reward_int = int_from_any(reward, 1 if bool_from_any(reward) else 0)
    task_id = first_nonempty(result.get("instance_id"), result.get("task_id"), seed_row.get("instance_id"), seed_row.get("task_id"))
    rollout_id = first_nonempty(result.get("rollout_id"), seed_row.get("rollout_id"))
    teacher = first_nonempty(result.get("model"), result.get("teacher"), seed_row.get("model"), seed_row.get("teacher"))
    agent_exception = result.get("agent_exception") if isinstance(result.get("agent_exception"), dict) else {}
    record_metadata = {
        "dataset": "",
        "source": "deepswe-generated-raw",
        "row_source": source_kind,
        "source_run_root": str(source_run_root) if source_run_root is not None else "",
        "workspace": str(workspace) if workspace is not None else "",
        "trajectory_path": str(trajectory_path),
        "result_path": str(result_path) if result_path is not None else "",
        "patch_path": patch_path_text,
        "task_id": task_id,
        "instance_id": task_id,
        "rollout_id": rollout_id,
        "repo": first_nonempty(result.get("repo"), seed_row.get("repo")),
        "difficulty": first_nonempty(result.get("difficulty"), seed_row.get("difficulty")),
        "language": first_nonempty(result.get("language"), seed_row.get("language")),
        "instruction_style": first_nonempty(result.get("instruction_style"), seed_row.get("instruction_style")),
        "benchmark_profile": first_nonempty(result.get("benchmark_profile"), seed_row.get("benchmark_profile")),
        "teacher": teacher,
        "model": teacher,
        "litellm_model": first_nonempty(result.get("litellm_model"), seed_row.get("litellm_model")),
        "passed": bool_from_any(result.get("passed", reward_int == 1)),
        "reward": reward_int,
        "agent_exit_status": first_nonempty(result.get("agent_exit_status"), info.get("exit_status")),
        "agent_exception_type": first_nonempty(result.get("agent_exception_type"), agent_exception.get("type")),
        "api_calls": int_from_any(result.get("api_calls", model_stats.get("api_calls", 0))),
        "cost_usd": float_from_any(result.get("cost_usd", model_stats.get("instance_cost", 0.0))),
        "message_count": len(raw_messages),
        "assistant_message_count": metrics["assistant_message_count"],
        "assistant_messages_with_reasoning": metrics["assistant_messages_with_reasoning"],
        "assistant_messages_without_reasoning": metrics["assistant_messages_without_reasoning"],
        "percent_messages_with_reasoning": metrics["percent_messages_with_reasoning"],
        "has_any_reasoning": metrics["assistant_messages_with_reasoning"] > 0,
        "has_all_assistant_reasoning": metrics["assistant_message_count"] > 0
        and metrics["assistant_messages_with_reasoning"] == metrics["assistant_message_count"],
        "reasoning_turns": len(reasoning),
        "reasoning_chars": sum(len(str(item.get("content", ""))) for item in reasoning),
        "trajectory_chars": len(json.dumps(raw_messages, ensure_ascii=False, separators=(",", ":"))),
        "trajectory_bytes": trajectory_path.stat().st_size,
        "trajectory_format": first_nonempty(trajectory.get("trajectory_format")),
        "mini_swe_agent_version": first_nonempty(info.get("mini_version")),
        "model_config": model_config,
        "model_patch_bytes": len(patch_text.encode("utf-8")) if patch_text else 0,
        "model_patch_sha256": sha256_text(patch_text) if patch_text else "",
        "code_diff_sha256": sha256_text(patch_text) if patch_text else "",
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "prompt_sha256": sha256_text(prompt) if prompt else "",
        "has_submit_marker": extract_submit_marker(raw_messages),
    }
    record = {
        "uuid": stable_uuid([str(trajectory_path), task_id, rollout_id, teacher, source_kind]),
        "task_id": task_id,
        "teacher": teacher,
        "reward": reward_int,
        "passed": bool_from_any(record_metadata["passed"]),
        "percent_messages_with_reasoning": metrics["percent_messages_with_reasoning"],
        "deepswe_prompt_augmentation": record_metadata["instruction_style"] == "deepswe",
        "prompt": prompt,
        "messages": raw_messages,
        "tools": [BASH_TOOL],
        "reasoning": reasoning,
        "metadata": record_metadata,
    }
    if patch_text:
        record["model_patch"] = patch_text
    return record, None


def record_index(row: dict[str, Any], line_number: int | None, new_line_number: int) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    prompt = row.get("prompt") if isinstance(row.get("prompt"), str) else metadata.get("prompt", "")
    return {
        "line_number": line_number,
        "new_line_number": new_line_number,
        "uuid": first_nonempty(row.get("uuid")),
        "task_id": first_nonempty(row.get("task_id"), metadata.get("task_id"), metadata.get("instance_id")),
        "teacher": first_nonempty(row.get("teacher"), metadata.get("teacher"), metadata.get("model")),
        "reward": int_from_any(row.get("reward", metadata.get("reward", 0))),
        "passed": bool_from_any(row.get("passed", metadata.get("passed", False))),
        "percent_messages_with_reasoning": float_from_any(
            row.get("percent_messages_with_reasoning", metadata.get("percent_messages_with_reasoning", 0.0))
        ),
        "assistant_message_count": int_from_any(metadata.get("assistant_message_count", 0)),
        "assistant_messages_with_reasoning": int_from_any(metadata.get("assistant_messages_with_reasoning", 0)),
        "assistant_messages_without_reasoning": int_from_any(metadata.get("assistant_messages_without_reasoning", 0)),
        "deepswe_prompt_augmentation": bool_from_any(
            row.get("deepswe_prompt_augmentation", metadata.get("deepswe_prompt_augmentation", False))
        ),
        "instruction_style": first_nonempty(metadata.get("instruction_style")),
        "difficulty": first_nonempty(metadata.get("difficulty")),
        "language": first_nonempty(metadata.get("language")),
        "repo": first_nonempty(metadata.get("repo")),
        "rollout_id": first_nonempty(metadata.get("rollout_id")),
        "source": first_nonempty(metadata.get("source")),
        "row_source": first_nonempty(metadata.get("row_source")),
        "source_run_root": first_nonempty(metadata.get("source_run_root")),
        "trajectory_path": first_nonempty(metadata.get("trajectory_path")),
        "result_path": first_nonempty(metadata.get("result_path")),
        "patch_path": first_nonempty(metadata.get("patch_path")),
        "model_patch_sha256": first_nonempty(metadata.get("model_patch_sha256"), metadata.get("code_diff_sha256")),
        "model_patch_bytes": int_from_any(metadata.get("model_patch_bytes", 0)),
        "message_count": int_from_any(metadata.get("message_count", 0)),
        "trajectory_chars": int_from_any(metadata.get("trajectory_chars", 0)),
        "trajectory_bytes": int_from_any(metadata.get("trajectory_bytes", 0)),
        "prompt_sha256": sha256_text(prompt) if prompt else first_nonempty(metadata.get("prompt_sha256")),
        "prompt_chars": len(prompt) if prompt else int_from_any(metadata.get("prompt_chars", 0)),
        "prompt_preview": prompt[:500].replace("\n", "\\n") if prompt else "",
        "agent_exit_status": first_nonempty(metadata.get("agent_exit_status")),
        "agent_exception_type": first_nonempty(metadata.get("agent_exception_type")),
        "api_calls": int_from_any(metadata.get("api_calls", 0)),
        "cost_usd": float_from_any(metadata.get("cost_usd", 0.0)),
        "has_submit_marker": bool_from_any(metadata.get("has_submit_marker", False)),
    }


def strict_quality(index_row: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if not bool_from_any(index_row.get("passed")):
        reasons.append("not_passed")
    if int_from_any(index_row.get("reward")) <= 0:
        reasons.append("reward_not_positive")
    if first_nonempty(index_row.get("agent_exit_status")) != "Submitted":
        reasons.append("not_submitted")
    if int_from_any(index_row.get("model_patch_bytes")) <= 0:
        reasons.append("empty_patch")
    if float_from_any(index_row.get("percent_messages_with_reasoning")) < STRICT_REASONING_THRESHOLD:
        reasons.append("reasoning_under_90pct")
    if int_from_any(index_row.get("api_calls")) <= 0:
        reasons.append("api_calls_not_positive")
    if int_from_any(index_row.get("assistant_message_count")) <= 0:
        reasons.append("missing_assistant_messages")
    return {
        **index_row,
        "strict_quality_passed": not reasons,
        "strict_quality_reject_reasons": reasons,
        "strict_quality_thresholds": {
            "reward_positive": True,
            "agent_exit_status": "Submitted",
            "model_patch_bytes_min": 1,
            "percent_messages_with_reasoning_min": STRICT_REASONING_THRESHOLD,
            "api_calls_min": 1,
        },
    }


def result_index_paths(run_root: Path) -> list[Path]:
    manifest = run_root / "manifest"
    candidates = [
        manifest / "result_index.jsonl",
        run_root / "result_index.jsonl",
        run_root / "metadata" / "result_index.jsonl",
    ]
    if manifest.is_dir():
        candidates.extend(sorted(manifest.glob("*result*index*.jsonl")))
    seen: set[Path] = set()
    paths: list[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def manifest_tsv_paths(run_root: Path) -> list[Path]:
    manifest = run_root / "manifest"
    if not manifest.is_dir():
        return []
    return sorted(manifest.glob("*.tsv"))


def rows_from_manifest_tsv(path: Path, run_root: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for manifest_line_number, raw in enumerate(handle):
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            workspace = Path(parts[4])
            if not workspace.is_absolute():
                workspace = (run_root / workspace).resolve()
            yield {
                "source_kind": "manifest_tsv",
                "source_manifest": str(path),
                "source_manifest_line_number": manifest_line_number,
                "result_path": str(result_path_for_workspace(workspace)),
                "trajectory_path": str(trajectory_path_for_workspace(workspace)),
                "rollout_id": parts[1] if len(parts) > 1 else "",
                "instance_id": parts[2] if len(parts) > 2 else "",
                "difficulty": parts[11] if len(parts) > 11 else "",
                "language": parts[12] if len(parts) > 12 else "",
                "instruction_style": parts[13] if len(parts) > 13 else "",
                "repo": parts[14] if len(parts) > 14 else "",
                "model": parts[6] if len(parts) > 6 else "",
                "litellm_model": parts[7] if len(parts) > 7 else "",
            }


def direct_result_row(result_path: Path) -> dict[str, Any]:
    result = load_json_dict(result_path)
    trajectory = first_nonempty(result.get("trajectory_path"))
    if not trajectory:
        trajectory = str(trajectory_path_for_workspace(result_path.parent))
    return {**result, "result_path": str(result_path), "trajectory_path": trajectory, "source_kind": "result_json"}


def direct_trajectory_row(trajectory_path: Path) -> dict[str, Any]:
    workspace = workspace_from_trajectory_path(trajectory_path)
    result_path = result_path_for_workspace(workspace) if workspace is not None else None
    result = load_json_dict(result_path) if result_path is not None and result_path.exists() else {}
    return {
        **result,
        "result_path": str(result_path) if result_path is not None else "",
        "trajectory_path": str(trajectory_path),
        "source_kind": "trajectory_path",
    }


def collection_rows_for_run_root(run_root: Path, allow_fallback_walk: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    index_paths = result_index_paths(run_root)
    for path in index_paths:
        for row in iter_jsonl(path):
            rows.append({**row, "source_kind": "result_index", "source_index_path": str(path)})
    if rows:
        return rows, diagnostics

    for path in manifest_tsv_paths(run_root):
        rows.extend(rows_from_manifest_tsv(path, run_root))
    if rows:
        return rows, diagnostics

    if allow_fallback_walk:
        for result_path in sorted(run_root.rglob("result.json")):
            rows.append(direct_result_row(result_path))
        if rows:
            diagnostics.append(
                {
                    "run_root": str(run_root),
                    "fallback_walk": True,
                    "result_json_rows": len(rows),
                }
            )
            return rows, diagnostics

    diagnostics.append(
        {
            "run_root": str(run_root),
            "collection_reject": True,
            "reject_reasons": ["no_result_index_or_manifest_tsv"],
        }
    )
    return rows, diagnostics


def discover_wave_summary_paths(root: Path) -> list[Path]:
    candidates: list[Path] = []
    if root.is_file():
        if "wave" in root.name and root.suffix in {".json", ".jsonl"}:
            candidates.append(root)
        return candidates
    for base in (root, root / "manifest", root / "metadata"):
        if not base.is_dir():
            continue
        candidates.extend(sorted(base.glob("wave_summaries.jsonl")))
        candidates.extend(sorted(base.glob("*wave*summary*.jsonl")))
        candidates.extend(sorted(base.glob("*wave*summary*.json")))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def read_wave_summaries(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in discover_wave_summary_paths(root):
        for row in iter_json_rows(path):
            rows.append({**row, "source_path": str(path), "source_root": str(root)})
    return rows


def compaction_index_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates: list[Path] = []
    for base in (path, path / "metadata", path / "manifest"):
        if not base.is_dir():
            continue
        candidates.extend(sorted(base.glob("compaction_index.jsonl")))
        candidates.extend(sorted(base.glob("*compaction_index*.jsonl")))
        candidates.extend(sorted(base.glob("compaction_records.jsonl")))
        candidates.extend(sorted(base.glob("*compaction_records*.jsonl")))
    seen: set[Path] = set()
    out: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def collect_compaction(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    compaction_rows: list[dict[str, Any]] = []
    collection_rows: list[dict[str, Any]] = []
    for index_path in compaction_index_paths(path):
        is_record_path = "record" in index_path.name
        for row in iter_json_rows(index_path):
            tagged = {**row, "compaction_source_path": str(index_path), "compaction_output_root": str(path)}
            if not is_record_path:
                compaction_rows.append(tagged)
            raw_record = None
            for key in ("raw_record", "record", "sft_record"):
                if isinstance(row.get(key), dict):
                    raw_record = dict(row[key])
                    break
            if raw_record is not None:
                metadata = raw_record.get("metadata") if isinstance(raw_record.get("metadata"), dict) else {}
                metadata = {
                    **metadata,
                    "source": first_nonempty(metadata.get("source"), "deepswe-compaction"),
                    "row_source": first_nonempty(metadata.get("row_source"), "compaction_record"),
                    "compaction_source_path": str(index_path),
                }
                raw_record["metadata"] = metadata
                collection_rows.append({"source_kind": "compaction_record", "raw_record": raw_record})
            elif isinstance(row.get("messages"), list):
                raw_record = dict(row)
                metadata = raw_record.get("metadata") if isinstance(raw_record.get("metadata"), dict) else {}
                metadata = {
                    **metadata,
                    "source": first_nonempty(metadata.get("source"), "deepswe-compaction"),
                    "row_source": first_nonempty(metadata.get("row_source"), "compaction_messages_row"),
                    "compaction_source_path": str(index_path),
                }
                raw_record["metadata"] = metadata
                collection_rows.append({"source_kind": "compaction_record", "raw_record": raw_record})
            elif is_record_path and (row.get("trajectory_path") or row.get("result_path")):
                collection_rows.append({**row, "source_kind": "compaction_index"})
    return compaction_rows, collection_rows


def base_data_files(base: Path) -> list[Path]:
    manifest = load_json_dict(base / "manifest.json")
    names: list[str] = []
    data_files = manifest.get("data_files")
    if isinstance(data_files, list):
        names.extend(str(item) for item in data_files if item)
    data_file = manifest.get("data_file")
    if data_file:
        names.append(str(data_file))
    paths: list[Path] = []
    for name in names:
        path = Path(name)
        if not path.is_absolute():
            path = base / path
        if path.exists() and path.is_file() and is_under(path, base):
            paths.append(path)
    if not paths:
        data_dir = base / "data"
        if data_dir.is_dir():
            for suffix in ("*.jsonl.zst", "*.jsonl", "*.parquet"):
                paths.extend(sorted(data_dir.rglob(suffix)))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def infer_base_rows(base: Path) -> int | None:
    for manifest_name in ("manifest.json", "dataset_info.json", "build_summary.json"):
        manifest = load_json_dict(base / manifest_name)
        for key in ("total_rows", "rows", "records", "base_raw_source_total_rows"):
            value = manifest.get(key)
            if isinstance(value, int):
                return value
    metadata = base / "metadata"
    for name in ("index.jsonl", "parent_index.jsonl"):
        path = metadata / name
        if path.exists():
            return count_jsonl_rows(path)
    index_paths = sorted(metadata.glob("*index.jsonl")) if metadata.is_dir() else []
    max_line = -1
    saw_line = False
    total_rows = 0
    for path in index_paths:
        if path.name in METADATA_OUTPUTS:
            continue
        for row in iter_jsonl(path):
            total_rows += 1
            line = row.get("line_number")
            if isinstance(line, int):
                saw_line = True
                max_line = max(max_line, line)
    if saw_line:
        return max_line + 1
    if total_rows:
        return total_rows
    return None


def next_shard_name(base_files: list[Path], new_shard_name: str) -> str:
    if new_shard_name:
        return new_shard_name
    max_idx = -1
    use_zst = any(path.name.endswith(".jsonl.zst") for path in base_files)
    for path in base_files:
        match = re.search(r"(\d+)(?=\.jsonl(?:\.zst)?$)", path.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    suffix = ".jsonl.zst" if use_zst else ".jsonl"
    return f"train-{max_idx + 1:05d}{suffix}" if max_idx >= 0 else f"appended-00000{suffix}"


def materialize_base_files(
    base: Path,
    output: Path,
    files: list[Path],
    *,
    copy_mode: str,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    copied: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    for src in files:
        rel = src.relative_to(base)
        dest = output / rel
        action = "copy"
        if copy_mode in {"auto", "hardlink"}:
            action = "hardlink"
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if action == "hardlink":
                try:
                    os.link(src, dest)
                except OSError:
                    if copy_mode == "hardlink":
                        raise
                    action = "copy"
                    shutil.copy2(src, dest)
            else:
                shutil.copy2(src, dest)
        actions[action] += 1
        copied.append(
            {
                "source_path": str(src),
                "path": str(rel),
                "bytes": src.stat().st_size,
                "materialization": action,
            }
        )
    return copied, actions


def copy_base_metadata(base: Path, output: Path, dry_run: bool) -> list[dict[str, Any]]:
    metadata = base / "metadata"
    if not metadata.is_dir():
        return []
    copied: list[dict[str, Any]] = []
    for src in sorted(path for path in metadata.iterdir() if path.is_file()):
        dest_name = src.name if src.name not in METADATA_OUTPUTS else f"base_{src.name}"
        dest = output / "metadata" / dest_name
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        copied.append({"source_path": str(src), "path": str(dest.relative_to(output)), "bytes": src.stat().st_size})
    return copied


def materialize_append_dataset(
    dataset: Path,
    output: Path,
    *,
    dataset_index: int,
    copy_mode: str,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]], int | None]:
    files = base_data_files(dataset)
    rows = infer_base_rows(dataset)
    copied: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    for src in files:
        rel = Path("data") / f"append{dataset_index:02d}-{src.name}"
        dest = output / rel
        action = "copy"
        if copy_mode in {"auto", "hardlink"}:
            action = "hardlink"
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if action == "hardlink":
                try:
                    os.link(src, dest)
                except OSError:
                    if copy_mode == "hardlink":
                        raise
                    action = "copy"
                    shutil.copy2(src, dest)
            else:
                shutil.copy2(src, dest)
        actions[action] += 1
        copied.append(
            {
                "append_dataset": str(dataset),
                "source_path": str(src),
                "path": str(rel),
                "bytes": src.stat().st_size,
                "materialization": action,
            }
        )

    copied_metadata: list[dict[str, Any]] = []
    metadata = dataset / "metadata"
    if metadata.is_dir():
        for src in sorted(path for path in metadata.iterdir() if path.is_file()):
            dest = output / "metadata" / f"append{dataset_index:02d}_{src.name}"
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            copied_metadata.append(
                {
                    "append_dataset": str(dataset),
                    "source_path": str(src),
                    "path": str(dest.relative_to(output)),
                    "bytes": src.stat().st_size,
                }
            )
    return copied, actions, copied_metadata, rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps(row) + "\n")
            count += 1
    return count


def write_new_shard(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    uncompressed_bytes = 0
    if path.name.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("zstandard is required to write .zst shards") from exc
        cctx = zstd.ZstdCompressor(level=3, threads=0)
        with path.open("wb") as raw:
            with cctx.stream_writer(raw) as compressor:
                for row in rows:
                    payload = (json_dumps(row) + "\n").encode("utf-8")
                    uncompressed_bytes += len(payload)
                    compressor.write(payload)
    else:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = json_dumps(row) + "\n"
                uncompressed_bytes += len(payload.encode("utf-8"))
                handle.write(payload)
    return {
        "path": str(path),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "uncompressed_bytes": uncompressed_bytes,
    }


def raw_record_from_compaction(row: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    record = dict(row["raw_record"])
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    metadata = {**metadata, "dataset": dataset_name}
    record["metadata"] = metadata
    if "uuid" not in record:
        record["uuid"] = stable_uuid([metadata.get("compaction_source_path"), metadata.get("task_id"), metadata.get("trajectory_path")])
    return record


def build_new_records(
    args: argparse.Namespace,
    dataset_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows_to_collect: list[tuple[dict[str, Any], Path | None]] = []
    collection_rejects: list[dict[str, Any]] = []
    compaction_rows: list[dict[str, Any]] = []
    wave_rows: list[dict[str, Any]] = []

    for run_root in args.run_root:
        rows, diagnostics = collection_rows_for_run_root(run_root, args.allow_fallback_walk)
        rows_to_collect.extend((row, run_root) for row in rows)
        collection_rejects.extend(diagnostics)
        wave_rows.extend(read_wave_summaries(run_root))

    for result_path in args.result_json:
        rows_to_collect.append((direct_result_row(result_path), None))

    for trajectory_path in args.trajectory_path:
        rows_to_collect.append((direct_trajectory_row(trajectory_path), None))

    for compaction_output in args.compaction_output:
        rows, collection_rows = collect_compaction(compaction_output)
        compaction_rows.extend(rows)
        rows_to_collect.extend((row, compaction_output if compaction_output.is_dir() else compaction_output.parent) for row in collection_rows)
        wave_rows.extend(read_wave_summaries(compaction_output))

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for seed_row, source_root in rows_to_collect:
        raw_record = seed_row.get("raw_record")
        if isinstance(raw_record, dict):
            record = raw_record_from_compaction(seed_row, dataset_name)
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            key = first_nonempty(record.get("uuid"), metadata.get("trajectory_path"), metadata.get("result_path"))
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            records.append(record)
            continue

        result_path_text = first_nonempty(seed_row.get("result_path"))
        trajectory_path_text = first_nonempty(seed_row.get("trajectory_path"))
        result_path = Path(result_path_text) if result_path_text else None
        trajectory_path = Path(trajectory_path_text) if trajectory_path_text else None
        source_kind = first_nonempty(seed_row.get("source_kind"), "generated_trace")
        key = first_nonempty(trajectory_path_text, result_path_text, seed_row.get("instance_id"), seed_row.get("task_id"))
        if key and key in seen:
            continue
        record, reject = build_record_from_paths(
            source_kind=source_kind,
            source_run_root=source_root,
            result_path=result_path,
            trajectory_path=trajectory_path,
            seed_row=seed_row,
        )
        if reject is not None:
            collection_rejects.append(reject)
            continue
        if record is None:
            continue
        if key:
            seen.add(key)
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        metadata["dataset"] = dataset_name
        record["metadata"] = metadata
        records.append(record)

    return records, collection_rejects, compaction_rows, wave_rows


def summarize_indexes(strict_rows: list[dict[str, Any]], rejected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_language = Counter(first_nonempty(row.get("language")) for row in strict_rows)
    by_source = Counter(first_nonempty(row.get("row_source")) for row in strict_rows)
    by_passed = Counter(str(bool_from_any(row.get("passed"))).lower() for row in strict_rows)
    reject_reasons: Counter[str] = Counter()
    for row in rejected_rows:
        reasons = row.get("strict_quality_reject_reasons") or row.get("reject_reasons") or []
        for reason in reasons:
            reject_reasons[str(reason)] += 1
    return {
        "new_rows": len(strict_rows),
        "strict_quality_rows": sum(1 for row in strict_rows if row.get("strict_quality_passed")),
        "rejected_rows": len(rejected_rows),
        "unique_new_tasks": len({row.get("task_id") for row in strict_rows if row.get("task_id")}),
        "new_by_language": dict(sorted(by_language.items())),
        "new_by_source": dict(sorted(by_source.items())),
        "new_by_passed": dict(sorted(by_passed.items())),
        "reject_reason_counts": dict(sorted(reject_reasons.items())),
    }


def remove_output_for_overwrite(output: Path, overwrite: bool, dry_run: bool) -> None:
    if not output.exists():
        return
    if not overwrite:
        raise SystemExit(f"Output already exists; pass --overwrite to rebuild: {output}")
    if not dry_run:
        shutil.rmtree(output)


def main() -> int:
    args = parse_args()
    base = args.base.resolve()
    output = args.output.resolve()
    dataset_name = args.dataset_name or output.name
    if not base.exists():
        raise SystemExit(f"Base dataset does not exist: {base}")
    append_datasets = [path.resolve() for path in args.append_dataset]
    for dataset in append_datasets:
        if not dataset.exists():
            raise SystemExit(f"Append dataset does not exist: {dataset}")

    remove_output_for_overwrite(output, args.overwrite, args.dry_run)
    base_files = base_data_files(base)
    if not base_files:
        raise SystemExit(f"No base data files found under {base}")

    base_rows = infer_base_rows(base)
    new_records, collection_rejects, compaction_rows, wave_rows = build_new_records(args, dataset_name)
    shard_name = next_shard_name(base_files, args.new_shard_name)
    new_shard_rel = Path("data") / shard_name

    copied_base, base_actions = materialize_base_files(
        base,
        output,
        base_files,
        copy_mode=args.copy_mode,
        dry_run=args.dry_run,
    )
    copied_metadata = [] if args.dry_run else copy_base_metadata(base, output, args.dry_run)
    appended_data_files: list[dict[str, Any]] = []
    appended_metadata_files: list[dict[str, Any]] = []
    appended_actions: Counter[str] = Counter()
    appended_dataset_rows = 0
    appended_dataset_summaries: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(append_datasets):
        copied_append, append_actions, append_metadata, append_rows = materialize_append_dataset(
            dataset,
            output,
            dataset_index=dataset_index,
            copy_mode=args.copy_mode,
            dry_run=args.dry_run,
        )
        appended_data_files.extend(copied_append)
        appended_metadata_files.extend(append_metadata)
        appended_actions.update(append_actions)
        if append_rows is not None:
            appended_dataset_rows += append_rows
        appended_dataset_summaries.append(
            {
                "path": str(dataset),
                "rows": append_rows,
                "data_files": len(copied_append),
                "metadata_files": len(append_metadata),
            }
        )

    base_offset = (base_rows if base_rows is not None else 0) + appended_dataset_rows
    new_index_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for new_line, record in enumerate(new_records):
        global_line = base_offset + new_line if base_rows is not None else None
        index_row = record_index(record, global_line, new_line)
        new_index_rows.append(index_row)
        quality_row = strict_quality(index_row)
        strict_rows.append(quality_row)
        if not quality_row["strict_quality_passed"]:
            rejected_rows.append(quality_row)

    rejected_rows.extend(collection_rejects)
    strict_summary = summarize_indexes(strict_rows, rejected_rows)

    if args.dry_run:
        new_shard = {
            "path": str(output / new_shard_rel),
            "rows": len(new_records),
            "bytes": None,
            "uncompressed_bytes": None,
            "dry_run": True,
        }
    else:
        (output / "data").mkdir(parents=True, exist_ok=True)
        (output / "metadata").mkdir(parents=True, exist_ok=True)
        new_shard = write_new_shard(output / new_shard_rel, new_records)
        new_shard["path"] = str(new_shard_rel)
        write_jsonl(output / "metadata" / "new_raw_index.jsonl", new_index_rows)
        write_jsonl(output / "metadata" / "strict_quality_index.jsonl", strict_rows)
        write_jsonl(output / "metadata" / "rejected_index.jsonl", rejected_rows)
        if compaction_rows:
            write_jsonl(output / "metadata" / "compaction_index.jsonl", compaction_rows)
        write_jsonl(output / "metadata" / "wave_summaries.jsonl", wave_rows)

    manifest = {
        "name": dataset_name,
        "created_at_unix": time.time(),
        "builder": str(Path(__file__).resolve()),
        "base_dataset": str(base),
        "output": str(output),
        "dry_run": bool(args.dry_run),
        "append_only_raw_source": True,
        "raw_data_filtering": "none",
        "strict_quality_filtering": "metadata_only",
        "strict_quality_rule": (
            "passed=true, reward>0, agent_exit_status=Submitted, model_patch_bytes>0, "
            "percent_messages_with_reasoning>=0.9, api_calls>0, assistant_message_count>0"
        ),
        "base_rows": base_rows,
        "append_datasets": appended_dataset_summaries,
        "append_dataset_rows": appended_dataset_rows,
        "new_rows": len(new_records),
        "total_rows": (base_rows + appended_dataset_rows + len(new_records)) if base_rows is not None else None,
        "base_data_files": copied_base,
        "base_data_materialization_counts": dict(sorted(base_actions.items())),
        "base_metadata_files_copied": copied_metadata,
        "append_dataset_data_files": appended_data_files,
        "append_dataset_data_materialization_counts": dict(sorted(appended_actions.items())),
        "append_dataset_metadata_files_copied": appended_metadata_files,
        "new_shard": new_shard,
        "run_roots": [str(path.resolve()) for path in args.run_root],
        "compaction_outputs": [str(path.resolve()) for path in args.compaction_output],
        "append_dataset_roots": [str(path) for path in append_datasets],
        "direct_result_json": [str(path.resolve()) for path in args.result_json],
        "direct_trajectory_paths": [str(path.resolve()) for path in args.trajectory_path],
        "metadata": {
            "new_raw_index": "metadata/new_raw_index.jsonl",
            "strict_quality_index": "metadata/strict_quality_index.jsonl",
            "rejected_index": "metadata/rejected_index.jsonl",
            "compaction_index": "metadata/compaction_index.jsonl" if compaction_rows else None,
            "wave_summaries": "metadata/wave_summaries.jsonl",
        },
        "counts": {
            **strict_summary,
            "collection_rejected_rows": len(collection_rejects),
            "compaction_index_rows": len(compaction_rows),
            "wave_summary_rows": len(wave_rows),
        },
    }
    if not args.dry_run:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(1)
