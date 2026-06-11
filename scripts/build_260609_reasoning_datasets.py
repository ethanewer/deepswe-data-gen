#!/usr/bin/env python3
"""Build 2026-06-09 high-quality reasoning datasets.

The expensive path is writing the full JSONL rows. This script therefore builds
`all-traces` once, records a compact index, and derives the unique/2x datasets
from line-number selections.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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

DATASET_NAMES = {
    "unique_90": "highquality-unique-reasoning-90pct",
    "unique_50": "highquality-unique-reasoning-50pct",
    "dup_90": "highquality-2x-duplicate-reasoning-90pct",
    "dup_50": "highquality-2x-duplicate-reasoning-50pct",
}

INDEX_FIELDS = [
    "line_number",
    "uuid",
    "task_id",
    "teacher",
    "reward",
    "passed",
    "percent_messages_with_reasoning",
    "assistant_message_count",
    "assistant_messages_with_reasoning",
    "assistant_messages_without_reasoning",
    "deepswe_prompt_augmentation",
    "instruction_style",
    "difficulty",
    "language",
    "repo",
    "rollout_id",
    "source_run_root",
    "trajectory_path",
    "result_path",
    "patch_path",
    "model_patch_sha256",
    "model_patch_bytes",
    "message_count",
    "trajectory_chars",
    "trajectory_bytes",
    "prompt_sha256",
    "prompt_chars",
    "prompt_preview",
    "sft_prompt_variant",
    "sft_prompt_substitution",
    "rollout_instruction_sha256",
    "sft_instruction_sha256",
    "agent_exit_status",
    "agent_exception_type",
    "api_calls",
    "cost_usd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-all-jsonl",
        type=Path,
        default=Path(
            "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/"
            "deepswe-highquality-all-trajectories/data/train.jsonl"
        ),
    )
    parser.add_argument(
        "--mimo-run-root",
        type=Path,
        default=Path(
            "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/"
            "datagen-20260608-pyxis-mimo-reasoning1"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260609"),
    )
    parser.add_argument("--limit", type=int, default=0, help="Debug limit for all-traces rows.")
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def stable_uuid(parts: list[str]) -> str:
    return sha256_text("\n".join(parts))


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


def planned_task_dir(workspace: Path | None, result: dict[str, Any], metadata: dict[str, Any]) -> Path | None:
    task_dir = first_nonempty(result.get("task_dir"), metadata.get("task_dir"))
    if not task_dir:
        return None
    path = Path(task_dir)
    if not path.is_absolute() and workspace is not None:
        path = workspace / path
    return path


def replace_first_user_prompt(
    messages: list[Any],
    rollout_instruction: str,
    sft_instruction: str,
) -> tuple[list[Any], bool]:
    if not rollout_instruction.strip() or not sft_instruction.strip():
        return messages, False
    replacements = [
        (rollout_instruction, sft_instruction),
        (rollout_instruction.strip(), sft_instruction.strip()),
        (rollout_instruction.rstrip("\n"), sft_instruction.rstrip("\n")),
    ]
    updated: list[Any] = []
    replaced = False
    for message in messages:
        if (
            not replaced
            and isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ):
            content = message["content"]
            for old, new in replacements:
                if old and old in content:
                    clean = dict(message)
                    clean["content"] = content.replace(old, new, 1)
                    updated.append(clean)
                    replaced = True
                    break
            if replaced:
                continue
        updated.append(message)
    return updated, replaced


def remove_rollout_hints_for_sft(
    messages: list[Any],
    workspace: Path | None,
    result: dict[str, Any],
    metadata: dict[str, Any],
    instruction_style: str,
) -> tuple[list[Any], dict[str, Any]]:
    if instruction_style != "planned":
        return messages, {"sft_prompt_variant": "rollout"}
    task_dir = planned_task_dir(workspace, result, metadata)
    if task_dir is None:
        return messages, {"sft_prompt_variant": "rollout", "sft_prompt_substitution": "missing_task_dir"}
    rollout_instruction = read_text_if_small(task_dir / "instruction.md")
    sft_instruction = read_text_if_small(task_dir / "instruction.sft.md")
    updated, replaced = replace_first_user_prompt(messages, rollout_instruction, sft_instruction)
    if not replaced:
        status = "missing_instruction_files" if not rollout_instruction or not sft_instruction else "rollout_text_not_found"
        return messages, {
            "sft_prompt_variant": "rollout",
            "sft_prompt_substitution": status,
            "rollout_instruction_sha256": sha256_text(rollout_instruction) if rollout_instruction else "",
            "sft_instruction_sha256": sha256_text(sft_instruction) if sft_instruction else "",
        }
    return updated, {
        "sft_prompt_variant": "unhinted",
        "sft_prompt_substitution": "replaced_first_user_prompt",
        "rollout_instruction_sha256": sha256_text(rollout_instruction),
        "sft_instruction_sha256": sha256_text(sft_instruction),
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_text_if_small(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        if path.exists() and path.stat().st_size <= max_bytes:
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return ""


def message_reasoning_content(message: dict[str, Any]) -> list[str]:
    contents: list[str] = []
    candidates: list[Any] = []
    for key in ("reasoning_content", "reasoning", "reasoning_details", "thinking"):
        candidates.append(message.get(key))
    provider = message.get("provider_specific_fields")
    if isinstance(provider, dict):
        for key in ("reasoning_content", "reasoning", "reasoning_details", "thinking"):
            candidates.append(provider.get(key))
    extra = message.get("extra")
    response = extra.get("response") if isinstance(extra, dict) else None
    choices = response.get("choices") if isinstance(response, dict) else None
    if isinstance(choices, list):
        for choice in choices:
            choice_message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(choice_message, dict):
                for key in ("reasoning_content", "reasoning", "reasoning_details", "thinking"):
                    candidates.append(choice_message.get(key))
    for value in candidates:
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, (list, dict)) and value:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            continue
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


def normalize_messages(messages: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        clean: dict[str, Any] = {}
        for key, value in message.items():
            if key == "extra" and isinstance(value, dict):
                # The provider response duplicates the assistant message and can
                # make rows much larger. Reasoning is extracted before this.
                extra = {extra_key: extra_value for extra_key, extra_value in value.items() if extra_key != "response"}
                if extra:
                    clean[key] = extra
            else:
                clean[key] = value
        normalized.append(clean)
    return normalized


def prompt_from_messages(messages: list[Any]) -> str:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def reasoning_indices_from_top_level(reasoning: Any) -> set[int]:
    indices: set[int] = set()
    if not isinstance(reasoning, list):
        return indices
    for item in reasoning:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str) and not content.strip():
            continue
        idx = item.get("message_index")
        if isinstance(idx, int):
            indices.add(idx)
    return indices


def reasoning_metrics(messages: list[Any], top_level_reasoning: Any = None) -> dict[str, Any]:
    top_indices = reasoning_indices_from_top_level(top_level_reasoning)
    assistant_count = 0
    with_reasoning = 0
    reasoning_chars = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        assistant_count += 1
        contents = message_reasoning_content(message)
        has_reasoning = bool(contents) or index in top_indices
        if has_reasoning:
            with_reasoning += 1
            reasoning_chars += sum(len(text) for text in contents)
    percent = (with_reasoning / assistant_count) if assistant_count else 0.0
    return {
        "assistant_message_count": assistant_count,
        "assistant_messages_with_reasoning": with_reasoning,
        "assistant_messages_without_reasoning": assistant_count - with_reasoning,
        "percent_messages_with_reasoning": percent,
        "has_any_reasoning": with_reasoning > 0,
        "has_all_assistant_reasoning": assistant_count > 0 and with_reasoning == assistant_count,
        "reasoning_chars_in_messages": reasoning_chars,
    }


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "passed"}
    return False


def build_existing_record(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    metadata = dict(row.get("metadata") or {})
    reasoning = row.get("reasoning") if isinstance(row.get("reasoning"), list) else collect_message_reasoning(messages)
    instruction_style = first_nonempty(metadata.get("instruction_style"), row.get("instruction_style"))
    workspace_text = first_nonempty(metadata.get("workspace"))
    workspace = Path(workspace_text) if workspace_text else None
    messages, prompt_variant_metadata = remove_rollout_hints_for_sft(
        messages,
        workspace,
        {},
        metadata,
        instruction_style,
    )
    metrics = reasoning_metrics(messages, reasoning)
    prompt = prompt_from_messages(messages)
    model_patch = row.get("model_patch") if isinstance(row.get("model_patch"), str) else ""
    model_patch_bytes = len(model_patch.encode("utf-8")) if model_patch else int(metadata.get("model_patch_bytes") or 0)
    model_patch_sha = sha256_text(model_patch) if model_patch else ""
    task_id = first_nonempty(metadata.get("task_id"), metadata.get("instance_id"), row.get("task_id"))
    teacher = first_nonempty(metadata.get("teacher"), metadata.get("model"), row.get("teacher"))
    reward = metadata.get("reward", row.get("reward", 0))
    try:
        reward_int = int(reward)
    except Exception:
        reward_int = 1 if bool_from_any(reward) else 0
    passed = bool_from_any(metadata.get("passed", row.get("passed", reward_int == 1)))
    deepswe_prompt_augmentation = instruction_style == "deepswe"
    trajectory_path = first_nonempty(metadata.get("trajectory_path"))
    trajectory_bytes = int(metadata.get("trajectory_bytes") or 0)
    if trajectory_path and not trajectory_bytes:
        try:
            trajectory_bytes = Path(trajectory_path).stat().st_size
        except OSError:
            pass
    trajectory_chars = int(metadata.get("trajectory_chars") or len(json.dumps(messages, ensure_ascii=False)))
    metadata.update(
        {
            "task_id": task_id,
            "instance_id": task_id,
            "teacher": teacher,
            "model": first_nonempty(metadata.get("model"), teacher),
            "reward": reward_int,
            "passed": passed,
            "instruction_style": instruction_style,
            "deepswe_prompt_augmentation": deepswe_prompt_augmentation,
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "prompt_sha256": sha256_text(prompt) if prompt else "",
            "assistant_message_count": metrics["assistant_message_count"],
            "assistant_messages_with_reasoning": metrics["assistant_messages_with_reasoning"],
            "assistant_messages_without_reasoning": metrics["assistant_messages_without_reasoning"],
            "percent_messages_with_reasoning": metrics["percent_messages_with_reasoning"],
            "has_any_reasoning": metrics["has_any_reasoning"],
            "has_all_assistant_reasoning": metrics["has_all_assistant_reasoning"],
            "reasoning_turns": len(reasoning),
            "reasoning_chars": sum(len(str(x.get("content", ""))) for x in reasoning if isinstance(x, dict)),
            "model_patch_bytes": model_patch_bytes,
            "model_patch_sha256": model_patch_sha,
            "code_diff_sha256": model_patch_sha,
            "trajectory_chars": trajectory_chars,
            "trajectory_bytes": trajectory_bytes,
            "row_source": first_nonempty(metadata.get("row_source"), "existing_all_trajectories"),
            **prompt_variant_metadata,
        }
    )
    out = dict(row)
    out["uuid"] = first_nonempty(row.get("uuid"), stable_uuid([trajectory_path, task_id, teacher, metadata.get("rollout_id", "")]))
    out["messages"] = messages
    out["tools"] = row.get("tools") if isinstance(row.get("tools"), list) else [BASH_TOOL]
    out["reasoning"] = reasoning
    if model_patch:
        out["model_patch"] = model_patch
    out["metadata"] = metadata
    out["task_id"] = task_id
    out["teacher"] = teacher
    out["reward"] = reward_int
    out["passed"] = passed
    out["percent_messages_with_reasoning"] = metrics["percent_messages_with_reasoning"]
    out["deepswe_prompt_augmentation"] = deepswe_prompt_augmentation
    out["prompt"] = prompt
    return out


def manifest_rows(mimo_run_root: Path) -> list[tuple[Path, list[str]]]:
    manifest_dir = mimo_run_root / "manifest"
    names = []
    for path in sorted(manifest_dir.glob("mimo-packed*.tsv")):
        if path.name == "mimo-packed-wave-r10-1024.tsv":
            names.append(path)
        elif path.name.startswith("mimo-packed-probe-"):
            names.append(path)
        elif path.name.startswith("mimo-packed-wave-"):
            names.append(path)
        elif path.name.startswith("mimo-packed-retry-"):
            names.append(path)
    rows: list[tuple[Path, list[str]]] = []
    seen_manifest_rows: set[tuple[str, str, str]] = set()
    for path in names:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                parts = raw.rstrip("\n").split("\t")
                if len(parts) < 15:
                    continue
                key = (path.name, parts[1], parts[2])
                if key in seen_manifest_rows:
                    continue
                seen_manifest_rows.add(key)
                rows.append((path, parts))
    return rows


def build_mimo_record(run_root: Path, manifest_path: Path, parts: list[str]) -> dict[str, Any] | None:
    workspace = Path(parts[4])
    if not workspace.is_absolute():
        workspace = Path.cwd() / workspace
    trajectory_path = workspace / "agent" / "mini-swe-agent.trajectory.json"
    if not trajectory_path.exists() or trajectory_path.stat().st_size <= 2:
        return None
    trajectory = load_json(trajectory_path)
    raw_messages = trajectory.get("messages")
    if not isinstance(raw_messages, list):
        return None
    reasoning = collect_message_reasoning(raw_messages)
    messages = normalize_messages(raw_messages)
    result = load_json(workspace / "result.json")
    run_metadata = load_json(workspace / "metadata.json")
    info = trajectory.get("info") if isinstance(trajectory.get("info"), dict) else {}
    model_stats = info.get("model_stats") if isinstance(info.get("model_stats"), dict) else {}
    config = info.get("config") if isinstance(info.get("config"), dict) else {}
    model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    task_id = first_nonempty(result.get("instance_id"), parts[2])
    rollout_id = first_nonempty(result.get("rollout_id"), parts[1])
    teacher = first_nonempty(result.get("model"), parts[6])
    litellm_model = first_nonempty(result.get("litellm_model"), parts[7])
    difficulty = first_nonempty(result.get("difficulty"), parts[11])
    language = first_nonempty(result.get("language"), parts[12])
    instruction_style = first_nonempty(result.get("instruction_style"), run_metadata.get("instruction_style"), parts[13])
    repo = first_nonempty(result.get("repo"), parts[14])
    messages, prompt_variant_metadata = remove_rollout_hints_for_sft(
        messages,
        workspace,
        result,
        run_metadata,
        instruction_style,
    )
    metrics = reasoning_metrics(messages, reasoning)
    reward = result.get("reward", 0)
    try:
        reward_int = int(reward)
    except Exception:
        reward_int = 1 if bool_from_any(reward) else 0
    passed = reward_int == 1 or bool_from_any(result.get("passed"))
    patch_text = ""
    patch_path = Path(first_nonempty(result.get("patch_path"), workspace / "logs" / "artifacts" / "model.patch"))
    if patch_path.is_absolute() and not patch_path.exists():
        try:
            patch_path = workspace / patch_path.relative_to("/workspace")
        except ValueError:
            pass
    patch_text = read_text_if_small(patch_path)
    if not patch_text and isinstance(result.get("patch"), str):
        patch_text = result.get("patch") or ""
    model_patch_sha = sha256_text(patch_text) if patch_text else ""
    prompt = prompt_from_messages(messages)
    agent_exception = result.get("agent_exception") if isinstance(result.get("agent_exception"), dict) else {}
    metadata = {
        "dataset": "deepswe-highquality-synthetic-260609",
        "source": "deepswe-data-gen",
        "source_run_root": str(run_root),
        "source_manifest": str(manifest_path),
        "source_manifest_row": int(parts[0]) if parts[0].isdigit() else parts[0],
        "workspace": str(workspace),
        "trajectory_path": str(trajectory_path),
        "result_path": str(workspace / "result.json"),
        "patch_path": str(patch_path) if patch_path else "",
        "task_id": task_id,
        "instance_id": task_id,
        "rollout_id": rollout_id,
        "repo": repo,
        "difficulty": difficulty,
        "language": language,
        "instruction_style": instruction_style,
        "deepswe_prompt_augmentation": instruction_style == "deepswe",
        "teacher": teacher,
        "model": teacher,
        "litellm_model": litellm_model,
        "docker_image": first_nonempty(result.get("docker_image"), parts[5]),
        "passed": passed,
        "reward": reward_int,
        "agent_exit_status": first_nonempty(result.get("agent_exit_status"), info.get("exit_status")),
        "agent_exception_type": first_nonempty(agent_exception.get("type")),
        "api_calls": result.get("api_calls", model_stats.get("api_calls", 0)),
        "cost_usd": result.get("cost_usd", model_stats.get("instance_cost", 0.0)),
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
        "trajectory_bytes": trajectory_path.stat().st_size,
        "trajectory_format": trajectory.get("trajectory_format", ""),
        "mini_swe_agent_version": info.get("mini_version", ""),
        "model_patch_bytes": len(patch_text.encode("utf-8")) if patch_text else 0,
        "model_patch_sha256": model_patch_sha,
        "code_diff_sha256": model_patch_sha,
        "model_config": model_config,
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "prompt_sha256": sha256_text(prompt) if prompt else "",
        "row_source": "mimo_manifest_trace",
        **prompt_variant_metadata,
    }
    record: dict[str, Any] = {
        "uuid": stable_uuid([str(trajectory_path), task_id, rollout_id, teacher, instruction_style]),
        "task_id": task_id,
        "teacher": teacher,
        "reward": reward_int,
        "passed": passed,
        "percent_messages_with_reasoning": metrics["percent_messages_with_reasoning"],
        "deepswe_prompt_augmentation": instruction_style == "deepswe",
        "prompt": prompt,
        "messages": messages,
        "tools": [BASH_TOOL],
        "reasoning": reasoning,
        "metadata": metadata,
    }
    if patch_text:
        record["model_patch"] = patch_text
    return record


def record_to_index(row: dict[str, Any], line_number: int) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    prompt = row.get("prompt") if isinstance(row.get("prompt"), str) else metadata.get("prompt", "")
    return {
        "line_number": line_number,
        "uuid": row.get("uuid", ""),
        "task_id": first_nonempty(row.get("task_id"), metadata.get("task_id"), metadata.get("instance_id")),
        "teacher": first_nonempty(row.get("teacher"), metadata.get("teacher"), metadata.get("model")),
        "reward": int(row.get("reward", metadata.get("reward", 0)) or 0),
        "passed": bool_from_any(row.get("passed", metadata.get("passed", False))),
        "percent_messages_with_reasoning": float(
            row.get("percent_messages_with_reasoning", metadata.get("percent_messages_with_reasoning", 0.0)) or 0.0
        ),
        "assistant_message_count": int(metadata.get("assistant_message_count", 0) or 0),
        "assistant_messages_with_reasoning": int(metadata.get("assistant_messages_with_reasoning", 0) or 0),
        "assistant_messages_without_reasoning": int(metadata.get("assistant_messages_without_reasoning", 0) or 0),
        "deepswe_prompt_augmentation": bool_from_any(
            row.get("deepswe_prompt_augmentation", metadata.get("deepswe_prompt_augmentation", False))
        ),
        "instruction_style": first_nonempty(metadata.get("instruction_style")),
        "difficulty": first_nonempty(metadata.get("difficulty")),
        "language": first_nonempty(metadata.get("language")),
        "repo": first_nonempty(metadata.get("repo")),
        "rollout_id": first_nonempty(metadata.get("rollout_id")),
        "source_run_root": first_nonempty(metadata.get("source_run_root")),
        "trajectory_path": first_nonempty(metadata.get("trajectory_path")),
        "result_path": first_nonempty(metadata.get("result_path")),
        "patch_path": first_nonempty(metadata.get("patch_path")),
        "model_patch_sha256": first_nonempty(metadata.get("model_patch_sha256"), metadata.get("code_diff_sha256")),
        "model_patch_bytes": int(metadata.get("model_patch_bytes", 0) or 0),
        "message_count": int(metadata.get("message_count", 0) or 0),
        "trajectory_chars": int(metadata.get("trajectory_chars", 0) or 0),
        "trajectory_bytes": int(metadata.get("trajectory_bytes", 0) or 0),
        "prompt_sha256": sha256_text(prompt) if prompt else first_nonempty(metadata.get("prompt_sha256")),
        "prompt_chars": len(prompt) if prompt else int(metadata.get("prompt_chars", 0) or 0),
        "prompt_preview": prompt[:500].replace("\n", "\\n") if prompt else "",
        "sft_prompt_variant": first_nonempty(metadata.get("sft_prompt_variant")),
        "sft_prompt_substitution": first_nonempty(metadata.get("sft_prompt_substitution")),
        "rollout_instruction_sha256": first_nonempty(metadata.get("rollout_instruction_sha256")),
        "sft_instruction_sha256": first_nonempty(metadata.get("sft_instruction_sha256")),
        "agent_exit_status": first_nonempty(metadata.get("agent_exit_status")),
        "agent_exception_type": first_nonempty(metadata.get("agent_exception_type")),
        "api_calls": int(metadata.get("api_calls", 0) or 0),
        "cost_usd": float(metadata.get("cost_usd", 0.0) or 0.0),
    }


def make_dirs(path: Path) -> None:
    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "metadata").mkdir(parents=True, exist_ok=True)


def write_index_files(dataset_dir: Path, rows: list[dict[str, Any]]) -> None:
    metadata_dir = dataset_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = metadata_dir / "index.jsonl"
    csv_path = metadata_dir / "index.csv"
    sqlite_path = metadata_dir / "index.sqlite"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in INDEX_FIELDS})
    if sqlite_path.exists():
        sqlite_path.unlink()
    conn = sqlite3.connect(sqlite_path)
    try:
        cols = ", ".join(f"{field} TEXT" for field in INDEX_FIELDS)
        conn.execute(f"CREATE TABLE traces ({cols})")
        placeholders = ", ".join("?" for _ in INDEX_FIELDS)
        conn.executemany(
            f"INSERT INTO traces VALUES ({placeholders})",
            [[str(row.get(field, "")) for field in INDEX_FIELDS] for row in rows],
        )
        conn.execute("CREATE INDEX traces_task_id_idx ON traces(task_id)")
        conn.execute("CREATE INDEX traces_teacher_idx ON traces(teacher)")
        conn.execute("CREATE INDEX traces_reasoning_pct_idx ON traces(percent_messages_with_reasoning)")
        conn.execute("CREATE INDEX traces_passed_idx ON traces(passed)")
        conn.commit()
    finally:
        conn.close()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_difficulty = Counter(row.get("difficulty", "") for row in rows)
    by_teacher = Counter(row.get("teacher", "") for row in rows)
    by_passed = Counter(str(row.get("passed", False)).lower() for row in rows)
    by_style = Counter(row.get("instruction_style", "") for row in rows)
    return {
        "records": len(rows),
        "unique_tasks": len({row.get("task_id", "") for row in rows if row.get("task_id")}),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "by_teacher": dict(sorted(by_teacher.items())),
        "by_passed": dict(sorted(by_passed.items())),
        "by_instruction_style": dict(sorted(by_style.items())),
        "mean_percent_messages_with_reasoning": (
            sum(float(row.get("percent_messages_with_reasoning", 0.0)) for row in rows) / len(rows) if rows else 0.0
        ),
    }


def write_manifest_and_readme(dataset_dir: Path, name: str, rows: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    summary = summarize(rows)
    manifest = {
        "name": name,
        "created_at_unix": time.time(),
        "data_file": str(dataset_dir / "data" / "train.jsonl"),
        "index_jsonl": str(dataset_dir / "metadata" / "index.jsonl"),
        "index_csv": str(dataset_dir / "metadata" / "index.csv"),
        "index_sqlite": str(dataset_dir / "metadata" / "index.sqlite"),
        **summary,
        **extra,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = [
        f"# {name}",
        "",
        "JSONL SFT records with `messages`, `tools`, `reasoning`, `model_patch`, top-level selection fields, and rich `metadata`.",
        "",
        "Important top-level fields: `task_id`, `teacher`, `reward`, `passed`, `percent_messages_with_reasoning`,",
        "`deepswe_prompt_augmentation`, and `prompt`.",
        "",
        "For fast inspection and filtering, use `metadata/index.sqlite` or `metadata/index.csv`.",
        "",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (dataset_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")


def unique_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        1 if row["passed"] else 0,
        -int(row["assistant_messages_without_reasoning"]),
        float(row["percent_messages_with_reasoning"]),
        int(row["reward"]),
        -int(row["trajectory_chars"]),
        -int(row["assistant_message_count"]),
        -int(row["line_number"]),
    )


def second_key(first_patch: str, row: dict[str, Any]) -> tuple[Any, ...]:
    patch = row.get("model_patch_sha256", "")
    different_patch = bool(patch and first_patch and patch != first_patch)
    return (
        1 if row["passed"] else 0,
        -int(row["assistant_messages_without_reasoning"]),
        float(row["percent_messages_with_reasoning"]),
        1 if different_patch else 0,
        int(row["reward"]),
        -int(row["trajectory_chars"]),
        -int(row["assistant_message_count"]),
        -int(row["line_number"]),
    )


def select_unique(index_rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in index_rows:
        if row["assistant_message_count"] <= 0 or row["percent_messages_with_reasoning"] < threshold:
            continue
        task_id = row["task_id"]
        if not task_id:
            continue
        current = best.get(task_id)
        if current is None or unique_key(row) > unique_key(current):
            best[task_id] = row
    return sorted(best.values(), key=lambda row: (row["difficulty"], row["language"], row["task_id"], row["line_number"]))


def select_duplicate(index_rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        if row["assistant_message_count"] <= 0 or row["percent_messages_with_reasoning"] < threshold:
            continue
        if row["task_id"]:
            grouped[row["task_id"]].append(row)
    selected: list[dict[str, Any]] = []
    for task_id, rows in grouped.items():
        first = max(rows, key=unique_key)
        chosen = [first]
        remaining = [row for row in rows if row["line_number"] != first["line_number"]]
        if remaining:
            chosen.append(max(remaining, key=lambda row: second_key(first.get("model_patch_sha256", ""), row)))
        selected.extend(chosen)
    return sorted(selected, key=lambda row: (row["difficulty"], row["language"], row["task_id"], row["line_number"]))


def route_selected_rows(all_jsonl: Path, selections: dict[str, set[int]], output_root: Path) -> None:
    handles: dict[str, Any] = {}
    tmp_paths: dict[str, Path] = {}
    try:
        for key, name in DATASET_NAMES.items():
            dataset_dir = output_root / name
            make_dirs(dataset_dir)
            path = dataset_dir / "data" / "train.jsonl"
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp_paths[key] = tmp
            handles[key] = tmp.open("w", encoding="utf-8")
        with all_jsonl.open(encoding="utf-8", errors="replace") as source:
            for line_number, line in enumerate(source):
                for key, selected in selections.items():
                    if line_number in selected:
                        handles[key].write(line)
    finally:
        for handle in handles.values():
            handle.close()
    for key, tmp in tmp_paths.items():
        tmp.replace(output_root / DATASET_NAMES[key] / "data" / "train.jsonl")


def main() -> None:
    args = parse_args()
    started = time.time()
    output_root = args.output_root
    all_dir = output_root / "all-traces"
    make_dirs(all_dir)
    all_jsonl = all_dir / "data" / "train.jsonl"
    all_tmp = all_jsonl.with_suffix(all_jsonl.suffix + ".tmp")
    index_rows: list[dict[str, Any]] = []
    seen_trajectory_paths: set[str] = set()
    skipped = Counter()

    with all_tmp.open("w", encoding="utf-8") as out:
        with args.base_all_jsonl.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                if args.limit and len(index_rows) >= args.limit:
                    break
                try:
                    row = build_existing_record(json.loads(line))
                except Exception:
                    skipped["existing_json_error"] += 1
                    continue
                if row is None:
                    skipped["existing_invalid"] += 1
                    continue
                trajectory_path = row["metadata"].get("trajectory_path", "")
                if trajectory_path:
                    seen_trajectory_paths.add(trajectory_path)
                line_number = len(index_rows)
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                index_rows.append(record_to_index(row, line_number))
                if line_number and line_number % 1000 == 0:
                    print(f"existing_progress rows={line_number} elapsed={time.time() - started:.1f}s", flush=True)
        if not args.limit:
            for manifest_path, parts in manifest_rows(args.mimo_run_root):
                row = build_mimo_record(args.mimo_run_root, manifest_path, parts)
                if row is None:
                    skipped["mimo_no_valid_trajectory"] += 1
                    continue
                trajectory_path = row["metadata"].get("trajectory_path", "")
                if trajectory_path in seen_trajectory_paths:
                    skipped["mimo_duplicate_trajectory_path"] += 1
                    continue
                seen_trajectory_paths.add(trajectory_path)
                line_number = len(index_rows)
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                index_rows.append(record_to_index(row, line_number))
                if line_number and line_number % 1000 == 0:
                    print(f"mimo_progress rows={line_number} elapsed={time.time() - started:.1f}s", flush=True)
    all_tmp.replace(all_jsonl)
    print(f"all_traces_written rows={len(index_rows)} elapsed={time.time() - started:.1f}s", flush=True)

    write_index_files(all_dir, index_rows)
    write_manifest_and_readme(
        all_dir,
        "all-traces",
        index_rows,
        {
            "builder": str(Path(__file__).resolve()),
            "base_all_jsonl": str(args.base_all_jsonl),
            "mimo_run_root": str(args.mimo_run_root),
            "skipped": dict(sorted(skipped.items())),
            "selection_rule": "all valid trajectory JSON files from the existing all-trajectories dataset plus current MiMo manifests",
        },
    )
    print(f"all_index_written elapsed={time.time() - started:.1f}s", flush=True)

    selected_rows_by_key = {
        "unique_90": select_unique(index_rows, 0.90),
        "unique_50": select_unique(index_rows, 0.50),
        "dup_90": select_duplicate(index_rows, 0.90),
        "dup_50": select_duplicate(index_rows, 0.50),
    }
    selections = {key: {int(row["line_number"]) for row in rows} for key, rows in selected_rows_by_key.items()}
    route_selected_rows(all_jsonl, selections, output_root)
    for key, rows in selected_rows_by_key.items():
        name = DATASET_NAMES[key]
        dataset_dir = output_root / name
        write_index_files(dataset_dir, rows)
        threshold = 0.90 if key.endswith("90") else 0.50
        max_per_task = 2 if key.startswith("dup") else 1
        write_manifest_and_readme(
            dataset_dir,
            name,
            rows,
            {
                "builder": str(Path(__file__).resolve()),
                "source_dataset": str(all_jsonl),
                "reasoning_threshold": threshold,
                "max_traces_per_task": max_per_task,
                "selection_rule": (
                    "filter assistant_message_count > 0 and percent_messages_with_reasoning >= threshold; "
                    "rank by passed, fewer assistant messages without reasoning, higher reasoning percent, "
                    "then shorter trajectory. For 2x datasets, the second trace additionally prefers a "
                    "different model_patch_sha256 after pass/reasoning quality."
                ),
            },
        )
        print(f"dataset_written name={name} rows={len(rows)} elapsed={time.time() - started:.1f}s", flush=True)

    final = {
        "output_root": str(output_root),
        "elapsed_sec": time.time() - started,
        "all_traces": summarize(index_rows),
        "datasets": {DATASET_NAMES[key]: summarize(rows) for key, rows in selected_rows_by_key.items()},
        "skipped": dict(sorted(skipped.items())),
    }
    (output_root / "manifest.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
