#!/usr/bin/env python3
"""Build a 90% reasoning, up-to-3-traces-per-task dataset.

Rows below the reasoning threshold are truncated to the longest message prefix
that still has >=90% assistant-message reasoning coverage. Truncated rows with
fewer than two assistant messages are excluded.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import build_260609_reasoning_datasets as base


DEFAULT_RUN_ROOTS = [
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260609-local-qwen36-missing",
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260609-local-qwen36-hq-allreason-retry1",
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260610-local-qwen36-direct-docker-hq-retry2",
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260610-local-qwen36-updated-1409-scale1",
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260610-openrouter-controlled-85be8b1",
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260610-openrouter-controlled-99f0ffb",
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260611-openrouter-controlled-postfix-906faa1-exact",
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260611-local-qwen36-supplemental-postfix-7164168",
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260611-openrouter-deepseek-original-throughput",
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260611-original-throughput-post-deepswe-disabled",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-all-jsonl",
        type=Path,
        default=Path(
            "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260609/"
            "all-traces/data/train.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260611/"
            "highquality-3x-duplicate-reasoning-90pct"
        ),
    )
    parser.add_argument("--run-root", action="append", type=Path, default=[])
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--max-per-task", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def result_patch_path(workspace: Path, result: dict[str, Any]) -> Path:
    raw = base.first_nonempty(result.get("patch_path"), workspace / "logs" / "artifacts" / "model.patch")
    patch_path = Path(raw)
    if patch_path.is_absolute() and not patch_path.exists():
        try:
            patch_path = workspace / patch_path.relative_to("/workspace")
        except ValueError:
            pass
    return patch_path


def provider_usage(messages: list[Any]) -> dict[str, Any]:
    out = {
        "provider_cost_usd": 0.0,
        "provider_api_calls": 0,
        "provider_costed_api_calls": 0,
        "provider_prompt_tokens": 0,
        "provider_completion_tokens": 0,
        "provider_reasoning_tokens": 0,
    }
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        extra = message.get("extra")
        response = extra.get("response") if isinstance(extra, dict) else None
        usage = response.get("usage") if isinstance(response, dict) and isinstance(response.get("usage"), dict) else None
        if not usage:
            continue
        out["provider_api_calls"] += 1
        out["provider_prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        out["provider_completion_tokens"] += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        if isinstance(details, dict):
            out["provider_reasoning_tokens"] += int(details.get("reasoning_tokens") or 0)
        if usage.get("cost") is not None:
            try:
                out["provider_cost_usd"] += float(usage.get("cost") or 0.0)
                out["provider_costed_api_calls"] += 1
            except Exception:
                pass
    return out


def reasoning_indices(reasoning: list[dict[str, Any]]) -> set[int]:
    indices: set[int] = set()
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


def longest_good_prefix(
    messages: list[Any],
    reasoning: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any], bool] | None:
    metrics = base.reasoning_metrics(messages, reasoning)
    if metrics["assistant_message_count"] >= 2 and metrics["percent_messages_with_reasoning"] >= threshold:
        return messages, reasoning, metrics, False

    top_indices = reasoning_indices(reasoning)
    assistant_count = 0
    assistant_with_reasoning = 0
    best_end = -1
    best_reasoning_count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        assistant_count += 1
        if base.message_reasoning_content(message) or index in top_indices:
            assistant_with_reasoning += 1
        percent = assistant_with_reasoning / assistant_count if assistant_count else 0.0
        if assistant_count >= 2 and percent >= threshold:
            best_end = index
            best_reasoning_count = assistant_with_reasoning

    if best_end < 0:
        return None

    prefix_messages = messages[: best_end + 1]
    prefix_reasoning = [
        item
        for item in reasoning
        if isinstance(item, dict) and isinstance(item.get("message_index"), int) and item["message_index"] <= best_end
    ]
    prefix_metrics = base.reasoning_metrics(prefix_messages, prefix_reasoning)
    if prefix_metrics["assistant_message_count"] < 2 or prefix_metrics["percent_messages_with_reasoning"] < threshold:
        return None
    if prefix_metrics["assistant_messages_with_reasoning"] != best_reasoning_count:
        prefix_metrics = base.reasoning_metrics(prefix_messages, prefix_reasoning)
    return prefix_messages, prefix_reasoning, prefix_metrics, True


def apply_reasoning_filter(row: dict[str, Any], threshold: float) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    reasoning = row.get("reasoning") if isinstance(row.get("reasoning"), list) else base.collect_message_reasoning(messages)
    original_metrics = base.reasoning_metrics(messages, reasoning)
    prefix = longest_good_prefix(messages, reasoning, threshold)
    if prefix is None:
        return None
    selected_messages, selected_reasoning, metrics, truncated = prefix
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "truncated": truncated,
            "truncated_for_reasoning_threshold": truncated,
            "truncation_reason": "reasoning_prefix" if truncated else "",
            "original_message_count": len(messages),
            "original_assistant_message_count": original_metrics["assistant_message_count"],
            "original_assistant_messages_with_reasoning": original_metrics["assistant_messages_with_reasoning"],
            "original_assistant_messages_without_reasoning": original_metrics["assistant_messages_without_reasoning"],
            "original_percent_messages_with_reasoning": original_metrics["percent_messages_with_reasoning"],
            "message_count": len(selected_messages),
            "assistant_message_count": metrics["assistant_message_count"],
            "assistant_messages_with_reasoning": metrics["assistant_messages_with_reasoning"],
            "assistant_messages_without_reasoning": metrics["assistant_messages_without_reasoning"],
            "percent_messages_with_reasoning": metrics["percent_messages_with_reasoning"],
            "has_any_reasoning": metrics["has_any_reasoning"],
            "has_all_assistant_reasoning": metrics["has_all_assistant_reasoning"],
            "reasoning_turns": len(selected_reasoning),
            "reasoning_chars": sum(len(str(x.get("content", ""))) for x in selected_reasoning if isinstance(x, dict)),
            "trajectory_chars": len(json.dumps(selected_messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
            "dataset_row_is_truncated": truncated,
        }
    )
    prompt = base.prompt_from_messages(selected_messages)
    metadata["prompt"] = prompt
    metadata["prompt_chars"] = len(prompt)
    metadata["prompt_sha256"] = base.sha256_text(prompt) if prompt else ""
    out = dict(row)
    out["messages"] = selected_messages
    out["reasoning"] = selected_reasoning
    out["metadata"] = metadata
    out["percent_messages_with_reasoning"] = metrics["percent_messages_with_reasoning"]
    out["prompt"] = prompt
    return out


def build_existing_record(raw: dict[str, Any], threshold: float) -> dict[str, Any] | None:
    row = base.build_existing_record(raw)
    if row is None:
        return None
    metadata = dict(row.get("metadata") or {})
    metadata.setdefault("has_result_json", bool(metadata.get("result_path")))
    metadata.setdefault("incomplete", False)
    metadata.setdefault("result_json_missing", False)
    metadata.setdefault("dataset_source_stage", "260609_all_traces")
    row["metadata"] = metadata
    return apply_reasoning_filter(row, threshold)


def build_trace_record(run_root: Path, workspace: Path, threshold: float) -> dict[str, Any] | None:
    trajectory_path = workspace / "agent" / "mini-swe-agent.trajectory.json"
    if not trajectory_path.exists() or trajectory_path.stat().st_size <= 2:
        return None
    result_path = workspace / "result.json"
    result = base.load_json(result_path) if result_path.exists() else {}
    metadata_json = base.load_json(workspace / "metadata.json")
    trajectory = base.load_json(trajectory_path)
    raw_messages = trajectory.get("messages")
    if not isinstance(raw_messages, list):
        return None

    reasoning = base.collect_message_reasoning(raw_messages)
    messages = base.normalize_messages(raw_messages)
    info = trajectory.get("info") if isinstance(trajectory.get("info"), dict) else {}
    model_stats = info.get("model_stats") if isinstance(info.get("model_stats"), dict) else {}
    config = info.get("config") if isinstance(info.get("config"), dict) else {}
    model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    agent_exception = result.get("agent_exception") if isinstance(result.get("agent_exception"), dict) else {}

    task_id = base.first_nonempty(result.get("instance_id"), metadata_json.get("instance_id"), workspace.name)
    rollout_id = base.first_nonempty(result.get("rollout_id"), metadata_json.get("rollout_id"), workspace.parent.name)
    teacher = base.first_nonempty(result.get("model"), result.get("teacher"), metadata_json.get("model"), workspace.parent.parent.name)
    litellm_model = base.first_nonempty(result.get("litellm_model"), metadata_json.get("litellm_model"))
    difficulty = base.first_nonempty(result.get("difficulty"), metadata_json.get("difficulty"))
    language = base.first_nonempty(result.get("language"), metadata_json.get("language"))
    instruction_style = base.first_nonempty(result.get("instruction_style"), metadata_json.get("instruction_style"), workspace.parent.parent.parent.name)
    repo = base.first_nonempty(result.get("repo"), metadata_json.get("repo"))
    try:
        reward_int = int(result.get("reward", 0) or 0)
    except Exception:
        reward_int = 1 if base.bool_from_any(result.get("reward")) else 0
    passed = reward_int == 1 or base.bool_from_any(result.get("passed"))

    patch_text = ""
    patch_path = result_patch_path(workspace, result)
    patch_text = base.read_text_if_small(patch_path)
    if not patch_text and isinstance(result.get("patch"), str):
        patch_text = result.get("patch") or ""
    model_patch_sha = base.sha256_text(patch_text) if patch_text else ""
    prompt = base.prompt_from_messages(messages)
    usage = provider_usage(raw_messages)
    cost_usd = result.get("cost_usd", model_stats.get("instance_cost", usage["provider_cost_usd"]))
    if not cost_usd and usage["provider_cost_usd"]:
        cost_usd = usage["provider_cost_usd"]
    has_result = result_path.exists()

    metadata = {
        "dataset": "deepswe-highquality-synthetic-260611-3x-90pct",
        "source": "deepswe-data-gen",
        "source_run_root": str(run_root),
        "workspace": str(workspace),
        "trajectory_path": str(trajectory_path),
        "result_path": str(result_path) if has_result else None,
        "patch_path": str(patch_path) if patch_path else None,
        "task_id": task_id,
        "instance_id": task_id,
        "rollout_id": rollout_id,
        "repo": repo or None,
        "difficulty": difficulty or None,
        "language": language or None,
        "instruction_style": instruction_style or None,
        "benchmark_profile": base.first_nonempty(result.get("benchmark_profile"), metadata_json.get("benchmark_profile")) or None,
        "deepswe_prompt_augmentation": instruction_style == "deepswe",
        "teacher": teacher,
        "model": teacher,
        "litellm_model": litellm_model or None,
        "api_base": base.first_nonempty(result.get("api_base"), metadata_json.get("api_base")) or None,
        "docker_image": base.first_nonempty(result.get("docker_image"), metadata_json.get("docker_image")) or None,
        "outside_original_high_quality_set": base.bool_from_any(
            result.get("outside_original_high_quality_set", metadata_json.get("outside_original_high_quality_set", False))
        ),
        "passed": passed,
        "reward": reward_int,
        "has_result_json": has_result,
        "incomplete": not has_result,
        "result_json_missing": not has_result,
        "agent_exit_status": base.first_nonempty(result.get("agent_exit_status"), info.get("exit_status")) or None,
        "agent_exception_type": base.first_nonempty(agent_exception.get("type")) or None,
        "api_calls": result.get("api_calls", model_stats.get("api_calls", usage["provider_api_calls"])),
        "cost_usd": cost_usd,
        "provider_usage": usage,
        "trajectory_bytes": trajectory_path.stat().st_size,
        "trajectory_format": trajectory.get("trajectory_format", ""),
        "mini_swe_agent_version": info.get("mini_version", ""),
        "mini_swe_agent_git_sha": base.first_nonempty(result.get("mini_swe_agent_git_sha"), metadata_json.get("mini_swe_agent_git_sha")) or None,
        "mini_swe_agent_config_file": base.first_nonempty(result.get("mini_swe_agent_config_file"), metadata_json.get("mini_swe_agent_config_file")) or None,
        "datagen_code_commit": base.first_nonempty(result.get("datagen_code_commit"), metadata_json.get("datagen_code_commit")) or None,
        "uses_updated_alignment": base.bool_from_any(result.get("uses_updated_alignment", metadata_json.get("uses_updated_alignment", False))),
        "eligible_for_controlled_comparison": base.bool_from_any(result.get("eligible_for_controlled_comparison", False)),
        "reason_excluded_from_comparison": base.first_nonempty(result.get("reason_excluded_from_comparison")) or None,
        "model_patch_bytes": len(patch_text.encode("utf-8")) if patch_text else 0,
        "model_patch_sha256": model_patch_sha,
        "code_diff_sha256": model_patch_sha,
        "model_config": model_config,
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "prompt_sha256": base.sha256_text(prompt) if prompt else "",
        "row_source": "post_260609_run_trace",
        "dataset_source_stage": "post_260609_run_trace",
    }
    record: dict[str, Any] = {
        "uuid": base.stable_uuid([str(trajectory_path), task_id, rollout_id, teacher, instruction_style]),
        "task_id": task_id,
        "teacher": teacher,
        "reward": reward_int,
        "passed": passed,
        "percent_messages_with_reasoning": 0.0,
        "deepswe_prompt_augmentation": instruction_style == "deepswe",
        "prompt": prompt,
        "messages": messages,
        "tools": [base.BASH_TOOL],
        "reasoning": reasoning,
        "metadata": metadata,
    }
    if patch_text:
        record["model_patch"] = patch_text
    return apply_reasoning_filter(record, threshold)


def iter_workspaces(run_root: Path) -> list[Path]:
    pyxis = run_root / "pyxis-traces"
    if not pyxis.exists():
        return []
    return sorted({path.parent.parent for path in pyxis.glob("*/*/r*/*/agent/mini-swe-agent.trajectory.json")})


def record_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    metadata = row.get("metadata") or {}
    return (
        1 if row.get("passed") else 0,
        0 if metadata.get("incomplete") else 1,
        -int(metadata.get("assistant_messages_without_reasoning") or 0),
        float(metadata.get("percent_messages_with_reasoning") or row.get("percent_messages_with_reasoning") or 0.0),
        0 if metadata.get("truncated") else 1,
        int(row.get("reward") or 0),
        -int(metadata.get("trajectory_chars") or 0),
        -int(metadata.get("assistant_message_count") or 0),
    )


def duplicate_rank(chosen_patches: set[str], row: dict[str, Any]) -> tuple[Any, ...]:
    metadata = row.get("metadata") or {}
    patch = metadata.get("model_patch_sha256") or metadata.get("code_diff_sha256") or ""
    new_patch = bool(patch and patch not in chosen_patches)
    return (*record_rank(row), 1 if new_patch else 0)


def select_records(records: list[dict[str, Any]], max_per_task: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        task_id = row.get("task_id") or (row.get("metadata") or {}).get("task_id")
        if task_id:
            grouped[str(task_id)].append(row)

    selected: list[dict[str, Any]] = []
    for task_id, rows in grouped.items():
        remaining = list(rows)
        chosen: list[dict[str, Any]] = []
        patches: set[str] = set()
        while remaining and len(chosen) < max_per_task:
            key_fn = record_rank if not chosen else lambda row: duplicate_rank(patches, row)
            row = max(remaining, key=key_fn)
            chosen.append(row)
            metadata = row.get("metadata") or {}
            patch = metadata.get("model_patch_sha256") or metadata.get("code_diff_sha256") or ""
            if patch:
                patches.add(str(patch))
            remaining = [candidate for candidate in remaining if candidate.get("uuid") != row.get("uuid")]
        selected.extend(chosen)
    return sorted(
        selected,
        key=lambda row: (
            (row.get("metadata") or {}).get("difficulty") or "",
            (row.get("metadata") or {}).get("language") or "",
            row.get("task_id") or "",
            (row.get("metadata") or {}).get("rollout_id") or "",
            row.get("uuid") or "",
        ),
    )


def full_index_row(row: dict[str, Any], line_number: int) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    compact = base.record_to_index(row, line_number)
    return {
        **compact,
        "has_result_json": metadata.get("has_result_json"),
        "incomplete": metadata.get("incomplete"),
        "result_json_missing": metadata.get("result_json_missing"),
        "truncated": metadata.get("truncated"),
        "truncated_for_reasoning_threshold": metadata.get("truncated_for_reasoning_threshold"),
        "truncation_reason": metadata.get("truncation_reason"),
        "original_message_count": metadata.get("original_message_count"),
        "original_assistant_message_count": metadata.get("original_assistant_message_count"),
        "original_percent_messages_with_reasoning": metadata.get("original_percent_messages_with_reasoning"),
        "uses_updated_alignment": metadata.get("uses_updated_alignment"),
        "datagen_code_commit": metadata.get("datagen_code_commit"),
        "benchmark_profile": metadata.get("benchmark_profile"),
        "outside_original_high_quality_set": metadata.get("outside_original_high_quality_set"),
        "provider_usage": metadata.get("provider_usage"),
    }


def write_dataset(dataset_dir: Path, rows: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    base.make_dirs(dataset_dir)
    data_path = dataset_dir / "data" / "train.jsonl"
    tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    index_rows: list[dict[str, Any]] = []
    full_index_rows: list[dict[str, Any]] = []
    task_ids: list[str] = []
    with tmp.open("w", encoding="utf-8") as handle:
        for line_number, row in enumerate(rows):
            metadata = row.get("metadata") or {}
            metadata["dataset"] = "deepswe-highquality-3x-duplicate-reasoning-90pct"
            row["metadata"] = metadata
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            index_rows.append(base.record_to_index(row, line_number))
            full_index_rows.append(full_index_row(row, line_number))
            task_id = row.get("task_id") or metadata.get("task_id")
            if task_id:
                task_ids.append(str(task_id))
    tmp.replace(data_path)
    base.write_index_files(dataset_dir, index_rows)
    full_index_path = dataset_dir / "metadata" / "full_index.jsonl"
    with full_index_path.open("w", encoding="utf-8") as handle:
        for row in full_index_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    unique_task_ids = sorted(set(task_ids))
    (dataset_dir / "metadata" / "task_ids.txt").write_text("\n".join(unique_task_ids) + "\n", encoding="utf-8")
    (dataset_dir / "metadata" / "exclude_task_ids_for_future_generation.txt").write_text(
        "\n".join(unique_task_ids) + "\n",
        encoding="utf-8",
    )
    base.write_manifest_and_readme(dataset_dir, dataset_dir.name, index_rows, extra)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "full_index_jsonl": str(full_index_path),
            "task_ids_file": str(dataset_dir / "metadata" / "task_ids.txt"),
            "exclude_task_ids_for_future_generation": str(
                dataset_dir / "metadata" / "exclude_task_ids_for_future_generation.txt"
            ),
            "truncated_rows": sum(1 for row in full_index_rows if row.get("truncated")),
            "incomplete_rows": sum(1 for row in full_index_rows if row.get("incomplete")),
            "rows_by_truncated": dict(Counter(str(row.get("truncated")).lower() for row in full_index_rows)),
            "rows_by_incomplete": dict(Counter(str(row.get("incomplete")).lower() for row in full_index_rows)),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    run_roots = args.run_root or [Path(path) for path in DEFAULT_RUN_ROOTS]
    records: list[dict[str, Any]] = []
    seen_trajectory_paths: set[str] = set()
    skipped: Counter[str] = Counter()

    if args.base_all_jsonl.exists():
        with args.base_all_jsonl.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if args.limit and len(records) >= args.limit:
                    break
                try:
                    row = build_existing_record(json.loads(line), args.threshold)
                except Exception:
                    skipped["existing_json_error"] += 1
                    continue
                if row is None:
                    skipped["existing_below_threshold_after_prefix"] += 1
                    continue
                trajectory_path = (row.get("metadata") or {}).get("trajectory_path")
                if trajectory_path:
                    seen_trajectory_paths.add(str(trajectory_path))
                records.append(row)
                if len(records) and len(records) % 1000 == 0:
                    print(f"existing_records={len(records)} elapsed={time.time() - started:.1f}s", flush=True)
    else:
        skipped["missing_base_all_jsonl"] += 1

    for run_root in run_roots:
        if args.limit and len(records) >= args.limit:
            break
        if not run_root.exists():
            skipped["missing_run_root"] += 1
            continue
        for workspace in iter_workspaces(run_root):
            if args.limit and len(records) >= args.limit:
                break
            trajectory_path = str(workspace / "agent" / "mini-swe-agent.trajectory.json")
            if trajectory_path in seen_trajectory_paths:
                skipped["duplicate_trajectory_path"] += 1
                continue
            seen_trajectory_paths.add(trajectory_path)
            row = build_trace_record(run_root, workspace, args.threshold)
            if row is None:
                skipped["run_trace_below_threshold_after_prefix_or_invalid"] += 1
                continue
            records.append(row)
        print(f"scanned_run_root={run_root} total_records={len(records)} elapsed={time.time() - started:.1f}s", flush=True)

    selected = select_records(records, args.max_per_task)
    write_dataset(
        args.output_dir,
        selected,
        {
            "builder": str(Path(__file__).resolve()),
            "base_all_jsonl": str(args.base_all_jsonl),
            "run_roots": [str(path) for path in run_roots],
            "reasoning_threshold": args.threshold,
            "max_traces_per_task": args.max_per_task,
            "candidate_records_after_prefix_filter": len(records),
            "candidate_unique_tasks": len({row.get("task_id") for row in records if row.get("task_id")}),
            "skipped": dict(sorted(skipped.items())),
            "selection_rule": (
                "Require >=2 assistant messages and >=90% assistant-message reasoning coverage. "
                "Rows below the threshold are truncated to the longest valid prefix and marked. "
                "Select up to 3 traces per task, prioritizing pass, complete result metadata, fewer "
                "assistant messages without reasoning, higher reasoning percent, non-truncated rows, "
                "shorter traces, and different code diffs for duplicate slots."
            ),
        },
    )
    final = {
        "output_dir": str(args.output_dir),
        "elapsed_sec": time.time() - started,
        "candidate_records_after_prefix_filter": len(records),
        "selected_records": len(selected),
        "selected_unique_tasks": len({row.get("task_id") for row in selected if row.get("task_id")}),
        "summary": base.summarize([base.record_to_index(row, i) for i, row in enumerate(selected)]),
        "skipped": dict(sorted(skipped.items())),
    }
    (args.output_dir / "build_summary.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
