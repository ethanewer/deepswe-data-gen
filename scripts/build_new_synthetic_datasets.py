#!/usr/bin/env python3
"""Build SFT-style datasets from generated SWE-rebench mini-swe-agent traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data"),
    )
    parser.add_argument(
        "--unique-name",
        default="deepswe-highquality-unique-reasoning",
    )
    parser.add_argument(
        "--all-name",
        default="deepswe-highquality-all-trajectories",
    )
    return parser.parse_args()


def iter_workspaces(runs_root: Path):
    for run_root in sorted(runs_root.glob("datagen-*")):
        traces_root = run_root / "pyxis-traces"
        if not traces_root.is_dir():
            continue
        for style_ent in sorted(os.scandir(traces_root), key=lambda e: e.name):
            if not style_ent.is_dir():
                continue
            for model_ent in sorted(os.scandir(style_ent.path), key=lambda e: e.name):
                if not model_ent.is_dir():
                    continue
                for rollout_ent in sorted(os.scandir(model_ent.path), key=lambda e: e.name):
                    if not rollout_ent.is_dir():
                        continue
                    for instance_ent in sorted(os.scandir(rollout_ent.path), key=lambda e: e.name):
                        if not instance_ent.is_dir():
                            continue
                        workspace = Path(instance_ent.path)
                        trajectory_path = workspace / "agent" / "mini-swe-agent.trajectory.json"
                        if trajectory_path.exists() and trajectory_path.stat().st_size > 2:
                            yield {
                                "run_root": run_root,
                                "workspace": workspace,
                                "trajectory_path": trajectory_path,
                                "instruction_style": style_ent.name,
                                "model_from_path": model_ent.name,
                                "rollout_id": rollout_ent.name,
                                "instance_id": instance_ent.name,
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


def stable_uuid(parts: list[str]) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8", errors="replace")).hexdigest()
    return digest


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


def collect_message_reasoning(messages: list[Any]) -> list[dict[str, Any]]:
    reasoning: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        candidates: list[tuple[str, Any]] = []
        for key in ("reasoning_content", "reasoning", "reasoning_details", "thinking"):
            candidates.append((f"messages[{index}].{key}", message.get(key)))
        provider = message.get("provider_specific_fields")
        if isinstance(provider, dict):
            for key in ("reasoning_content", "reasoning", "reasoning_details", "thinking"):
                candidates.append((f"messages[{index}].provider_specific_fields.{key}", provider.get(key)))
        # Some mini-swe-agent messages preserve the raw provider response under
        # extra.response. Inspect immediate response choices here before the
        # normalizer drops that duplicate payload from the SFT record.
        extra = message.get("extra")
        response = extra.get("response") if isinstance(extra, dict) else None
        choices = response.get("choices") if isinstance(response, dict) else None
        if isinstance(choices, list):
            for choice_index, choice in enumerate(choices):
                choice_message = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(choice_message, dict):
                    for key in ("reasoning_content", "reasoning", "reasoning_details", "thinking"):
                        candidates.append(
                            (
                                f"messages[{index}].extra.response.choices[{choice_index}].message.{key}",
                                choice_message.get(key),
                            )
                        )
        for path, value in candidates:
            if isinstance(value, str):
                content = value.strip()
            elif isinstance(value, (list, dict)) and value:
                content = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            else:
                continue
            if not content:
                continue
            key = (index, content)
            if key in seen:
                continue
            seen.add(key)
            reasoning.append(
                {
                    "message_index": index,
                    "path": path,
                    "content": content,
                }
            )
    return reasoning


def normalize_messages(messages: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        clean = {}
        for key, value in message.items():
            if key == "extra" and isinstance(value, dict):
                extra = {extra_key: extra_value for extra_key, extra_value in value.items() if extra_key != "response"}
                if extra:
                    clean[key] = extra
            else:
                clean[key] = value
        normalized.append(clean)
    return normalized


def trajectory_length(messages: list[Any]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def build_record(item: dict[str, Any]) -> dict[str, Any] | None:
    trajectory = load_json(item["trajectory_path"])
    raw_messages = trajectory.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return None
    reasoning = collect_message_reasoning(raw_messages)
    messages = normalize_messages(raw_messages)

    workspace = item["workspace"]
    result = load_json(workspace / "result.json")
    metadata = load_json(workspace / "metadata.json")
    info = trajectory.get("info") if isinstance(trajectory.get("info"), dict) else {}
    model_stats = info.get("model_stats") if isinstance(info.get("model_stats"), dict) else {}
    config = info.get("config") if isinstance(info.get("config"), dict) else {}

    instance_id = first_nonempty(
        result.get("instance_id"),
        metadata.get("instance_id"),
        item["instance_id"],
    )
    rollout_id = first_nonempty(
        result.get("rollout_id"),
        metadata.get("rollout_id"),
        item["rollout_id"],
    )
    model = first_nonempty(
        result.get("model"),
        metadata.get("model"),
        item["model_from_path"],
    )
    litellm_model = first_nonempty(result.get("litellm_model"), metadata.get("litellm_model"))
    difficulty = first_nonempty(result.get("difficulty"), metadata.get("difficulty"))
    language = first_nonempty(result.get("language"), metadata.get("language"))
    repo = first_nonempty(result.get("repo"), metadata.get("repo"))
    instruction_style = first_nonempty(
        result.get("instruction_style"),
        metadata.get("instruction_style"),
        item["instruction_style"],
    )

    reward = result.get("reward", 0)
    passed = reward == 1
    length = trajectory_length(messages)

    patch_path = Path(first_nonempty(result.get("patch_path"), workspace / "model.patch"))
    if patch_path.is_absolute() and not patch_path.exists():
        # Older container-side results sometimes record /workspace paths.
        try:
            rel = patch_path.relative_to("/workspace")
            patch_path = workspace / rel
        except ValueError:
            pass
    patch_text = read_text_if_small(patch_path)

    record = {
        "uuid": stable_uuid(
            [
                str(item["trajectory_path"]),
                instance_id,
                rollout_id,
                model,
                instruction_style,
            ]
        ),
        "messages": messages,
        "tools": [BASH_TOOL],
        "reasoning": reasoning,
        "metadata": {
            "dataset": "deepswe-highquality-synthetic",
            "source": "deepswe-data-gen",
            "source_run_root": str(item["run_root"]),
            "workspace": str(workspace),
            "trajectory_path": str(item["trajectory_path"]),
            "result_path": str(workspace / "result.json"),
            "patch_path": str(patch_path) if patch_path else "",
            "instance_id": instance_id,
            "rollout_id": rollout_id,
            "repo": repo,
            "difficulty": difficulty,
            "language": language,
            "instruction_style": instruction_style,
            "benchmark_profile": first_nonempty(result.get("benchmark_profile"), metadata.get("benchmark_profile")),
            "mini_swe_agent_config_file": first_nonempty(
                result.get("mini_swe_agent_config_file"),
                metadata.get("mini_swe_agent_config_file"),
            ),
            "model": model,
            "litellm_model": litellm_model,
            "passed": passed,
            "reward": reward,
            "agent_exit_status": first_nonempty(result.get("agent_exit_status"), info.get("exit_status")),
            "agent_exception_type": (
                (result.get("agent_exception") or {}).get("type")
                if isinstance(result.get("agent_exception"), dict)
                else ""
            ),
            "api_calls": result.get("api_calls", model_stats.get("api_calls", 0)),
            "cost_usd": result.get("cost_usd", model_stats.get("instance_cost", 0.0)),
            "message_count": len(messages),
            "trajectory_chars": length,
            "trajectory_bytes": item["trajectory_path"].stat().st_size,
            "has_reasoning": bool(reasoning),
            "reasoning_turns": len(reasoning),
            "reasoning_chars": sum(len(x["content"]) for x in reasoning),
            "trajectory_format": trajectory.get("trajectory_format", ""),
            "mini_swe_agent_version": info.get("mini_version", ""),
            "model_patch_bytes": len(patch_text.encode("utf-8")) if patch_text else 0,
            "model_config": config.get("model", {}) if isinstance(config.get("model"), dict) else {},
        },
    }
    if patch_text:
        record["model_patch"] = patch_text
    return record


def write_readme(path: Path, title: str, summary: dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        "Synthetic SWE-rebench mini-swe-agent trajectories generated under `/wbl-fast`.",
        "",
        "Format: JSONL records with `uuid`, `messages`, `tools`, `reasoning`, and `metadata`.",
        "Assistant reasoning is preserved in-place inside `messages` when present and is also",
        "extracted into the top-level `reasoning` list for convenience.",
        "The duplicated raw provider payload at `message.extra.response` is omitted from",
        "`messages`; the original trajectory path is preserved in `metadata.trajectory_path`.",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def empty_summary() -> dict[str, Any]:
    return {
        "records": 0,
        "with_reasoning": 0,
        "by_difficulty": Counter(),
        "by_model": Counter(),
        "by_instruction_style": Counter(),
        "by_passed": Counter(),
    }


def update_summary(summary: dict[str, Any], record: dict[str, Any]) -> None:
    metadata = record["metadata"]
    summary["records"] += 1
    summary["by_difficulty"][metadata.get("difficulty", "")] += 1
    summary["by_model"][metadata.get("model", "")] += 1
    summary["by_instruction_style"][metadata.get("instruction_style", "")] += 1
    summary["by_passed"][str(metadata.get("passed", False)).lower()] += 1
    if metadata.get("has_reasoning"):
        summary["with_reasoning"] += 1


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "records": summary["records"],
        "with_reasoning": summary["with_reasoning"],
        "by_difficulty": dict(sorted(summary["by_difficulty"].items())),
        "by_model": dict(sorted(summary["by_model"].items())),
        "by_instruction_style": dict(sorted(summary["by_instruction_style"].items())),
        "by_passed": dict(sorted(summary["by_passed"].items())),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    skipped = Counter()
    best_by_task: dict[str, dict[str, Any]] = {}
    all_summary_work = empty_summary()

    all_dir = args.output_root / args.all_name
    all_data = all_dir / "data" / "train.jsonl"
    all_data.parent.mkdir(parents=True, exist_ok=True)
    all_tmp = all_data.with_suffix(all_data.suffix + ".tmp")
    all_handle = all_tmp.open("w", encoding="utf-8")
    for item in iter_workspaces(args.runs_root):
        record = build_record(item)
        if record is None:
            skipped["invalid_trajectory"] += 1
            continue
        all_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        update_summary(all_summary_work, record)
        metadata = record["metadata"]
        if not metadata.get("has_reasoning"):
            continue
        instance_id = metadata.get("instance_id", "")
        if not instance_id:
            continue
        current = best_by_task.get(instance_id)
        candidate_key = (
            1 if metadata.get("passed") else 0,
            -int(metadata.get("trajectory_chars", 0)),
        )
        if current is None:
            best_by_task[instance_id] = {
                "key": candidate_key,
                "item": item,
                "instance_id": instance_id,
                "difficulty": metadata.get("difficulty", ""),
                "language": metadata.get("language", ""),
            }
            continue
        if candidate_key > current["key"]:
            best_by_task[instance_id] = {
                "key": candidate_key,
                "item": item,
                "instance_id": instance_id,
                "difficulty": metadata.get("difficulty", ""),
                "language": metadata.get("language", ""),
            }
    all_handle.close()
    all_tmp.replace(all_data)

    unique_entries = sorted(
        best_by_task.values(),
        key=lambda r: (
            r.get("difficulty", ""),
            r.get("language", ""),
            r.get("instance_id", ""),
        ),
    )

    unique_dir = args.output_root / args.unique_name
    unique_data = unique_dir / "data" / "train.jsonl"
    unique_data.parent.mkdir(parents=True, exist_ok=True)
    unique_tmp = unique_data.with_suffix(unique_data.suffix + ".tmp")

    unique_summary_work = empty_summary()
    with unique_tmp.open("w", encoding="utf-8") as handle:
        for entry in unique_entries:
            record = build_record(entry["item"])
            if record is None:
                skipped["invalid_unique_rebuild"] += 1
                continue
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            update_summary(unique_summary_work, record)
    unique_tmp.replace(unique_data)
    unique_summary = finalize_summary(unique_summary_work)
    all_summary = finalize_summary(all_summary_work)
    common = {
        "created_at_unix": time.time(),
        "builder": str(Path(__file__).resolve()),
        "runs_root": str(args.runs_root),
        "skipped": dict(sorted(skipped.items())),
        "selection_rule_unique": "one per instance_id; only has_reasoning; max(passed, -trajectory_chars)",
    }
    unique_manifest = {
        **common,
        "name": args.unique_name,
        "data_file": str(unique_data),
        **unique_summary,
    }
    all_manifest = {
        **common,
        "name": args.all_name,
        "data_file": str(all_data),
        **all_summary,
    }
    unique_dir.mkdir(parents=True, exist_ok=True)
    all_dir.mkdir(parents=True, exist_ok=True)
    (unique_dir / "manifest.json").write_text(
        json.dumps(unique_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (all_dir / "manifest.json").write_text(
        json.dumps(all_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(unique_dir / "README.md", args.unique_name, unique_manifest)
    write_readme(all_dir / "README.md", args.all_name, all_manifest)
    print(json.dumps({"unique": unique_manifest, "all": all_manifest, "elapsed_sec": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
