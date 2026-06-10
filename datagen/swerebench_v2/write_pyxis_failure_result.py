#!/usr/bin/env python3
"""Write a result.json when a Pyxis task container cannot start."""

from __future__ import annotations

import argparse
import fcntl
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--rollout-id", default="r00")
    parser.add_argument("--model", required=True)
    parser.add_argument("--litellm-model", required=True)
    parser.add_argument("--instruction-style", required=True)
    parser.add_argument("--difficulty", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--repo", default="")
    parser.add_argument("--outside-original-high-quality-set", default="false")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--pyxis-image", required=True)
    parser.add_argument("--error-type", default="PyxisContainerStartError")
    parser.add_argument("--error-message", default="")
    parser.add_argument("--exit-status", type=int, required=True)
    parser.add_argument("--stdout-log", default="")
    parser.add_argument("--stderr-log", default="")
    return parser.parse_args()


def result_index_path(workspace: Path) -> Path | None:
    for parent in workspace.parents:
        if parent.name == "pyxis-traces":
            return parent.parent / "manifest" / "result_index.jsonl"
    return None


def append_result_index(workspace: Path, result: dict) -> None:
    index_path = result_index_path(workspace)
    if index_path is None:
        return
    trajectory_path = Path(result["trajectory_path"])
    try:
        trajectory_saved = trajectory_path.exists() and trajectory_path.stat().st_size > 2
    except OSError:
        trajectory_saved = False
    record = {
        "instance_id": result.get("instance_id"),
        "rollout_id": result.get("rollout_id"),
        "model": result.get("model"),
        "litellm_model": result.get("litellm_model"),
        "instruction_style": result.get("instruction_style"),
        "difficulty": result.get("difficulty"),
        "language": result.get("language"),
        "repo": result.get("repo"),
        "outside_original_high_quality_set": result.get("outside_original_high_quality_set", False),
        "finished_at": result.get("finished_at"),
        "agent_exit_status": result.get("agent_exit_status"),
        "agent_exception_type": (result.get("agent_exception") or {}).get("type"),
        "api_calls": result.get("api_calls", 0),
        "cost_usd": result.get("cost_usd", 0.0),
        "reward": result.get("reward", 0),
        "trajectory_saved": trajectory_saved,
        "result_path": str(workspace / "result.json"),
        "trajectory_path": str(trajectory_path),
        "patch_path": result.get("patch_path"),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        fcntl.flock(handle, fcntl.LOCK_UN)


def write_setup_failure_trajectory(args: argparse.Namespace, result: dict) -> None:
    trajectory_path = Path(result["trajectory_path"])
    if trajectory_path.exists():
        return
    instruction_path = Path(args.task_dir) / "instruction.md"
    if instruction_path.exists():
        instruction = instruction_path.read_text(encoding="utf-8", errors="replace")
    else:
        instruction = ""
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = {
        "info": {
            "setup_failure": result.get("agent_exception"),
            "container_image": args.image,
            "runtime_image": args.pyxis_image,
            "stdout_log": args.stdout_log,
            "stderr_log": args.stderr_log,
        },
        "messages": [{"role": "user", "content": instruction}],
        "trajectory_format": "mini-swe-agent-v2-setup-failure",
    }
    trajectory_path.write_text(json.dumps(trajectory, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    (args.workspace / "agent").mkdir(parents=True, exist_ok=True)
    message = args.error_message or f"srun/Pyxis exited with status {args.exit_status}"
    result = {
        "instance_id": args.instance_id,
        "rollout_id": args.rollout_id,
        "model": args.model,
        "litellm_model": args.litellm_model,
        "instruction_style": args.instruction_style,
        "difficulty": args.difficulty,
        "language": args.language,
        "repo": args.repo,
        "outside_original_high_quality_set": str(args.outside_original_high_quality_set).lower()
        in {"1", "true", "yes"},
        "task_dir": args.task_dir,
        "docker_image": args.image,
        "pyxis_image": args.pyxis_image,
        "finished_at": utc_now(),
        "agent_exit_status": args.error_type,
        "agent_exception": {
            "type": args.error_type,
            "message": message,
        },
        "api_calls": 0,
        "cost_usd": 0.0,
        "patch_path": str(args.workspace / "model.patch"),
        "trajectory_path": str(args.workspace / "agent" / "mini-swe-agent.trajectory.json"),
        "reward": 0,
        "verifier": None,
        "stdout_log": args.stdout_log,
        "stderr_log": args.stderr_log,
    }
    write_setup_failure_trajectory(args, result)
    (args.workspace / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    try:
        append_result_index(args.workspace, result)
    except Exception:  # noqa: BLE001 - result.json has already been written
        (args.workspace / "result_index_error.log").write_text(traceback.format_exc())


if __name__ == "__main__":
    main()
