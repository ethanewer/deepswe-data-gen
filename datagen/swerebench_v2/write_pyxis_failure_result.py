#!/usr/bin/env python3
"""Write a result.json when a Pyxis task container cannot start."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--pyxis-image", required=True)
    parser.add_argument("--exit-status", type=int, required=True)
    parser.add_argument("--stdout-log", default="")
    parser.add_argument("--stderr-log", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    result = {
        "instance_id": args.instance_id,
        "rollout_id": args.rollout_id,
        "model": args.model,
        "litellm_model": args.litellm_model,
        "instruction_style": args.instruction_style,
        "difficulty": args.difficulty,
        "language": args.language,
        "repo": args.repo,
        "task_dir": args.task_dir,
        "docker_image": args.image,
        "pyxis_image": args.pyxis_image,
        "finished_at": utc_now(),
        "agent_exit_status": "PyxisContainerStartError",
        "agent_exception": {
            "type": "PyxisContainerStartError",
            "message": f"srun/Pyxis exited with status {args.exit_status}",
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
    (args.workspace / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
