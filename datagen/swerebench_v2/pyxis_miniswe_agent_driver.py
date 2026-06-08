#!/usr/bin/env python3
"""Run one SWE-rebench task inside its Pyxis task container.

This intentionally avoids Pier's Docker-Compose environment layer. Slurm/Pyxis
already started the task image, so mini-swe-agent runs against the local shell
inside that container and the generated patch is verified with the task's
existing Harbor verifier script.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
import tomllib
import traceback
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.utils.serialize import recursive_merge


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--litellm-model", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--extra-body-json", default="")
    parser.add_argument("--instruction-style", default="original")
    parser.add_argument("--difficulty", default="")
    parser.add_argument("--language", default="")
    parser.add_argument("--repo", default="")
    parser.add_argument("--rollout-id", default="r00")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--model-timeout", type=int, default=600)
    parser.add_argument("--agent-wall-time-limit", type=int, default=2700)
    parser.add_argument("--command-timeout", type=int, default=180)
    return parser.parse_args()


def run_shell(
    command: str,
    *,
    cwd: Path | str | None = None,
    timeout: int = 300,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {command}\n{result.stdout[-4000:]}"
        )
    return result


def load_task_toml(task_dir: Path) -> dict[str, Any]:
    path = task_dir / "task.toml"
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")

        def field(name: str) -> str:
            match = re.search(rf'(?m)^{re.escape(name)}\s*=\s*"((?:\\.|[^"])*)"', text)
            if not match:
                raise
            return bytes(match.group(1), "utf-8").decode("unicode_escape", errors="replace")

        return {
            "metadata": {"base_commit_hash": field("base_commit_hash")},
            "environment": {
                "docker_image": field("docker_image"),
                "workdir": field("workdir"),
            },
        }


def configure_runtime_env(args: argparse.Namespace) -> None:
    home = args.workspace / "home"
    cache_root = Path("/wbl-fast/usrs/ee/code-swe-data/cache")
    for path in (
        home,
        cache_root / "hf",
        cache_root / "xdg",
        cache_root / "uv",
        cache_root / "pip",
    ):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HOME"] = str(home)
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("UV_CACHE_DIR", str(cache_root / "uv"))
    os.environ.setdefault("PIP_CACHE_DIR", str(cache_root / "pip"))
    os.environ.setdefault("PAGER", "cat")
    os.environ.setdefault("MANPAGER", "cat")
    os.environ.setdefault("LESS", "-R")
    os.environ.setdefault("PIP_PROGRESS_BAR", "off")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = args.api_base
        os.environ["OPENAI_API_BASE"] = args.api_base
    if args.api_key_env == "DEEPSEEK_API_KEY" and os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]


def prepare_aliases(args: argparse.Namespace) -> None:
    logs = args.workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    for link_path, target in (
        (Path("/logs"), logs),
        (Path("/tests"), args.task_dir / "tests"),
    ):
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink():
                link_path.unlink()
            elif link_path.is_dir() and not any(link_path.iterdir()):
                link_path.rmdir()
            else:
                continue
        if not link_path.exists():
            link_path.symlink_to(target, target_is_directory=True)


def build_agent(args: argparse.Namespace, workdir: str, trajectory_path: Path) -> DefaultAgent:
    config = yaml.safe_load(args.config_file.read_text(encoding="utf-8")) or {}
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None

    thinking_enabled = (
        isinstance(extra_body, dict)
        and isinstance(extra_body.get("thinking"), dict)
        and extra_body["thinking"].get("type") == "enabled"
    )
    model_kwargs: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "timeout": args.model_timeout,
    }
    if thinking_enabled:
        model_kwargs["reasoning_effort"] = args.reasoning_effort
    else:
        model_kwargs["temperature"] = args.temperature
    if args.api_base:
        model_kwargs["api_base"] = args.api_base
    if extra_body:
        model_kwargs["extra_body"] = extra_body

    model_config = recursive_merge(
        config.get("model", {}),
        {
            "model_name": args.litellm_model,
            "model_class": "litellm",
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
        },
    )
    agent_config = recursive_merge(
        config.get("agent", {}),
        {
            "output_path": trajectory_path,
            "wall_time_limit_seconds": args.agent_wall_time_limit,
        },
    )
    model = get_model(config=model_config)
    env = LocalEnvironment(
        cwd=workdir,
        timeout=args.command_timeout,
        env={
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
        },
    )
    return DefaultAgent(model, env, **agent_config)


def reset_repo(workdir: str, base_commit: str) -> dict[str, Any]:
    quoted = shlex.quote(workdir)
    base = shlex.quote(base_commit)
    command = (
        f"cd {quoted} && "
        "git config --global --add safe.directory \"$(pwd)\" >/dev/null 2>&1 || true && "
        f"git rev-parse --verify {base}^{{commit}} >/dev/null && "
        f"git reset --hard {base} && "
        f"git checkout {base} && "
        "git clean -fd"
    )
    result = run_shell(command, timeout=300)
    return {"returncode": result.returncode, "output_tail": result.stdout[-4000:]}


def collect_patch(workdir: str, patch_path: Path) -> dict[str, Any]:
    quoted = shlex.quote(workdir)
    command = f"cd {quoted} && git add -A -- . && git diff --cached --binary"
    result = run_shell(command, timeout=120)
    patch_path.write_text(result.stdout, encoding="utf-8")
    return {
        "returncode": result.returncode,
        "patch_bytes": patch_path.stat().st_size if patch_path.exists() else 0,
    }


def run_verifier(args: argparse.Namespace) -> dict[str, Any]:
    stdout_path = args.workspace / "verifier.stdout.log"
    started = utc_now()
    try:
        result = run_shell("bash /tests/test.sh", timeout=3600)
        stdout_path.write_text(result.stdout, encoding="utf-8")
        returncode = result.returncode
        exception = None
    except Exception as exc:  # noqa: BLE001 - record every verifier failure
        stdout_path.write_text(traceback.format_exc(), encoding="utf-8")
        returncode = -1
        exception = {"type": type(exc).__name__, "message": str(exc)}

    reward_path = args.workspace / "logs" / "verifier" / "reward.txt"
    reward = 0
    if reward_path.exists():
        reward = 1 if reward_path.read_text(encoding="utf-8", errors="replace").strip() == "1" else 0
    return {
        "started_at": started,
        "finished_at": utc_now(),
        "returncode": returncode,
        "reward": reward,
        "stdout_path": str(stdout_path),
        "exception": exception,
    }


def result_index_path(workspace: Path) -> Path | None:
    for parent in workspace.parents:
        if parent.name == "pyxis-traces":
            return parent.parent / "manifest" / "result_index.jsonl"
    return None


def append_result_index(workspace: Path, result: dict[str, Any]) -> None:
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
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    (args.workspace / "agent").mkdir(parents=True, exist_ok=True)
    (args.workspace / "artifacts").mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    configure_runtime_env(args)
    prepare_aliases(args)

    task_data = load_task_toml(args.task_dir)
    metadata = task_data["metadata"]
    environment = task_data["environment"]
    workdir = environment.get("workdir") or "/testbed"
    base_commit = metadata["base_commit_hash"]
    instruction = (args.task_dir / "instruction.md").read_text(encoding="utf-8")

    metadata_record = {
        "instance_id": args.instance_id,
        "rollout_id": args.rollout_id,
        "model": args.model,
        "litellm_model": args.litellm_model,
        "instruction_style": args.instruction_style,
        "difficulty": args.difficulty,
        "language": args.language,
        "repo": args.repo,
        "task_dir": str(args.task_dir),
        "docker_image": environment.get("docker_image"),
        "workdir": workdir,
        "base_commit": base_commit,
        "started_at": started_at,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "extra_body_json": args.extra_body_json,
    }
    (args.workspace / "metadata.json").write_text(json.dumps(metadata_record, indent=2) + "\n")

    result: dict[str, Any] = {
        **metadata_record,
        "finished_at": None,
        "agent_exit_status": "",
        "agent_exception": None,
        "api_calls": 0,
        "cost_usd": 0.0,
        "patch_path": str(args.workspace / "model.patch"),
        "trajectory_path": str(args.workspace / "agent" / "mini-swe-agent.trajectory.json"),
        "reward": 0,
        "verifier": None,
    }

    try:
        result["repo_reset"] = reset_repo(workdir, base_commit)
        agent = build_agent(args, workdir, Path(result["trajectory_path"]))
        try:
            agent_info = agent.run(instruction)
            result["agent_exit_status"] = agent_info.get("exit_status", "")
        except Exception as exc:  # noqa: BLE001 - failed traces are still useful
            result["agent_exit_status"] = type(exc).__name__
            result["agent_exception"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            trajectory = agent.save(Path(result["trajectory_path"]))
            model_stats = trajectory.get("info", {}).get("model_stats", {})
            result["api_calls"] = model_stats.get("api_calls", 0)
            result["cost_usd"] = model_stats.get("instance_cost", 0.0)

        result["patch"] = collect_patch(workdir, Path(result["patch_path"]))
        verifier = run_verifier(args)
        result["verifier"] = verifier
        result["reward"] = verifier.get("reward", 0)
        verifier_patch = args.workspace / "logs" / "artifacts" / "model.patch"
        if verifier_patch.exists():
            result["patch_path"] = str(verifier_patch)
            result["patch"]["patch_bytes"] = verifier_patch.stat().st_size
    except Exception as exc:  # noqa: BLE001 - write a result for setup failures too
        result["agent_exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        result["finished_at"] = utc_now()
        (args.workspace / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        try:
            append_result_index(args.workspace, result)
        except Exception:  # noqa: BLE001 - result.json has already been written
            (args.workspace / "result_index_error.log").write_text(traceback.format_exc())


if __name__ == "__main__":
    main()
