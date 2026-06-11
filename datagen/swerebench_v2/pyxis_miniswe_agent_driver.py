#!/usr/bin/env python3
"""Run one SWE-rebench task inside its Pyxis task container.

This intentionally avoids Pier's Docker-Compose environment layer. Slurm/Pyxis
already started the task image, so mini-swe-agent runs against the local shell
inside that container and the generated patch is verified with the task's
existing Harbor verifier script.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
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

from eval.minisweagent_pin import MINI_SWE_AGENT_GIT_SHA
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.utils.serialize import recursive_merge


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PROFILE_BY_STYLE = {
    "original": "swebench-multilingual",
    "swe_rebench": "swebench-multilingual",
    "deepswe": "deepswe",
    "rewritten": "deepswe",
    "planned": "deepswe",
}
BENCHMARK_PROFILE_CHOICES = ("auto", "swebench-multilingual", "deepswe", "datagen-strict")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def str_to_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def datagen_code_commit() -> str:
    from_env = os.environ.get("DATAGEN_CODE_COMMIT", "").strip()
    if from_env:
        return from_env
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--benchmark-profile", choices=BENCHMARK_PROFILE_CHOICES, default="auto")
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
    parser.add_argument("--outside-original-high-quality-set", default="false")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--model-timeout", type=int, default=600)
    parser.add_argument("--agent-wall-time-limit", type=int, default=2700)
    parser.add_argument("--uses-updated-alignment", choices=("true", "false"), default="true")
    parser.add_argument("--eligible-for-controlled-comparison", choices=("true", "false"), default="false")
    parser.add_argument("--reason-excluded-from-comparison", default="")
    parser.add_argument("--command-timeout", type=int)
    return parser.parse_args()


def resolve_benchmark_profile(args: argparse.Namespace) -> str:
    if args.benchmark_profile != "auto":
        return args.benchmark_profile
    return BENCHMARK_PROFILE_BY_STYLE.get(args.instruction_style, "datagen-strict")


def resolve_model_class_and_name(profile: str, litellm_model: str) -> tuple[str, str]:
    if profile == "swebench-multilingual":
        return "litellm", litellm_model
    if profile == "deepswe":
        if litellm_model.startswith("openai/"):
            return "litellm_response", litellm_model
        if litellm_model.startswith("openrouter/"):
            return "openrouter", litellm_model.removeprefix("openrouter/")
        return "litellm", litellm_model
    if litellm_model.startswith("openrouter/"):
        return "openrouter", litellm_model.removeprefix("openrouter/")
    return "litellm", litellm_model


def build_model_kwargs(
    args: argparse.Namespace,
    *,
    benchmark_profile: str,
    extra_body: dict[str, Any] | None,
    use_native_openrouter: bool,
) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.api_base:
        model_kwargs["api_base"] = args.api_base

    if benchmark_profile == "datagen-strict":
        if use_native_openrouter:
            model_kwargs["request_timeout"] = args.model_timeout
        else:
            model_kwargs["timeout"] = args.model_timeout
        thinking_enabled = (
            isinstance(extra_body, dict)
            and isinstance(extra_body.get("thinking"), dict)
            and extra_body["thinking"].get("type") == "enabled"
        )
        qwen_thinking_enabled = (
            isinstance(extra_body, dict)
            and isinstance(extra_body.get("chat_template_kwargs"), dict)
            and extra_body["chat_template_kwargs"].get("enable_thinking") is True
        )
        reasoning_config = extra_body.get("reasoning") if isinstance(extra_body, dict) else None
        openrouter_reasoning_enabled = (
            isinstance(reasoning_config, dict)
            and not reasoning_config.get("exclude", False)
            and (
                reasoning_config.get("enabled") is True
                or bool(reasoning_config.get("effort"))
                or bool(reasoning_config.get("max_tokens"))
            )
        )
        reasoning_enabled = thinking_enabled or qwen_thinking_enabled or openrouter_reasoning_enabled
        if thinking_enabled:
            model_kwargs["reasoning_effort"] = args.reasoning_effort
        if reasoning_enabled and not qwen_thinking_enabled:
            model_kwargs.pop("temperature", None)
        if extra_body and use_native_openrouter:
            model_kwargs.update(extra_body)
        elif extra_body:
            model_kwargs["extra_body"] = extra_body
        return model_kwargs

    if extra_body:
        model_kwargs["extra_body"] = extra_body
    return model_kwargs


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
    if (
        args.api_key_env == "OPENAI_API_KEY"
        and args.api_base.startswith("http://")
        and ".integrated.pcluster:" in args.api_base
        and not os.environ.get("OPENAI_API_KEY")
    ):
        # Local OpenAI-compatible serving does not require auth, but the OpenAI
        # client still requires a non-empty key before sending the request.
        os.environ["OPENAI_API_KEY"] = "local-model-no-auth-required"
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


def ensure_testbed_alias(workdir: str, alias: Path = Path("/testbed")) -> dict[str, Any]:
    """Make /testbed resolve to the task workdir when the image uses another path."""
    target = Path(workdir)
    record: dict[str, Any] = {
        "alias": str(alias),
        "target": str(target),
        "created": False,
        "usable": False,
        "error": None,
    }
    try:
        if str(target) == str(alias):
            record["usable"] = alias.exists()
            return record
        if alias.is_symlink():
            if alias.resolve(strict=False) == target:
                record["usable"] = True
                return record
            alias.unlink()
        elif alias.exists():
            record["usable"] = True
            record["target"] = str(alias)
            return record
        if target.exists():
            alias.symlink_to(target, target_is_directory=True)
            record["created"] = True
            record["usable"] = True
        return record
    except Exception as exc:  # noqa: BLE001 - surface this in metadata, but do not fail setup
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["usable"] = alias.exists()
        return record


def prepare_agent_bin(args: argparse.Namespace) -> Path:
    agent_bin = args.workspace / "agent-bin"
    agent_bin.mkdir(parents=True, exist_ok=True)
    find_wrapper = agent_bin / "find"
    find_wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
  case "$arg" in
    /|/home|/home/*|/wbl-fast|/wbl-fast/*|/mnt|/mnt/*|/proc|/proc/*|/sys|/sys/*|/dev|/dev/*|/var|/var/*)
      echo "find: command timed out" >&2
      exit 2
      ;;
  esac
done
exec /usr/bin/find "$@"
""",
        encoding="utf-8",
    )
    find_wrapper.chmod(0o755)
    return agent_bin


def build_agent(args: argparse.Namespace, workdir: str, trajectory_path: Path) -> DefaultAgent:
    config = yaml.safe_load(args.config_file.read_text(encoding="utf-8")) or {}
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None
    benchmark_profile = resolve_benchmark_profile(args)
    environment_config = config.get("environment", {}) if isinstance(config.get("environment"), dict) else {}

    model_class, model_name = resolve_model_class_and_name(benchmark_profile, args.litellm_model)
    use_native_openrouter = model_class == "openrouter"
    model_kwargs = build_model_kwargs(
        args,
        benchmark_profile=benchmark_profile,
        extra_body=extra_body,
        use_native_openrouter=use_native_openrouter,
    )
    model_config = recursive_merge(
        config.get("model", {}),
        {
            "model_name": model_name,
            "model_class": model_class,
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
        },
    )
    agent_overrides: dict[str, Any] = {
        "output_path": trajectory_path,
        "wall_time_limit_seconds": args.agent_wall_time_limit,
    }
    if benchmark_profile == "deepswe":
        agent_overrides["cost_limit"] = 0

    agent_config = recursive_merge(
        config.get("agent", {}),
        agent_overrides,
    )
    model = get_model(config=model_config)
    agent_bin = prepare_agent_bin(args)
    configured_env = environment_config.get("env") if isinstance(environment_config.get("env"), dict) else {}
    local_env = {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
        **configured_env,
        "PATH": f"{agent_bin}:{os.environ.get('PATH', '')}",
    }
    command_timeout = args.command_timeout
    if command_timeout is None:
        configured_timeout = environment_config.get("timeout")
        command_timeout = int(configured_timeout) if configured_timeout is not None else 30
    env = LocalEnvironment(
        cwd=workdir,
        timeout=command_timeout,
        env=local_env,
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


def host_workspace_path(workspace: Path) -> Path:
    """Return the host-visible workspace when running under a /workspace mount."""
    home = os.environ.get("HOME")
    if home:
        home_path = Path(home)
        if home_path.name == "home":
            return home_path.parent
    return workspace


def result_index_path(workspace: Path) -> Path | None:
    for parent in host_workspace_path(workspace).parents:
        if parent.name == "pyxis-traces":
            return parent.parent / "manifest" / "result_index.jsonl"
    return None


def append_result_index(workspace: Path, result: dict[str, Any]) -> None:
    index_path = result_index_path(workspace)
    if index_path is None:
        return
    host_workspace = host_workspace_path(workspace)
    trajectory_path = Path(result["trajectory_path"])
    try:
        trajectory_saved = trajectory_path.exists() and trajectory_path.stat().st_size > 2
    except OSError:
        trajectory_saved = False
    trajectory_record_path = host_workspace / "agent" / "mini-swe-agent.trajectory.json"
    patch_path = result.get("patch_path")
    if patch_path:
        patch_path_obj = Path(patch_path)
        if patch_path_obj.is_absolute() and str(patch_path_obj).startswith("/workspace/"):
            patch_path = str(host_workspace / patch_path_obj.relative_to("/workspace"))
    record = {
        "instance_id": result.get("instance_id"),
        "rollout_id": result.get("rollout_id"),
        "model": result.get("model"),
        "litellm_model": result.get("litellm_model"),
        "instruction_style": result.get("instruction_style"),
        "benchmark_profile": result.get("benchmark_profile"),
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
        "api_base": result.get("api_base", ""),
        "max_tokens": result.get("max_tokens"),
        "reasoning_effort": result.get("reasoning_effort"),
        "extra_body_json": result.get("extra_body_json", ""),
        "datagen_code_commit": result.get("datagen_code_commit", ""),
        "mini_swe_agent_git_sha": result.get("mini_swe_agent_git_sha", ""),
        "mini_swe_agent_config_file": result.get("mini_swe_agent_config_file", ""),
        "uses_updated_alignment": result.get("uses_updated_alignment", False),
        "eligible_for_controlled_comparison": result.get("eligible_for_controlled_comparison", False),
        "reason_excluded_from_comparison": result.get("reason_excluded_from_comparison", ""),
        "trajectory_saved": trajectory_saved,
        "result_path": str(host_workspace_path(workspace) / "result.json"),
        "trajectory_path": str(trajectory_record_path),
        "patch_path": patch_path,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        fcntl.flock(handle, fcntl.LOCK_UN)


def write_setup_failure_trajectory(
    args: argparse.Namespace,
    result: dict[str, Any],
    instruction: str,
) -> None:
    """Record a trace artifact even when setup fails before the agent exists."""
    trajectory_path = Path(result["trajectory_path"])
    if trajectory_path.exists():
        return
    try:
        mini_version = importlib.metadata.version("mini-swe-agent")
    except importlib.metadata.PackageNotFoundError:
        mini_version = "unknown"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    failure = {
        "info": {
            "mini_version": mini_version,
            "model_stats": {
                "api_calls": result.get("api_calls", 0),
                "instance_cost": result.get("cost_usd", 0.0),
            },
            "setup_failure": result.get("agent_exception"),
            "datagen_code_commit": result.get("datagen_code_commit", ""),
            "mini_swe_agent_git_sha": result.get("mini_swe_agent_git_sha", ""),
            "uses_updated_alignment": result.get("uses_updated_alignment", False),
            "eligible_for_controlled_comparison": result.get("eligible_for_controlled_comparison", False),
            "reason_excluded_from_comparison": result.get("reason_excluded_from_comparison", ""),
            "config": {
                "model": {
                    "model_name": args.litellm_model,
                    "api_base": args.api_base,
                    "max_tokens": args.max_tokens,
                    "reasoning_effort": args.reasoning_effort,
                    "extra_body_json": args.extra_body_json,
                },
                "agent": {
                    "output_path": str(trajectory_path),
                    "wall_time_limit_seconds": args.agent_wall_time_limit,
                },
            },
        },
        "messages": [{"role": "user", "content": instruction}],
        "trajectory_format": "mini-swe-agent-v2-setup-failure",
    }
    trajectory_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    (args.workspace / "agent").mkdir(parents=True, exist_ok=True)
    (args.workspace / "artifacts").mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    configure_runtime_env(args)
    prepare_aliases(args)
    benchmark_profile = resolve_benchmark_profile(args)

    task_data = load_task_toml(args.task_dir)
    metadata = task_data["metadata"]
    environment = task_data["environment"]
    workdir = environment.get("workdir") or "/testbed"
    base_commit = metadata["base_commit_hash"]
    instruction = (args.task_dir / "instruction.md").read_text(encoding="utf-8")
    testbed_alias = ensure_testbed_alias(workdir)
    effective_workdir = (
        "/testbed"
        if benchmark_profile == "swebench-multilingual" and testbed_alias.get("usable")
        else workdir
    )

    metadata_record = {
        "instance_id": args.instance_id,
        "rollout_id": args.rollout_id,
        "model": args.model,
        "litellm_model": args.litellm_model,
        "api_base": args.api_base,
        "instruction_style": args.instruction_style,
        "benchmark_profile": benchmark_profile,
        "mini_swe_agent_config_file": str(args.config_file),
        "mini_swe_agent_git_sha": MINI_SWE_AGENT_GIT_SHA,
        "datagen_code_commit": datagen_code_commit(),
        "uses_updated_alignment": str_to_bool(args.uses_updated_alignment),
        "eligible_for_controlled_comparison": str_to_bool(args.eligible_for_controlled_comparison),
        "reason_excluded_from_comparison": args.reason_excluded_from_comparison,
        "difficulty": args.difficulty,
        "language": args.language,
        "repo": args.repo,
        "outside_original_high_quality_set": str(args.outside_original_high_quality_set).lower()
        in {"1", "true", "yes"},
        "task_dir": str(args.task_dir),
        "docker_image": environment.get("docker_image"),
        "task_workdir": workdir,
        "workdir": effective_workdir,
        "testbed_alias": testbed_alias,
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
        result["repo_reset"] = reset_repo(effective_workdir, base_commit)
        agent = build_agent(args, effective_workdir, Path(result["trajectory_path"]))
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

        result["patch"] = collect_patch(effective_workdir, Path(result["patch_path"]))
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
        write_setup_failure_trajectory(args, result, instruction)
    finally:
        result["finished_at"] = utc_now()
        (args.workspace / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        try:
            append_result_index(args.workspace, result)
        except Exception:  # noqa: BLE001 - result.json has already been written
            (args.workspace / "result_index_error.log").write_text(traceback.format_exc())


if __name__ == "__main__":
    main()
