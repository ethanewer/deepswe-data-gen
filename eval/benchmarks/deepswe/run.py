#!/usr/bin/env python3
"""Run generated tasks with the same harness DeepSWE uses.

DeepSWE leaderboard runs use Pier with the model-agnostic mini-swe-agent
harness. This script deliberately shells out to that harness instead of calling
the OpenAI API directly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from eval.model.config import add_model_args, model_from_defaults
from eval.minisweagent_pin import MINI_SWE_AGENT_PIER_EXTRA_PACKAGES
from eval.paths import REPO_ROOT


MODULE_DIR = Path(__file__).resolve().parent
DEFAULTS_PATH = MODULE_DIR / "defaults.json"
DEFAULT_TASK_SPLIT = MODULE_DIR / "data" / "easiest_5_eval_split.json"
DEFAULT_TASKS_DIR = Path("/tmp/deep-swe/tasks")
DEFAULT_JOBS_DIR = REPO_ROOT / "runs" / "pier-jobs"
DEFAULT_MODEL_CONFIG = {
    "model_config": {
        "name": "gpt-5.4-mini",
        "litellm_model": "openai/gpt-5.4-mini",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0,
        "max_tokens": 4096,
    },
    "n_attempts": 1,
    "n_concurrent": 5,
    "timeout_multiplier": 1.0,
}


def load_defaults() -> dict:
    if DEFAULTS_PATH.exists():
        return json.loads(DEFAULTS_PATH.read_text())
    return DEFAULT_MODEL_CONFIG


def default_task_names(path: Path = DEFAULT_TASK_SPLIT) -> list[str]:
    data = json.loads(path.read_text())
    return [task["task_id"] for task in data["tasks"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SWE-rebench-derived tasks through Pier/mini-swe-agent."
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help="DeepSWE/Pier task directory to pass to pier -p.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum tasks to run.")
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=DEFAULT_JOBS_DIR,
        help="Pier jobs output directory.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        help="Optional deterministic Pier sample seed.",
    )
    parser.add_argument(
        "--include-task-name",
        action="append",
        default=[],
        help="Task name or glob to include. Defaults to the DeepSWE easiest-5 subset.",
    )
    parser.add_argument(
        "--disable-verification",
        action="store_true",
        help="Skip verifier execution after the agent run. Useful only for smoke tests.",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=None,
        help="Number of concurrent Pier trials.",
    )
    parser.add_argument("--n-attempts", type=int, default=None, help="Pier attempts per task.")
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Pier timeout multiplier.",
    )
    parser.add_argument(
        "--agent-kwarg",
        action="append",
        default=[],
        help="Additional Pier --agent-kwarg value, e.g. key=value.",
    )
    parser.add_argument(
        "--agent-env",
        action="append",
        default=[],
        help="Additional Pier --agent-env value, e.g. KEY=VALUE.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional .env file to pass to Pier.",
    )
    parser.add_argument(
        "--prepare-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch DeepSWE task definitions if the default task dir is missing.",
    )
    add_model_args(parser)
    return parser.parse_args()


def ensure_prerequisites() -> None:
    if not pier_executable():
        raise SystemExit(
            "pier is not installed. Install it with: "
            "uv tool install git+https://github.com/datacurve-ai/pier"
        )


def pier_executable() -> str | None:
    candidate = Path(sys.executable).resolve().parent / "pier"
    if candidate.exists():
        return str(candidate)
    return shutil.which("pier")


def prepare_tasks_if_needed(args: argparse.Namespace) -> None:
    if args.tasks_dir.exists():
        return
    if not args.prepare_tasks or args.tasks_dir != DEFAULT_TASKS_DIR:
        raise SystemExit(f"{args.tasks_dir} does not exist.")
    subprocess.run([str(MODULE_DIR / "prepare_tasks.sh")], cwd=REPO_ROOT, check=True)


def build_command(args: argparse.Namespace, defaults: dict, model_config) -> list[str]:
    if args.include_task_name:
        include_task_names = args.include_task_name
    elif args.tasks_dir == DEFAULT_TASKS_DIR:
        include_task_names = default_task_names()
    else:
        include_task_names = []
    limit = args.limit or (len(include_task_names) if include_task_names else 1)
    n_concurrent = args.n_concurrent or defaults.get("n_concurrent", 5)
    n_attempts = args.n_attempts or defaults.get("n_attempts", 1)

    command = [
        pier_executable() or "pier",
        "run",
        "-p",
        str(args.tasks_dir),
        "--agent",
        "mini-swe-agent",
        "--model",
        model_config.pier_model,
        "--jobs-dir",
        str(args.jobs_dir),
        "--n-tasks",
        str(limit),
        "--n-attempts",
        str(n_attempts),
        "--n-concurrent",
        str(n_concurrent),
        "--timeout-multiplier",
        str(args.timeout_multiplier),
        "--yes",
    ]
    if args.sample_seed is not None:
        command.extend(["--sample-seed", str(args.sample_seed)])
    for include in include_task_names:
        command.extend(["--include-task-name", include])
    model_kwargs = {
        "temperature": model_config.temperature,
        "max_tokens": model_config.max_tokens,
    }
    if model_config.extra_body:
        model_kwargs["extra_body"] = model_config.extra_body
    command.extend(
        [
            "--agent-kwarg",
            "extra_python_packages="
            f"{json.dumps(MINI_SWE_AGENT_PIER_EXTRA_PACKAGES, separators=(',', ':'))}",
        ]
    )
    command.extend(["--agent-kwarg", f"model_kwargs={json.dumps(model_kwargs, separators=(',', ':'))}"])
    if model_config.api_base:
        command.extend(["--agent-env", f"OPENAI_BASE_URL={model_config.api_base}"])
        command.extend(["--agent-env", f"OPENAI_API_BASE={model_config.api_base}"])
    for value in args.agent_kwarg:
        command.extend(["--agent-kwarg", value])
    for value in args.agent_env:
        command.extend(["--agent-env", value])
    if args.env_file:
        command.extend(["--env-file", str(args.env_file)])
    if args.disable_verification:
        command.append("--disable-verification")
    return command


def main() -> None:
    args = parse_args()
    defaults = load_defaults()
    model_config = model_from_defaults(defaults, args)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    prepare_tasks_if_needed(args)
    ensure_prerequisites()

    command = build_command(args, defaults, model_config)
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, env=model_config.generation_env(), check=True)


if __name__ == "__main__":
    main()
