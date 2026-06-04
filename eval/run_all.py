#!/usr/bin/env python
"""Run configured eval benchmarks with one OpenAI-compatible model config."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval.model.config import ModelConfig
from eval.paths import REPO_ROOT, python_executable


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "all_benchmarks.example.json"
BENCHMARK_MODULES = {
    "deepswe": "eval.benchmarks.deepswe.run",
    "swebench_multilingual": "eval.benchmarks.swebench_multilingual.run",
    "livecodebench_v6": "eval.benchmarks.livecodebench_v6.run",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured eval benchmarks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=sorted(BENCHMARK_MODULES),
        help="Benchmark to skip. Can be passed multiple times.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        choices=sorted(BENCHMARK_MODULES),
        help="Benchmark to run exclusively. Can be passed multiple times.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=JSON",
        help="Override a config value, e.g. model.name=\"Qwen/Qwen3-8B\".",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise SystemExit(f"override must be PATH=JSON: {override}")
    path, raw_value = override.split("=", 1)
    keys = path.split(".")
    target = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = parse_value(raw_value)


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    result = copy.deepcopy(base)
    if not override:
        return result
    for key, value in override.items():
        if isinstance(value, dict) and not value:
            result[key] = {}
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def append_flag(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, list):
        for item in value:
            command.extend([flag, str(item)])
        return
    command.extend([flag, str(value)])


def benchmark_args(name: str, settings: dict[str, Any]) -> list[str]:
    args: list[str] = []
    if name == "deepswe":
        for key, flag in [
            ("tasks_dir", "--tasks-dir"),
            ("jobs_dir", "--jobs-dir"),
            ("limit", "--limit"),
            ("sample_seed", "--sample-seed"),
            ("n_concurrent", "--n-concurrent"),
            ("n_attempts", "--n-attempts"),
            ("timeout_multiplier", "--timeout-multiplier"),
            ("include_task_name", "--include-task-name"),
            ("agent_kwarg", "--agent-kwarg"),
            ("agent_env", "--agent-env"),
            ("env_file", "--env-file"),
        ]:
            append_flag(args, flag, settings.get(key))
        append_flag(args, "--disable-verification", settings.get("disable_verification"))
        if settings.get("prepare_tasks") is False:
            args.append("--no-prepare-tasks")
    elif name == "swebench_multilingual":
        for key, flag in [
            ("harness", "--harness"),
            ("instance_ids", "--instance-ids"),
            ("output", "--output"),
            ("run_id", "--run-id"),
            ("generation_workers", "--generation-workers"),
            ("eval_workers", "--eval-workers"),
            ("openhands_infer_command", "--openhands-infer-command"),
            ("openhands_command_cwd", "--openhands-command-cwd"),
            ("openhands_llm_config", "--openhands-llm-config"),
            ("openhands_output_json", "--openhands-output-json"),
            ("openhands_workspace", "--openhands-workspace"),
            ("openhands_max_iterations", "--openhands-max-iterations"),
            ("openhands_n_critic_runs", "--openhands-n-critic-runs"),
            ("openhands_max_retries", "--openhands-max-retries"),
            ("openhands_tool_preset", "--openhands-tool-preset"),
            ("opencode_command", "--opencode-command"),
            ("opencode_model", "--opencode-model"),
            ("opencode_config", "--opencode-config"),
            ("opencode_workspace", "--opencode-workspace"),
            ("opencode_timeout", "--opencode-timeout"),
            ("opencode_agent", "--opencode-agent"),
            ("opencode_variant", "--opencode-variant"),
            ("opencode_command_template", "--opencode-command-template"),
        ]:
            append_flag(args, flag, settings.get(key))
        for extra_arg in settings.get("openhands_extra_arg") or []:
            args.append(f"--openhands-extra-arg={extra_arg}")
        for extra_arg in settings.get("opencode_extra_arg") or []:
            args.append(f"--opencode-extra-arg={extra_arg}")
        append_flag(args, "--openhands-enable-delegation", settings.get("openhands_enable_delegation"))
        append_flag(args, "--skip-generation", settings.get("skip_generation"))
        append_flag(args, "--skip-evaluation", settings.get("skip_evaluation"))
    elif name == "livecodebench_v6":
        for key, flag in [
            ("generation_workers", "--generation-workers"),
            ("eval_workers", "--eval-workers"),
            ("timeout", "--timeout"),
            ("retries", "--retries"),
            ("n", "--n"),
            ("output_dir", "--output-dir"),
        ]:
            append_flag(args, flag, settings.get(key))
        append_flag(args, "--all-tasks", settings.get("all_tasks"))
        append_flag(args, "--skip-generation", settings.get("skip_generation"))
        append_flag(args, "--skip-evaluation", settings.get("skip_evaluation"))
    else:
        raise ValueError(name)
    return args


def command_for(name: str, global_model: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    model_config = ModelConfig.from_dict(deep_merge(global_model, settings.get("model")))
    return [
        python_executable(),
        "-m",
        BENCHMARK_MODULES[name],
        *model_config.to_cli_args(),
        *benchmark_args(name, settings),
    ]


def enabled_benchmarks(config: dict[str, Any], only: list[str], skip: list[str]) -> list[str]:
    selected = []
    for name in BENCHMARK_MODULES:
        settings = config.get("benchmarks", {}).get(name, {})
        if only and name not in only:
            continue
        if name in skip:
            continue
        if settings.get("enabled", True):
            selected.append(name)
    return selected


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    for override in args.set:
        apply_override(config, override)

    model = config.get("model")
    if not model:
        raise SystemExit("config must define a top-level `model` object")

    for name in enabled_benchmarks(config, args.only, args.skip):
        settings = config.get("benchmarks", {}).get(name, {})
        command = command_for(name, model, settings)
        print("+ " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
