#!/usr/bin/env python
"""Run DeepSeek V4 Flash on the predictive SWE-bench Multilingual subset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBSET_DIR = Path(__file__).resolve().parent
DEFAULTS_PATH = SUBSET_DIR / "defaults.json"
DEFAULT_INSTANCE_IDS_PATH = SUBSET_DIR / "predictive_30_instance_ids.txt"
DEFAULT_CA_BUNDLE_PATH = REPO_ROOT / "runs" / "system-ca-bundle.pem"


def load_defaults(path: Path) -> dict:
    return json.loads(path.read_text())


def read_instance_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def make_filter_regex(instance_ids: list[str]) -> str:
    return "^(" + "|".join(instance_ids) + ")$"


def python_executable() -> str:
    venv_python = REPO_ROOT / ".venv-swe-uv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def make_env(use_deepseek_key: bool) -> dict[str, str]:
    env = os.environ.copy()
    if DEFAULT_CA_BUNDLE_PATH.exists():
        env.setdefault("REQUESTS_CA_BUNDLE", str(DEFAULT_CA_BUNDLE_PATH))
        env.setdefault("SSL_CERT_FILE", str(DEFAULT_CA_BUNDLE_PATH))
    if use_deepseek_key:
        deepseek_key = env.get("DEEPSEEK_API_KEY")
        if not deepseek_key:
            raise SystemExit("DEEPSEEK_API_KEY must be set in the environment.")
        env["OPENAI_API_KEY"] = deepseek_key
        env["MSWEA_COST_TRACKING"] = "ignore_errors"
    return env


def main() -> None:
    defaults = load_defaults(DEFAULTS_PATH)
    parser = argparse.ArgumentParser(
        description="Run DeepSeek V4 Flash on the 30-task predictive subset."
    )
    parser.add_argument("--instance-ids", type=Path, default=DEFAULT_INSTANCE_IDS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--generation-workers",
        type=int,
        default=defaults["generation_workers"],
        help="mini-swe-agent generation workers.",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=defaults["eval_workers"],
        help="SWE-bench evaluation workers.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output or REPO_ROOT / "runs" / f"deepseek-v4-flash-multilingual-30-{timestamp}"
    run_id = args.run_id or f"deepseek-v4-flash-multilingual-30-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_ids = read_instance_ids(args.instance_ids)
    filter_regex = make_filter_regex(instance_ids)
    python = python_executable()

    if not args.skip_generation:
        generation_env = make_env(use_deepseek_key=True)
        extra_body = json.dumps({"thinking": {"type": "disabled"}}, separators=(",", ":"))
        generation_cmd = [
            python,
            "-m",
            "minisweagent.run.benchmarks.swebench",
            "--subset",
            "multilingual",
            "--split",
            "test",
            "--filter",
            filter_regex,
            "--output",
            str(output_dir),
            "--workers",
            str(args.generation_workers),
            "--model",
            defaults["model"],
            "--model-class",
            "litellm",
            "-c",
            "swebench.yaml",
            "-c",
            f"model.model_kwargs.api_base={defaults['api_base']}",
            "-c",
            f"model.model_kwargs.extra_body={extra_body}",
            "-c",
            "model.model_kwargs.temperature=0",
            "-c",
            "model.model_kwargs.max_tokens=4096",
            "-c",
            "environment.pull_timeout=1800",
            "-c",
            f"agent.step_limit={defaults['generation_step_limit']}",
        ]
        run(generation_cmd, generation_env)

    if not args.skip_evaluation:
        evaluation_env = make_env(use_deepseek_key=False)
        predictions_path = output_dir / "preds.json"
        evaluation_cmd = [
            python,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            "swe-bench/SWE-Bench_Multilingual",
            "--split",
            "test",
            "--instance_ids",
            *instance_ids,
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            str(args.eval_workers),
            "--run_id",
            run_id,
            "--cache_level",
            defaults["evaluation_cache_level"],
            "--clean",
            "False",
            "--timeout",
            str(defaults["evaluation_timeout_seconds"]),
        ]
        run(evaluation_cmd, evaluation_env)


if __name__ == "__main__":
    main()
