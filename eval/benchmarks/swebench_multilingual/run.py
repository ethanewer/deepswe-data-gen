#!/usr/bin/env python
"""Run a model on the predictive SWE-bench Multilingual subset."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from eval.model.config import add_model_args, model_from_defaults
from eval.paths import REPO_ROOT, configure_ca_bundle, python_executable


SUBSET_DIR = Path(__file__).resolve().parent
DEFAULTS_PATH = SUBSET_DIR / "defaults.json"
DEFAULT_INSTANCE_IDS_PATH = SUBSET_DIR / "predictive_30_instance_ids.txt"


def load_defaults(path: Path) -> dict:
    return json.loads(path.read_text())


def read_instance_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def make_filter_regex(instance_ids: list[str]) -> str:
    return "^(" + "|".join(re.escape(instance_id) for instance_id in instance_ids) + ")$"


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    defaults = load_defaults(DEFAULTS_PATH)
    parser = argparse.ArgumentParser(
        description="Run an OpenAI-compatible model on the 30-task predictive subset."
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
    add_model_args(parser)
    args = parser.parse_args()
    model_config = model_from_defaults(defaults, args)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output or REPO_ROOT / "runs" / f"{model_config.slug}-multilingual-30-{timestamp}"
    run_id = args.run_id or f"{model_config.slug}-multilingual-30-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_ids = read_instance_ids(args.instance_ids)
    filter_regex = make_filter_regex(instance_ids)
    python = python_executable()

    if not args.skip_generation:
        generation_env = model_config.generation_env()
        extra_body = json.dumps(model_config.extra_body, separators=(",", ":"))
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
            model_config.litellm_name,
            "--model-class",
            "litellm",
            "-c",
            "swebench.yaml",
            "-c",
            f"model.model_kwargs.temperature={model_config.temperature}",
            "-c",
            f"model.model_kwargs.max_tokens={model_config.max_tokens}",
            "-c",
            "environment.pull_timeout=1800",
            "-c",
            f"agent.step_limit={defaults['generation_step_limit']}",
        ]
        if model_config.api_base:
            generation_cmd.extend(["-c", f"model.model_kwargs.api_base={model_config.api_base}"])
        if model_config.extra_body:
            generation_cmd.extend(["-c", f"model.model_kwargs.extra_body={extra_body}"])
        run(generation_cmd, generation_env)

    if not args.skip_evaluation:
        evaluation_env = dict(generation_env if not args.skip_generation else {})
        if not evaluation_env:
            import os

            evaluation_env = os.environ.copy()
            configure_ca_bundle(evaluation_env)
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
