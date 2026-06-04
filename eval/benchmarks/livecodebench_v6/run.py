#!/usr/bin/env python
"""Run an OpenAI-compatible model on LiveCodeBench v6."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import hf_hub_download
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from eval.model.config import ModelConfig, add_model_args, model_from_defaults
from eval.paths import CACHE_DIR, RUNS_DIR, configure_ca_bundle

MODULE_DIR = Path(__file__).resolve().parent
DEFAULTS_PATH = MODULE_DIR / "defaults.json"
IDS_PATH = MODULE_DIR / "predictive_50_question_ids.txt"
LCB_SOURCE_DIR = CACHE_DIR / "livecodebench-src"
LCB_SOURCE_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LCB_SOURCE_URL = "https://github.com/LiveCodeBench/LiveCodeBench.git"


@dataclass
class GenerationResult:
    question_id: str
    sample_index: int
    raw_output: str
    code: str
    ok: bool
    error: str | None
    elapsed_s: float
    attempts: int


def load_defaults() -> dict[str, Any]:
    return json.loads(DEFAULTS_PATH.read_text())


def ensure_lcb_source() -> None:
    marker = LCB_SOURCE_DIR / "lcb_runner" / "evaluation" / "compute_code_generation_metrics.py"
    if not marker.exists():
        LCB_SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", LCB_SOURCE_URL, str(LCB_SOURCE_DIR)], check=True)
    subprocess.run(["git", "-C", str(LCB_SOURCE_DIR), "checkout", LCB_SOURCE_COMMIT], check=True)
    sys.path.insert(0, str(LCB_SOURCE_DIR))


def load_subset_ids(path: Path = IDS_PATH) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_v6_problems(defaults: dict[str, Any]):
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    dataset_path = hf_hub_download(
        defaults["dataset_repo"],
        defaults["dataset_file"],
        repo_type="dataset",
    )
    problems = []
    for line in Path(dataset_path).read_text().splitlines():
        problems.append(CodeGenerationProblem(**json.loads(line)))
    return problems


def filter_problems(problems: list[Any], subset_ids: list[str]) -> list[Any]:
    by_id = {problem.question_id: problem for problem in problems}
    missing = [question_id for question_id in subset_ids if question_id not in by_id]
    if missing:
        raise RuntimeError(f"Selected ids missing from dataset: {missing}")
    return [by_id[question_id] for question_id in subset_ids]


def prompt_messages(problem: Any) -> list[dict[str, str]]:
    if problem.starter_code:
        format_text = (
            "You will use the following starter code to write the solution to the "
            "problem and enclose your code within delimiters."
        )
        template = f"### Format: {format_text}\n```python\n{problem.starter_code}\n```\n\n"
    else:
        format_text = (
            "Read the inputs from stdin solve the problem and write the answer to stdout "
            "(do not directly test on the sample inputs). Enclose your code within "
            "delimiters as follows. Ensure that when the python program runs, it reads "
            "the inputs, runs the algorithm and writes output to STDOUT."
        )
        template = f"### Format: {format_text}\n```python\n# YOUR CODE HERE\n```\n\n"

    user = (
        f"### Question:\n{problem.question_content}\n\n"
        f"{template}"
        "### Answer: (use the provided format with backticks)\n\n"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert Python programmer. You will be given a question "
                "(problem specification) and will generate a correct Python program "
                "that matches the specification and passes all tests."
            ),
        },
        {"role": "user", "content": user},
    ]


def extract_code(output: str) -> str:
    blocks = re.findall(r"```(?:python|Python|py)?\s*\n(.*?)```", output, flags=re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return output.strip()


def existing_generations(path: Path, *, include_failed: bool = False) -> dict[str, GenerationResult]:
    if not path.exists():
        return {}
    loaded: dict[str, GenerationResult] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row.setdefault("sample_index", 0)
        if not include_failed and not row.get("ok", False):
            continue
        key = generation_key(row["question_id"], row["sample_index"])
        loaded[key] = GenerationResult(**row)
    return loaded


def generation_key(question_id: str, sample_index: int) -> str:
    return f"{question_id}::{sample_index}"


async def generate_one(
    client: AsyncOpenAI,
    problem: Any,
    sample_index: int,
    model_config: ModelConfig,
    semaphore: asyncio.Semaphore,
    retries: int,
) -> GenerationResult:
    started = time.perf_counter()
    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model_config.openai_model,
                    messages=prompt_messages(problem),
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    timeout=300,
                    extra_body=model_config.extra_body or None,
                )
                raw_output = response.choices[0].message.content or ""
                return GenerationResult(
                    question_id=problem.question_id,
                    sample_index=sample_index,
                    raw_output=raw_output,
                    code=extract_code(raw_output),
                    ok=True,
                    error=None,
                    elapsed_s=time.perf_counter() - started,
                    attempts=attempt,
                )
            except (APIConnectionError, APIError, APIStatusError, APITimeoutError, RateLimitError) as exc:
                if attempt == retries:
                    return GenerationResult(
                        question_id=problem.question_id,
                        sample_index=sample_index,
                        raw_output="",
                        code="",
                        ok=False,
                        error=repr(exc),
                        elapsed_s=time.perf_counter() - started,
                        attempts=attempt,
                    )
                delay = min(90.0, 2.0 * (2 ** (attempt - 1))) + random.random()
                await asyncio.sleep(delay)

    raise AssertionError("unreachable")


async def generate_all(
    problems: list[Any],
    model_config: ModelConfig,
    output_path: Path,
    workers: int,
    retries: int,
    n: int,
) -> list[list[GenerationResult]]:
    cached = existing_generations(output_path)
    expected = [
        (problem, sample_index)
        for problem in problems
        for sample_index in range(n)
    ]
    pending = [
        (problem, sample_index)
        for problem, sample_index in expected
        if generation_key(problem.question_id, sample_index) not in cached
    ]
    if not pending:
        return [
            [cached[generation_key(problem.question_id, sample_index)] for sample_index in range(n)]
            for problem in problems
        ]

    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    http_client = httpx.AsyncClient(limits=limits)
    client_kwargs: dict[str, Any] = {
        "api_key": model_config.api_key(),
        "http_client": http_client,
    }
    if model_config.api_base:
        client_kwargs["base_url"] = model_config.api_base
    client = AsyncOpenAI(**client_kwargs)
    semaphore = asyncio.Semaphore(workers)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        asyncio.create_task(generate_one(client, problem, sample_index, model_config, semaphore, retries))
        for problem, sample_index in pending
    ]
    with output_path.open("a") as f:
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            result = await task
            cached[generation_key(result.question_id, result.sample_index)] = result
            f.write(json.dumps(result.__dict__) + "\n")
            f.flush()
            status = "ok" if result.ok else "failed"
            print(
                f"[generation {completed}/{len(tasks)}] {result.question_id}#{result.sample_index} {status} "
                f"{result.elapsed_s:.1f}s attempts={result.attempts}",
                flush=True,
            )

    await client.close()
    await http_client.aclose()
    return [
        [cached[generation_key(problem.question_id, sample_index)] for sample_index in range(n)]
        for problem in problems
    ]


def evaluate(problems: list[Any], generations: list[list[GenerationResult]], eval_workers: int, timeout: int):
    from lcb_runner.evaluation import codegen_metrics, extract_instance_results

    samples = [problem.get_evaluation_sample() for problem in problems]
    generation_lists = [
        [generation.code if generation.ok else "" for generation in problem_generations]
        for problem_generations in generations
    ]
    sample_count = len(generation_lists[0]) if generation_lists else 1
    k_list = [1]
    if sample_count >= 3:
        k_list.append(3)
    metrics = codegen_metrics(
        samples,
        generation_lists,
        k_list=k_list,
        num_process_evaluate=eval_workers,
        timeout=timeout,
        debug=False,
    )
    graded = extract_instance_results(metrics[1])
    return metrics, graded


def write_custom_outputs(path: Path, generations: list[list[GenerationResult]]) -> None:
    payload = [
        {
            "question_id": problem_generations[0].question_id,
            "code_list": [
                generation.code if generation.ok else ""
                for generation in problem_generations
            ],
        }
        for problem_generations in generations
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-workers", type=int, default=None)
    parser.add_argument("--eval-workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    add_model_args(parser)
    args = parser.parse_args()

    import os

    env = os.environ.copy()
    configure_ca_bundle(env)
    os.environ.update({key: env[key] for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE") if key in env})
    defaults = load_defaults()
    model_config = model_from_defaults(defaults, args)
    n = args.n or defaults["n"]
    generation_workers = args.generation_workers or defaults["generation_workers"]
    eval_workers = args.eval_workers or defaults["evaluation_workers"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (RUNS_DIR / "livecodebench-v6" / f"{model_config.slug}-{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_lcb_source()
    all_problems = load_v6_problems(defaults)
    if args.all_tasks:
        problems = sorted(all_problems, key=lambda problem: problem.question_id)
    else:
        subset_ids = load_subset_ids()
        problems = filter_problems(all_problems, subset_ids)

    raw_path = output_dir / "raw_generations.jsonl"
    custom_outputs_path = output_dir / "custom_outputs.json"
    eval_path = output_dir / "evaluation.json"
    timing_path = output_dir / "timing.json"

    print(
        f"Running {model_config.openai_model} on {len(problems)} LiveCodeBench v6 tasks "
        f"with n={n}, generation_workers={generation_workers}, eval_workers={eval_workers}",
        flush=True,
    )

    start_total = time.perf_counter()
    start_generation = time.perf_counter()
    if args.skip_generation:
        cached = existing_generations(raw_path, include_failed=True)
        generations = [
            [cached[generation_key(problem.question_id, sample_index)] for sample_index in range(n)]
            for problem in problems
        ]
    else:
        generations = asyncio.run(
            generate_all(problems, model_config, raw_path, generation_workers, args.retries, n)
        )
    generation_s = time.perf_counter() - start_generation
    write_custom_outputs(custom_outputs_path, generations)

    metrics = None
    graded = None
    evaluation_s = 0.0
    if not args.skip_evaluation:
        start_evaluation = time.perf_counter()
        metrics, graded = evaluate(problems, generations, eval_workers, args.timeout)
        evaluation_s = time.perf_counter() - start_evaluation
        eval_payload = {
            "metrics": metrics[0],
            "graded": [
                {"question_id": problem.question_id, "passed": [bool(value) for value in grade]}
                for problem, grade in zip(problems, graded)
            ],
            "metadata": metrics[2],
        }
        eval_path.write_text(json.dumps(eval_payload, indent=2) + "\n")

    total_s = time.perf_counter() - start_total
    passed = None
    if graded is not None:
        passed = sum(1 for grade in graded if any(bool(value) for value in grade))
    timing = {
        "model": model_config.openai_model,
        "api_base": model_config.api_base,
        "task_count": len(problems),
        "n": n,
        "generation_workers": generation_workers,
        "evaluation_workers": eval_workers,
        "timeout": args.timeout,
        "generation_s": generation_s,
        "evaluation_s": evaluation_s,
        "total_s": total_s,
        "passed": passed,
        "pass_at_1": None if metrics is None else metrics[0].get("pass@1"),
        "pass_at_3": None if metrics is None else metrics[0].get("pass@3"),
        "failed_generations": [
            f"{generation.question_id}#{generation.sample_index}"
            for problem_generations in generations
            for generation in problem_generations
            if not generation.ok
        ],
        "output_dir": str(output_dir),
    }
    timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2), flush=True)


if __name__ == "__main__":
    main()
