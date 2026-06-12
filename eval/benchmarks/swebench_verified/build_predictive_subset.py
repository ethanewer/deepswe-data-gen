#!/usr/bin/env python
"""Build a 20-task predictive subset of SWE-bench Verified.

The subset is optimized against public SWE-Router full 500-task result datasets.
It follows the same score-preservation approach used by the other predictive
subsets in this repo: minimize known model score error while balancing task
difficulty, repository mix, and task creation date.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


TASK_DATASET = "SWE-bench/SWE-bench_Verified"
TASK_FILE = "data/test-00000-of-00001.parquet"
SUBSET_SIZE = 20
MAX_PER_REPO = 4
RANDOM_SEED = 20260612
RESTARTS = 100
LOCAL_SEARCH_STEPS = 4500
RANDOM_BASELINE_SAMPLES = 5000

FULL_RUNS = [
    {
        "name": "gpt-5-nano",
        "repo": "SWE-Router/swebench-verified-gpt-5-nano",
        "file": "data/test-00000-of-00001.parquet",
        "source": "SWE-Router",
    },
    {
        "name": "gpt-5-mini",
        "repo": "SWE-Router/swebench-verified-gpt-5-mini",
        "file": "data/test-00000-of-00001.parquet",
        "source": "SWE-Router",
    },
    {
        "name": "gpt-5.2",
        "repo": "SWE-Router/swebench-verified-gpt-5.2",
        "file": "data/test-00000-of-00001.parquet",
        "source": "SWE-Router",
    },
    {
        "name": "deepseek-v4-flash",
        "repo": "SWE-Router/swebench-verified-deepseek-v4-flash",
        "file": "data/test-00000-of-00001.parquet",
        "source": "SWE-Router",
    },
    {
        "name": "deepseek-v3.2",
        "repo": "SWE-Router/swebench-verified-deepseek-v3.2",
        "file": "data/test-00000-of-00001.parquet",
        "source": "SWE-Router",
    },
    {
        "name": "gemini-3-pro",
        "repo": "SWE-Router/swebench-verified-gemini-3-pro",
        "file": "data/test-00000-of-00001.parquet",
        "source": "SWE-Router",
    },
    {
        "name": "gemini-2.5-pro",
        "repo": "SWE-Router/swebench-verified-gemini-2.5-pro",
        "file": "data/test-00000-of-00001.parquet",
        "source": "SWE-Router",
    },
    {
        "name": "claude-opus-4.7",
        "repo": "SWE-Router/swebench-verified-claude-opus-4.7",
        "file": "data/test-00000-of-00001.parquet",
        "source": "SWE-Router",
    },
    {
        "name": "gemini-3.1-pro-preview",
        "repo": "SWE-Router/swebench-verified-gemini-3.1-pro-preview",
        "file": "data/test-00000-of-00001.parquet",
        "source": "SWE-Router",
    },
]


@dataclass(frozen=True)
class Task:
    index: int
    instance_id: str
    repo: str
    created_at: str
    year_bucket: str
    patch_changed_lines: int
    fail_to_pass_count: int
    pass_to_pass_count: int


def count_changed_lines(patch: str | None) -> int:
    total = 0
    for line in (patch or "").splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            total += 1
    return total


def parse_test_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if not value.strip():
            return []
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    raise TypeError(
        "Expected a test list or JSON-encoded test list, "
        f"got {type(value).__name__}"
    )


def year_bucket(created_at: str | None) -> str:
    if not created_at:
        return "unknown"
    year = int(created_at[:4])
    if year <= 2018:
        return "<=2018"
    return str(year)


def download_parquet(repo_id: str, filename: str) -> str:
    return hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)


def load_tasks() -> list[Task]:
    path = download_parquet(TASK_DATASET, TASK_FILE)
    rows = pq.read_table(
        path,
        columns=[
            "repo",
            "instance_id",
            "created_at",
            "patch",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        ],
    ).to_pylist()
    tasks: list[Task] = []
    for index, row in enumerate(rows):
        tasks.append(
            Task(
                index=index,
                instance_id=row["instance_id"],
                repo=row["repo"],
                created_at=row["created_at"] or "",
                year_bucket=year_bucket(row["created_at"]),
                patch_changed_lines=count_changed_lines(row["patch"]),
                fail_to_pass_count=len(parse_test_list(row["FAIL_TO_PASS"])),
                pass_to_pass_count=len(parse_test_list(row["PASS_TO_PASS"])),
            )
        )
    return tasks


def load_run(run: dict, all_ids: set[str]) -> tuple[dict[str, bool], dict]:
    path = download_parquet(run["repo"], run["file"])
    rows = pq.read_table(path, columns=["instance_id", "resolved"]).to_pylist()
    labels = {row["instance_id"]: bool(row["resolved"]) for row in rows}
    unknown = sorted(set(labels) - all_ids)
    return labels, {"rows": len(rows), "unknown_ids": unknown}


def build_outcomes(tasks: list[Task]) -> tuple[dict, np.ndarray]:
    all_ids = {task.instance_id for task in tasks}
    metadata = {"full_runs": {}}
    vectors = []
    for run in FULL_RUNS:
        labels, info = load_run(run, all_ids)
        vector = np.zeros(len(tasks), dtype=bool)
        missing = []
        for task in tasks:
            if task.instance_id in labels:
                vector[task.index] = labels[task.instance_id]
            else:
                missing.append(task.instance_id)
        vectors.append(vector)
        metadata["full_runs"][run["name"]] = {
            **info,
            "source": run["source"],
            "dataset": run["repo"],
            "coverage": len(tasks) - len(missing),
            "missing_counted_unresolved": missing,
            "score": float(vector.mean()),
        }
    return metadata, np.column_stack(vectors).astype(float)


def proportions(values: Iterable[str]) -> dict[str, float]:
    counts = Counter(values)
    total = sum(counts.values())
    return {key: value / total for key, value in sorted(counts.items())}


def l1_counter_distance(sample: Iterable[str], population_props: dict[str, float]) -> float:
    sample_props = proportions(sample)
    keys = set(sample_props) | set(population_props)
    return sum(
        abs(sample_props.get(key, 0.0) - population_props.get(key, 0.0))
        for key in sorted(keys)
    )


def bin_difficulty(values: np.ndarray) -> list[str]:
    bins = []
    for value in values:
        if value <= 0.001:
            bins.append("0")
        elif value <= 0.25:
            bins.append("(0,.25]")
        elif value <= 0.50:
            bins.append("(.25,.50]")
        elif value <= 0.75:
            bins.append("(.50,.75]")
        elif value < 0.999:
            bins.append("(.75,1)")
        else:
            bins.append("1")
    return bins


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(rankdata(a), rankdata(b))


def make_random_subset(rng: random.Random, tasks: list[Task]) -> tuple[int, ...]:
    indices = list(range(len(tasks)))
    for _ in range(10000):
        rng.shuffle(indices)
        selected = []
        repo_counts: Counter[str] = Counter()
        for index in indices:
            if repo_counts[tasks[index].repo] >= MAX_PER_REPO:
                continue
            selected.append(index)
            repo_counts[tasks[index].repo] += 1
            if len(selected) == SUBSET_SIZE:
                return tuple(sorted(selected))
    raise RuntimeError("Could not sample a subset under repo constraints")


def build_objective(tasks: list[Task], matrix: np.ndarray):
    full_scores = matrix.mean(axis=0)
    task_scores = matrix.mean(axis=1)
    population_bins = proportions(bin_difficulty(task_scores))
    population_years = proportions(task.year_bucket for task in tasks)
    population_repos = proportions(task.repo for task in tasks)

    def objective(selected: tuple[int, ...]) -> float:
        selected_array = np.array(selected, dtype=int)
        subset_scores = matrix[selected_array, :].mean(axis=0)
        errors = subset_scores - full_scores
        rmse = float(np.sqrt(np.mean(errors * errors)))
        max_abs = float(np.max(np.abs(errors)))
        repo_penalty = 0.0
        for count in Counter(tasks[i].repo for i in selected).values():
            if count > MAX_PER_REPO:
                repo_penalty += (count - MAX_PER_REPO) * 10.0
        difficulty_l1 = l1_counter_distance(
            bin_difficulty(task_scores[selected_array]), population_bins
        )
        year_l1 = l1_counter_distance((tasks[i].year_bucket for i in selected), population_years)
        repo_l1 = l1_counter_distance((tasks[i].repo for i in selected), population_repos)
        return (
            rmse
            + 0.40 * max_abs
            + 0.030 * difficulty_l1
            + 0.015 * year_l1
            + 0.020 * repo_l1
            + repo_penalty
        )

    return objective, full_scores, task_scores


def optimize_subset(tasks: list[Task], matrix: np.ndarray) -> tuple[tuple[int, ...], dict]:
    rng = random.Random(RANDOM_SEED)
    objective, full_scores, task_scores = build_objective(tasks, matrix)
    best_subset: tuple[int, ...] | None = None
    best_score = float("inf")

    all_indices = set(range(len(tasks)))
    for _ in range(RESTARTS):
        current = make_random_subset(rng, tasks)
        current_set = set(current)
        current_score = objective(current)
        for step in range(LOCAL_SEARCH_STEPS):
            selected_index = rng.choice(sorted(current_set))
            candidates = sorted(all_indices - current_set)
            rng.shuffle(candidates)
            repo_counts = Counter(tasks[i].repo for i in current_set)
            repo_counts[tasks[selected_index].repo] -= 1
            replacement = None
            for candidate in candidates:
                if repo_counts[tasks[candidate].repo] < MAX_PER_REPO:
                    replacement = candidate
                    break
            if replacement is None:
                continue
            proposed_set = set(current_set)
            proposed_set.remove(selected_index)
            proposed_set.add(replacement)
            proposed = tuple(sorted(proposed_set))
            proposed_score = objective(proposed)
            temperature = 0.002 * (1.0 - step / LOCAL_SEARCH_STEPS)
            accept = proposed_score < current_score
            if not accept and temperature > 0:
                accept = rng.random() < math.exp((current_score - proposed_score) / temperature)
            if accept:
                current = proposed
                current_set = proposed_set
                current_score = proposed_score
        if current_score < best_score:
            best_subset = current
            best_score = current_score

    if best_subset is None:
        raise RuntimeError("No subset found")

    selected_array = np.array(best_subset, dtype=int)
    subset_scores = matrix[selected_array, :].mean(axis=0)
    errors = subset_scores - full_scores
    return best_subset, {
        "objective": best_score,
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mae": float(np.mean(np.abs(errors))),
        "max_abs_error": float(np.max(np.abs(errors))),
        "pearson": pearson(full_scores, subset_scores),
        "spearman": spearman(full_scores, subset_scores),
        "task_solve_rate": task_scores.tolist(),
    }


def random_baseline(tasks: list[Task], matrix: np.ndarray) -> dict:
    rng = random.Random(RANDOM_SEED + 1)
    full_scores = matrix.mean(axis=0)
    rmses = []
    maes = []
    maxes = []
    pearsons = []
    for _ in range(RANDOM_BASELINE_SAMPLES):
        subset = make_random_subset(rng, tasks)
        scores = matrix[np.array(subset), :].mean(axis=0)
        errors = scores - full_scores
        rmses.append(float(np.sqrt(np.mean(errors * errors))))
        maes.append(float(np.mean(np.abs(errors))))
        maxes.append(float(np.max(np.abs(errors))))
        pearsons.append(pearson(full_scores, scores))

    def pct(values: list[float], q: float) -> float:
        return float(np.percentile(np.array(values), q))

    return {
        "samples": RANDOM_BASELINE_SAMPLES,
        "rmse": {"p05": pct(rmses, 5), "median": pct(rmses, 50), "p95": pct(rmses, 95)},
        "mae": {"p05": pct(maes, 5), "median": pct(maes, 50), "p95": pct(maes, 95)},
        "max_abs_error": {
            "p05": pct(maxes, 5),
            "median": pct(maxes, 50),
            "p95": pct(maxes, 95),
        },
        "pearson": {
            "p05": pct(pearsons, 5),
            "median": pct(pearsons, 50),
            "p95": pct(pearsons, 95),
        },
    }


def write_outputs(
    output_dir: Path,
    tasks: list[Task],
    selected: tuple[int, ...],
    matrix: np.ndarray,
    metadata: dict,
    diagnostics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [run["name"] for run in FULL_RUNS]
    full_scores = matrix.mean(axis=0)
    selected_array = np.array(selected, dtype=int)
    subset_scores = matrix[selected_array, :].mean(axis=0)
    selected_sorted = sorted(selected, key=lambda i: (tasks[i].repo, tasks[i].instance_id))

    task_solve_rates = diagnostics["task_solve_rate"]
    tasks_csv = output_dir / "predictive_20_tasks.csv"
    fields = [
        "rank",
        "instance_id",
        "repo",
        "created_at",
        "year_bucket",
        "patch_changed_lines",
        "fail_to_pass_count",
        "pass_to_pass_count",
        "full_model_solve_rate",
    ] + [f"resolved_{name}" for name in model_names]
    with tasks_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, index in enumerate(selected_sorted, start=1):
            task = tasks[index]
            row = {
                "rank": rank,
                "instance_id": task.instance_id,
                "repo": task.repo,
                "created_at": task.created_at,
                "year_bucket": task.year_bucket,
                "patch_changed_lines": task.patch_changed_lines,
                "fail_to_pass_count": task.fail_to_pass_count,
                "pass_to_pass_count": task.pass_to_pass_count,
                "full_model_solve_rate": round(float(task_solve_rates[index]), 6),
            }
            for model_index, name in enumerate(model_names):
                row[f"resolved_{name}"] = int(matrix[index, model_index])
            writer.writerow(row)

    ids_path = output_dir / "predictive_20_instance_ids.txt"
    with ids_path.open("w", encoding="utf-8") as handle:
        for index in selected_sorted:
            handle.write(f"{tasks[index].instance_id}\n")

    comparison_path = output_dir / "predictive_20_model_score_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "model",
            "source",
            "coverage",
            "full_score_pct",
            "subset_score_pct",
            "error_pct_points",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_index, run in enumerate(FULL_RUNS):
            name = run["name"]
            writer.writerow(
                {
                    "model": name,
                    "source": metadata["full_runs"][name]["dataset"],
                    "coverage": metadata["full_runs"][name]["coverage"],
                    "full_score_pct": round(float(full_scores[model_index] * 100), 4),
                    "subset_score_pct": round(float(subset_scores[model_index] * 100), 4),
                    "error_pct_points": round(
                        float((subset_scores[model_index] - full_scores[model_index]) * 100),
                        4,
                    ),
                }
            )

    baseline = random_baseline(tasks, matrix)
    repo_counts = Counter(tasks[i].repo for i in selected)
    summary = {
        "subset_size": SUBSET_SIZE,
        "random_seed": RANDOM_SEED,
        "task_dataset": TASK_DATASET,
        "selection_method": {
            "target": (
                "Minimize 20-task score error against known full per-instance "
                "SWE-bench Verified model runs."
            ),
            "constraints": {"max_per_repo": MAX_PER_REPO},
            "objective_terms": [
                "model score RMSE",
                "model score max absolute error",
                "task difficulty histogram balance",
                "year bucket balance",
                "repository distribution balance",
            ],
        },
        "selected_instance_ids": [tasks[i].instance_id for i in selected_sorted],
        "selected_repo_counts": dict(sorted(repo_counts.items())),
        "known_full_runs": metadata["full_runs"],
        "validation": {
            "rmse_pct_points": diagnostics["rmse"] * 100,
            "mae_pct_points": diagnostics["mae"] * 100,
            "max_abs_error_pct_points": diagnostics["max_abs_error"] * 100,
            "pearson_across_known_model_scores": diagnostics["pearson"],
            "spearman_across_known_model_scores": diagnostics["spearman"],
            "random_repo_capped_baseline_pct_points": {
                "rmse": {k: v * 100 for k, v in baseline["rmse"].items()},
                "mae": {k: v * 100 for k, v in baseline["mae"].items()},
                "max_abs_error": {
                    k: v * 100 for k, v in baseline["max_abs_error"].items()
                },
                "pearson": baseline["pearson"],
                "samples": baseline["samples"],
            },
        },
        "files": [
            "predictive_20_tasks.csv",
            "predictive_20_instance_ids.txt",
            "predictive_20_model_score_comparison.csv",
            "predictive_20_summary.json",
        ],
    }
    with (output_dir / "predictive_20_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    tasks = load_tasks()
    metadata, matrix = build_outcomes(tasks)
    selected, diagnostics = optimize_subset(tasks, matrix)
    write_outputs(output_dir, tasks, selected, matrix, metadata, diagnostics)
    print(f"Wrote {SUBSET_SIZE} tasks to {output_dir}")
    print(f"RMSE: {diagnostics['rmse'] * 100:.3f} percentage points")
    print(f"Max absolute error: {diagnostics['max_abs_error'] * 100:.3f} percentage points")


if __name__ == "__main__":
    main()
