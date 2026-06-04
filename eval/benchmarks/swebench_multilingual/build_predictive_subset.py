#!/usr/bin/env python
"""Build a 30-task predictive subset of SWE-bench Multilingual.

The script uses public per-instance result datasets where available:
- SWE-Router full 300-task runs for four models.
- AlienKevin GLM-5 full 300-task run.
- AlienKevin MiniMax M2.5 299-task run; the one missing task is counted as
  unresolved for 300-task score consistency.
- AlienKevin Rust-only trajectory runs for three 43-task models, used as
  additional task-difficulty signal but not as full-benchmark score targets.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


TASK_DATASET = "SWE-bench/SWE-bench_Multilingual"
TASK_FILE = "data/test-00000-of-00001.parquet"
SUBSET_SIZE = 30
MAX_PER_REPO = 2
RANDOM_SEED = 20260603
RESTARTS_PER_QUOTA = 12
LOCAL_SEARCH_STEPS = 1200
RANDOM_BASELINE_SAMPLES = 5000

FULL_RUNS = [
    {
        "name": "gpt-5-nano",
        "repo": "SWE-Router/swebench-multilingual-gpt-5-nano",
        "file": "data/test-00000-of-00001.parquet",
        "kind": "direct",
        "source": "SWE-Router",
    },
    {
        "name": "gpt-5-mini",
        "repo": "SWE-Router/swebench-multilingual-gpt-5-mini",
        "file": "data/test-00000-of-00001.parquet",
        "kind": "direct",
        "source": "SWE-Router",
    },
    {
        "name": "claude-opus-4.7",
        "repo": "SWE-Router/swebench-multilingual-claude-opus-4.7",
        "file": "data/test-00000-of-00001.parquet",
        "kind": "direct",
        "source": "SWE-Router",
    },
    {
        "name": "deepseek-v4-flash",
        "repo": "SWE-Router/swebench-multilingual-deepseek-v4-flash",
        "file": "data/test-00000-of-00001.parquet",
        "kind": "direct",
        "source": "SWE-Router",
    },
    {
        "name": "glm-5",
        "repo": "AlienKevin/SWE-bench-multilingual-glm-5-trajectories",
        "file": "data/train-00000-of-00001.parquet",
        "kind": "problem_text",
        "source": "AlienKevin",
    },
    {
        "name": "minimax-m2.5",
        "repo": "AlienKevin/SWE-bench-multilingual-minimax-m2.5-trajectories",
        "file": "data/train-00000-of-00001.parquet",
        "kind": "problem_text",
        "source": "AlienKevin",
    },
]

PARTIAL_RUNS = [
    {
        "name": "multi-swe-agent-32b-rust",
        "repo": "AlienKevin/SWE-bench-Multilingual-trajectories",
        "file": "data/Multi_SWE_agent_32B_Rust-00000-of-00001.parquet",
        "source": "AlienKevin",
    },
    {
        "name": "qwen2.5-coder-32b-instruct-rust",
        "repo": "AlienKevin/SWE-bench-Multilingual-trajectories",
        "file": "data/Qwen2.5_Coder_32B_Instruct-00000-of-00001.parquet",
        "source": "AlienKevin",
    },
    {
        "name": "swe-agent-lm-32b-rust",
        "repo": "AlienKevin/SWE-bench-Multilingual-trajectories",
        "file": "data/SWE_agent_LM_32B-00000-of-00001.parquet",
        "source": "AlienKevin",
    },
]

LANGUAGE_BY_REPO = {
    "redis/redis": "C",
    "jqlang/jq": "C",
    "micropython/micropython": "C",
    "valkey-io/valkey": "C",
    "nlohmann/json": "C++",
    "fmtlib/fmt": "C++",
    "caddyserver/caddy": "Go",
    "hashicorp/terraform": "Go",
    "prometheus/prometheus": "Go",
    "gohugoio/hugo": "Go",
    "gin-gonic/gin": "Go",
    "google/gson": "Java",
    "apache/druid": "Java",
    "projectlombok/lombok": "Java",
    "apache/lucene": "Java",
    "reactivex/rxjava": "Java",
    "javaparser/javaparser": "Java",
    "babel/babel": "JavaScript/TypeScript",
    "vuejs/core": "JavaScript/TypeScript",
    "facebook/docusaurus": "JavaScript/TypeScript",
    "immutable-js/immutable-js": "JavaScript/TypeScript",
    "mrdoob/three.js": "JavaScript/TypeScript",
    "preactjs/preact": "JavaScript/TypeScript",
    "axios/axios": "JavaScript/TypeScript",
    "phpoffice/phpspreadsheet": "PHP",
    "laravel/framework": "PHP",
    "php-cs-fixer/php-cs-fixer": "PHP",
    "briannesbitt/carbon": "PHP",
    "jekyll/jekyll": "Ruby",
    "fluent/fluentd": "Ruby",
    "fastlane/fastlane": "Ruby",
    "jordansissel/fpm": "Ruby",
    "faker-ruby/faker": "Ruby",
    "rubocop/rubocop": "Ruby",
    "tokio-rs/tokio": "Rust",
    "uutils/coreutils": "Rust",
    "nushell/nushell": "Rust",
    "tokio-rs/axum": "Rust",
    "burntsushi/ripgrep": "Rust",
    "sharkdp/bat": "Rust",
    "astral-sh/ruff": "Rust",
}


@dataclass(frozen=True)
class Task:
    index: int
    instance_id: str
    repo: str
    language: str
    language_group: str
    created_at: str
    year_bucket: str
    patch_changed_lines: int
    fail_to_pass_count: int
    pass_to_pass_count: int
    problem_statement_norm: str


def normalize_text(value: str | None) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def count_changed_lines(patch: str | None) -> int:
    total = 0
    for line in (patch or "").splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            total += 1
    return total


def year_bucket(created_at: str | None) -> str:
    if not created_at:
        return "unknown"
    year = int(created_at[:4])
    if year <= 2021:
        return "<=2021"
    return str(year)


def language_group(language: str) -> str:
    if language in {"C", "C++"}:
        return "C/C++"
    return language


def download_parquet(repo_id: str, filename: str) -> str:
    return hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)


def load_tasks() -> list[Task]:
    path = download_parquet(TASK_DATASET, TASK_FILE)
    rows = pq.read_table(
        path,
        columns=[
            "repo",
            "instance_id",
            "problem_statement",
            "created_at",
            "patch",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        ],
    ).to_pylist()
    tasks: list[Task] = []
    for index, row in enumerate(rows):
        language = LANGUAGE_BY_REPO[row["repo"]]
        tasks.append(
            Task(
                index=index,
                instance_id=row["instance_id"],
                repo=row["repo"],
                language=language,
                language_group=language_group(language),
                created_at=row["created_at"] or "",
                year_bucket=year_bucket(row["created_at"]),
                patch_changed_lines=count_changed_lines(row["patch"]),
                fail_to_pass_count=len(row["FAIL_TO_PASS"] or []),
                pass_to_pass_count=len(row["PASS_TO_PASS"] or []),
                problem_statement_norm=normalize_text(row["problem_statement"]),
            )
        )
    return tasks


def extract_problem_description(messages: list[dict] | None) -> str:
    if not messages or len(messages) < 2:
        return ""
    content = messages[1].get("content") or ""
    match = re.search(
        r"<pr_description>\s*Consider the following PR description:\s*(.*?)\s*</pr_description>",
        content,
        flags=re.S,
    )
    return match.group(1) if match else content


def load_direct_run(run: dict, all_ids: set[str]) -> tuple[dict[str, bool], dict]:
    path = download_parquet(run["repo"], run["file"])
    rows = pq.read_table(path, columns=["instance_id", "resolved"]).to_pylist()
    labels = {row["instance_id"]: bool(row["resolved"]) for row in rows}
    unknown = sorted(set(labels) - all_ids)
    return labels, {"rows": len(rows), "unknown_ids": unknown}


def load_problem_text_run(
    run: dict, problem_to_id: dict[str, str]
) -> tuple[dict[str, bool], dict]:
    path = download_parquet(run["repo"], run["file"])
    rows = pq.read_table(path, columns=["messages", "resolved"]).to_pylist()
    labels: dict[str, bool] = {}
    misses = []
    for row in rows:
        problem = normalize_text(extract_problem_description(row["messages"]))
        instance_id = problem_to_id.get(problem)
        if instance_id is None:
            misses.append(problem[:120])
            continue
        labels[instance_id] = bool(row["resolved"])
    return labels, {"rows": len(rows), "mapped": len(labels), "misses": misses[:5]}


def load_partial_run(run: dict, all_ids: set[str]) -> tuple[dict[str, bool], dict]:
    path = download_parquet(run["repo"], run["file"])
    rows = pq.read_table(path, columns=["id", "resolved"]).to_pylist()
    labels = {row["id"]: bool(row["resolved"]) for row in rows}
    unknown = sorted(set(labels) - all_ids)
    return labels, {"rows": len(rows), "unknown_ids": unknown}


def build_outcomes(tasks: list[Task]) -> tuple[dict, np.ndarray, dict[str, np.ndarray]]:
    all_ids = {task.instance_id for task in tasks}
    id_to_index = {task.instance_id: task.index for task in tasks}
    problem_to_id = {task.problem_statement_norm: task.instance_id for task in tasks}

    full_labels: dict[str, np.ndarray] = {}
    partial_labels: dict[str, np.ndarray] = {}
    metadata = {"full_runs": {}, "partial_runs": {}}

    for run in FULL_RUNS:
        if run["kind"] == "direct":
            labels, info = load_direct_run(run, all_ids)
        else:
            labels, info = load_problem_text_run(run, problem_to_id)

        vector = np.zeros(len(tasks), dtype=bool)
        missing = []
        for task in tasks:
            if task.instance_id in labels:
                vector[task.index] = labels[task.instance_id]
            else:
                missing.append(task.instance_id)
        full_labels[run["name"]] = vector
        metadata["full_runs"][run["name"]] = {
            **info,
            "source": run["source"],
            "dataset": run["repo"],
            "coverage": len(tasks) - len(missing),
            "missing_counted_unresolved": missing,
            "score": float(vector.mean()),
        }

    for run in PARTIAL_RUNS:
        labels, info = load_partial_run(run, all_ids)
        vector = np.full(len(tasks), np.nan)
        for instance_id, resolved in labels.items():
            if instance_id in id_to_index:
                vector[id_to_index[instance_id]] = float(resolved)
        partial_labels[run["name"]] = vector
        metadata["partial_runs"][run["name"]] = {
            **info,
            "source": run["source"],
            "dataset": run["repo"],
            "coverage": int(np.isfinite(vector).sum()),
            "score_on_available_tasks": float(np.nanmean(vector)),
        }

    full_matrix = np.column_stack([full_labels[name] for name in full_labels]).astype(float)
    all_vectors = dict(full_labels)
    all_vectors.update(partial_labels)
    return metadata, full_matrix, all_vectors


def proportions(values: Iterable[str]) -> dict[str, float]:
    counts = Counter(values)
    total = sum(counts.values())
    return {key: value / total for key, value in sorted(counts.items())}


def l1_counter_distance(sample: Iterable[str], population_props: dict[str, float]) -> float:
    sample_props = proportions(sample)
    keys = set(sample_props) | set(population_props)
    return sum(abs(sample_props.get(key, 0.0) - population_props.get(key, 0.0)) for key in keys)


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


def quota_options(tasks: list[Task]) -> list[dict[str, int]]:
    counts = Counter(task.language_group for task in tasks)
    floors = {
        language: math.floor(count / len(tasks) * SUBSET_SIZE)
        for language, count in counts.items()
    }
    remaining = SUBSET_SIZE - sum(floors.values())
    options = []
    for extras in itertools.combinations(sorted(counts), remaining):
        quota = dict(floors)
        for language in extras:
            quota[language] += 1
        options.append(quota)
    return options


def make_random_subset(
    rng: random.Random,
    quota: dict[str, int],
    by_language: dict[str, list[int]],
    tasks: list[Task],
) -> tuple[int, ...]:
    selected: list[int] = []
    repo_counts: Counter[str] = Counter()
    for language, count in quota.items():
        candidates = list(by_language[language])
        rng.shuffle(candidates)
        picked = []
        for index in candidates:
            if repo_counts[tasks[index].repo] >= MAX_PER_REPO:
                continue
            picked.append(index)
            repo_counts[tasks[index].repo] += 1
            if len(picked) == count:
                break
        if len(picked) != count:
            raise RuntimeError(f"Could not sample quota for {language}")
        selected.extend(picked)
    return tuple(sorted(selected))


def build_objective(
    tasks: list[Task],
    full_matrix: np.ndarray,
    all_vectors: dict[str, np.ndarray],
):
    full_scores = full_matrix.mean(axis=0)
    full_task_scores = full_matrix.mean(axis=1)
    all_available_scores = []
    for i in range(len(tasks)):
        labels = []
        for vector in all_vectors.values():
            value = vector[i]
            if np.isfinite(value):
                labels.append(float(value))
        all_available_scores.append(sum(labels) / len(labels))
    all_available_scores_array = np.array(all_available_scores)
    population_full_bins = proportions(bin_difficulty(full_task_scores))
    population_all_bins = proportions(bin_difficulty(all_available_scores_array))
    population_years = proportions(task.year_bucket for task in tasks)

    def objective(selected: tuple[int, ...]) -> float:
        selected_array = np.array(selected, dtype=int)
        subset_scores = full_matrix[selected_array, :].mean(axis=0)
        errors = subset_scores - full_scores
        rmse = float(np.sqrt(np.mean(errors * errors)))
        max_abs = float(np.max(np.abs(errors)))
        repo_penalty = 0.0
        for count in Counter(tasks[i].repo for i in selected).values():
            if count > MAX_PER_REPO:
                repo_penalty += (count - MAX_PER_REPO) * 10.0
        full_bin_l1 = l1_counter_distance(
            (bin_difficulty(full_task_scores[selected_array])), population_full_bins
        )
        all_bin_l1 = l1_counter_distance(
            (bin_difficulty(all_available_scores_array[selected_array])),
            population_all_bins,
        )
        year_l1 = l1_counter_distance(
            (tasks[i].year_bucket for i in selected), population_years
        )
        return (
            rmse
            + 0.40 * max_abs
            + 0.030 * full_bin_l1
            + 0.020 * all_bin_l1
            + 0.010 * year_l1
            + repo_penalty
        )

    return objective, full_scores, all_available_scores_array


def optimize_subset(
    tasks: list[Task],
    full_matrix: np.ndarray,
    all_vectors: dict[str, np.ndarray],
) -> tuple[tuple[int, ...], dict]:
    rng = random.Random(RANDOM_SEED)
    by_language: dict[str, list[int]] = defaultdict(list)
    for task in tasks:
        by_language[task.language_group].append(task.index)

    objective, full_scores, all_available_scores = build_objective(
        tasks, full_matrix, all_vectors
    )

    best_subset: tuple[int, ...] | None = None
    best_score = float("inf")
    best_quota: dict[str, int] | None = None
    quota_results = []

    for quota in quota_options(tasks):
        quota_best = float("inf")
        quota_best_subset: tuple[int, ...] | None = None
        for _ in range(RESTARTS_PER_QUOTA):
            current = make_random_subset(rng, quota, by_language, tasks)
            current_score = objective(current)
            current_set = set(current)
            for step in range(LOCAL_SEARCH_STEPS):
                selected_index = rng.choice(tuple(current_set))
                language = tasks[selected_index].language_group
                candidates = [
                    i
                    for i in by_language[language]
                    if i not in current_set and tasks[i].repo != tasks[selected_index].repo
                ]
                if not candidates:
                    continue
                replacement = rng.choice(candidates)
                repo_counts = Counter(tasks[i].repo for i in current_set)
                repo_counts[tasks[selected_index].repo] -= 1
                if repo_counts[tasks[replacement].repo] >= MAX_PER_REPO:
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
            if current_score < quota_best:
                quota_best = current_score
                quota_best_subset = current
        quota_results.append({"quota": quota, "objective": quota_best})
        if quota_best_subset is not None and quota_best < best_score:
            best_subset = quota_best_subset
            best_score = quota_best
            best_quota = quota

    if best_subset is None or best_quota is None:
        raise RuntimeError("No subset found")

    selected_array = np.array(best_subset, dtype=int)
    subset_scores = full_matrix[selected_array, :].mean(axis=0)
    errors = subset_scores - full_scores
    diagnostics = {
        "objective": best_score,
        "quota": best_quota,
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mae": float(np.mean(np.abs(errors))),
        "max_abs_error": float(np.max(np.abs(errors))),
        "pearson": pearson(full_scores, subset_scores),
        "spearman": spearman(full_scores, subset_scores),
        "quota_search": quota_results,
        "all_available_task_solve_rate": all_available_scores.tolist(),
    }
    return best_subset, diagnostics


def random_baseline(
    tasks: list[Task],
    full_matrix: np.ndarray,
    quota: dict[str, int],
) -> dict:
    rng = random.Random(RANDOM_SEED + 1)
    by_language: dict[str, list[int]] = defaultdict(list)
    for task in tasks:
        by_language[task.language_group].append(task.index)
    full_scores = full_matrix.mean(axis=0)
    rmses = []
    maes = []
    maxes = []
    pearsons = []
    for _ in range(RANDOM_BASELINE_SAMPLES):
        subset = make_random_subset(rng, quota, by_language, tasks)
        scores = full_matrix[np.array(subset), :].mean(axis=0)
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
    full_matrix: np.ndarray,
    all_vectors: dict[str, np.ndarray],
    metadata: dict,
    diagnostics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [run["name"] for run in FULL_RUNS]
    full_scores = full_matrix.mean(axis=0)
    selected_array = np.array(selected, dtype=int)
    subset_scores = full_matrix[selected_array, :].mean(axis=0)

    selected_sorted = sorted(
        selected,
        key=lambda i: (tasks[i].language_group, tasks[i].repo, tasks[i].instance_id),
    )

    all_available_scores = diagnostics["all_available_task_solve_rate"]
    csv_path = output_dir / "predictive_30_tasks.csv"
    fields = [
        "rank",
        "instance_id",
        "repo",
        "language",
        "language_group",
        "created_at",
        "year_bucket",
        "patch_changed_lines",
        "fail_to_pass_count",
        "pass_to_pass_count",
        "full_model_solve_rate",
        "all_available_solve_rate",
    ] + [f"resolved_{name}" for name in model_names]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, index in enumerate(selected_sorted, start=1):
            task = tasks[index]
            row = {
                "rank": rank,
                "instance_id": task.instance_id,
                "repo": task.repo,
                "language": task.language,
                "language_group": task.language_group,
                "created_at": task.created_at,
                "year_bucket": task.year_bucket,
                "patch_changed_lines": task.patch_changed_lines,
                "fail_to_pass_count": task.fail_to_pass_count,
                "pass_to_pass_count": task.pass_to_pass_count,
                "full_model_solve_rate": round(float(full_matrix[index, :].mean()), 6),
                "all_available_solve_rate": round(float(all_available_scores[index]), 6),
            }
            for model_index, name in enumerate(model_names):
                row[f"resolved_{name}"] = int(full_matrix[index, model_index])
            writer.writerow(row)

    ids_path = output_dir / "predictive_30_instance_ids.txt"
    with ids_path.open("w", encoding="utf-8") as handle:
        for index in selected_sorted:
            handle.write(f"{tasks[index].instance_id}\n")

    comparison_path = output_dir / "predictive_30_model_score_comparison.csv"
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

    baseline = random_baseline(tasks, full_matrix, diagnostics["quota"])
    language_counts = Counter(tasks[i].language_group for i in selected)
    repo_counts = Counter(tasks[i].repo for i in selected)
    summary = {
        "subset_size": SUBSET_SIZE,
        "random_seed": RANDOM_SEED,
        "task_dataset": TASK_DATASET,
        "selection_method": {
            "target": "Minimize 30-task score error against known full/nearly-full per-instance model runs.",
            "constraints": {
                "language_group_counts": "floor/ceil proportional quotas; all choices tried",
                "max_per_repo": MAX_PER_REPO,
            },
            "objective_terms": [
                "model score RMSE",
                "model score max absolute error",
                "full-model task difficulty histogram balance",
                "all-available task difficulty histogram balance",
                "year bucket balance",
            ],
        },
        "selected_instance_ids": [tasks[i].instance_id for i in selected_sorted],
        "selected_language_group_counts": dict(sorted(language_counts.items())),
        "selected_repo_counts": dict(sorted(repo_counts.items())),
        "known_full_runs": metadata["full_runs"],
        "known_partial_runs_used_for_difficulty_signal": metadata["partial_runs"],
        "validation": {
            "rmse_pct_points": diagnostics["rmse"] * 100,
            "mae_pct_points": diagnostics["mae"] * 100,
            "max_abs_error_pct_points": diagnostics["max_abs_error"] * 100,
            "pearson_across_known_model_scores": diagnostics["pearson"],
            "spearman_across_known_model_scores": diagnostics["spearman"],
            "random_stratified_baseline_pct_points": {
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
            "predictive_30_tasks.csv",
            "predictive_30_instance_ids.txt",
            "predictive_30_model_score_comparison.csv",
            "predictive_30_summary.json",
        ],
    }
    with (output_dir / "predictive_30_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    tasks = load_tasks()
    metadata, full_matrix, all_vectors = build_outcomes(tasks)
    selected, diagnostics = optimize_subset(tasks, full_matrix, all_vectors)
    write_outputs(output_dir, tasks, selected, full_matrix, all_vectors, metadata, diagnostics)
    print(f"Wrote {SUBSET_SIZE} tasks to {output_dir}")
    print(f"RMSE: {diagnostics['rmse'] * 100:.3f} percentage points")
    print(f"Max absolute error: {diagnostics['max_abs_error'] * 100:.3f} percentage points")


if __name__ == "__main__":
    main()
