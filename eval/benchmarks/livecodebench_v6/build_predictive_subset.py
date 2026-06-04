#!/usr/bin/env python
"""Build a 50-problem predictive subset for LiveCodeBench v6.

LiveCodeBench's `v6` config is the 175-problem slice newly added in the
release_v6 era. This script selects 50 problems whose per-model pass@1 averages
best preserve full-v6 scores over the public leaderboard's per-problem records.
"""

from __future__ import annotations

import csv
import json
import math
import random
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download


MODULE_DIR = Path(__file__).resolve().parent
DEFAULTS_PATH = MODULE_DIR / "defaults.json"
TASKS_CSV = MODULE_DIR / "predictive_50_tasks.csv"
IDS_TXT = MODULE_DIR / "predictive_50_question_ids.txt"
COMPARISON_CSV = MODULE_DIR / "predictive_50_model_score_comparison.csv"
SUMMARY_JSON = MODULE_DIR / "predictive_50_summary.json"


@dataclass(frozen=True)
class Problem:
    index: int
    question_id: str
    title: str
    platform: str
    contest_id: str
    contest_date: str
    difficulty: str
    statement_chars: int
    public_tests: int


def load_defaults() -> dict:
    return json.loads(DEFAULTS_PATH.read_text())


def load_v6_problems(defaults: dict) -> list[Problem]:
    path = hf_hub_download(
        defaults["dataset_repo"],
        defaults["dataset_file"],
        repo_type="dataset",
    )
    problems: list[Problem] = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        row = json.loads(line)
        public_tests = len(json.loads(row["public_test_cases"] or "[]"))
        problems.append(
            Problem(
                index=index,
                question_id=row["question_id"],
                title=row["question_title"],
                platform=row["platform"],
                contest_id=row["contest_id"],
                contest_date=row["contest_date"],
                difficulty=row["difficulty"],
                statement_chars=len(row["question_content"] or ""),
                public_tests=public_tests,
            )
        )
    return problems


def load_leaderboard_records(defaults: dict) -> list[dict]:
    with urllib.request.urlopen(defaults["leaderboard_url"], timeout=60) as response:
        data = json.loads(response.read())
    return data["performances"]


def build_matrix(problems: list[Problem], records: list[dict]) -> tuple[list[str], np.ndarray]:
    qids = [p.question_id for p in problems]
    qid_to_index = {qid: i for i, qid in enumerate(qids)}
    by_model: dict[str, list[float | None]] = defaultdict(lambda: [None] * len(qids))
    for record in records:
        idx = qid_to_index.get(record["question_id"])
        if idx is None:
            continue
        by_model[record["model"]][idx] = float(record["pass@1"]) / 100.0

    models = sorted(model for model, values in by_model.items() if all(v is not None for v in values))
    matrix = np.array([[by_model[model][i] for i in range(len(qids))] for model in models], dtype=float)
    if matrix.shape[1] != 175:
        raise RuntimeError(f"Expected 175 v6 problems, got {matrix.shape[1]}")
    if len(models) < 2:
        raise RuntimeError("Need at least two complete model result vectors.")
    return models, matrix


def quotas(problems: list[Problem], subset_size: int) -> dict[tuple[str, str], int]:
    counts = Counter((p.difficulty, p.platform) for p in problems)
    raw = {key: subset_size * value / len(problems) for key, value in counts.items()}
    base = {key: math.floor(value) for key, value in raw.items()}
    remaining = subset_size - sum(base.values())
    for key, _ in sorted(raw.items(), key=lambda kv: kv[1] - math.floor(kv[1]), reverse=True)[:remaining]:
        base[key] += 1
    return base


def rmse_for_indices(matrix: np.ndarray, indices: list[int]) -> float:
    full = matrix.mean(axis=1)
    subset = matrix[:, indices].mean(axis=1)
    return float(np.sqrt(np.mean((subset - full) ** 2)))


def max_abs_for_indices(matrix: np.ndarray, indices: list[int]) -> float:
    full = matrix.mean(axis=1)
    subset = matrix[:, indices].mean(axis=1)
    return float(np.max(np.abs(subset - full)))


def greedy_start(
    problems: list[Problem],
    matrix: np.ndarray,
    quota: dict[tuple[str, str], int],
    rng: random.Random,
) -> list[int]:
    selected: list[int] = []
    used = Counter()
    target = matrix.mean(axis=1)
    candidates_by_bucket: dict[tuple[str, str], list[int]] = defaultdict(list)
    for problem in problems:
        candidates_by_bucket[(problem.difficulty, problem.platform)].append(problem.index)

    for bucket, count in sorted(quota.items()):
        for _ in range(count):
            best: tuple[float, float, int] | None = None
            remaining = [i for i in candidates_by_bucket[bucket] if i not in selected]
            rng.shuffle(remaining)
            for idx in remaining:
                trial = selected + [idx]
                estimate = matrix[:, trial].mean(axis=1)
                error = float(np.sqrt(np.mean((estimate - target) ** 2)))
                tie = float(np.var(matrix[:, idx]))
                key = (error, -tie, idx)
                if best is None or key < best:
                    best = key
            if best is None:
                raise RuntimeError(f"No candidates left for bucket {bucket}")
            selected.append(best[2])
            used[bucket] += 1
    return selected


def local_search(
    problems: list[Problem],
    matrix: np.ndarray,
    selected: list[int],
    rng: random.Random,
    steps: int = 4000,
) -> list[int]:
    selected_set = set(selected)
    bucket_by_index = {p.index: (p.difficulty, p.platform) for p in problems}
    by_bucket: dict[tuple[str, str], list[int]] = defaultdict(list)
    for p in problems:
        by_bucket[bucket_by_index[p.index]].append(p.index)
    current = selected[:]
    current_score = rmse_for_indices(matrix, current)
    for _ in range(steps):
        out_pos = rng.randrange(len(current))
        out_idx = current[out_pos]
        bucket = bucket_by_index[out_idx]
        choices = [idx for idx in by_bucket[bucket] if idx not in selected_set]
        if not choices:
            continue
        in_idx = rng.choice(choices)
        trial = current[:]
        trial[out_pos] = in_idx
        score = rmse_for_indices(matrix, trial)
        if score < current_score or (score == current_score and rng.random() < 0.05):
            selected_set.remove(out_idx)
            selected_set.add(in_idx)
            current = trial
            current_score = score
    return sorted(current)


def random_baseline(
    problems: list[Problem],
    matrix: np.ndarray,
    quota: dict[tuple[str, str], int],
    rng: random.Random,
    samples: int = 5000,
) -> dict:
    by_bucket: dict[tuple[str, str], list[int]] = defaultdict(list)
    for problem in problems:
        by_bucket[(problem.difficulty, problem.platform)].append(problem.index)
    scores = []
    for _ in range(samples):
        chosen = []
        for bucket, count in quota.items():
            chosen.extend(rng.sample(by_bucket[bucket], count))
        scores.append(rmse_for_indices(matrix, chosen))
    arr = np.array(scores)
    return {
        "samples": samples,
        "median_rmse_pp": float(np.median(arr) * 100),
        "p10_rmse_pp": float(np.percentile(arr, 10) * 100),
        "best_rmse_pp": float(np.min(arr) * 100),
    }


def main() -> None:
    defaults = load_defaults()
    rng = random.Random(defaults["random_seed"])
    problems = load_v6_problems(defaults)
    records = load_leaderboard_records(defaults)
    models, matrix = build_matrix(problems, records)
    quota = quotas(problems, defaults["subset_size"])

    best: list[int] | None = None
    best_score = float("inf")
    for _ in range(40):
        start = greedy_start(problems, matrix, quota, rng)
        candidate = local_search(problems, matrix, start, rng)
        score = rmse_for_indices(matrix, candidate)
        if score < best_score:
            best = candidate
            best_score = score
    assert best is not None

    selected = [problems[i] for i in best]
    full_scores = matrix.mean(axis=1)
    subset_scores = matrix[:, best].mean(axis=1)

    with IDS_TXT.open("w") as f:
        for problem in selected:
            f.write(problem.question_id + "\n")

    with TASKS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "question_id",
                "title",
                "platform",
                "contest_id",
                "contest_date",
                "difficulty",
                "statement_chars",
                "public_tests",
                "mean_pass_rate",
                "std_pass_rate",
            ],
        )
        writer.writeheader()
        for rank, problem in enumerate(selected, start=1):
            values = matrix[:, problem.index]
            writer.writerow(
                {
                    "rank": rank,
                    "question_id": problem.question_id,
                    "title": problem.title,
                    "platform": problem.platform,
                    "contest_id": problem.contest_id,
                    "contest_date": problem.contest_date,
                    "difficulty": problem.difficulty,
                    "statement_chars": problem.statement_chars,
                    "public_tests": problem.public_tests,
                    "mean_pass_rate": round(float(values.mean()), 4),
                    "std_pass_rate": round(float(values.std()), 4),
                }
            )

    with COMPARISON_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "full_v6_score", "subset_50_score", "delta"],
        )
        writer.writeheader()
        for model, full, subset in zip(models, full_scores, subset_scores):
            writer.writerow(
                {
                    "model": model,
                    "full_v6_score": round(float(full) * 100, 4),
                    "subset_50_score": round(float(subset) * 100, 4),
                    "delta": round(float(subset - full) * 100, 4),
                }
            )

    summary = {
        "dataset": defaults["dataset_repo"],
        "dataset_file": defaults["dataset_file"],
        "leaderboard_url": defaults["leaderboard_url"],
        "total_problems": len(problems),
        "subset_size": len(selected),
        "models_used": len(models),
        "rmse_pp": best_score * 100,
        "max_abs_error_pp": max_abs_for_indices(matrix, best) * 100,
        "difficulty_counts_full": dict(Counter(p.difficulty for p in problems)),
        "difficulty_counts_subset": dict(Counter(p.difficulty for p in selected)),
        "platform_counts_full": dict(Counter(p.platform for p in problems)),
        "platform_counts_subset": dict(Counter(p.platform for p in selected)),
        "quota": {f"{difficulty}/{platform}": count for (difficulty, platform), count in quota.items()},
        "random_baseline": random_baseline(problems, matrix, quota, rng),
        "question_ids": [p.question_id for p in selected],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
