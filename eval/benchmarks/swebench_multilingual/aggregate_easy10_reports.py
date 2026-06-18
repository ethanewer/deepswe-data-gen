#!/usr/bin/env python3
"""Aggregate SWE-bench Multilingual easy-10 evaluation reports across N trials.

Usage:
    python aggregate_easy10_reports.py REPORT.json [REPORT.json ...]

Each REPORT.json is an official swebench.harness.run_evaluation report
(contains resolved_ids / empty_patch_ids / error_ids). Prints per-trial
resolved rates, the mean, pass@N (union) and pass^N (intersection), and a
per-instance solve-count table over the easy-10 subset.
"""
import json
import statistics
import sys
from pathlib import Path

SUBSET = Path(__file__).with_name("swebench_multilingual_easy_10_instance_ids.txt")


def main(paths: list[str]) -> None:
    ids = [
        l.strip()
        for l in SUBSET.read_text().splitlines()
        if l.strip() and not l.lstrip().startswith("#")
    ]
    N = len(ids)
    reports = [json.loads(Path(p).read_text()) for p in paths]
    resolved = [set(r.get("resolved_ids", [])) for r in reports]
    counts = [len(r) for r in resolved]

    print(f"Trials: {len(reports)}   Instances: {N}\n")
    for i, c in enumerate(counts, 1):
        print(f"Trial {i}: {c}/{N} = {100 * c / N:.0f}%")
    print(
        f"\nMean: {statistics.mean(counts):.2f}/{N} = "
        f"{100 * statistics.mean(counts) / N:.1f}%  "
        f"(min {min(counts)}, max {max(counts)}, stdev {statistics.pstdev(counts):.2f})"
    )
    union = set().union(*resolved)
    inter = set.intersection(*resolved) if resolved else set()
    print(f"pass@{len(reports)} (any):  {len(union)}/{N} = {100 * len(union) / N:.0f}%")
    print(f"pass^{len(reports)} (all):  {len(inter)}/{N} = {100 * len(inter) / N:.0f}%")

    print("\nPer-instance solve count:")
    for iid in sorted(ids, key=lambda i: -sum(i in r for r in resolved)):
        c = sum(iid in r for r in resolved)
        print(f"  {c}/{len(reports)}  {iid}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
