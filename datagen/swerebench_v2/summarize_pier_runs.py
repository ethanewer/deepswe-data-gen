#!/usr/bin/env python3
"""Summarize Pier/mini-swe-agent data-generation runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


TEST_FILE_RE = re.compile(
    r"(^|/)(tests?|spec|__tests__)(/|$)|"
    r"(^|/)(test_[^/]+|[^/]+_(test|spec)|[^/]+\.(test|spec))\.[A-Za-z0-9]+$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--assignments-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--max-patch-lines", type=int, default=5000)
    return parser.parse_args()


def load_assignments(path: Path | None) -> dict[str, dict[str, str]]:
    if not path or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["instance_id"]: row for row in csv.DictReader(handle)}


def read_reward(trial_dir: Path) -> int | None:
    reward_path = trial_dir / "verifier" / "reward.txt"
    if not reward_path.exists():
        return None
    text = reward_path.read_text(errors="replace").strip()
    try:
        return int(float(text))
    except ValueError:
        return None


def patch_headers(patch: str) -> list[str]:
    paths = []
    for line in patch.splitlines():
        match = re.match(r"^diff --git a/(.*?) b/(.*?)$", line)
        if match:
            paths.append(match.group(2).strip('"'))
    return paths


def patch_quality(trial_dir: Path) -> dict:
    patch_path = trial_dir / "artifacts" / "model.patch"
    if not patch_path.exists():
        return {"patch_exists": False, "patch_lines": 0, "test_file_edits": []}
    patch = patch_path.read_text(errors="replace")
    files = patch_headers(patch)
    test_files = [path for path in files if TEST_FILE_RE.search(path)]
    return {
        "patch_exists": True,
        "patch_lines": len(patch.splitlines()),
        "test_file_edits": test_files,
    }


def trial_records(
    run_root: Path,
    assignments: dict[str, dict[str, str]],
    *,
    max_patch_lines: int,
) -> list[dict]:
    records = []
    for result_path in sorted((run_root / "pier-jobs").glob("*/*/*/*/result.json")):
        trial_dir = result_path.parent
        try:
            result = json.loads(result_path.read_text())
        except Exception as exc:
            result = {"error": f"could not parse result.json: {exc}"}
        task_name = result.get("task_name") or ""
        instance_id = task_name.rsplit("/", 1)[-1] if task_name else trial_dir.name.rsplit("__", 1)[0]
        assignment = assignments.get(instance_id, {})
        style = assignment.get("instruction_style") or result_path.parents[3].name
        model = assignment.get("assigned_model") or result_path.parents[2].name
        quality = patch_quality(trial_dir)
        reward = read_reward(trial_dir)
        trajectory = trial_dir / "agent" / "trajectory.json"
        mini_trajectory = trial_dir / "agent" / "mini-swe-agent.trajectory.json"
        records.append(
            {
                "instance_id": instance_id,
                "trial_dir": str(trial_dir),
                "model": model,
                "difficulty": assignment.get("difficulty", ""),
                "language": assignment.get("language", ""),
                "style": style,
                "reward": reward,
                "has_trajectory": trajectory.exists() or mini_trajectory.exists(),
                "has_exception": (trial_dir / "exception.txt").exists(),
                "patch_exists": quality["patch_exists"],
                "patch_lines": quality["patch_lines"],
                "patch_too_large": quality["patch_lines"] > max_patch_lines,
                "test_file_edits": quality["test_file_edits"],
                "clean_quality_pass": bool(
                    reward == 1
                    and (trajectory.exists() or mini_trajectory.exists())
                    and quality["patch_exists"]
                    and quality["patch_lines"] > 0
                    and quality["patch_lines"] <= max_patch_lines
                    and not quality["test_file_edits"]
                    and not (trial_dir / "exception.txt").exists()
                ),
            }
        )
    return records


def grouped(records: list[dict], keys: tuple[str, ...]) -> dict[str, dict[str, int | float]]:
    out = {}
    for record in records:
        label = "/".join(record.get(key, "") or "unknown" for key in keys)
        out.setdefault(label, {"total": 0, "reward_pass": 0, "clean_quality_pass": 0})
        out[label]["total"] += 1
        out[label]["reward_pass"] += int(record.get("reward") == 1)
        out[label]["clean_quality_pass"] += int(record.get("clean_quality_pass"))
    for stats in out.values():
        total = stats["total"] or 1
        stats["reward_pass_rate"] = stats["reward_pass"] / total
        stats["clean_quality_pass_rate"] = stats["clean_quality_pass"] / total
    return dict(sorted(out.items()))


def main() -> None:
    args = parse_args()
    assignments = load_assignments(args.assignments_csv)
    records = trial_records(
        args.run_root,
        assignments,
        max_patch_lines=args.max_patch_lines,
    )
    summary = {
        "run_root": str(args.run_root),
        "total_trials_with_result": len(records),
        "reward_pass": sum(record["reward"] == 1 for record in records),
        "clean_quality_pass": sum(record["clean_quality_pass"] for record in records),
        "pending_assignments": max(len(assignments) - len({r["instance_id"] for r in records}), 0),
        "quality_reject_reasons": {
            "test_file_edits": sum(bool(record["test_file_edits"]) for record in records),
            "empty_or_missing_patch": sum(
                not record["patch_exists"] or record["patch_lines"] == 0 for record in records
            ),
            "patch_too_large": sum(record["patch_too_large"] for record in records),
            "missing_trajectory": sum(not record["has_trajectory"] for record in records),
            "exceptions": sum(record["has_exception"] for record in records),
        },
        "by_model": grouped(records, ("model",)),
        "by_difficulty": grouped(records, ("difficulty",)),
        "by_style": grouped(records, ("style",)),
        "by_model_style": grouped(records, ("model", "style")),
        "by_language": grouped(records, ("language",)),
        "records": records,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
