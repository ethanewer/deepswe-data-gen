#!/usr/bin/env python3
"""Assert runnable quality for prepared other-source task images."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


SOURCE_EXTENSIONS = {
    "c": {".c", ".h"},
    "cpp": {".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ipp", ".tpp", ".inl"},
}
TEST_PATH_PARTS = (
    "test/",
    "tests/",
    "_test.",
    "test_",
    "fixture",
    "golden",
    "testdata/",
    "expected",
)


def parse_manifest(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 16:
                continue
            rows.append(
                {
                    "index": fields[0],
                    "rollout_id": fields[1],
                    "instance_id": fields[2],
                    "task_dir": fields[3],
                    "workspace": fields[4],
                    "image": fields[5],
                    "model": fields[6],
                    "litellm_model": fields[7],
                    "difficulty": fields[11],
                    "language": fields[12],
                    "instruction_style": fields[13],
                    "repo": fields[14],
                }
            )
    return rows


def patch_paths(patch: str) -> list[str]:
    paths = [match[1] for match in re.findall(r"(?m)^diff --git a/(.*?) b/(.*?)$", patch)]
    if paths:
        return paths
    parsed = []
    old_path = ""
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            old_path = line[len("--- a/") :].split("\t", 1)[0]
        elif line.startswith("+++ b/") and old_path:
            parsed.append(line[len("+++ b/") :].split("\t", 1)[0])
            old_path = ""
    return parsed


def source_patch_quality(task_dir: Path, language: str) -> tuple[bool, list[str], list[str]]:
    reasons = []
    patch_path = task_dir / "solution" / "solution.patch"
    patch = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.exists() else ""
    if not patch.strip():
        reasons.append("missing_solution_patch")
    paths = patch_paths(patch)
    if not paths:
        reasons.append("missing_patch_paths")
    allowed = SOURCE_EXTENSIONS.get(language, set())
    for path in paths:
        lowered = path.lower()
        if any(part in lowered for part in TEST_PATH_PARTS):
            reasons.append(f"test_or_fixture_path:{path}")
        if allowed and Path(path).suffix.lower() not in allowed:
            reasons.append(f"non_source_extension:{path}")
    return not reasons, sorted(set(reasons)), paths


def inspect_image(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def run_dynamic_quality(row: dict[str, str], timeout: int) -> tuple[bool, str, int]:
    task_dir = Path(row["task_dir"])
    command = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "-e",
        "SWE_SMITH_CPP_BUILD_JOBS=4",
        "-v",
        f"{task_dir / 'tests'}:/tests:ro",
        "-v",
        f"{task_dir / 'solution' / 'solution.patch'}:/tmp/solution.patch:ro",
        row["image"],
        "bash",
        "-lc",
        r"""
set -uo pipefail
cd /testbed || exit 10
git config --global --add safe.directory /testbed 2>/dev/null || true
git rev-parse --is-inside-work-tree >/dev/null || exit 11
git rev-parse --verify buggy >/dev/null || exit 12
git reset --hard buggy >/dev/null
git clean -fd >/dev/null
bash /tests/test.sh >/tmp/buggy-verifier.log 2>&1
buggy_status=$?
git reset --hard buggy >/dev/null
git clean -fd >/dev/null
git apply /tmp/solution.patch || exit 13
bash /tests/test.sh >/tmp/gold-verifier.log 2>&1
gold_status=$?
if [ "$buggy_status" -eq 0 ]; then
  echo "BUGGY_UNEXPECTED_PASS"
  tail -n 80 /tmp/buggy-verifier.log
  exit 20
fi
if [ "$gold_status" -ne 0 ]; then
  echo "GOLD_FAILED status=$gold_status"
  tail -n 120 /tmp/gold-verifier.log
  exit 21
fi
echo "QUALITY_OK buggy_status=$buggy_status gold_status=$gold_status"
""",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"timeout after {timeout}s\n{exc.stdout or ''}", 124
    return result.returncode == 0, result.stdout[-6000:], result.returncode


def check_row(row: dict[str, str], timeout: int) -> dict[str, Any]:
    started = time.time()
    task_dir = Path(row["task_dir"])
    reasons = []
    paths: list[str] = []
    if not inspect_image(row["image"]):
        reasons.append("image_missing")
    patch_ok, patch_reasons, paths = source_patch_quality(task_dir, row["language"])
    if not patch_ok:
        reasons.extend(patch_reasons)
    dynamic_ok = False
    output = ""
    returncode = None
    if not reasons:
        dynamic_ok, output, returncode = run_dynamic_quality(row, timeout)
        if not dynamic_ok:
            reasons.append("dynamic_quality_failed")
    return {
        "instance_id": row["instance_id"],
        "task_dir": row["task_dir"],
        "image": row["image"],
        "language": row["language"],
        "repo": row["repo"],
        "accepted": not reasons,
        "reject_reasons": sorted(set(reasons)),
        "patch_paths": paths,
        "dynamic_returncode": returncode,
        "dynamic_output_tail": output,
        "elapsed_sec": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    rows = parse_manifest(args.manifest_tsv)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(check_row, row, args.timeout) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "instance_id": result["instance_id"],
                        "accepted": result["accepted"],
                        "reject_reasons": result["reject_reasons"],
                        "elapsed_sec": round(result["elapsed_sec"], 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    results.sort(key=lambda row: row["instance_id"])
    summary = {
        "manifest_tsv": str(args.manifest_tsv),
        "count": len(results),
        "accepted": sum(1 for row in results if row["accepted"]),
        "rejected": sum(1 for row in results if not row["accepted"]),
        "reject_reason_counts": {},
        "rows": results,
    }
    for row in results:
        for reason in row["reject_reasons"]:
            summary["reject_reason_counts"][reason] = summary["reject_reason_counts"].get(reason, 0) + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("count", "accepted", "rejected", "reject_reason_counts")}, indent=2, sort_keys=True))
    if summary["rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
