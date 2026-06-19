#!/usr/bin/env python3
"""Audit other-source mini-swe-agent traces for patch and SFT quality."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


FORBIDDEN_SUBMISSION_SNIPPETS = (
    "patch.txt",
    "/tmp/",
    "# ... existing code ...",
    "... existing code",
)
SCRATCH_PATH_PARTS = (
    "/tmp/",
    "scratch",
    "repro",
    "reproduce",
    "debug",
)
TEST_PATH_PARTS = (
    "test/",
    "tests/",
    "_test.",
    "test_",
    ".snap",
    "fixture",
    "golden",
    "testdata/",
    "_result.",
    "expected",
)
LANGUAGE_SOURCE_EXTENSIONS = {
    "go": {".go"},
    "java": {".java"},
    "python": {".py"},
    "c": {".c", ".h"},
    "cpp": {".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ipp", ".tpp", ".inl"},
}
MAX_CLEAN_TRAJECTORY_BYTES = 32 * 1024 * 1024


def patch_paths(patch: str) -> list[str]:
    return [match[1] for match in re.findall(r"(?m)^diff --git a/(.*?) b/(.*?)$", patch)]


def source_paths_only(paths: list[str], language: str = "") -> bool:
    if not paths:
        return False
    allowed_extensions = LANGUAGE_SOURCE_EXTENSIONS.get(language.lower())
    for path in paths:
        lowered = path.lower()
        if any(part in lowered for part in SCRATCH_PATH_PARTS):
            return False
        if any(part in lowered for part in TEST_PATH_PARTS):
            return False
        if allowed_extensions and Path(path).suffix.lower() not in allowed_extensions:
            return False
    return True


def clean_submission(submission: str, language: str = "") -> tuple[bool, list[str]]:
    reasons = []
    if not submission.strip():
        reasons.append("empty_submission")
    if "diff --git " not in submission:
        reasons.append("missing_diff_header")
    paths = patch_paths(submission)
    if not source_paths_only(paths, language):
        reasons.append("bad_changed_paths")
    for snippet in FORBIDDEN_SUBMISSION_SNIPPETS:
        if snippet in submission:
            reasons.append(f"forbidden_snippet:{snippet}")
    if re.search(r"(?m)^diff --git a/patch\.txt b/patch\.txt$", submission):
        reasons.append("patch_txt_diff")
    return not reasons, reasons


def load_trajectory(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reasoning_fraction(trajectory: dict[str, Any]) -> float:
    assistants = [message for message in trajectory.get("messages", []) if message.get("role") == "assistant"]
    if not assistants:
        return 0.0
    with_reasoning = [
        message
        for message in assistants
        if str(message.get("reasoning_content") or "").strip()
    ]
    return len(with_reasoning) / len(assistants)


def final_submission(trajectory: dict[str, Any]) -> str:
    return str((trajectory.get("info") or {}).get("submission") or "")


def apply_check_in_image(task_dir: Path, image: str, patch: str, timeout: int) -> tuple[bool, str]:
    if not patch.strip():
        return False, "empty patch"
    with tempfile.TemporaryDirectory(prefix="other-source-audit.") as tmp:
        patch_path = Path(tmp) / "submission.patch"
        patch_path.write_text(patch, encoding="utf-8")
        command = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "-v",
            f"{patch_path}:/tmp/submission.patch:ro",
            image,
            "bash",
            "-lc",
            "cd /testbed && git reset --hard buggy >/dev/null && git clean -fd >/dev/null && git apply --check /tmp/submission.patch",
        ]
        try:
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return result.returncode == 0, result.stdout[-4000:]


def task_image(task_dir: Path) -> str:
    text = (task_dir / "task.toml").read_text(encoding="utf-8", errors="replace")
    match = re.search(r'(?m)^\s*docker_image\s*=\s*"((?:\\.|[^"])*)"', text)
    if not match:
        return ""
    return bytes(match.group(1), "utf-8").decode("unicode_escape", errors="replace")


def task_language(task_dir: Path) -> str:
    text = (task_dir / "task.toml").read_text(encoding="utf-8", errors="replace")
    match = re.search(r'(?m)^\s*language\s*=\s*"((?:\\.|[^"])*)"', text)
    if not match:
        return ""
    return bytes(match.group(1), "utf-8").decode("unicode_escape", errors="replace")


def audit_result(result_path: Path, apply_check: bool, timeout: int) -> dict[str, Any]:
    workspace = result_path.parent
    result = json.loads(result_path.read_text(encoding="utf-8"))
    trajectory_path = workspace / "agent" / "mini-swe-agent.trajectory.json"
    trajectory = load_trajectory(trajectory_path) if trajectory_path.exists() else {"messages": [], "info": {}}
    trajectory_bytes = trajectory_path.stat().st_size if trajectory_path.exists() else 0
    submission = final_submission(trajectory)
    task_dir = Path(result.get("task_dir") or "")
    language = task_language(task_dir) if task_dir.exists() else ""
    clean, reasons = clean_submission(submission, language)
    if trajectory_bytes > MAX_CLEAN_TRAJECTORY_BYTES:
        reasons.append("trajectory_too_large")
    image = task_image(task_dir) if task_dir.exists() else str(result.get("docker_image") or "")
    applies = None
    apply_output = ""
    if apply_check and image:
        applies, apply_output = apply_check_in_image(task_dir, image, submission, timeout)
        if not applies:
            reasons.append("patch_apply_check_failed")
    patch_paths_value = patch_paths(submission)
    return {
        "instance_id": result.get("instance_id") or workspace.name,
        "workspace": str(workspace),
        "reward": result.get("reward"),
        "status": result.get("agent_exit_status"),
        "api_calls": result.get("api_calls"),
        "trajectory_bytes": trajectory_bytes,
        "submission_bytes": len(submission.encode("utf-8")),
        "changed_paths": patch_paths_value,
        "language": language,
        "assistant_reasoning_fraction": reasoning_fraction(trajectory),
        "clean_submission": clean,
        "patch_applies": applies,
        "accepted": (
            result.get("reward") == 1
            and clean
            and trajectory_bytes <= MAX_CLEAN_TRAJECTORY_BYTES
            and (applies is not False)
            and reasoning_fraction(trajectory) >= 0.9
        ),
        "reject_reasons": sorted(set(reasons)),
        "apply_check_output_tail": apply_output[-1000:] if apply_output else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--apply-check", action="store_true")
    parser.add_argument("--apply-timeout", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [
        audit_result(path, args.apply_check, args.apply_timeout)
        for path in sorted(args.trace_root.glob("*/result.json"))
    ]
    summary = {
        "trace_root": str(args.trace_root),
        "rows": rows,
        "count": len(rows),
        "accepted": sum(1 for row in rows if row["accepted"]),
        "passrate": (sum(1 for row in rows if row["reward"] == 1) / len(rows)) if rows else 0.0,
        "clean_submission_fraction": (sum(1 for row in rows if row["clean_submission"]) / len(rows)) if rows else 0.0,
        "apply_check_fraction": (
            sum(1 for row in rows if row["patch_applies"] is True) / len(rows)
        ) if rows and args.apply_check else None,
        "reasoning_ge_90_fraction": (
            sum(1 for row in rows if row["assistant_reasoning_fraction"] >= 0.9) / len(rows)
        ) if rows else 0.0,
        "reject_reason_counts": {},
    }
    for row in rows:
        for reason in row["reject_reasons"]:
            summary["reject_reason_counts"][reason] = summary["reject_reason_counts"].get(reason, 0) + 1
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
