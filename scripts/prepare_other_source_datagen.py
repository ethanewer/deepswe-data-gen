#!/usr/bin/env python3
"""Prepare non-overlapping non-V2 task-source batches for mini-swe-agent datagen."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download


CURRENT_INDEX = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swerebench-traces-highquality-2x-duplicate-reasoning-90pct/metadata/index.jsonl"
)
ENV_FILE = Path("/wbl-fast/usrs/ee/code-swe-data/.env")
SWE_REBENCH_V1 = "nebius/SWE-rebench"
OPENTHINK_TASKTROVE = "open-thoughts/TaskTrove"
OPENSWE_PARQUET = "laion__openswe-tasks-patched-v5-oracle-success/tasks.parquet"
SWEGYM_PARQUET = "laion__swegym-tasks-patched-validated-v5/tasks.parquet"
SWE_SMITH_HARBOR = "ricdomolm/SWE-smith-trajectories-harbor-found-235B"
SWE_SMITH_CPP = "SWE-bench/SWE-smith-cpp"


@dataclass(frozen=True)
class CurrentIndex:
    task_ids: set[str]
    repo_issue_keys: set[str]


def load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_READ_TOKEN")
    )


def read_current_index(path: Path = CURRENT_INDEX) -> CurrentIndex:
    task_ids: set[str] = set()
    repo_issue_keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or row.get("instance_id") or "")
            repo = str(row.get("repo") or "")
            if task_id:
                task_ids.add(task_id)
            issue = issue_number(task_id)
            if repo and issue:
                repo_issue_keys.add(f"{repo}::{issue}")
    return CurrentIndex(task_ids, repo_issue_keys)


def issue_number(instance_id: str) -> str:
    match = re.search(r"-(\d+)$", instance_id)
    return match.group(1) if match else ""


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def toml_string(value: str) -> str:
    return json.dumps(value)


def task_slug(instance_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", instance_id).strip("-").lower()


def patch_paths(patch: str) -> list[str]:
    paths = re.findall(r"(?m)^diff --git a/(.*?) b/(.*?)$", patch)
    if paths:
        return paths
    # SWE-smith C++ stores plain unified patches without `diff --git` lines.
    parsed: list[tuple[str, str]] = []
    old_path = ""
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            old_path = line[len("--- a/") :].split("\t", 1)[0]
        elif line.startswith("+++ b/") and old_path:
            new_path = line[len("+++ b/") :].split("\t", 1)[0]
            parsed.append((old_path, new_path))
            old_path = ""
    return parsed


def patch_file_count(patch: str) -> int:
    return len(patch_paths(patch))


def patch_net_line_delta(patch: str) -> int:
    delta = 0
    for line in patch.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            delta += 1
        elif line.startswith("-"):
            delta -= 1
    return delta


def patch_touches_non_test(patch: str) -> bool:
    for _, path in patch_paths(patch):
        lowered = path.lower()
        if not any(x in lowered for x in ("tests/", "/test", "test_", "_test.", "fixture", "golden")):
            return True
    return False


def patch_languages(patch: str) -> set[str]:
    ext_lang = {
        ".py": "python",
        ".js": "js",
        ".jsx": "js",
        ".ts": "ts",
        ".tsx": "ts",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".php": "php",
        ".rb": "ruby",
        ".cs": "csharp",
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cxx": "cpp",
        ".h": "cpp",
        ".hh": "cpp",
        ".hpp": "cpp",
        ".hxx": "cpp",
        ".ipp": "cpp",
        ".tpp": "cpp",
    }
    langs = set()
    for _, path in patch_paths(patch):
        filename = path.rsplit("/", 1)[-1]
        ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
        if ext in ext_lang:
            langs.add(ext_lang[ext])
    return langs


def clean_issue_prompt(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def reverse_zero_delta_patch(patch: str) -> str:
    """Return an inverse patch for simple unified diffs."""
    out: list[str] = []
    block: list[str] = []

    def flush_block() -> None:
        if not block:
            return
        for line in block:
            if line.startswith("+"):
                out.append("-" + line[1:])
        for line in block:
            if line.startswith("-"):
                out.append("+" + line[1:])
        block.clear()

    for line in patch.splitlines():
        if line.startswith(("+++", "---")):
            flush_block()
            out.append(line)
        elif line.startswith("index "):
            flush_block()
            parts = line.split()
            if len(parts) >= 2 and ".." in parts[1]:
                before, after = parts[1].split("..", 1)
                rest = " ".join(parts[2:])
                out.append(" ".join(part for part in ["index", f"{after}..{before}", rest] if part))
            else:
                out.append(line)
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            block.append(line)
        else:
            flush_block()
            out.append(line)
    flush_block()
    return "\n".join(out) + "\n"


def build_swe_rebench_task_toml(row: dict[str, Any]) -> str:
    return f"""schema_version = "1.1"
artifacts = []

[task]
name = {toml_string("swe-rebench-v1/" + row["instance_id"])}
description = ""
authors = []
keywords = []

[metadata]
task_id = {toml_string(row["instance_id"])}
display_title = {toml_string((row.get("problem_statement") or row["instance_id"]).splitlines()[0][:180])}
display_description = ""
original_title = {toml_string((row.get("problem_statement") or row["instance_id"]).splitlines()[0][:180])}
category = "software-engineering"
language = "python"
repository_url = {toml_string("https://github.com/" + row["repo"])}
base_commit_hash = {toml_string(row["base_commit"])}
source_dataset = "nebius/SWE-rebench"
source_split = "filtered"

[verifier]
timeout_sec = 1800.0

[agent]
timeout_sec = 5400.0

[environment]
build_timeout_sec = 1800.0
docker_image = {toml_string(row["image_name"])}
os = "linux"
cpus = 2
memory_mb = 8192
storage_mb = 20480
gpus = 0
allow_internet = false
mcp_servers = []
workdir = "/testbed"

[environment.env]

[solution.env]
"""


def build_swe_rebench_test_sh(row: dict[str, Any]) -> str:
    install_config = row.get("install_config") or {}
    test_cmd = install_config.get("test_cmd") or "pytest -q"

    def command_list(value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]

    def normalize_shell_command(command: str) -> str:
        command = command.strip()
        if command.startswith("pip "):
            return "python -m " + command
        if command.startswith("pytest "):
            return "python -m " + command
        return command

    setup_lines = [
        "if [ -f /opt/conda/etc/profile.d/conda.sh ]; then",
        "  . /opt/conda/etc/profile.d/conda.sh",
        "  conda activate testbed 2>/dev/null || true",
        "fi",
        "python -m pip install --upgrade pip setuptools wheel",
    ]
    reqs_path = install_config.get("reqs_path")
    if isinstance(reqs_path, str):
        reqs_paths = [reqs_path]
    elif isinstance(reqs_path, list):
        reqs_paths = [str(path) for path in reqs_path if path]
    else:
        reqs_paths = []
    for req_path in reqs_paths:
        setup_lines.append(f"if [ -f {shell_quote(req_path)} ]; then python -m pip install -r {shell_quote(req_path)}; fi")
    pip_packages = [str(package) for package in (install_config.get("pip_packages") or []) if package]
    if pip_packages:
        setup_lines.append("python -m pip install " + " ".join(shell_quote(package) for package in pip_packages))
    for command in command_list(install_config.get("pre_install")):
        setup_lines.append(normalize_shell_command(command))
    for command in command_list(install_config.get("install")):
        setup_lines.append(normalize_shell_command(command))
    setup_block = "\n".join(setup_lines)
    test_command = normalize_shell_command(str(test_cmd))
    return f"""#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

cd /testbed || exit 6
git config --global --add safe.directory /testbed 2>/dev/null || true

{setup_block}

if [ -s /tests/test.patch ]; then
  git apply --whitespace=nowarn /tests/test.patch || exit 3
fi

{test_command}
status=$?
if [ "$status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
exit "$status"
"""


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def materialize_swe_rebench(
    output_dir: Path,
    current: CurrentIndex,
    limit: int,
    summary: dict[str, Any],
) -> list[dict[str, str]]:
    rows = []
    if limit <= 0:
        return rows
    dataset = load_dataset(SWE_REBENCH_V1, split="filtered", streaming=True, token=hf_token())
    for row in dataset:
        instance_id = row["instance_id"]
        repo = row["repo"]
        patch = row.get("patch") or ""
        issue = issue_number(instance_id)
        if instance_id in current.task_ids or (repo and issue and f"{repo}::{issue}" in current.repo_issue_keys):
            summary["rejected"]["swe_rebench_overlap"] += 1
            continue
        if not row.get("image_name") or not row.get("base_commit"):
            summary["rejected"]["swe_rebench_missing_image_or_commit"] += 1
            continue
        if not patch.strip() or not row.get("test_patch"):
            summary["rejected"]["swe_rebench_missing_patch_or_tests"] += 1
            continue
        if not patch_touches_non_test(patch):
            summary["rejected"]["swe_rebench_test_only_patch"] += 1
            continue
        if len(patch) > 2200 or patch_file_count(patch) > 2:
            summary["rejected"]["swe_rebench_complex_patch"] += 1
            continue
        if len(row.get("problem_statement") or "") > 2200:
            summary["rejected"]["swe_rebench_long_prompt"] += 1
            continue
        task_dir = output_dir / task_slug(instance_id)
        if task_dir.exists():
            shutil.rmtree(task_dir)
        write_text(task_dir / "task.toml", build_swe_rebench_task_toml(row))
        write_text(task_dir / "instruction.md", clean_issue_prompt(row.get("problem_statement") or ""))
        write_text(task_dir / "tests" / "test.patch", row.get("test_patch") or "")
        write_text(task_dir / "tests" / "test.sh", build_swe_rebench_test_sh(row), executable=True)
        write_text(task_dir / "solution" / "solution.patch", patch)
        write_text(task_dir / "solution" / "solve.sh", "#!/bin/bash\nset -euo pipefail\ngit apply /solution/solution.patch\n", executable=True)
        rows.append(
            {
                "instance_id": instance_id,
                "source": "swe_rebench_v1",
                "repo": repo,
                "language": "python",
                "difficulty": "easy",
                "instruction_style": "original",
                "task_dir": str(task_dir),
                "image": row["image_name"],
                "patch_bytes": str(len(patch)),
                "prompt_chars": str(len(row.get("problem_statement") or "")),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def public_repo_and_commit_from_swesmith(instance_id: str) -> tuple[str, str]:
    match = re.match(r"(?P<owner>[^_]+(?:-[^_]+)*)__(?P<repo>[^.]+)\.(?P<commit>[0-9a-f]{7,40})\.", instance_id)
    if not match:
        return "", ""
    return f"{match.group('owner')}/{match.group('repo')}", match.group("commit")


def build_swesmith_go_task_toml(row: dict[str, Any], public_repo: str, base_ref: str, image: str) -> str:
    return f"""schema_version = "1.1"
artifacts = []

[task]
name = {toml_string("swesmith-go/" + row["instance_id"])}
description = ""
authors = []
keywords = []

[metadata]
task_id = {toml_string(row["instance_id"])}
display_title = {toml_string((row.get("problem_statement") or row["instance_id"]).splitlines()[0][:180])}
display_description = ""
original_title = {toml_string((row.get("problem_statement") or row["instance_id"]).splitlines()[0][:180])}
category = "software-engineering"
language = "go"
repository_url = {toml_string("https://github.com/" + public_repo)}
base_commit_hash = {toml_string(base_ref)}
source_dataset = "ricdomolm/SWE-smith-trajectories-harbor-found-235B"
source_image_name = {toml_string(str(row.get("image_name") or ""))}
source_repo = {toml_string(str(row.get("repo") or ""))}

[verifier]
timeout_sec = 1800.0

[agent]
timeout_sec = 5400.0

[environment]
build_timeout_sec = 1800.0
docker_image = {toml_string(image)}
os = "linux"
cpus = 2
memory_mb = 8192
storage_mb = 20480
gpus = 0
allow_internet = false
mcp_servers = []
workdir = "/testbed"

[environment.env]

[solution.env]
"""


def build_swesmith_go_test_sh(row: dict[str, Any]) -> str:
    tests = sorted({str(test) for test in (row.get("FAIL_TO_PASS") or []) + (row.get("PASS_TO_PASS") or []) if test})
    regex = "^(" + "|".join(re.escape(test) for test in tests) + ")$" if tests else ""
    test_command = "go test ./... -count=1 -v"
    if regex:
        test_command += " -run " + shell_quote(regex)
    return f"""#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

cd /testbed || exit 6
git config --global --add safe.directory /testbed 2>/dev/null || true

export GO111MODULE=on
export GOPATH="${{GOPATH:-/go}}"
export PATH="/usr/local/go/bin:/go/bin:$PATH"
export GOPROXY="${{GOPROXY:-https://proxy.golang.org,direct}}"

{test_command}
status=$?
if [ "$status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
exit "$status"
"""


def build_swesmith_go_dockerfile(public_repo: str, upstream_commit: str) -> str:
    return f"""FROM golang:1.24-bookworm

ENV GOPATH=/go
ENV PATH=/usr/local/go/bin:/go/bin:$PATH

RUN rm -rf /testbed /logs && mkdir -p /testbed /output && chmod 777 /output
RUN ln -sf /usr/local/go/bin/go /usr/local/bin/go && \\
    ln -sf /usr/local/go/bin/gofmt /usr/local/bin/gofmt
WORKDIR /testbed
COPY setup_files/bug.patch /tmp/bug.patch
RUN git clone {shell_quote("https://github.com/" + public_repo + ".git")} /testbed && \\
    cd /testbed && \\
    git checkout {shell_quote(upstream_commit)} && \\
    git apply --whitespace=nowarn /tmp/bug.patch && \\
    git config user.email "synthetic@example.invalid" && \\
    git config user.name "Synthetic Bug" && \\
    git add -A && \\
    GIT_AUTHOR_DATE="2024-01-01T00:00:00+00:00" \\
    GIT_COMMITTER_DATE="2024-01-01T00:00:00+00:00" \\
    git commit -m "Apply synthetic SWE-smith bug" && \\
    git checkout -B buggy && \\
    go test ./... >/tmp/swesmith-go-build-smoke.log 2>&1 || true
WORKDIR /testbed
"""


def build_swesmith_java_task_toml(row: dict[str, Any], public_repo: str, base_ref: str, image: str) -> str:
    return f"""schema_version = "1.1"
artifacts = []

[task]
name = {toml_string("swesmith-java/" + row["instance_id"])}
description = ""
authors = []
keywords = []

[metadata]
task_id = {toml_string(row["instance_id"])}
display_title = {toml_string((row.get("problem_statement") or row["instance_id"]).splitlines()[0][:180])}
display_description = ""
original_title = {toml_string((row.get("problem_statement") or row["instance_id"]).splitlines()[0][:180])}
category = "software-engineering"
language = "java"
repository_url = {toml_string("https://github.com/" + public_repo)}
base_commit_hash = {toml_string(base_ref)}
source_dataset = "ricdomolm/SWE-smith-trajectories-harbor-found-235B"
source_image_name = {toml_string(str(row.get("image_name") or ""))}
source_repo = {toml_string(str(row.get("repo") or ""))}

[verifier]
timeout_sec = 1800.0

[agent]
timeout_sec = 5400.0

[environment]
build_timeout_sec = 1800.0
docker_image = {toml_string(image)}
os = "linux"
cpus = 2
memory_mb = 16384
storage_mb = 20480
gpus = 0
allow_internet = false
mcp_servers = []
workdir = "/testbed"

[environment.env]

[solution.env]
"""


def java_surefire_selectors(row: dict[str, Any], pass_sample_size: int = 5) -> list[str]:
    tests = []
    for test in row.get("FAIL_TO_PASS") or []:
        if isinstance(test, str) and "." in test:
            class_name, method = test.rsplit(".", 1)
            tests.append(f"{class_name}#{method}")
    for test in (row.get("PASS_TO_PASS") or [])[:pass_sample_size]:
        if isinstance(test, str) and "." in test:
            class_name, method = test.rsplit(".", 1)
            tests.append(f"{class_name}#{method}")
    return tests


def build_swesmith_java_test_sh(row: dict[str, Any]) -> str:
    selectors = java_surefire_selectors(row)
    selector_arg = ",".join(selectors)
    return f"""#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

cd /testbed || exit 6
git config --global --add safe.directory /testbed 2>/dev/null || true

mvn -q -pl gson -Dproguard.skip=true -Dtest={shell_quote(selector_arg)} test
status=$?
if [ "$status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
exit "$status"
"""


def build_swesmith_java_dockerfile(public_repo: str, upstream_commit: str) -> str:
    return f"""FROM swebench/sweb.eval.x86_64.google_1776_gson-2061:latest

RUN rm -rf /testbed /logs && mkdir -p /testbed /output && chmod 777 /output
WORKDIR /testbed
COPY setup_files/bug.patch /tmp/bug.patch
RUN git clone {shell_quote("https://github.com/" + public_repo + ".git")} /testbed && \\
    cd /testbed && \\
    git checkout {shell_quote(upstream_commit)} && \\
    git apply --whitespace=nowarn /tmp/bug.patch && \\
    git config user.email "synthetic@example.invalid" && \\
    git config user.name "Synthetic Bug" && \\
    git add -A && \\
    GIT_AUTHOR_DATE="2024-01-01T00:00:00+00:00" \\
    GIT_COMMITTER_DATE="2024-01-01T00:00:00+00:00" \\
    git commit -m "Apply synthetic SWE-smith bug" && \\
    git checkout -B buggy && \\
    mvn -q -pl gson -Dproguard.skip=true -DskipTests compile >/tmp/swesmith-java-build-smoke.log 2>&1 || true
WORKDIR /testbed
"""


def build_swesmith_cpp_task_toml(row: dict[str, Any], image: str) -> str:
    problem_statement = row.get("problem_statement") or row["instance_id"]
    return f"""schema_version = "1.1"
artifacts = []

[task]
name = {toml_string("swesmith-cpp/" + row["instance_id"])}
description = ""
authors = []
keywords = []

[metadata]
task_id = {toml_string(row["instance_id"])}
display_title = {toml_string(problem_statement.splitlines()[0][:180])}
display_description = ""
original_title = {toml_string(problem_statement.splitlines()[0][:180])}
category = "software-engineering"
language = "cpp"
repository_url = {toml_string("https://github.com/" + str(row.get("repo") or ""))}
base_commit_hash = "buggy"
source_dataset = "SWE-bench/SWE-smith-cpp"
source_image_name = {toml_string(str(row.get("image_name") or ""))}
source_repo = {toml_string(str(row.get("repo") or ""))}

[verifier]
timeout_sec = 1800.0

[agent]
timeout_sec = 5400.0

[environment]
build_timeout_sec = 1800.0
docker_image = {toml_string(image)}
os = "linux"
cpus = 4
memory_mb = 16384
storage_mb = 40960
gpus = 0
allow_internet = false
mcp_servers = []
workdir = "/testbed"

[environment.env]

[solution.env]
"""


def ctest_regex(tests: list[str]) -> str:
    unique = []
    seen = set()
    for test in tests:
        if not isinstance(test, str):
            continue
        text = test.strip()
        if text and text not in seen:
            unique.append(text)
            seen.add(text)
    return "^(" + "|".join(re.escape(test) for test in unique) + ")$" if unique else ""


def build_swesmith_cpp_test_sh(row: dict[str, Any], pass_sample_size: int = 3) -> str:
    fail_tests = [str(test) for test in (row.get("FAIL_TO_PASS") or []) if str(test).strip()]
    pass_tests = [str(test) for test in (row.get("PASS_TO_PASS") or [])[:pass_sample_size] if str(test).strip()]
    regex = ctest_regex(fail_tests + pass_tests)
    selector = f"-R {shell_quote(regex)}" if regex else ""
    return f"""#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

cd /testbed || exit 6
git config --global --add safe.directory /testbed 2>/dev/null || true

jobs="${{SWE_SMITH_CPP_BUILD_JOBS:-4}}"
if [ -d build ]; then
  cmake --build build -j "$jobs" || exit 2
else
  cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo || exit 2
  cmake --build build -j "$jobs" || exit 2
fi

cd build || exit 6
ctest --output-on-failure {selector}
status=$?
if [ "$status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
exit "$status"
"""


def build_swesmith_cpp_dockerfile(source_image: str) -> str:
    return f"""FROM {source_image}

WORKDIR /testbed
COPY setup_files/bug.patch /tmp/bug.patch
RUN git config user.email "synthetic@example.invalid" && \\
    git config user.name "Synthetic Bug" && \\
    git checkout -B clean-base && \\
    git apply --whitespace=nowarn /tmp/bug.patch && \\
    git add -A && \\
    GIT_AUTHOR_DATE="2024-01-01T00:00:00+00:00" \\
    GIT_COMMITTER_DATE="2024-01-01T00:00:00+00:00" \\
    git commit -m "Apply synthetic SWE-smith C++ bug" && \\
    git checkout -B buggy
WORKDIR /testbed
"""


def materialize_swesmith_go(
    output_dir: Path,
    current: CurrentIndex,
    limit: int,
    skip: int,
    summary: dict[str, Any],
) -> list[dict[str, str]]:
    rows = []
    if limit <= 0:
        return rows
    eligible_seen = 0
    dataset = load_dataset(SWE_SMITH_HARBOR, split="train", streaming=True, token=hf_token())
    for row in dataset:
        instance_id = row["instance_id"]
        public_repo, upstream_commit = public_repo_and_commit_from_swesmith(instance_id)
        patch = row.get("patch") or ""
        problem_statement = row.get("problem_statement") or ""
        if instance_id in current.task_ids:
            summary["rejected"]["swesmith_go_exact_overlap"] += 1
            continue
        if public_repo and public_repo in {key.rsplit("::", 1)[0] for key in current.repo_issue_keys}:
            summary["rejected"]["swesmith_go_repo_overlap"] += 1
            continue
        if not public_repo or not upstream_commit:
            summary["rejected"]["swesmith_go_missing_repo_or_commit"] += 1
            continue
        if patch_languages(patch) != {"go"}:
            summary["rejected"]["swesmith_go_not_go_only"] += 1
            continue
        if not patch_touches_non_test(patch):
            summary["rejected"]["swesmith_go_test_only_patch"] += 1
            continue
        if patch_file_count(patch) != 1 or len(patch) > 900 or patch_net_line_delta(patch) != 0:
            summary["rejected"]["swesmith_go_complex_patch"] += 1
            continue
        if not (300 <= len(problem_statement) <= 1800):
            summary["rejected"]["swesmith_go_bad_prompt_length"] += 1
            continue
        if len(row.get("FAIL_TO_PASS") or []) > 4:
            summary["rejected"]["swesmith_go_many_fail_tests"] += 1
            continue

        eligible_seen += 1
        if eligible_seen <= skip:
            summary["rejected"]["swesmith_go_skipped_selected_offset"] += 1
            continue

        task_dir = output_dir / task_slug(instance_id)
        if task_dir.exists():
            shutil.rmtree(task_dir)
        image = f"other-src-swesmith-go-{task_slug(instance_id)}:latest"
        write_text(task_dir / "task.toml", build_swesmith_go_task_toml(row, public_repo, "buggy", image))
        write_text(task_dir / "instruction.md", clean_issue_prompt(problem_statement))
        write_text(task_dir / "tests" / "test.sh", build_swesmith_go_test_sh(row), executable=True)
        write_text(task_dir / "setup_files" / "bug.patch", patch)
        write_text(task_dir / "environment" / "Dockerfile.prepared", build_swesmith_go_dockerfile(public_repo, upstream_commit))
        write_text(task_dir / "solution" / "bug.patch", patch)
        write_text(task_dir / "solution" / "solution.patch", reverse_zero_delta_patch(patch))
        write_text(task_dir / "solution" / "solve.sh", "#!/bin/bash\nset -euo pipefail\ngit apply /solution/solution.patch\n", executable=True)
        rows.append(
            {
                "instance_id": instance_id,
                "source": "swesmith_go",
                "repo": public_repo,
                "language": "go",
                "difficulty": "easy",
                "instruction_style": "original",
                "task_dir": str(task_dir),
                "image": image,
                "patch_bytes": str(len(patch)),
                "prompt_chars": str(len(problem_statement)),
                "source_instance_id": instance_id,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def materialize_swesmith_java(
    output_dir: Path,
    current: CurrentIndex,
    limit: int,
    skip: int,
    summary: dict[str, Any],
) -> list[dict[str, str]]:
    rows = []
    if limit <= 0:
        return rows
    eligible_seen = 0
    dataset = load_dataset(SWE_SMITH_HARBOR, split="train", streaming=True, token=hf_token())
    for row in dataset:
        instance_id = row["instance_id"]
        public_repo, upstream_commit = public_repo_and_commit_from_swesmith(instance_id)
        patch = row.get("patch") or ""
        problem_statement = row.get("problem_statement") or ""
        if instance_id in current.task_ids:
            summary["rejected"]["swesmith_java_exact_overlap"] += 1
            continue
        if public_repo != "google/gson":
            summary["rejected"]["swesmith_java_not_gson"] += 1
            continue
        if patch_languages(patch) != {"java"}:
            summary["rejected"]["swesmith_java_not_java_only"] += 1
            continue
        if not patch_touches_non_test(patch):
            summary["rejected"]["swesmith_java_test_only_patch"] += 1
            continue
        if patch_file_count(patch) != 1 or len(patch) > 1200 or abs(patch_net_line_delta(patch)) > 2:
            summary["rejected"]["swesmith_java_complex_patch"] += 1
            continue
        if not (300 <= len(problem_statement) <= 2200):
            summary["rejected"]["swesmith_java_bad_prompt_length"] += 1
            continue
        if not java_surefire_selectors(row) or len(row.get("FAIL_TO_PASS") or []) > 5:
            summary["rejected"]["swesmith_java_bad_tests"] += 1
            continue

        eligible_seen += 1
        if eligible_seen <= skip:
            summary["rejected"]["swesmith_java_skipped_selected_offset"] += 1
            continue

        task_dir = output_dir / task_slug(instance_id)
        if task_dir.exists():
            shutil.rmtree(task_dir)
        image = f"other-src-swesmith-java-{task_slug(instance_id)}:latest"
        write_text(task_dir / "task.toml", build_swesmith_java_task_toml(row, public_repo, "buggy", image))
        write_text(task_dir / "instruction.md", clean_issue_prompt(problem_statement))
        write_text(task_dir / "tests" / "test.sh", build_swesmith_java_test_sh(row), executable=True)
        write_text(task_dir / "setup_files" / "bug.patch", patch)
        write_text(task_dir / "environment" / "Dockerfile.prepared", build_swesmith_java_dockerfile(public_repo, upstream_commit))
        write_text(task_dir / "solution" / "bug.patch", patch)
        write_text(task_dir / "solution" / "solution.patch", reverse_zero_delta_patch(patch))
        write_text(task_dir / "solution" / "solve.sh", "#!/bin/bash\nset -euo pipefail\ngit apply /solution/solution.patch\n", executable=True)
        rows.append(
            {
                "instance_id": instance_id,
                "source": "swesmith_java",
                "repo": public_repo,
                "language": "java",
                "difficulty": "easy",
                "instruction_style": "original",
                "task_dir": str(task_dir),
                "image": image,
                "patch_bytes": str(len(patch)),
                "prompt_chars": str(len(problem_statement)),
                "source_instance_id": instance_id,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def materialize_swesmith_cpp(
    output_dir: Path,
    current: CurrentIndex,
    limit: int,
    skip: int,
    summary: dict[str, Any],
) -> list[dict[str, str]]:
    rows = []
    if limit <= 0:
        return rows
    eligible_seen = 0
    dataset = load_dataset(SWE_SMITH_CPP, split="train", streaming=True, token=hf_token())
    for row in dataset:
        instance_id = row["instance_id"]
        patch = row.get("patch") or ""
        problem_statement = row.get("problem_statement") or ""
        image_name = str(row.get("image_name") or "")
        repo = str(row.get("repo") or "")
        if instance_id in current.task_ids:
            summary["rejected"]["swesmith_cpp_exact_overlap"] += 1
            continue
        if not image_name.startswith("swebench/swesmith."):
            summary["rejected"]["swesmith_cpp_missing_image"] += 1
            continue
        if patch_languages(patch) != {"cpp"}:
            summary["rejected"]["swesmith_cpp_not_cpp_only"] += 1
            continue
        if not patch_touches_non_test(patch):
            summary["rejected"]["swesmith_cpp_test_only_patch"] += 1
            continue
        if patch_file_count(patch) != 1 or len(patch) > 900 or patch_net_line_delta(patch) != 0:
            summary["rejected"]["swesmith_cpp_complex_patch"] += 1
            continue
        if not (250 <= len(problem_statement) <= 2200):
            summary["rejected"]["swesmith_cpp_bad_prompt_length"] += 1
            continue
        if not row.get("FAIL_TO_PASS") or len(row.get("FAIL_TO_PASS") or []) > 4:
            summary["rejected"]["swesmith_cpp_bad_tests"] += 1
            continue

        eligible_seen += 1
        if eligible_seen <= skip:
            summary["rejected"]["swesmith_cpp_skipped_selected_offset"] += 1
            continue

        task_dir = output_dir / task_slug(instance_id)
        if task_dir.exists():
            shutil.rmtree(task_dir)
        image = f"other-src-swesmith-cpp-{task_slug(instance_id)}:latest"
        write_text(task_dir / "task.toml", build_swesmith_cpp_task_toml(row, image))
        write_text(task_dir / "instruction.md", clean_issue_prompt(problem_statement))
        write_text(task_dir / "tests" / "test.sh", build_swesmith_cpp_test_sh(row), executable=True)
        write_text(task_dir / "setup_files" / "bug.patch", patch)
        write_text(task_dir / "environment" / "Dockerfile.prepared", build_swesmith_cpp_dockerfile(image_name))
        write_text(task_dir / "solution" / "bug.patch", patch)
        write_text(task_dir / "solution" / "solution.patch", reverse_zero_delta_patch(patch))
        write_text(task_dir / "solution" / "solve.sh", "#!/bin/bash\nset -euo pipefail\ngit apply /solution/solution.patch\n", executable=True)
        rows.append(
            {
                "instance_id": instance_id,
                "source": "swesmith_cpp",
                "repo": repo,
                "language": "cpp",
                "difficulty": "easy",
                "instruction_style": "original",
                "task_dir": str(task_dir),
                "image": image,
                "patch_bytes": str(len(patch)),
                "prompt_chars": str(len(problem_statement)),
                "source_instance_id": instance_id,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def extract_embedded_patches(text: str) -> list[str]:
    chunks = []
    for match in re.finditer(r"(?m)^diff --git ", text):
        tail = text[match.start():]
        marker = re.search(r"(?m)^(__SOLUTION__|PATCH_EOF|EOF|SOLUTION_PATCH_EOF)\s*$", tail)
        chunks.append((tail[: marker.start()] if marker else tail).strip() + "\n")
    return [chunk for chunk in chunks if chunk.strip()]


def extract_embedded_patch(text: str) -> str:
    chunks = extract_embedded_patches(text)
    for chunk in chunks:
        if patch_touches_non_test(chunk):
            return chunk
    return chunks[0] if chunks else ""


def strip_swegym_setup_instruction(text: str) -> str:
    match = re.search(r"(?s)## Problem Statement\s*(.*)", text)
    if match:
        text = match.group(1)
    text = re.sub(r"(?s)### [^\n]*version checks\s*.*?(?=\n### |\Z)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?s)### Installed Versions\s*.*?(?=\n### |\Z)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?s)<details>\s*.*?</details>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text + "\n"


def swegym_install_commands(instruction: str, version: str = "") -> str:
    match = re.search(r"(?s)# Install dependencies\s*\n(.*?)\n```", instruction)
    if not match:
        return ""
    return pin_swegym_dependency_text(match.group(1).strip(), version)


def pin_swegym_dependency_text(text: str, version: str = "", no_index: bool = False) -> str:
    commands = text
    commands = commands.replace('"numpy"', '"numpy<2"')
    commands = commands.replace("'numpy'", "'numpy<2'")
    if no_index:
        commands = commands.replace(
            "pip install --prefer-binary --upgrade pip setuptools wheel",
            "PIP_NO_INDEX=1 pip install --prefer-binary pip setuptools wheel",
        )
        commands = commands.replace("python -m pip install ", "PIP_NO_INDEX=1 python -m pip install ")
        commands = commands.replace("pip install \"pytest>=8.0.0\"", "PIP_NO_INDEX=1 pip install \"pytest>=8.0.0\"")
    # Older Pandas SWEGym tasks predate current Cython/NumPy releases, which
    # break editable builds with stricter generated Cython signatures.
    if version.startswith("2.0"):
        return re.sub(r"cython>=3(?:,<3\\.1)?", "cython<3", commands)
    return re.sub(r"cython>=3(?:,<3\\.1)?", "cython>=3,<3.1", commands)


def build_swegym_task_toml(
    task_id: str,
    source_path: str,
    repo: str,
    base_commit: str,
    language: str,
    image: str,
    prompt: str,
) -> str:
    title = prompt.splitlines()[0][:180] if prompt.splitlines() else task_id
    return f"""schema_version = "1.1"
artifacts = []

[task]
name = {toml_string("swegym-c-cpp/" + source_path)}
description = ""
authors = []
keywords = []

[metadata]
task_id = {toml_string(source_path)}
display_title = {toml_string(title)}
display_description = ""
original_title = {toml_string(title)}
category = "software-engineering"
language = {toml_string(language)}
repository_url = {toml_string("https://github.com/" + repo)}
base_commit_hash = {toml_string(base_commit)}
source_dataset = "open-thoughts/TaskTrove"
source_path = {toml_string(SWEGYM_PARQUET)}
source_instance_id = {toml_string(task_id)}

[verifier]
timeout_sec = 1800.0

[agent]
timeout_sec = 5400.0

[environment]
build_timeout_sec = 3600.0
docker_image = {toml_string(image)}
os = "linux"
cpus = 4
memory_mb = 16384
storage_mb = 40960
gpus = 0
allow_internet = false
mcp_servers = []
workdir = "/testbed/repo"

[environment.env]

[solution.env]
"""


def build_swegym_setup_script(repo: str, base_commit: str, install_commands: str) -> str:
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail

source /opt/miniconda3/bin/activate
conda activate testbed

cd /testbed
git clone {shell_quote("https://github.com/" + repo + ".git")} repo
cd repo
git checkout {shell_quote(base_commit)}

{install_commands}

git config user.email "synthetic@example.invalid"
git config user.name "Synthetic Task"
git checkout -B buggy
"""


def apply_swegym_test_patch_in_test_script(text: str, has_test_patch: bool) -> str:
    if not has_test_patch:
        return text
    needle = '    cd "$REPO_DIR"\n    ensure_dependencies'
    replacement = (
        '    cd "$REPO_DIR"\n'
        '    git apply --whitespace=nowarn /tests/test.patch\n'
        '    ensure_dependencies'
    )
    if needle in text:
        return text.replace(needle, replacement, 1)
    return text + '\n# Apply SWEGym verifier patch before running tests.\ngit apply --whitespace=nowarn /tests/test.patch\n'


def materialize_swegym_c_cpp(
    output_dir: Path,
    current: CurrentIndex,
    limit: int,
    skip: int,
    summary: dict[str, Any],
) -> list[dict[str, str]]:
    rows = []
    if limit <= 0:
        return rows
    parquet_path = hf_hub_download(
        repo_id=OPENTHINK_TASKTROVE,
        repo_type="dataset",
        filename=SWEGYM_PARQUET,
        token=hf_token(),
    )
    eligible_seen = 0
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=1):
        record = batch.to_pylist()[0]
        source_path = record["path"]
        data = bytes(record["task_binary"])
        try:
            data = gzip.decompress(data)
        except OSError:
            pass
        texts: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            for member in tf.getmembers():
                clean = member.name.strip("./")
                if member.isfile() and (
                    clean in {
                        "instruction.md",
                        "metadata.json",
                        "task.toml",
                        "environment/Dockerfile",
                        "tests/test.sh",
                        "solution/solve.sh",
                    }
                    or clean.startswith("tests/")
                ):
                    extracted = tf.extractfile(member)
                    if extracted is not None:
                        texts[clean] = extracted.read().decode("utf-8", errors="replace")
        metadata = json.loads(texts.get("metadata.json", "{}") or "{}")
        task_id = str(metadata.get("instance_id") or source_path)
        repo = str(metadata.get("repo") or "")
        base_commit = str(metadata.get("base_commit") or "")
        embedded_patches = extract_embedded_patches(texts.get("solution/solve.sh", ""))
        solution_patch = next((patch for patch in embedded_patches if patch_touches_non_test(patch)), "")
        test_patch = next(
            (patch for patch in embedded_patches if patch_paths(patch) and not patch_touches_non_test(patch)),
            "",
        )
        langs = patch_languages(solution_patch)
        matching_langs = langs.intersection({"c", "cpp"})
        if not matching_langs:
            summary["rejected"]["swegym_c_cpp_not_c_cpp"] += 1
            continue
        if task_id in current.task_ids:
            summary["rejected"]["swegym_c_cpp_exact_overlap"] += 1
            continue
        issue = issue_number(task_id)
        if repo and issue and f"{repo}::{issue}" in current.repo_issue_keys:
            summary["rejected"]["swegym_c_cpp_repo_issue_overlap"] += 1
            continue
        if not repo or not base_commit:
            summary["rejected"]["swegym_c_cpp_missing_repo_or_commit"] += 1
            continue
        if not patch_touches_non_test(solution_patch):
            summary["rejected"]["swegym_c_cpp_no_gold_source_patch"] += 1
            continue
        if patch_file_count(solution_patch) > 3 or len(solution_patch) > 2500:
            summary["rejected"]["swegym_c_cpp_complex_patch"] += 1
            continue
        prompt = strip_swegym_setup_instruction(texts.get("instruction.md", ""))
        if len(prompt) > 16000:
            summary["rejected"]["swegym_c_cpp_long_prompt"] += 1
            continue
        install_commands = swegym_install_commands(texts.get("instruction.md", ""), str(metadata.get("version") or ""))
        if not install_commands:
            summary["rejected"]["swegym_c_cpp_missing_install"] += 1
            continue

        eligible_seen += 1
        if eligible_seen <= skip:
            summary["rejected"]["swegym_c_cpp_skipped_selected_offset"] += 1
            continue

        task_dir = output_dir / task_slug(source_path)
        if task_dir.exists():
            shutil.rmtree(task_dir)
        image = f"other-src-swegym-c-cpp-{task_slug(source_path)}:latest"
        language = sorted(matching_langs)[0]
        dockerfile = (
            texts["environment/Dockerfile"].rstrip()
            + "\n\nCOPY setup_files/setup_repo.sh /setup_repo.sh\n"
            + "RUN bash /setup_repo.sh\n"
            + "WORKDIR /testbed/repo\n"
        )
        write_text(task_dir / "task.toml", build_swegym_task_toml(task_id, source_path, repo, base_commit, language, image, prompt))
        write_text(task_dir / "instruction.md", prompt)
        for rel, content in texts.items():
            if rel.startswith("tests/"):
                if rel.endswith(".sh"):
                    content = pin_swegym_dependency_text(content, str(metadata.get("version") or ""), no_index=True)
                    content = apply_swegym_test_patch_in_test_script(content, bool(test_patch.strip()))
                write_text(task_dir / rel, content, executable=rel.endswith(".sh"))
        if test_patch.strip():
            write_text(task_dir / "tests" / "test.patch", test_patch)
        write_text(task_dir / "environment" / "Dockerfile.prepared", dockerfile)
        write_text(task_dir / "setup_files" / "setup_repo.sh", build_swegym_setup_script(repo, base_commit, install_commands), executable=True)
        write_text(task_dir / "solution" / "solution.patch", solution_patch)
        write_text(task_dir / "solution" / "solve.sh", "#!/bin/bash\nset -euo pipefail\ngit apply /solution/solution.patch\n", executable=True)
        rows.append(
            {
                "instance_id": source_path,
                "source": "swegym_c_cpp",
                "repo": repo,
                "language": language,
                "difficulty": "medium",
                "instruction_style": "original",
                "task_dir": str(task_dir),
                "image": image,
                "patch_bytes": str(len(solution_patch)),
                "prompt_chars": str(len(prompt)),
                "source_instance_id": task_id,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def strip_openswe_setup_instruction(text: str) -> str:
    text = re.sub(r"(?s)^## Environment Setup.*?---\s*", "", text).strip()
    text = re.sub(r"(?m)^The code is located in `/testbed`\.\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text + "\n"


def checkout_commit_from_setup(setup_sh: str) -> str:
    match = re.search(r"\bgit checkout\s+([0-9a-f]{7,40})", setup_sh)
    return match.group(1) if match else ""


def repo_from_setup(setup_sh: str) -> str:
    match = re.search(r"https://github\.com/([^/\s]+/[^/\s.]+)(?:\.git)?", setup_sh)
    return match.group(1) if match else ""


def materialize_openswe(
    output_dir: Path,
    current: CurrentIndex,
    limit: int,
    summary: dict[str, Any],
    allowed_languages: set[str],
) -> list[dict[str, str]]:
    rows = []
    if limit <= 0:
        return rows
    parquet_path = hf_hub_download(
        repo_id=OPENTHINK_TASKTROVE,
        repo_type="dataset",
        filename=OPENSWE_PARQUET,
        token=hf_token(),
    )
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=1):
        record = batch.to_pylist()[0]
        task_path = record["path"]
        data = bytes(record["task_binary"])
        try:
            data = gzip.decompress(data)
        except OSError:
            pass
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            texts: dict[str, str] = {}
            members = tf.getmembers()
            for member in members:
                clean = member.name.strip("./")
                if member.isfile() and (
                    clean in {
                        "instruction.md",
                        "task.toml",
                        "environment/Dockerfile",
                        "setup_files/setup.sh",
                        "tests/test.sh",
                        "tests/test_patch.diff",
                        "solution/solve.sh",
                    }
                    or clean.startswith("tests/")
                    or clean.startswith("setup_files/")
                ):
                    extracted = tf.extractfile(member)
                    if extracted is not None:
                        texts[clean] = extracted.read().decode("utf-8", errors="replace")
        setup_sh = texts.get("setup_files/setup.sh", "")
        patch = extract_embedded_patch(texts.get("solution/solve.sh", ""))
        langs = patch_languages(patch)
        matching_langs = langs.intersection(allowed_languages)
        if not matching_langs:
            summary["rejected"]["openswe_not_allowed_language"] += 1
            continue
        if not patch_touches_non_test(patch):
            summary["rejected"]["openswe_no_gold_source_patch"] += 1
            continue
        if len(patch) > 3000 or patch_file_count(patch) > 2:
            summary["rejected"]["openswe_complex_patch"] += 1
            continue
        instruction = strip_openswe_setup_instruction(texts.get("instruction.md", ""))
        if len(instruction) > 1800:
            summary["rejected"]["openswe_long_prompt"] += 1
            continue
        base_commit = checkout_commit_from_setup(setup_sh)
        repo = repo_from_setup(setup_sh)
        source_instance = ""
        match = re.search(r"\(instance:\s*`([^`]+)`\)", texts.get("instruction.md", ""))
        if match:
            source_instance = match.group(1)
        if source_instance in current.task_ids:
            summary["rejected"]["openswe_exact_overlap"] += 1
            continue
        issue = issue_number(source_instance)
        if repo and issue and f"{repo}::{issue}" in current.repo_issue_keys:
            summary["rejected"]["openswe_repo_issue_overlap"] += 1
            continue
        if not base_commit or not repo:
            summary["rejected"]["openswe_missing_repo_or_commit"] += 1
            continue
        task_dir = output_dir / task_slug(task_path)
        if task_dir.exists():
            shutil.rmtree(task_dir)
        for rel, content in texts.items():
            if rel == "instruction.md":
                continue
            write_text(task_dir / rel, content, executable=rel.endswith(".sh"))
        image = f"other-src-openswe-{task_slug(task_path)}:latest"
        dockerfile = texts["environment/Dockerfile"].rstrip() + "\n\nCOPY setup_files /setup_files\nRUN bash /setup_files/setup.sh\nWORKDIR /testbed\n"
        write_text(task_dir / "environment" / "Dockerfile.prepared", dockerfile)
        write_text(task_dir / "instruction.md", instruction)
        task_language = sorted(matching_langs)[0]
        write_text(
            task_dir / "task.toml",
            f"""schema_version = "1.1"
artifacts = []

[task]
name = {toml_string("openswe/" + task_path)}
description = ""
authors = []
keywords = []

[metadata]
task_id = {toml_string(task_path)}
display_title = {toml_string(instruction.splitlines()[0][:180] if instruction.splitlines() else task_path)}
display_description = ""
original_title = {toml_string(instruction.splitlines()[0][:180] if instruction.splitlines() else task_path)}
category = "software-engineering"
language = {toml_string(task_language)}
repository_url = {toml_string("https://github.com/" + repo)}
base_commit_hash = {toml_string(base_commit)}
source_dataset = "open-thoughts/TaskTrove"
source_path = {toml_string(OPENSWE_PARQUET)}
source_instance_id = {toml_string(source_instance)}

[verifier]
timeout_sec = 1800.0

[agent]
timeout_sec = 5400.0

[environment]
build_timeout_sec = 3600.0
docker_image = {toml_string(image)}
os = "linux"
cpus = 2
memory_mb = 8192
storage_mb = 20480
gpus = 0
allow_internet = false
mcp_servers = []
workdir = "/testbed"

[environment.env]

[solution.env]
""",
        )
        write_text(task_dir / "solution" / "solution.patch", patch)
        rows.append(
            {
                "instance_id": task_path,
                "source": "openswe_tasktrove",
                "repo": repo,
                "language": task_language,
                "difficulty": "easy",
                "instruction_style": "original",
                "task_dir": str(task_dir),
                "image": image,
                "patch_bytes": str(len(patch)),
                "prompt_chars": str(len(instruction)),
                "source_instance_id": source_instance,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def write_assignments_and_manifest(run_root: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    manifest_dir = run_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    assignments = manifest_dir / "assignments.csv"
    manifest_tsv = manifest_dir / "manifest.tsv"
    fields = [
        "instance_id",
        "source",
        "repo",
        "language",
        "difficulty",
        "instruction_style",
        "assigned_model",
        "outside_original_high_quality_set",
        "patch_bytes",
        "prompt_chars",
        "source_instance_id",
    ]
    with assignments.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fields},
                    "assigned_model": "xiaomi/mimo-v2.5",
                    "outside_original_high_quality_set": "true",
                }
            )

    extra_body = json.dumps({"reasoning": {"effort": "high", "exclude": False}}, separators=(",", ":"))
    with manifest_tsv.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            workspace = (
                run_root
                / "pyxis-traces"
                / row["source"]
                / "xiaomi-mimo-v2.5"
                / "r00"
                / task_slug(row["instance_id"])
            )
            values = [
                str(index),
                "r00",
                row["instance_id"],
                str(Path(row["task_dir"]).resolve()),
                str(workspace.resolve()),
                row["image"],
                "xiaomi/mimo-v2.5",
                "openrouter/xiaomi/mimo-v2.5",
                "OPENROUTER_API_KEY",
                "-",
                extra_body,
                row["difficulty"],
                row["language"],
                row["instruction_style"],
                row["repo"],
                "true",
            ]
            handle.write("\t".join(value.replace("\t", " ").replace("\n", " ") for value in values) + "\n")
    return assignments, manifest_tsv


def docker_image_exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def build_openswe_images(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        if row["source"] not in {"openswe_tasktrove", "swesmith_go", "swesmith_java", "swesmith_cpp", "swegym_c_cpp"}:
            continue
        image = row["image"]
        if docker_image_exists(image):
            results.append({"instance_id": row["instance_id"], "image": image, "status": "exists"})
            continue
        task_dir = Path(row["task_dir"])
        command = [
            "docker",
            "build",
            "-t",
            image,
            "-f",
            str(task_dir / "environment" / "Dockerfile.prepared"),
            str(task_dir),
        ]
        log_path = task_dir / "docker-build.log"
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            status = subprocess.run(command, stdout=log, stderr=log).returncode
        results.append(
            {
                "instance_id": row["instance_id"],
                "image": image,
                "status": "built" if status == 0 else "failed",
                "returncode": status,
                "log_path": str(log_path),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--swe-rebench-limit", type=int, default=8)
    parser.add_argument("--openswe-limit", type=int, default=2)
    parser.add_argument(
        "--openswe-languages",
        default="js,ts,go,rust,java,php,csharp,c,cpp",
        help="Comma-separated OpenSWE patch languages to select.",
    )
    parser.add_argument("--swesmith-go-limit", type=int, default=0)
    parser.add_argument("--swesmith-go-skip", type=int, default=0)
    parser.add_argument("--swesmith-java-limit", type=int, default=0)
    parser.add_argument("--swesmith-java-skip", type=int, default=0)
    parser.add_argument("--swesmith-cpp-limit", type=int, default=0)
    parser.add_argument("--swesmith-cpp-skip", type=int, default=0)
    parser.add_argument("--swegym-cpp-limit", type=int, default=0)
    parser.add_argument("--swegym-cpp-skip", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--build-openswe-images", action="store_true")
    parser.add_argument("--build-local-images", action="store_true")
    args = parser.parse_args()

    load_env_file()
    if args.clean and args.run_root.exists():
        shutil.rmtree(args.run_root)
    args.run_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"selected": {}, "rejected": {}, "notes": []}
    summary["rejected"] = {key: 0 for key in [
        "swe_rebench_overlap",
        "swe_rebench_missing_image_or_commit",
        "swe_rebench_missing_patch_or_tests",
        "swe_rebench_test_only_patch",
        "swe_rebench_complex_patch",
        "swe_rebench_long_prompt",
        "openswe_not_non_python",
        "openswe_not_allowed_language",
        "openswe_no_gold_source_patch",
        "openswe_complex_patch",
        "openswe_long_prompt",
        "openswe_exact_overlap",
        "openswe_repo_issue_overlap",
        "openswe_missing_repo_or_commit",
        "swegym_c_cpp_not_c_cpp",
        "swegym_c_cpp_exact_overlap",
        "swegym_c_cpp_repo_issue_overlap",
        "swegym_c_cpp_missing_repo_or_commit",
        "swegym_c_cpp_no_gold_source_patch",
        "swegym_c_cpp_complex_patch",
        "swegym_c_cpp_long_prompt",
        "swegym_c_cpp_missing_install",
        "swegym_c_cpp_skipped_selected_offset",
        "swesmith_go_exact_overlap",
        "swesmith_go_repo_overlap",
        "swesmith_go_missing_repo_or_commit",
        "swesmith_go_not_go_only",
        "swesmith_go_test_only_patch",
        "swesmith_go_complex_patch",
        "swesmith_go_bad_prompt_length",
        "swesmith_go_many_fail_tests",
        "swesmith_go_skipped_selected_offset",
        "swesmith_java_exact_overlap",
        "swesmith_java_not_gson",
        "swesmith_java_not_java_only",
        "swesmith_java_test_only_patch",
        "swesmith_java_complex_patch",
        "swesmith_java_bad_prompt_length",
        "swesmith_java_bad_tests",
        "swesmith_java_skipped_selected_offset",
        "swesmith_cpp_exact_overlap",
        "swesmith_cpp_missing_image",
        "swesmith_cpp_not_cpp_only",
        "swesmith_cpp_test_only_patch",
        "swesmith_cpp_complex_patch",
        "swesmith_cpp_bad_prompt_length",
        "swesmith_cpp_bad_tests",
        "swesmith_cpp_skipped_selected_offset",
    ]}
    current = read_current_index()
    openswe_languages = {
        language.strip().lower()
        for language in args.openswe_languages.split(",")
        if language.strip()
    }
    tasks_dir = args.run_root / "tasks"
    rows = []
    rows.extend(materialize_swe_rebench(tasks_dir / "swe-rebench-v1", current, args.swe_rebench_limit, summary))
    rows.extend(materialize_openswe(tasks_dir / "openswe", current, args.openswe_limit, summary, openswe_languages))
    rows.extend(
        materialize_swegym_c_cpp(
            tasks_dir / "swegym-c-cpp",
            current,
            args.swegym_cpp_limit,
            args.swegym_cpp_skip,
            summary,
        )
    )
    rows.extend(
        materialize_swesmith_go(
            tasks_dir / "swesmith-go",
            current,
            args.swesmith_go_limit,
            args.swesmith_go_skip,
            summary,
        )
    )
    rows.extend(
        materialize_swesmith_java(
            tasks_dir / "swesmith-java",
            current,
            args.swesmith_java_limit,
            args.swesmith_java_skip,
            summary,
        )
    )
    rows.extend(
        materialize_swesmith_cpp(
            tasks_dir / "swesmith-cpp",
            current,
            args.swesmith_cpp_limit,
            args.swesmith_cpp_skip,
            summary,
        )
    )
    assignments, manifest_tsv = write_assignments_and_manifest(args.run_root, rows)
    summary["selected"] = {
        "total": len(rows),
        "by_source": {},
        "by_language": {},
    }
    for row in rows:
        summary["selected"]["by_source"][row["source"]] = summary["selected"]["by_source"].get(row["source"], 0) + 1
        summary["selected"]["by_language"][row["language"]] = summary["selected"]["by_language"].get(row["language"], 0) + 1
    summary["assignments_csv"] = str(assignments)
    summary["manifest_tsv"] = str(manifest_tsv)
    summary["notes"].append("SWE-smith images under swesmith/* were not pullable after successful docker login; excluded from runnable manifest.")
    if args.build_openswe_images or args.build_local_images:
        summary["local_image_builds"] = build_openswe_images(rows)
    (args.run_root / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
