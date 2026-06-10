#!/usr/bin/env python3
"""Run packed SWE-rebench mini-swe-agent data generation with Docker.

This runner is intended for direct execution on local serving nodes over SSH.
It avoids Slurm task allocations while still running every rollout inside the
task's Docker image.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import re
import shutil
import shlex
import subprocess
import threading
import time
from pathlib import Path

from eval.minisweagent_pin import MINI_SWE_AGENT_GIT_SHA, MINI_SWE_AGENT_OVERLAY_ENV, require_pinned_minisweagent_overlay
from eval.paths import REPO_ROOT


DEFAULT_PYTHON = Path("/wbl-fast/usrs/ee/code-swe-data/runtime/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12")
DEFAULT_DATAGEN_STRICT_CONFIG = REPO_ROOT / "datagen" / "swerebench_v2" / "minisweagent_datagen_strict.yaml"
DEFAULT_SWEBENCH_MULTILINGUAL_CONFIG = (
    REPO_ROOT / "datagen" / "swerebench_v2" / "minisweagent_swebench_multilingual.yaml"
)
DEFAULT_DEEPSWE_CONFIG = REPO_ROOT / "datagen" / "swerebench_v2" / "minisweagent_deepswe_pier.yaml"
DEFAULT_ENV_FILE = Path("/wbl-fast/usrs/ee/code-swe-data/.env")
DRIVER = REPO_ROOT / "datagen" / "swerebench_v2" / "pyxis_miniswe_agent_driver.py"
FAILURE_WRITER = REPO_ROOT / "datagen" / "swerebench_v2" / "write_pyxis_failure_result.py"
BENCHMARK_PROFILE_CHOICES = ("auto", "swebench-multilingual", "deepswe", "datagen-strict")
SECRET_ENV_NAMES = {"OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "DOCKER_PAT"}


@dataclasses.dataclass(frozen=True)
class ManifestRow:
    index: str
    rollout_id: str
    instance_id: str
    task_dir: Path
    workspace: Path
    image: str
    model: str
    litellm_model: str
    api_key_env: str
    api_base: str
    extra_body_json: str
    difficulty: str
    language: str
    instruction_style: str
    repo: str
    outside_original_high_quality_set: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpus-per-worker", type=float, default=4.0)
    parser.add_argument("--memory-per-worker", default="48g")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--api-base-filter", default="")
    parser.add_argument("--skip-existing-result", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--config-file",
        type=Path,
        help=(
            "Mini-swe-agent config to use for every row. By default the runner "
            "selects the benchmark-matching config from instruction_style."
        ),
    )
    parser.add_argument(
        "--benchmark-profile",
        choices=BENCHMARK_PROFILE_CHOICES,
        default="auto",
        help=(
            "Model-class/profile behavior to match. 'auto' maps original/swe_rebench "
            "rows to SWE-bench Multilingual and deepswe/rewritten/planned rows to DeepSWE."
        ),
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--model-timeout", type=int, default=600)
    parser.add_argument("--agent-wall-time-limit", type=int, default=2700)
    parser.add_argument("--uses-updated-alignment", choices=("true", "false"), default="true")
    parser.add_argument("--eligible-for-controlled-comparison", action="store_true")
    parser.add_argument("--reason-excluded-from-comparison", default="")
    parser.add_argument(
        "--command-timeout",
        type=int,
        help="Override the command timeout from the selected mini-swe-agent config.",
    )
    parser.add_argument("--pull-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def parse_manifest(path: Path, api_base_filter: str, limit: int) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 16:
                raise ValueError(f"manifest row has {len(fields)} fields, expected 16: {line[:200]}")
            api_base = "" if fields[9] == "-" else fields[9]
            if api_base_filter and api_base_filter not in api_base:
                continue
            extra_body_json = "" if fields[10] == "-" else fields[10]
            rows.append(
                ManifestRow(
                    index=fields[0],
                    rollout_id=fields[1],
                    instance_id=fields[2],
                    task_dir=Path(fields[3]),
                    workspace=Path(fields[4]),
                    image=fields[5],
                    model=fields[6],
                    litellm_model=fields[7],
                    api_key_env=fields[8],
                    api_base=api_base,
                    extra_body_json=extra_body_json,
                    difficulty=fields[11],
                    language=fields[12],
                    instruction_style=fields[13],
                    repo=fields[14],
                    outside_original_high_quality_set=fields[15] if len(fields) > 15 else "false",
                )
            )
            if limit and len(rows) >= limit:
                break
    return rows


def docker_image_ref(image: str) -> str:
    ref = image.removeprefix("docker://")
    return ref


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()[:80]


def resolve_benchmark_profile(style: str, override: str = "auto") -> str:
    if override != "auto":
        return override
    if style in {"original", "swe_rebench"}:
        return "swebench-multilingual"
    if style in {"deepswe", "rewritten", "planned"}:
        return "deepswe"
    return "datagen-strict"


def selected_config_file(args: argparse.Namespace, profile: str) -> Path:
    if args.config_file is not None:
        return args.config_file
    if profile == "swebench-multilingual":
        return DEFAULT_SWEBENCH_MULTILINGUAL_CONFIG
    if profile == "deepswe":
        return DEFAULT_DEEPSWE_CONFIG
    return DEFAULT_DATAGEN_STRICT_CONFIG


def datagen_code_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def run_command(command: list[str], stdout: Path, stderr: Path, timeout: int | None = None) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("ab") as out, stderr.open("ab") as err:
        process = subprocess.run(command, stdout=out, stderr=err, timeout=timeout)
    return process.returncode


def result_has_model_trace(workspace: Path) -> bool:
    result_path = workspace / "result.json"
    if not result_path.exists():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(result.get("api_calls") or 0) > 0


def archive_retryable_result(workspace: Path) -> None:
    result_path = workspace / "result.json"
    trajectory_path = workspace / "agent" / "mini-swe-agent.trajectory.json"
    if not result_path.exists() and not trajectory_path.exists():
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    history_dir = workspace / "retry-history" / stamp
    history_dir.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        shutil.copy2(result_path, history_dir / "result.json")
    if trajectory_path.exists():
        (history_dir / "agent").mkdir(parents=True, exist_ok=True)
        shutil.copy2(trajectory_path, history_dir / "agent" / "mini-swe-agent.trajectory.json")


pull_locks: dict[str, threading.Lock] = {}
pull_locks_guard = threading.Lock()
docker_login_lock = threading.Lock()
docker_login_attempted = False


def pull_lock(image: str) -> threading.Lock:
    with pull_locks_guard:
        return pull_locks.setdefault(image, threading.Lock())


def ensure_docker_login(log_path: Path) -> None:
    global docker_login_attempted
    username = os.environ.get("DOCKER_USERNAME", "")
    password = os.environ.get("DOCKER_PAT", "")
    if not username or not password:
        return
    with docker_login_lock:
        if docker_login_attempted:
            return
        docker_login_attempted = True
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                ["docker", "login", "--username", username, "--password-stdin"],
                input=password,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
                check=False,
            )
            with log_path.open("ab") as log:
                log.write(f"docker_login_status={result.returncode}\n".encode())
        except Exception as exc:
            with log_path.open("ab") as log:
                log.write(f"docker_login_error={type(exc).__name__}\n".encode())


def ensure_image(image: str, log_path: Path, retries: int) -> tuple[bool, str]:
    ref = docker_image_ref(image)
    inspect = subprocess.run(
        ["docker", "image", "inspect", ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if inspect.returncode == 0:
        return True, "present"
    with pull_lock(ref):
        inspect = subprocess.run(
            ["docker", "image", "inspect", ref],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if inspect.returncode == 0:
            return True, "present"
        ensure_docker_login(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, retries + 1):
            with log_path.open("ab") as log:
                log.write(f"docker_pull_attempt={attempt}\n".encode())
                status = subprocess.run(["docker", "pull", ref], stdout=log, stderr=log).returncode
            if status == 0:
                return True, "pulled"
            if attempt < retries:
                time.sleep(20 * attempt)
        return False, f"pull_failed:{status}"


def base_environment(args: argparse.Namespace, workspace: Path, row: ManifestRow) -> dict[str, str]:
    cache_root = Path("/wbl-fast/usrs/ee/code-swe-data/cache")
    pydeps_overlay = require_pinned_minisweagent_overlay().resolve()
    pythonpath = f"{pydeps_overlay}:{REPO_ROOT}"
    ca_bundle = pydeps_overlay / "certifi" / "cacert.pem"
    env = {
        "HF_HOME": str(cache_root / "hf"),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
        "UV_CACHE_DIR": str(cache_root / "uv"),
        "PIP_CACHE_DIR": str(cache_root / "pip"),
        "PYDEPS_OVERLAY": str(pydeps_overlay),
        "PYTHONPATH": pythonpath,
        "DATAGEN_CODE_COMMIT": datagen_code_commit(),
        "SSL_CERT_FILE": str(ca_bundle),
        "REQUESTS_CA_BUNDLE": str(ca_bundle),
        "CURL_CA_BUNDLE": str(ca_bundle),
        "HOME": "/workspace/home",
        "MSWEA_COST_TRACKING": "ignore_errors",
        "MSWEA_SILENT_STARTUP": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NVIDIA_VISIBLE_DEVICES": "none",
    }
    if row.api_base:
        env["OPENAI_BASE_URL"] = row.api_base
        env["OPENAI_API_BASE"] = row.api_base
    api_value = os.environ.get(row.api_key_env, "")
    if (
        row.api_key_env == "OPENAI_API_KEY"
        and row.api_base.startswith("http://")
        and ".integrated.pcluster:" in row.api_base
        and not api_value
    ):
        api_value = "local-model-no-auth-required"
    if api_value:
        env[row.api_key_env] = api_value
        if row.api_key_env in {"OPENAI_API_KEY", "DEEPSEEK_API_KEY"}:
            env["OPENAI_API_KEY"] = api_value
    return env


def docker_run_command(args: argparse.Namespace, row: ManifestRow) -> list[str]:
    shared_root = REPO_ROOT.parent
    env = base_environment(args, row.workspace, row)
    benchmark_profile = resolve_benchmark_profile(row.instruction_style, args.benchmark_profile)
    config_file = selected_config_file(args, benchmark_profile)
    uses_updated_alignment = getattr(args, "uses_updated_alignment", "true")
    eligible_for_controlled_comparison = (
        "true" if getattr(args, "eligible_for_controlled_comparison", False) else "false"
    )
    reason_excluded_from_comparison = getattr(args, "reason_excluded_from_comparison", "")
    command = [
        "docker",
        "run",
        "--rm",
        "--runtime=runc",
        "--network=host",
        f"--cpus={args.cpus_per_worker}",
        f"--memory={args.memory_per_worker}",
        "--name",
        f"{safe_name(args.job_name)}-{safe_name(row.instance_id)}-{int(time.time())}",
        "-v",
        f"{shared_root}:{shared_root}:rw",
        "-v",
        f"{row.workspace}:/workspace:rw",
        "-v",
        f"{row.task_dir / 'tests'}:/tests:ro",
    ]
    for key, value in sorted(env.items()):
        if key in SECRET_ENV_NAMES:
            os.environ[key] = value
            command.extend(["-e", key])
        else:
            command.extend(["-e", f"{key}={value}"])
    command.extend(
        [
            docker_image_ref(row.image),
            str(args.python),
            str(DRIVER),
            "--task-dir",
            str(row.task_dir),
            "--workspace",
            "/workspace",
            "--config-file",
            str(config_file),
            "--benchmark-profile",
            benchmark_profile,
            "--instance-id",
            row.instance_id,
            "--rollout-id",
            row.rollout_id,
            "--model",
            row.model,
            "--litellm-model",
            row.litellm_model,
            "--api-key-env",
            row.api_key_env,
            "--api-base",
            row.api_base,
            "--extra-body-json",
            row.extra_body_json,
            "--instruction-style",
            row.instruction_style,
            "--difficulty",
            row.difficulty,
            "--language",
            row.language,
            "--repo",
            row.repo,
            "--outside-original-high-quality-set",
            row.outside_original_high_quality_set,
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--reasoning-effort",
            args.reasoning_effort,
            "--model-timeout",
            str(args.model_timeout),
            "--agent-wall-time-limit",
            str(args.agent_wall_time_limit),
            "--uses-updated-alignment",
            uses_updated_alignment,
            "--eligible-for-controlled-comparison",
            eligible_for_controlled_comparison,
            "--reason-excluded-from-comparison",
            reason_excluded_from_comparison,
        ]
    )
    if args.command_timeout is not None:
        command.extend(["--command-timeout", str(args.command_timeout)])
    return command


def write_failure(args: argparse.Namespace, row: ManifestRow, status: int, stdout_log: Path, stderr_log: Path, message: str) -> None:
    error_type = "DockerContainerStartError" if status == 125 else "TaskContainerRunError"
    benchmark_profile = resolve_benchmark_profile(row.instruction_style, args.benchmark_profile)
    uses_updated_alignment = getattr(args, "uses_updated_alignment", "true")
    eligible_for_controlled_comparison = (
        "true" if getattr(args, "eligible_for_controlled_comparison", False) else "false"
    )
    reason_excluded_from_comparison = getattr(args, "reason_excluded_from_comparison", "")
    command = [
        str(args.python),
        str(FAILURE_WRITER),
        "--workspace",
        str(row.workspace),
        "--instance-id",
        row.instance_id,
        "--rollout-id",
        row.rollout_id,
        "--model",
        row.model,
        "--litellm-model",
        row.litellm_model,
        "--instruction-style",
        row.instruction_style,
        "--benchmark-profile",
        benchmark_profile,
        "--difficulty",
        row.difficulty,
        "--language",
        row.language,
        "--repo",
        row.repo,
        "--outside-original-high-quality-set",
        row.outside_original_high_quality_set,
        "--mini-swe-agent-config-file",
        str(selected_config_file(args, benchmark_profile)),
        "--uses-updated-alignment",
        uses_updated_alignment,
        "--eligible-for-controlled-comparison",
        eligible_for_controlled_comparison,
        "--reason-excluded-from-comparison",
        reason_excluded_from_comparison,
        "--task-dir",
        str(row.task_dir),
        "--image",
        row.image,
        "--pyxis-image",
        docker_image_ref(row.image),
        "--error-type",
        error_type,
        "--error-message",
        message,
        "--exit-status",
        str(status),
        "--stdout-log",
        str(stdout_log),
        "--stderr-log",
        str(stderr_log),
    ]
    subprocess.run(command, check=False)


def run_row(args: argparse.Namespace, row: ManifestRow, log_dir: Path) -> dict[str, str | int]:
    row.workspace.mkdir(parents=True, exist_ok=True)
    (row.workspace / "home").mkdir(parents=True, exist_ok=True)
    row_id = f"{row.index}.{safe_name(row.instruction_style)}.{safe_name(row.instance_id)}"
    stdout_log = log_dir / f"{args.job_name}.{row_id}.out"
    stderr_log = log_dir / f"{args.job_name}.{row_id}.err"
    pull_log = log_dir / f"{args.job_name}.{row_id}.docker-pull.log"
    if args.skip_existing_result and result_has_model_trace(row.workspace):
        return {"instance_id": row.instance_id, "status": 0, "state": "skipped_existing"}
    archive_retryable_result(row.workspace)
    ok, pull_state = ensure_image(row.image, pull_log, args.pull_retries)
    if not ok:
        write_failure(args, row, 125, stdout_log, stderr_log, pull_state)
        return {"instance_id": row.instance_id, "status": 125, "state": pull_state}
    command = docker_run_command(args, row)
    status = run_command(command, stdout_log, stderr_log, timeout=args.agent_wall_time_limit + 4200)
    if status != 0 and not (row.workspace / "result.json").exists():
        write_failure(args, row, status, stdout_log, stderr_log, f"docker run exited with status {status}")
    return {"instance_id": row.instance_id, "status": status, "state": "done" if status == 0 else "docker_failed"}


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    load_env_file(args.env_file)
    rows = parse_manifest(args.manifest_tsv, args.api_base_filter, args.limit)
    if args.skip_existing_result:
        rows = [row for row in rows if not result_has_model_trace(row.workspace)]
    if not rows:
        print("rows=0")
        return
    log_dir = args.run_root / "docker" / args.job_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"node={os.uname().nodename}")
    print(f"rows={len(rows)}")
    print(f"workers={args.workers}")
    print(f"cpus_per_worker={args.cpus_per_worker}")
    print(f"mini_swe_agent_git_sha={MINI_SWE_AGENT_GIT_SHA}")
    print(f"mini_swe_agent_overlay_env={MINI_SWE_AGENT_OVERLAY_ENV}")
    print(f"log_dir={log_dir}")
    if args.dry_run:
        for row in rows[:5]:
            print(f"dry_row={row.instance_id} image={row.image} workspace={row.workspace}")
        return
    counts: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_row, args, row, log_dir) for row in rows]
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            state = str(result["state"])
            counts[state] = counts.get(state, 0) + 1
            print(
                f"completed={i}/{len(rows)} instance_id={result['instance_id']} "
                f"state={state} status={result['status']} counts={json.dumps(counts, sort_keys=True)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
