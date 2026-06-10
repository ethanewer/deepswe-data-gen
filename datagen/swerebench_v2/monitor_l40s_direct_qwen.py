#!/usr/bin/env python3
"""Adaptive monitor for direct Docker datagen on L40S Qwen serving nodes."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path("/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen")
PYTHON = Path("/wbl-fast/usrs/ee/code-swe-data/runtime/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12")
RUNNER = REPO_ROOT / "datagen" / "swerebench_v2" / "run_docker_datagen_packed.py"


@dataclass(frozen=True)
class Node:
    name: str
    api_filter: str
    metrics_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--low-active", type=int, default=7)
    parser.add_argument("--max-queue-sum", type=float, default=0.0)
    parser.add_argument("--max-token-usage", type=float, default=0.72)
    parser.add_argument("--container-prefix", default="hq-")
    parser.add_argument("--job-prefix", default="hq-direct-auto")
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def log(message: str, log_file: Path | None) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
    print(line, flush=True)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def run(command: list[str], *, timeout: int = 60, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=check)


def ssh(node: Node, remote: str, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return run(["ssh", node.name, remote], timeout=timeout)


def metrics(node: Node) -> dict[str, float]:
    text = urllib.request.urlopen(node.metrics_url, timeout=10).read().decode("utf-8", errors="replace")
    values: dict[str, list[float]] = {
        "num_running_reqs": [],
        "num_queue_reqs": [],
        "token_usage": [],
    }
    for line in text.splitlines():
        for key in values:
            if line.startswith(f"sglang:{key}"):
                try:
                    values[key].append(float(line.rsplit(" ", 1)[1]))
                except ValueError:
                    pass
    return {
        "running_sum": sum(values["num_running_reqs"]),
        "queue_sum": sum(values["num_queue_reqs"]),
        "token_max": max(values["token_usage"] or [0.0]),
    }


def read_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parts = line.split("\t")
            if len(parts) >= 16:
                rows.append(parts)
    return rows


def active_container_names(node: Node, prefix: str) -> list[str]:
    command = f"docker ps --format '{{{{.Names}}}}' | grep -E {shlex.quote('^' + prefix)} || true"
    result = ssh(node, command, timeout=60)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def active_ids_from_names(names: list[str], instance_ids: set[str]) -> set[str]:
    active: set[str] = set()
    ordered = sorted(instance_ids, key=len, reverse=True)
    for name in names:
        for instance_id in ordered:
            if f"-{instance_id}-" in name:
                active.add(instance_id)
                break
    return active


def parent_pid_path(run_root: Path, node: Node) -> Path:
    return run_root / "remote" / f"monitor-{node.api_filter}.pid"


def parent_running(node: Node, pid_file: Path) -> bool:
    if not pid_file.exists():
        return False
    pid = pid_file.read_text(encoding="utf-8").strip()
    if not pid:
        return False
    result = ssh(node, f"kill -0 {shlex.quote(pid)} 2>/dev/null", timeout=20)
    return result.returncode == 0


def stop_parent(node: Node, pid_file: Path, log_file: Path | None) -> None:
    if not pid_file.exists():
        return
    pid = pid_file.read_text(encoding="utf-8").strip()
    if not pid:
        return
    ssh(node, f"kill -TERM {shlex.quote(pid)} 2>/dev/null || true", timeout=20)
    log(f"stopped_parent node={node.api_filter} pid={pid}", log_file)


def write_manifest(
    run_root: Path,
    source_rows: list[list[str]],
    node: Node,
    active_ids: set[str],
) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = run_root / "manifest" / f"monitor-{node.api_filter}-{timestamp}.tsv"
    kept: list[str] = []
    for parts in source_rows:
        instance_id = parts[2]
        workspace = Path(parts[4])
        api_base = parts[9]
        if node.api_filter not in api_base:
            continue
        if instance_id in active_ids:
            continue
        if (workspace / "result.json").exists():
            continue
        kept.append("\t".join(parts))
    out.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return out


def launch_parent(args: argparse.Namespace, node: Node, manifest: Path, log_file: Path | None) -> None:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    job = f"{args.job_prefix}-{node.api_filter}-{timestamp}"
    remote_log_dir = args.run_root / "remote"
    pid_file = parent_pid_path(args.run_root, node)
    command = (
        f"mkdir -p {shlex.quote(str(remote_log_dir))}; "
        "nohup bash -lc "
        + shlex.quote(
            "cd /wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen && "
            f"PYTHONPATH={REPO_ROOT} {PYTHON} {RUNNER} "
            f"--run-root {args.run_root} "
            f"--manifest-tsv {manifest} "
            f"--job-name {job} "
            f"--workers {args.workers} "
            "--cpus-per-worker 4 --memory-per-worker 48g "
            f"--api-base-filter {node.api_filter} "
            "--skip-existing-result --pull-retries 3 "
            "--temperature 0.0 --max-tokens 16384 --reasoning-effort high "
            "--model-timeout 600 --agent-wall-time-limit 2700 --command-timeout 180"
        )
        + f" > {shlex.quote(str(remote_log_dir / (job + '.out')))}"
        + f" 2> {shlex.quote(str(remote_log_dir / (job + '.err')))}"
        + f" < /dev/null & echo $! > {shlex.quote(str(pid_file))}"
    )
    ssh(node, command, timeout=30)
    log(f"launched node={node.api_filter} job={job} manifest={manifest}", log_file)


def main() -> None:
    args = parse_args()
    args.run_root = args.run_root.resolve()
    args.source_manifest = args.source_manifest.resolve()
    log_file = args.log_file or (args.run_root / "remote" / "monitor-l40s-direct-qwen.log")
    nodes = [
        Node(
            "l40s-8gpu-dy-l40s-8gpu-cr-0-2.integrated.pcluster",
            "cr-0-2",
            "http://l40s-8gpu-dy-l40s-8gpu-cr-0-2.integrated.pcluster:20010/metrics",
        ),
        Node(
            "l40s-8gpu-dy-l40s-8gpu-cr-0-3.integrated.pcluster",
            "cr-0-3",
            "http://l40s-8gpu-dy-l40s-8gpu-cr-0-3.integrated.pcluster:20010/metrics",
        ),
    ]
    source_rows = read_rows(args.source_manifest)
    instance_ids = {row[2] for row in source_rows}
    log(
        "monitor_started "
        + json.dumps(
            {
                "source_rows": len(source_rows),
                "workers": args.workers,
                "low_active": args.low_active,
                "max_queue_sum": args.max_queue_sum,
                "max_token_usage": args.max_token_usage,
            },
            sort_keys=True,
        ),
        log_file,
    )
    while True:
        any_work_or_active = False
        for node in nodes:
            pid_file = parent_pid_path(args.run_root, node)
            names = active_container_names(node, args.container_prefix)
            active_ids = active_ids_from_names(names, instance_ids)
            current_metrics = metrics(node)
            running_parent = parent_running(node, pid_file)
            manifest = write_manifest(args.run_root, source_rows, node, active_ids)
            pending = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
            any_work_or_active = any_work_or_active or bool(names) or pending > 0 or running_parent
            log(
                "state "
                + json.dumps(
                    {
                        "node": node.api_filter,
                        "active_containers": len(names),
                        "active_ids": len(active_ids),
                        "pending": pending,
                        "parent_running": running_parent,
                        **current_metrics,
                    },
                    sort_keys=True,
                ),
                log_file,
            )
            overloaded = (
                current_metrics["queue_sum"] > args.max_queue_sum
                or current_metrics["token_max"] > args.max_token_usage
            )
            if overloaded and running_parent:
                stop_parent(node, pid_file, log_file)
                continue
            if (
                not running_parent
                and pending > 0
                and len(names) <= args.low_active
                and not overloaded
            ):
                launch_parent(args, node, manifest, log_file)
        if not any_work_or_active:
            log("monitor_finished_no_work", log_file)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
