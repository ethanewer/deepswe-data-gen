#!/usr/bin/env python3
"""One active hourly datagen check for the foreground monitoring loop."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen")
BASE = REPO / "runs" / "swerebench-v2"
MONITOR_ROOT = BASE / "monitor-deepseek-20260607"
PROGRESS = REPO / "eval" / "benchmarks" / "deepswe" / "progress-report.md"
PYTHON = Path("/wbl-fast/usrs/ee/code-swe-data/runtime/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12")
WAVES = [1, 2, 3, 4, 5, 6]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run(cmd: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def run_root(wave: int) -> Path:
    return BASE / f"datagen-20260607-pyxis-deepseek-afterreset{wave}"


def schedule_rows(wave: int) -> list[dict[str, str]]:
    path = run_root(wave) / "slurm" / "scheduled_jobs.tsv"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    headers = lines[0].split("\t")
    return [dict(zip(headers, line.split("\t"))) for line in lines[1:] if line.strip()]


def all_job_ids() -> list[str]:
    ids = []
    for wave in WAVES:
        ids.extend(row["job_id"] for row in schedule_rows(wave))
    return ids


def monitor_alive() -> bool:
    pid_path = MONITOR_ROOT / "monitor.pid"
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return False
    return Path(f"/proc/{pid}").exists()


def restart_monitor() -> str:
    script = MONITOR_ROOT / "monitor.py"
    result = run([str(PYTHON), "-m", "py_compile", str(script)], timeout=60)
    if result.returncode != 0:
        return f"compile_failed: {result.stdout[-500:]}"
    with (MONITOR_ROOT / "nohup.out").open("a", encoding="utf-8") as out:
        proc = subprocess.Popen(
            ["setsid", str(PYTHON), str(script)],
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=False,
        )
    (MONITOR_ROOT / "monitor.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
    return f"restarted pid={proc.pid}"


def squeue_summary() -> tuple[Counter, str]:
    ids = all_job_ids()
    if not ids:
        return Counter(), ""
    result = run(["squeue", "-h", "-j", ",".join(ids), "-o", "%.24i\t%j\t%T\t%M\t%R"], timeout=120)
    counts = Counter()
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            counts[parts[2]] += 1
            rows.append(line)
    return counts, "\n".join(rows[:80])


def result_summary() -> dict:
    records = []
    for wave in WAVES:
        for path in run_root(wave).glob("pyxis-traces/*/*/r00/*/result.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                record = {"model": "unknown", "difficulty": "unknown", "instruction_style": "unknown", "reward": 0, "read_error": str(exc)}
            record["_wave"] = wave
            record["_result_path"] = str(path)
            records.append(record)

    by_wave = Counter()
    by_model = Counter()
    reward_by_model = Counter()
    by_difficulty = Counter()
    reward_by_difficulty = Counter()
    by_style = Counter()
    exception_types = Counter()
    trajectories = 0
    rewards = 0
    for record in records:
        wave = f"wave{record.get('_wave')}"
        model = record.get("model", "unknown")
        difficulty = record.get("difficulty", "unknown")
        style = record.get("instruction_style", "unknown")
        reward = int(record.get("reward") == 1)
        by_wave[wave] += 1
        by_model[model] += 1
        reward_by_model[model] += reward
        by_difficulty[difficulty] += 1
        reward_by_difficulty[difficulty] += reward
        by_style[style] += 1
        rewards += reward
        exc = record.get("agent_exception") or {}
        exception_types[exc.get("type", "")] += 1
        traj = record.get("trajectory_path")
        if traj:
            traj_path = Path(traj)
            if not traj_path.exists() and traj_path.is_absolute():
                try:
                    result_dir = Path(record["_result_path"]).parent
                    traj_path = result_dir / traj_path.relative_to("/workspace")
                except ValueError:
                    pass
            if traj_path.exists():
                trajectories += 1
    return {
        "total": len(records),
        "reward": rewards,
        "trajectories": trajectories,
        "by_wave": dict(sorted(by_wave.items())),
        "by_model": dict(sorted(by_model.items())),
        "reward_by_model": dict(sorted(reward_by_model.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "reward_by_difficulty": dict(sorted(reward_by_difficulty.items())),
        "by_style": dict(sorted(by_style.items())),
        "exceptions": dict(sorted(exception_types.items())),
    }


def scheduled_summary() -> dict:
    totals = Counter()
    model = Counter()
    difficulty = Counter()
    for wave in WAVES:
        path = run_root(wave) / "manifest" / f"deepseek-afterreset{wave}-summary.json"
        if wave == 1:
            path = run_root(wave) / "manifest" / "deepseek-afterreset-summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        totals[f"wave{wave}"] = data["total"]
        model.update(data["by_model"])
        difficulty.update(data["by_difficulty"])
    return {
        "total": sum(totals.values()),
        "by_wave": dict(sorted(totals.items())),
        "by_model": dict(sorted(model.items())),
        "by_difficulty": dict(sorted(difficulty.items())),
    }


def unique_container_remaining_count() -> dict:
    hq = REPO / "datagen" / "swerebench_v2" / "data" / "high_quality_conf_ge_0.95_tasks.csv"
    with hq.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    excluded_ids: set[str] = set()
    excluded_images: set[str] = set()
    for path in BASE.glob("datagen-*/manifest/*.tsv"):
        with path.open(errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 15 and parts[2] and parts[2] != "instance_id":
                    excluded_ids.add(parts[2])
                    excluded_images.add(parts[5])
    remaining = [
        row
        for row in rows
        if row["instance_id"] not in excluded_ids
        and row["image_name"] not in excluded_images
    ]
    return {
        "total": len(remaining),
        "by_difficulty": dict(sorted(Counter(row["difficulty"] for row in remaining).items())),
        "by_language": dict(sorted(Counter(row["language"] for row in remaining).items())),
    }


def progress_report_age_minutes() -> float | None:
    if not PROGRESS.exists():
        return None
    return (time.time() - PROGRESS.stat().st_mtime) / 60.0


def git_head() -> str:
    result = run(["git", "log", "--oneline", "-1"], timeout=60)
    return result.stdout.strip()


def main() -> None:
    print(f"=== active hourly check: {utc_now()} ===", flush=True)
    alive = monitor_alive()
    print(f"detached_monitor_alive={alive}", flush=True)
    if not alive:
        print(f"detached_monitor_action={restart_monitor()}", flush=True)

    schedule = scheduled_summary()
    print("scheduled=" + json.dumps(schedule, sort_keys=True), flush=True)

    q_counts, q_sample = squeue_summary()
    print("squeue_counts=" + json.dumps(dict(sorted(q_counts.items())), sort_keys=True), flush=True)
    if q_sample:
        print("squeue_sample_begin", flush=True)
        print(q_sample, flush=True)
        print("squeue_sample_end", flush=True)

    results = result_summary()
    print("results=" + json.dumps(results, sort_keys=True), flush=True)

    state_path = MONITOR_ROOT / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print("monitor_release_state=" + json.dumps(state.get("released", {}), sort_keys=True), flush=True)
        print("monitor_probe_state=" + json.dumps(state.get("probe_status", {}), sort_keys=True), flush=True)

    remaining = unique_container_remaining_count()
    print("unique_container_remaining=" + json.dumps(remaining, sort_keys=True), flush=True)
    age = progress_report_age_minutes()
    print(f"progress_report_age_minutes={age:.1f}" if age is not None else "progress_report_missing=true", flush=True)
    print("git_head=" + git_head(), flush=True)


if __name__ == "__main__":
    main()
