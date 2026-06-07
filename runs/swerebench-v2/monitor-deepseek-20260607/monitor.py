#!/usr/bin/env python3
"""Monitor delayed DeepSeek Pyxis waves for the Docker reset window."""

from __future__ import annotations

import json
import os
import base64
import shlex
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen")
BASE = REPO / "runs" / "swerebench-v2"
MONITOR_ROOT = BASE / "monitor-deepseek-20260607"
PROGRESS = REPO / "eval" / "benchmarks" / "deepswe" / "progress-report.md"
LOG = MONITOR_ROOT / "monitor.log"
STATE_PATH = MONITOR_ROOT / "state.json"
PYTHON = Path("/wbl-fast/usrs/ee/code-swe-data/runtime/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12")
SHARDS = {
    "ewe": "ethanewer",
    "och": "ethanoch",
    "oew": "ethanoewer",
}
ALL_WAVES = [1, 2, 3, 4, 5, 6, 7]
WINDOWS = {
    "window1": [1, 2],
    "window1b": [5, 6],
    "window2": [3, 4],
}
PROBE_INTERVAL_SEC = 15 * 60
REPORT_INTERVAL_SEC = 60 * 60
RUN_FOR_SEC = 12 * 60 * 60
MIN_REMAINING_FOR_EARLY_RELEASE = 120


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stamp() -> str:
    return utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp()}] {message}\n")


def run(cmd: list[str], *, timeout: int | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    log("$ " + " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=env,
    )


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "started_at": stamp(),
            "last_probe": {},
            "probe_success": {},
            "released": {},
            "last_report_ts": 0,
        }
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_root(wave: int) -> Path:
    return BASE / f"datagen-20260607-pyxis-deepseek-afterreset{wave}"


def schedule_rows(wave: int) -> list[dict[str, str]]:
    path = run_root(wave) / "slurm" / "scheduled_jobs.tsv"
    rows = []
    if not path.exists():
        return rows
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return rows
    headers = lines[0].split("\t")
    for line in lines[1:]:
        if line.strip():
            rows.append(dict(zip(headers, line.split("\t"))))
    return rows


def all_job_ids(waves: list[int] | None = None) -> list[str]:
    ids = []
    for wave in (waves or ALL_WAVES):
        for row in schedule_rows(wave):
            ids.append(row["job_id"])
    return ids


def window_begin(window: str) -> str:
    for wave in WINDOWS[window]:
        rows = schedule_rows(wave)
        if rows:
            return rows[0]["begin_utc"]
    return ""


def parse_begin(begin: str) -> float:
    return datetime.fromisoformat(begin).replace(tzinfo=timezone.utc).timestamp()


def first_manifest_record(wave: int, shard: str) -> dict[str, str]:
    if wave == 1:
        path = run_root(wave) / "manifest" / f"deepseek-afterreset-{shard}.tsv"
    else:
        path = run_root(wave) / "manifest" / f"deepseek-afterreset{wave}-{shard}.tsv"
    fields = [
        "index",
        "rollout_id",
        "instance_id",
        "task_dir",
        "workspace",
        "image",
        "model",
        "litellm_model",
        "api_key_env",
        "api_base",
        "extra_body_json",
        "difficulty",
        "language",
        "instruction_style",
        "repo",
    ]
    first = path.read_text(encoding="utf-8").splitlines()[0].split("\t")
    return dict(zip(fields, first))


def pyxis_image(image: str, docker_user: str) -> str:
    value = image.removeprefix("docker.io/").removeprefix("docker://")
    return f"{docker_user}@registry-1.docker.io#{value}"


def docker_credentials(docker_user: str) -> tuple[str, str]:
    path = Path("/wbl-fast/usrs/ee/code-swe-data/credentials/dockerhub") / docker_user / "enroot" / ".credentials"
    tokens = shlex.split(path.read_text(encoding="utf-8"))
    try:
        login = tokens[tokens.index("login") + 1]
        password = tokens[tokens.index("password") + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"cannot parse Docker credential file for {docker_user}") from exc
    return login, password


def registry_manifest_probe(image: str, docker_user: str, log_path: Path) -> tuple[bool, dict[str, str]]:
    image = image.removeprefix("docker.io/").removeprefix("docker://")
    repository, tag = image.rsplit(":", 1)
    login, password = docker_credentials(docker_user)
    auth = base64.b64encode(f"{login}:{password}".encode("utf-8")).decode("ascii")
    token_url = (
        "https://auth.docker.io/token?"
        + urllib.parse.urlencode(
            {
                "service": "registry.docker.io",
                "scope": f"repository:{repository}:pull",
            }
        )
    )
    token_request = urllib.request.Request(token_url, headers={"Authorization": f"Basic {auth}"})
    status: dict[str, str] = {"docker_user": docker_user, "repository": repository, "tag": tag}
    try:
        with urllib.request.urlopen(token_request, timeout=60) as response:
            token_data = json.loads(response.read().decode("utf-8"))
        token = token_data["token"]
        manifest_url = f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}"
        request = urllib.request.Request(
            manifest_url,
            method="HEAD",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.docker.distribution.manifest.v2+json",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            status.update(
                {
                    "status": str(response.status),
                    "ratelimit-limit": response.headers.get("ratelimit-limit", ""),
                    "ratelimit-remaining": response.headers.get("ratelimit-remaining", ""),
                    "docker-ratelimit-source": response.headers.get("docker-ratelimit-source", ""),
                }
            )
        remaining = status.get("ratelimit-remaining", "").split(";", 1)[0]
        try:
            remaining_count = int(remaining)
        except ValueError:
            remaining_count = -1
        status["min_remaining_required_for_early_release"] = str(MIN_REMAINING_FOR_EARLY_RELEASE)
        ok = status["status"] == "200" and remaining_count >= MIN_REMAINING_FOR_EARLY_RELEASE
    except urllib.error.HTTPError as exc:
        status.update(
            {
                "status": str(exc.code),
                "reason": exc.reason,
                "ratelimit-limit": exc.headers.get("ratelimit-limit", ""),
                "ratelimit-remaining": exc.headers.get("ratelimit-remaining", ""),
                "docker-ratelimit-source": exc.headers.get("docker-ratelimit-source", ""),
            }
        )
        ok = False
    except Exception as exc:  # noqa: BLE001
        status.update({"status": "exception", "exception_type": type(exc).__name__, "message": str(exc)})
        ok = False
    # Do not write auth headers or bearer tokens.
    log_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ok, status


def probe_shard(window: str, shard: str, state: dict) -> bool:
    key = f"{window}:{shard}"
    now_ts = time.time()
    if state["probe_success"].get(key):
        return True
    if now_ts - float(state["last_probe"].get(key, 0)) < PROBE_INTERVAL_SEC:
        return False
    state["last_probe"][key] = now_ts
    save_state(state)

    wave = WINDOWS[window][0]
    docker_user = SHARDS[shard]
    record = first_manifest_record(wave, shard)
    out = MONITOR_ROOT / "probes" / f"{window}-{shard}-{int(now_ts)}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, status = registry_manifest_probe(record["image"], docker_user, out)
    state["probe_success"][key] = ok
    state.setdefault("probe_logs", {})[key] = str(out)
    state.setdefault("probe_status", {})[key] = status
    if ok:
        log(f"registry probe succeeded: {key} remaining={status.get('ratelimit-remaining', '')}")
    else:
        log(f"registry probe failed: {key} status={status.get('status')} remaining={status.get('ratelimit-remaining', '')} log={out}")
    save_state(state)
    return ok


def release_window(window: str, state: dict) -> None:
    if state["released"].get(window):
        return
    ids = all_job_ids(WINDOWS[window])
    for job_id in ids:
        result = run(["scontrol", "update", f"JobId={job_id}", "StartTime=now"], timeout=60)
        log(f"release {window} job={job_id} rc={result.returncode} out={result.stdout.strip()[-500:]}")
    state["released"][window] = stamp()
    state[f"{window}_release_ts"] = time.time()
    save_state(state)


def maybe_release_early(state: dict) -> None:
    now_ts = time.time()
    for window in WINDOWS:
        if state["released"].get(window):
            continue
        begin = window_begin(window)
        if not begin:
            continue
        begin_ts = parse_begin(begin)
        if now_ts >= begin_ts:
            state["released"][window] = f"scheduled-start {stamp()}"
            save_state(state)
            continue
        if window == "window2":
            release1_ts = state.get("window1_release_ts")
            if release1_ts and now_ts < float(release1_ts) + 5.5 * 60 * 60:
                continue
            if not release1_ts and now_ts < begin_ts - 45 * 60:
                continue
        successes = [probe_shard(window, shard, state) for shard in SHARDS]
        if all(successes):
            release_window(window, state)


def squeue_text() -> str:
    ids = all_job_ids()
    if not ids:
        return ""
    result = run(
        [
            "squeue",
            "-h",
            "-j",
            ",".join(ids),
            "-o",
            "%.18i %.9P %.28j %.10T %.10M %R",
        ],
        timeout=60,
    )
    return result.stdout.strip()


def summarize_results() -> dict:
    records = []
    for wave in ALL_WAVES:
        root = run_root(wave)
        if not root.exists():
            continue
        for path in root.glob("pyxis-traces/*/*/r00/*/result.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                record = {"model": "unknown", "difficulty": "unknown", "instruction_style": "unknown", "reward": 0, "read_error": str(exc)}
            record["_wave"] = wave
            record["_path"] = str(path)
            records.append(record)
    by_model = Counter()
    by_difficulty = Counter()
    by_style = Counter()
    by_wave = Counter()
    reward_by_model = Counter()
    reward_by_difficulty = Counter()
    pyxis_failures = 0
    trajectories = 0
    for record in records:
        model = record.get("model", "unknown")
        difficulty = record.get("difficulty", "unknown")
        style = record.get("instruction_style", "unknown")
        wave = f"wave{record.get('_wave')}"
        reward = int(record.get("reward") == 1)
        by_model[model] += 1
        by_difficulty[difficulty] += 1
        by_style[style] += 1
        by_wave[wave] += 1
        reward_by_model[model] += reward
        reward_by_difficulty[difficulty] += reward
        exc = record.get("agent_exception") or {}
        if exc.get("type") == "PyxisContainerStartError":
            pyxis_failures += 1
        traj = record.get("trajectory_path")
        if traj:
            traj_path = Path(traj)
            if not traj_path.exists() and traj_path.is_absolute():
                try:
                    result_dir = Path(record["_path"]).parent
                    traj_path = result_dir / traj_path.relative_to("/workspace")
                except ValueError:
                    pass
            if traj_path.exists():
                trajectories += 1
    return {
        "results": len(records),
        "reward": sum(int(r.get("reward") == 1) for r in records),
        "trajectories": trajectories,
        "pyxis_start_failures": pyxis_failures,
        "by_model": dict(sorted(by_model.items())),
        "reward_by_model": dict(sorted(reward_by_model.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "reward_by_difficulty": dict(sorted(reward_by_difficulty.items())),
        "by_style": dict(sorted(by_style.items())),
        "by_wave": dict(sorted(by_wave.items())),
    }


def scheduled_totals() -> dict:
    totals = Counter()
    model = Counter()
    difficulty = Counter()
    style = Counter()
    for wave in ALL_WAVES:
        summary_path = run_root(wave) / "manifest" / f"deepseek-afterreset{wave}-summary.json"
        if wave == 1:
            summary_path = run_root(wave) / "manifest" / "deepseek-afterreset-summary.json"
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        totals[f"wave{wave}"] = data["total"]
        model.update(data["by_model"])
        difficulty.update(data["by_difficulty"])
        style.update(data["by_style"])
    return {
        "by_wave": dict(sorted(totals.items())),
        "by_model": dict(sorted(model.items())),
        "by_difficulty": dict(sorted(difficulty.items())),
        "by_style": dict(sorted(style.items())),
        "total": sum(totals.values()),
    }


def append_report(state: dict) -> None:
    summary = summarize_results()
    schedule = scheduled_totals()
    queue = squeue_text()
    probe_success = state.get("probe_success", {})
    release = state.get("released", {})
    text = [
        "",
        f"## {utc_now().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "DeepSeek Docker-reset monitor update:",
        "",
        f"- Scheduled DeepSeek-only unique-container trials: `{schedule['total']}` across waves 1-7.",
        f"- Scheduled by wave: `{schedule['by_wave']}`.",
        f"- Scheduled by model: `{schedule['by_model']}`.",
        f"- Scheduled by difficulty: `{schedule['by_difficulty']}`.",
        f"- Scheduled prompt styles: `{schedule['by_style']}`.",
        f"- Completed result records so far: `{summary['results']}`; reward-pass: `{summary['reward']}`; saved trajectories: `{summary['trajectories']}`; Pyxis start failures: `{summary['pyxis_start_failures']}`.",
        f"- Results by model: total `{summary['by_model']}`, reward `{summary['reward_by_model']}`.",
        f"- Results by difficulty: total `{summary['by_difficulty']}`, reward `{summary['reward_by_difficulty']}`.",
        f"- Results by style: `{summary['by_style']}`.",
        f"- Probe success state: `{probe_success}`.",
        f"- Release state: `{release}`.",
        "",
        "Queue snapshot:",
        "",
        "```",
        queue[:6000] if queue else "No scheduled DeepSeek datagen jobs currently visible in squeue.",
        "```",
    ]
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(text) + "\n")
    log("appended progress report")


def commit_report() -> None:
    run(["git", "add", str(PROGRESS.relative_to(REPO))], timeout=60)
    diff = run(["git", "diff", "--cached", "--quiet", "--", str(PROGRESS.relative_to(REPO))], timeout=60)
    if diff.returncode == 0:
        log("no progress report diff to commit")
        return
    message = f"Monitor DeepSeek datagen {utc_now().strftime('%Y-%m-%d %H:%M UTC')}"
    commit = run(["git", "commit", "-m", message], timeout=120)
    log(f"git commit rc={commit.returncode} out={commit.stdout[-1000:]}")
    if commit.returncode == 0:
        push = run(["git", "push"], timeout=300)
        log(f"git push rc={push.returncode} out={push.stdout[-1000:]}")


def main() -> None:
    MONITOR_ROOT.mkdir(parents=True, exist_ok=True)
    state = load_state()
    state.setdefault("started_at", stamp())
    save_state(state)
    end_ts = time.time() + RUN_FOR_SEC
    log("monitor started")
    # Write an immediate report on the first launch, then hourly. Restarts inside
    # the same hour should not spam duplicate report sections.
    if time.time() - float(state.get("last_report_ts", 0)) >= REPORT_INTERVAL_SEC:
        append_report(state)
        commit_report()
        state["last_report_ts"] = time.time()
        save_state(state)

    while time.time() < end_ts:
        try:
            state = load_state()
            maybe_release_early(state)
            now_ts = time.time()
            if now_ts - float(state.get("last_report_ts", 0)) >= REPORT_INTERVAL_SEC:
                append_report(state)
                commit_report()
                state["last_report_ts"] = now_ts
                save_state(state)
        except Exception as exc:  # noqa: BLE001
            log(f"monitor exception: {type(exc).__name__}: {exc}")
        time.sleep(5 * 60)

    state = load_state()
    state["finished_at"] = stamp()
    save_state(state)
    append_report(state)
    commit_report()
    log("monitor finished")


if __name__ == "__main__":
    main()
