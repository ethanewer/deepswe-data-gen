#!/usr/bin/env python3
"""Submit packed direct-Docker mini-swe-agent data-generation arrays.

This launcher is for CPU-only Slurm nodes with a host Docker daemon. It avoids
Pyxis image materialization, which can exhaust /run/pyxis when many unique task
images are started on one node.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
from pathlib import Path

from eval.minisweagent_pin import require_pinned_minisweagent_overlay
from eval.paths import REPO_ROOT


DEFAULT_PYTHON = Path("/wbl-fast/usrs/ee/code-swe-data/runtime/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12")
RUNNER = REPO_ROOT / "datagen" / "swerebench_v2" / "run_docker_datagen_packed.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--partition", default="m7i-cpu2")
    parser.add_argument("--rows-per-job", type=int, default=256)
    parser.add_argument("--array-concurrency", type=int, default=0, help="0 or >= array size omits the Slurm % throttle.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--mem", default="56G")
    parser.add_argument("--memory-per-worker", default="3g")
    parser.add_argument("--time", default="14:00:00")
    parser.add_argument("--stagger-seconds", type=float, default=15.0)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--model-timeout", type=int, default=600)
    parser.add_argument("--agent-wall-time-limit", type=int, default=2700)
    parser.add_argument("--command-timeout", type=int, default=180)
    parser.add_argument("--max-concurrent-pulls", type=int, default=2)
    parser.add_argument("--min-docker-free-gb", type=float, default=18.0)
    parser.add_argument("--docker-space-check-interval", type=int, default=30)
    parser.add_argument("--benchmark-profile", default="auto")
    parser.add_argument("--config-file", type=Path, help="Optional mini-swe-agent config passed to the Docker runner.")
    parser.add_argument("--env-file", type=Path, default=Path("/wbl-fast/usrs/ee/code-swe-data/.env"))
    parser.add_argument("--keep-images", action="store_true", help="Do not remove task images after rows finish.")
    parser.add_argument("--rerun-existing", action="store_true", help="Do not skip workspaces that already contain model traces.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def count_manifest_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_chunks(manifest_tsv: Path, chunk_dir: Path, rows_per_job: int) -> int:
    if rows_per_job < 1:
        raise SystemExit("--rows-per-job must be positive")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for old in chunk_dir.glob("chunk-*.tsv"):
        old.unlink()
    chunk_index = 0
    rows: list[str] = []
    with manifest_tsv.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            rows.append(line)
            if len(rows) >= rows_per_job:
                (chunk_dir / f"chunk-{chunk_index:03d}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
                chunk_index += 1
                rows = []
    if rows:
        (chunk_dir / f"chunk-{chunk_index:03d}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        chunk_index += 1
    return chunk_index


def write_sbatch(args: argparse.Namespace, chunk_count: int, chunk_dir: Path) -> Path:
    if args.partition.lower().startswith("h") or "gpu" in args.partition.lower():
        raise SystemExit(f"refusing non-CPU partition: {args.partition}")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.cpus_per_task < args.workers:
        raise SystemExit("--cpus-per-task should be >= --workers for full-node Docker packing")
    run_root = args.run_root.resolve()
    log_dir = run_root / "slurm" / "docker"
    log_dir.mkdir(parents=True, exist_ok=True)
    script_path = run_root / "slurm" / f"{args.job_name}.docker-packed-array.sh"
    pydeps = require_pinned_minisweagent_overlay().resolve()
    pythonpath = f"{pydeps}:{REPO_ROOT}"
    array_spec = f"0-{chunk_count - 1}"
    if args.array_concurrency and args.array_concurrency < chunk_count:
        array_spec = f"{array_spec}%{args.array_concurrency}"
    remove_image_arg = "" if args.keep_images else " \\\n  --remove-image-after-run"
    skip_existing_arg = "" if args.rerun_existing else " \\\n  --skip-existing-result"
    config_file_arg = f" \\\n  --config-file {shell_quote(args.config_file.resolve())}" if args.config_file else ""
    script = f"""#!/usr/bin/env bash
#SBATCH -J {args.job_name}
#SBATCH -p {args.partition}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --time={args.time}
#SBATCH --array={array_spec}
#SBATCH --output={log_dir}/{args.job_name}.%A_%a.out
#SBATCH --error={log_dir}/{args.job_name}.%A_%a.err

set -euo pipefail

RUN_ROOT={shell_quote(run_root)}
REPO_ROOT={shell_quote(REPO_ROOT)}
PYTHON_BIN={shell_quote(args.python)}
PYDEPS={shell_quote(pydeps)}
RUNNER={shell_quote(RUNNER)}
CHUNK_DIR={shell_quote(chunk_dir.resolve())}
CHUNK="$(printf '%s/chunk-%03d.tsv' "$CHUNK_DIR" "$SLURM_ARRAY_TASK_ID")"
JOB_LABEL="$(printf '%s-%03d' {shell_quote(args.job_name)} "$SLURM_ARRAY_TASK_ID")"

export PYDEPS_OVERLAY="$PYDEPS"
export PYTHONPATH={shell_quote(pythonpath)}
export MSWEA_SILENT_STARTUP=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/wbl-fast/usrs/ee/code-swe-data/cache/hf
export XDG_CACHE_HOME=/wbl-fast/usrs/ee/code-swe-data/cache/xdg
export UV_CACHE_DIR=/wbl-fast/usrs/ee/code-swe-data/cache/uv
export PIP_CACHE_DIR=/wbl-fast/usrs/ee/code-swe-data/cache/pip

echo "node=$(hostname)"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "chunk=$CHUNK"
echo "job_label=$JOB_LABEL"
echo "workers={args.workers}"
echo "cpus_per_worker={args.cpus_per_worker}"
echo "memory_per_worker={args.memory_per_worker}"
echo "stagger_seconds={args.stagger_seconds}"
echo "repo_root=$REPO_ROOT"

"$PYTHON_BIN" "$RUNNER" \\
  --run-root "$RUN_ROOT" \\
  --manifest-tsv "$CHUNK" \\
  --job-name "$JOB_LABEL" \\
  --workers {args.workers} \\
  --cpus-per-worker {args.cpus_per_worker} \\
  --memory-per-worker {shell_quote(args.memory_per_worker)} \\
  --stagger-seconds {args.stagger_seconds} \\
  --temperature {args.temperature} \\
  --max-tokens {args.max_tokens} \\
  --reasoning-effort {shell_quote(args.reasoning_effort)} \\
  --model-timeout {args.model_timeout} \\
  --agent-wall-time-limit {args.agent_wall_time_limit} \\
  --command-timeout {args.command_timeout} \\
  --max-concurrent-pulls {args.max_concurrent_pulls} \\
  --min-docker-free-gb {args.min_docker_free_gb} \\
  --docker-space-check-interval {args.docker_space_check_interval} \\
  --benchmark-profile {shell_quote(args.benchmark_profile)} \\
  --env-file {shell_quote(args.env_file)}{config_file_arg}{skip_existing_arg}{remove_image_arg}
"""
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def main() -> None:
    args = parse_args()
    n_rows = count_manifest_rows(args.manifest_tsv)
    if n_rows < 1:
        raise SystemExit(f"manifest is empty: {args.manifest_tsv}")
    chunk_dir = args.run_root / "manifest" / f"{args.job_name}-chunks-{args.rows_per_job}"
    chunk_count = write_chunks(args.manifest_tsv, chunk_dir, args.rows_per_job)
    script_path = write_sbatch(args, chunk_count, chunk_dir)
    command = ["sbatch", "--parsable", str(script_path)]
    print(f"script={script_path}")
    print(f"rows={n_rows}")
    print(f"rows_per_job={args.rows_per_job}")
    print(f"array_elements={chunk_count}")
    print("command=" + " ".join(shlex.quote(x) for x in command))
    if args.dry_run:
        return
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        env=os.environ.copy(),
    )
    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise SystemExit("sbatch succeeded but did not print a job id")
    job_id = output_lines[-1]
    (args.run_root / "slurm" / f"{args.job_name}.jobid").write_text(job_id + "\n", encoding="utf-8")
    print(f"job_id={job_id}")


if __name__ == "__main__":
    main()
