#!/usr/bin/env python3
"""Submit CPU-only Slurm/Pyxis mini-swe-agent data-generation arrays."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from eval.minisweagent_pin import (
    MINI_SWE_AGENT_GIT_SHA,
    MINI_SWE_AGENT_OVERLAY_ENV,
    require_pinned_minisweagent_overlay,
)
from eval.paths import REPO_ROOT


DEFAULT_PYTHON = Path("/wbl-fast/usrs/ee/code-swe-data/runtime/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12")
DEFAULT_CONFIG = REPO_ROOT / "datagen" / "swerebench_v2" / "minisweagent_datagen_strict.yaml"
DEFAULT_ENV_FILE = Path("/wbl-fast/usrs/ee/code-swe-data/.env")
DRIVER = REPO_ROOT / "datagen" / "swerebench_v2" / "pyxis_miniswe_agent_driver.py"
FAILURE_WRITER = REPO_ROOT / "datagen" / "swerebench_v2" / "write_pyxis_failure_result.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--partition", default="m7i-cpu2")
    parser.add_argument("--array-concurrency", type=int, default=100)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem", default="16G")
    parser.add_argument(
        "--tmp",
        default="",
        help=(
            "Temporary disk requested per Slurm task, in MB. Empty by default because "
            "some CPU partitions report BadConstraints for nonzero tmp requests."
        ),
    )
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--enroot-config-path",
        type=Path,
        help=(
            "Credential/config directory for Pyxis/enroot image imports. "
            "When --docker-user is set, generated jobs try an anonymous import "
            "first and only use this path on the authenticated retry."
        ),
    )
    parser.add_argument(
        "--docker-user",
        default="",
        help="Docker Hub username to include in Pyxis image URIs; token stays in enroot credentials.",
    )
    parser.add_argument(
        "--container-source",
        choices=("registry", "dockerd"),
        default="registry",
        help=(
            "Use registry imports directly through Pyxis, or pre-pull with the "
            "node Docker daemon and run Pyxis from dockerd://. The dockerd path "
            "avoids rootless Enroot layer-permission failures on some images."
        ),
    )
    parser.add_argument(
        "--auth-first",
        action="store_true",
        help="Try the authenticated Docker Hub path before anonymous pulls/imports.",
    )
    parser.add_argument(
        "--docker-login-from-enroot",
        action="store_true",
        help="For --container-source=dockerd, run docker login using the enroot .credentials file.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--model-timeout", type=int, default=600)
    parser.add_argument("--agent-wall-time-limit", type=int, default=2700)
    parser.add_argument("--command-timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def count_manifest_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def write_array_script(args: argparse.Namespace, n_rows: int) -> Path:
    script_path = args.run_root / "slurm" / f"{args.job_name}.array.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.run_root / "slurm" / "pyxis"
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path("/wbl-fast/usrs/ee/code-swe-data/cache")
    repo_root = REPO_ROOT
    shared_root = REPO_ROOT.parent
    pydeps_overlay = require_pinned_minisweagent_overlay()
    pydantic_stack = shared_root / "runtime" / "manual-pydeps" / "pydantic-stack-clean"
    pythonpath_parts = [
        *([pydantic_stack] if pydantic_stack.exists() else []),
        pydeps_overlay,
        repo_root / ".venv" / "lib" / "python3.12" / "site-packages",
        repo_root,
    ]
    pythonpath = ":".join(str(path) for path in pythonpath_parts)

    script = f"""#!/usr/bin/env bash
#SBATCH -J {args.job_name}
#SBATCH -p {args.partition}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
{f"#SBATCH --tmp={args.tmp}" if args.tmp else ""}
#SBATCH --time={args.time}
#SBATCH --array=0-{n_rows - 1}%{args.array_concurrency}
#SBATCH --output={log_dir}/{args.job_name}.%A_%a.out
#SBATCH --error={log_dir}/{args.job_name}.%A_%a.err

set -euo pipefail

MANIFEST={shell_quote(args.manifest_tsv)}
ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
if [[ -z "$ROW" ]]; then
  echo "empty manifest row for task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

IFS=$'\\t' read -r IDX ROLLOUT_ID INSTANCE_ID TASK_DIR WORKSPACE IMAGE MODEL LITELLM_MODEL API_KEY_ENV API_BASE EXTRA_BODY_JSON DIFFICULTY LANGUAGE STYLE REPO <<<"$ROW"
SUBMIT_DIR="${{SLURM_SUBMIT_DIR:-$PWD}}"
if [[ "$TASK_DIR" != /* ]]; then
  TASK_DIR="$SUBMIT_DIR/$TASK_DIR"
fi
if [[ "$WORKSPACE" != /* ]]; then
  WORKSPACE="$SUBMIT_DIR/$WORKSPACE"
fi
mkdir -p "$WORKSPACE"
if [[ "$API_BASE" == "-" ]]; then
  API_BASE=""
fi
if [[ "$EXTRA_BODY_JSON" == "-" ]]; then
  EXTRA_BODY_JSON=""
fi
PYXIS_IMAGE_REF="${{IMAGE#docker.io/}}"
PYXIS_IMAGE_REF="${{PYXIS_IMAGE_REF#docker://}}"
DOCKER_PULL_REF="$PYXIS_IMAGE_REF"
ANON_PYXIS_IMAGE="registry-1.docker.io#${{PYXIS_IMAGE_REF}}"
DOCKER_USER={shell_quote(args.docker_user)}
AUTH_ENROOT_CONFIG_PATH={shell_quote(str(args.enroot_config_path) if args.enroot_config_path else "")}
AUTH_PYXIS_IMAGE="$ANON_PYXIS_IMAGE"
if [[ -n "$DOCKER_USER" ]]; then
  AUTH_PYXIS_IMAGE="${{DOCKER_USER}}@registry-1.docker.io#${{PYXIS_IMAGE_REF}}"
fi
DOCKERD_PYXIS_IMAGE="dockerd://${{PYXIS_IMAGE_REF}}"
CONTAINER_SOURCE={shell_quote(args.container_source)}
AUTH_FIRST={1 if args.auth_first else 0}
DOCKER_LOGIN_FROM_ENROOT={1 if args.docker_login_from_enroot else 0}
PYXIS_IMAGE="$ANON_PYXIS_IMAGE"
LAST_PYXIS_IMAGE="$PYXIS_IMAGE"
ANON_ENROOT_CONFIG_PATH="$WORKSPACE/enroot-anonymous"
STDOUT_LOG={log_dir}/{args.job_name}.${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out
STDERR_LOG={log_dir}/{args.job_name}.${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.err

export HF_HOME={shell_quote(cache_root / "hf")}
export XDG_CACHE_HOME={shell_quote(cache_root / "xdg")}
export UV_CACHE_DIR={shell_quote(cache_root / "uv")}
export PIP_CACHE_DIR={shell_quote(cache_root / "pip")}
export PYDEPS_OVERLAY={shell_quote(pydeps_overlay)}
export PYTHONPATH={shell_quote(pythonpath)}
export HOME="$WORKSPACE/home"
export ENROOT_TEMP_PATH="$WORKSPACE/enroot-tmp"
export MSWEA_COST_TRACKING=ignore_errors
export MSWEA_SILENT_STARTUP=1
export ENROOT_REMAP_ROOT=yes
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HOME"
mkdir -p "$ENROOT_TEMP_PATH"
mkdir -p "$ANON_ENROOT_CONFIG_PATH"

set -a
source {shell_quote(args.env_file)}
set +a

if [[ "$API_KEY_ENV" == "DEEPSEEK_API_KEY" && -n "${{DEEPSEEK_API_KEY:-}}" ]]; then
  export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
fi
if [[ -n "$API_BASE" ]]; then
  export OPENAI_BASE_URL="$API_BASE"
  export OPENAI_API_BASE="$API_BASE"
fi

MOUNTS="{shared_root}:{shared_root}:rw,$WORKSPACE:/workspace:rw,$TASK_DIR/tests:/tests:ro"

echo "node=$(hostname)"
echo "partition={args.partition}"
echo "instance_id=$INSTANCE_ID"
echo "rollout_id=$ROLLOUT_ID"
echo "model=$MODEL"
echo "image=$IMAGE"
echo "container_source=$CONTAINER_SOURCE"
echo "auth_first=$AUTH_FIRST"
echo "mini_swe_agent_git_sha={MINI_SWE_AGENT_GIT_SHA}"
echo "mini_swe_agent_overlay_env={MINI_SWE_AGENT_OVERLAY_ENV}"
echo "mini_swe_agent_pydeps_overlay=$PYDEPS_OVERLAY"
echo "anonymous_pyxis_image=$ANON_PYXIS_IMAGE"
echo "authenticated_pyxis_image=$AUTH_PYXIS_IMAGE"
echo "dockerd_pyxis_image=$DOCKERD_PYXIS_IMAGE"
echo "workspace=$WORKSPACE"

docker_login_from_enroot() {{
  local credentials_file="$AUTH_ENROOT_CONFIG_PATH/.credentials"
  if [[ -z "$DOCKER_USER" ]]; then
    return 0
  fi
  if [[ -z "$AUTH_ENROOT_CONFIG_PATH" || ! -r "$credentials_file" ]]; then
    echo "docker_login_skipped=missing_enroot_credentials"
    return 1
  fi
  local password
  password="$(
    awk -v user="$DOCKER_USER" '
      $1 == "machine" && $2 == "registry-1.docker.io" {{
        login = ""; password = "";
        for (i = 1; i <= NF; i++) {{
          if ($i == "login") login = $(i + 1);
          if ($i == "password") password = $(i + 1);
        }}
        if (login == user && password != "") {{
          print password;
          exit;
        }}
      }}
    ' "$credentials_file"
  )"
  if [[ -z "$password" ]]; then
    echo "docker_login_skipped=no_matching_credential"
    return 1
  fi
  local auth
  auth="$(printf '%s:%s' "$DOCKER_USER" "$password" | base64 | tr -d '\\n')"
  unset password
  mkdir -p "$HOME/.docker"
  umask 077
  cat >"$HOME/.docker/config.json" <<JSON
{{"auths":{{"https://index.docker.io/v1/":{{"auth":"$auth"}},"registry-1.docker.io":{{"auth":"$auth"}},"docker.io":{{"auth":"$auth"}}}}}}
JSON
  unset auth
  echo "docker_auth_config=ok" >"$WORKSPACE/docker-login.log"
  echo "docker_login=ok"
}}

docker_pull_image() {{
  local status=0
  local logged_in=0
  if [[ "$AUTH_FIRST" -eq 1 && "$DOCKER_LOGIN_FROM_ENROOT" -eq 1 && -n "$DOCKER_USER" ]]; then
    docker_login_from_enroot || true
    logged_in=1
  fi
  if docker image inspect "$DOCKER_PULL_REF" >/dev/null 2>&1; then
    echo "docker_image_present=$DOCKER_PULL_REF"
    return 0
  fi
  : >"$WORKSPACE/docker-pull.log"
  for attempt in 1 2 3 4 5 6; do
    refreshed_auth=0
    echo "docker_pull_attempt=$attempt" | tee -a "$WORKSPACE/docker-pull.log"
    docker pull "$DOCKER_PULL_REF" >>"$WORKSPACE/docker-pull.log" 2>&1
    status=$?
    if [[ "$status" -eq 0 ]]; then
      echo "docker_pull=ok"
      return 0
    fi
    if [[ "$logged_in" -eq 0 && "$DOCKER_LOGIN_FROM_ENROOT" -eq 1 && -n "$DOCKER_USER" ]]; then
      echo "docker_pull_initial_failed status=$status; refreshing auth config" | tee -a "$WORKSPACE/docker-pull.log"
      docker_login_from_enroot || true
      logged_in=1
      refreshed_auth=1
    fi
    if [[ "$refreshed_auth" -eq 1 && "$attempt" -lt 6 ]]; then
      echo "docker_pull_retry_after_auth=1" | tee -a "$WORKSPACE/docker-pull.log"
      continue
    fi
    if [[ "$attempt" -lt 6 ]] && grep -Eiq '429|Too Many Requests|rate limit|TLS handshake timeout|connection reset|unexpected EOF|temporarily unavailable|timeout' "$WORKSPACE/docker-pull.log"; then
      sleep_seconds=$((20 * attempt + RANDOM % 20))
      echo "docker_pull_retry_sleep=$sleep_seconds" | tee -a "$WORKSPACE/docker-pull.log"
      sleep "$sleep_seconds"
      continue
    fi
    break
  done
  if [[ "$status" -eq 0 ]]; then
    echo "docker_pull=ok"
  else
    echo "docker_pull=failed status=$status"
  fi
  return "$status"
}}

run_agent_attempt() {{
  local attempt_label="$1"
  local attempt_image="$2"
  local attempt_enroot_config="$3"
  LAST_PYXIS_IMAGE="$attempt_image"
  echo "pyxis_attempt=$attempt_label"
  echo "pyxis_image=$attempt_image"
  if [[ -n "$attempt_enroot_config" ]]; then
    export ENROOT_CONFIG_PATH="$attempt_enroot_config"
  else
    unset ENROOT_CONFIG_PATH
  fi
  srun --container-image="$attempt_image" \\
  --container-writable \\
  --container-remap-root \\
  --no-container-mount-home \\
  --container-mounts="$MOUNTS" \\
  --container-env=OPENAI_API_KEY,DEEPSEEK_API_KEY,OPENROUTER_API_KEY,OPENAI_BASE_URL,OPENAI_API_BASE,HF_HOME,XDG_CACHE_HOME,UV_CACHE_DIR,PIP_CACHE_DIR,PYTHONPATH,HOME,MSWEA_COST_TRACKING,MSWEA_SILENT_STARTUP,PYTHONDONTWRITEBYTECODE \\
  {shell_quote(args.python)} {shell_quote(DRIVER)} \\
    --task-dir "$TASK_DIR" \\
    --workspace /workspace \\
    --config-file {shell_quote(args.config_file)} \\
    --instance-id "$INSTANCE_ID" \\
    --rollout-id "$ROLLOUT_ID" \\
    --model "$MODEL" \\
    --litellm-model "$LITELLM_MODEL" \\
    --api-key-env "$API_KEY_ENV" \\
    --api-base "$API_BASE" \\
    --extra-body-json "$EXTRA_BODY_JSON" \\
    --instruction-style "$STYLE" \\
    --difficulty "$DIFFICULTY" \\
    --language "$LANGUAGE" \\
    --repo "$REPO" \\
    --temperature {args.temperature} \\
    --max-tokens {args.max_tokens} \\
    --reasoning-effort {shell_quote(args.reasoning_effort)} \\
    --model-timeout {args.model_timeout} \\
    --agent-wall-time-limit {args.agent_wall_time_limit} \\
    --command-timeout {args.command_timeout}
}}

set +e
if [[ "$CONTAINER_SOURCE" == "dockerd" ]]; then
  LAST_PYXIS_IMAGE="$DOCKERD_PYXIS_IMAGE"
  docker_pull_image
  STATUS=$?
  if [[ "$STATUS" -eq 0 ]]; then
    run_agent_attempt dockerd "$DOCKERD_PYXIS_IMAGE" ""
    STATUS=$?
  fi
elif [[ "$AUTH_FIRST" -eq 1 && -n "$DOCKER_USER" ]]; then
  run_agent_attempt authenticated "$AUTH_PYXIS_IMAGE" "$AUTH_ENROOT_CONFIG_PATH"
  STATUS=$?
  if [[ "$STATUS" -ne 0 && ! -f "$WORKSPACE/result.json" ]]; then
    echo "authenticated import/run failed before result.json; retrying anonymous"
    run_agent_attempt anonymous "$ANON_PYXIS_IMAGE" "$ANON_ENROOT_CONFIG_PATH"
    STATUS=$?
  fi
else
  run_agent_attempt anonymous "$ANON_PYXIS_IMAGE" "$ANON_ENROOT_CONFIG_PATH"
  STATUS=$?
  if [[ "$STATUS" -ne 0 && ! -f "$WORKSPACE/result.json" && -n "$DOCKER_USER" ]]; then
    echo "anonymous import/run failed before result.json; retrying with Docker Hub credentials"
    run_agent_attempt authenticated "$AUTH_PYXIS_IMAGE" "$AUTH_ENROOT_CONFIG_PATH"
    STATUS=$?
  fi
fi
set -e

if [[ "$STATUS" -ne 0 && ! -f "$WORKSPACE/result.json" ]]; then
  {shell_quote(args.python)} {shell_quote(FAILURE_WRITER)} \\
    --workspace "$WORKSPACE" \\
    --instance-id "$INSTANCE_ID" \\
    --rollout-id "$ROLLOUT_ID" \\
    --model "$MODEL" \\
    --litellm-model "$LITELLM_MODEL" \\
    --instruction-style "$STYLE" \\
    --difficulty "$DIFFICULTY" \\
    --language "$LANGUAGE" \\
    --repo "$REPO" \\
    --task-dir "$TASK_DIR" \\
    --image "$IMAGE" \\
    --pyxis-image "$LAST_PYXIS_IMAGE" \\
    --exit-status "$STATUS" \\
    --stdout-log "$STDOUT_LOG" \\
    --stderr-log "$STDERR_LOG"
fi
exit "$STATUS"
"""
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def main() -> None:
    args = parse_args()
    if args.partition.lower().startswith("h") or "gpu" in args.partition.lower():
        raise SystemExit(f"refusing non-CPU partition: {args.partition}")
    n_rows = count_manifest_rows(args.manifest_tsv)
    if n_rows < 1:
        raise SystemExit(f"manifest is empty: {args.manifest_tsv}")
    script_path = write_array_script(args, n_rows)
    command = ["sbatch", "--parsable", str(script_path)]
    print(f"script={script_path}")
    print(f"rows={n_rows}")
    print("command=" + " ".join(shlex.quote(x) for x in command))
    if args.dry_run:
        return
    env = os.environ.copy()
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, env=env)
    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise SystemExit("sbatch succeeded but did not print a job id")
    job_id = output_lines[-1]
    (args.run_root / "slurm" / f"{args.job_name}.jobid").write_text(job_id + "\n", encoding="utf-8")
    print(f"job_id={job_id}")


if __name__ == "__main__":
    main()
