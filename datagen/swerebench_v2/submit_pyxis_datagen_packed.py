#!/usr/bin/env python3
"""Submit packed CPU-only Slurm/Pyxis mini-swe-agent data-generation arrays."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--partition", default="m7i-cpu2")
    parser.add_argument("--array-concurrency", type=int, default=0, help="0 or >= array size omits the Slurm % throttle.")
    parser.add_argument("--rows-per-job", type=int, default=12)
    parser.add_argument("--parallel-rows", type=int, default=2)
    parser.add_argument("--cpus-per-row", type=int, default=8)
    parser.add_argument(
        "--stagger-seconds",
        type=float,
        default=0.0,
        help="Sleep this many seconds between launching row containers inside one packed array element.",
    )
    parser.add_argument("--mem", default="56G")
    parser.add_argument("--tmp", default="")
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--config-file",
        type=Path,
        help=(
            "Mini-swe-agent config to use for every row. By default the launcher "
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
    parser.add_argument("--enroot-config-path", type=Path)
    parser.add_argument("--docker-user", default="")
    parser.add_argument("--container-source", choices=("registry", "dockerd"), default="registry")
    parser.add_argument("--auth-first", action="store_true")
    parser.add_argument("--docker-login-from-enroot", action="store_true")
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def count_manifest_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


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


def write_array_script(args: argparse.Namespace, n_rows: int) -> Path:
    if args.rows_per_job < 1:
        raise SystemExit("--rows-per-job must be positive")
    if args.parallel_rows < 1:
        raise SystemExit("--parallel-rows must be positive")
    if args.parallel_rows > args.rows_per_job:
        raise SystemExit("--parallel-rows must be <= --rows-per-job")

    run_root = args.run_root.resolve()
    manifest_tsv = args.manifest_tsv.resolve()
    script_path = run_root / "slurm" / f"{args.job_name}.packed-array.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = (run_root / "slurm" / "pyxis").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    repo_root = REPO_ROOT
    shared_root = REPO_ROOT.parent
    cache_root = Path("/wbl-fast/usrs/ee/code-swe-data/cache")
    pydeps_overlay = require_pinned_minisweagent_overlay().resolve()
    pythonpath_parts: list[Path] = [
        pydeps_overlay,
        repo_root,
    ]
    pythonpath = ":".join(str(path) for path in pythonpath_parts)
    ca_bundle = pydeps_overlay / "certifi" / "cacert.pem"
    code_commit = datagen_code_commit()
    command_timeout_arg = (
        f" \\\n          --command-timeout {args.command_timeout}"
        if args.command_timeout is not None
        else ""
    )
    uses_updated_alignment = getattr(args, "uses_updated_alignment", "true")
    eligible_for_controlled_comparison = (
        "true" if getattr(args, "eligible_for_controlled_comparison", False) else "false"
    )
    reason_excluded_from_comparison = getattr(args, "reason_excluded_from_comparison", "")
    array_size = math.ceil(n_rows / args.rows_per_job)
    if args.array_concurrency and args.array_concurrency < array_size:
        array_spec = f"0-{array_size - 1}%{args.array_concurrency}"
    else:
        array_spec = f"0-{array_size - 1}"

    script = f"""#!/usr/bin/env bash
#SBATCH -J {args.job_name}
#SBATCH -p {args.partition}
#SBATCH -N 1
#SBATCH -n {args.parallel_rows}
#SBATCH --cpus-per-task={args.cpus_per_row}
#SBATCH --mem={args.mem}
{f"#SBATCH --tmp={args.tmp}" if args.tmp else ""}
#SBATCH --time={args.time}
#SBATCH --array={array_spec}
#SBATCH --output={log_dir}/{args.job_name}.%A_%a.controller.out
#SBATCH --error={log_dir}/{args.job_name}.%A_%a.controller.err

set -euo pipefail

MANIFEST={shell_quote(manifest_tsv)}
TOTAL_ROWS={n_rows}
ROWS_PER_JOB={args.rows_per_job}
PARALLEL_ROWS={args.parallel_rows}
CPUS_PER_ROW={args.cpus_per_row}
STAGGER_SECONDS={shell_quote(str(args.stagger_seconds))}
SUBMIT_DIR="${{SLURM_SUBMIT_DIR:-$PWD}}"
LOG_DIR={shell_quote(log_dir)}
SHARED_ROOT={shell_quote(shared_root)}
REPO_ROOT={shell_quote(repo_root)}
CACHE_ROOT={shell_quote(cache_root)}
AUTH_ENROOT_CONFIG_PATH={shell_quote(str(args.enroot_config_path) if args.enroot_config_path else "")}
DOCKER_USER={shell_quote(args.docker_user)}
CONTAINER_SOURCE={shell_quote(args.container_source)}
AUTH_FIRST={1 if args.auth_first else 0}
DOCKER_LOGIN_FROM_ENROOT={1 if args.docker_login_from_enroot else 0}
PYTHON_BIN={shell_quote(args.python)}
DRIVER={shell_quote(DRIVER)}
FAILURE_WRITER={shell_quote(FAILURE_WRITER)}
CUSTOM_CONFIG_FILE={shell_quote(str(args.config_file) if args.config_file else "")}
SWEBENCH_MULTILINGUAL_CONFIG={shell_quote(DEFAULT_SWEBENCH_MULTILINGUAL_CONFIG)}
DEEPSWE_CONFIG={shell_quote(DEFAULT_DEEPSWE_CONFIG)}
DATAGEN_STRICT_CONFIG={shell_quote(DEFAULT_DATAGEN_STRICT_CONFIG)}
BENCHMARK_PROFILE_OVERRIDE={shell_quote(args.benchmark_profile)}
ENV_FILE={shell_quote(args.env_file)}
USES_UPDATED_ALIGNMENT={shell_quote(uses_updated_alignment)}
ELIGIBLE_FOR_CONTROLLED_COMPARISON={shell_quote(eligible_for_controlled_comparison)}
REASON_EXCLUDED_FROM_COMPARISON={shell_quote(reason_excluded_from_comparison)}

export HF_HOME={shell_quote(cache_root / "hf")}
export XDG_CACHE_HOME={shell_quote(cache_root / "xdg")}
export UV_CACHE_DIR={shell_quote(cache_root / "uv")}
export PIP_CACHE_DIR={shell_quote(cache_root / "pip")}
export PYDEPS_OVERLAY={shell_quote(pydeps_overlay)}
export PYTHONPATH={shell_quote(pythonpath)}
export DATAGEN_CODE_COMMIT={shell_quote(code_commit)}
export SSL_CERT_FILE={shell_quote(ca_bundle)}
export REQUESTS_CA_BUNDLE={shell_quote(ca_bundle)}
export CURL_CA_BUNDLE={shell_quote(ca_bundle)}
export MSWEA_COST_TRACKING=ignore_errors
export MSWEA_SILENT_STARTUP=1
export ENROOT_REMAP_ROOT=yes
export PYTHONDONTWRITEBYTECODE=1

echo "node=$(hostname)"
echo "partition={args.partition}"
echo "array_task_id=${{SLURM_ARRAY_TASK_ID}}"
echo "rows_per_job=$ROWS_PER_JOB"
echo "parallel_rows=$PARALLEL_ROWS"
echo "stagger_seconds=$STAGGER_SECONDS"
echo "mini_swe_agent_git_sha={MINI_SWE_AGENT_GIT_SHA}"
echo "mini_swe_agent_overlay_env={MINI_SWE_AGENT_OVERLAY_ENV}"
echo "mini_swe_agent_pydeps_overlay=$PYDEPS_OVERLAY"
echo "ssl_cert_file=$SSL_CERT_FILE"

run_row() {{
  local row_number="$1"
  local row
  row="$(sed -n "${{row_number}}p" "$MANIFEST")"
  if [[ -z "$row" ]]; then
    echo "empty manifest row $row_number" >&2
    return 2
  fi

  local IDX ROLLOUT_ID INSTANCE_ID TASK_DIR WORKSPACE IMAGE MODEL LITELLM_MODEL API_KEY_ENV API_BASE EXTRA_BODY_JSON DIFFICULTY LANGUAGE STYLE REPO OUTSIDE_ORIGINAL_HIGH_QUALITY_SET
  IFS=$'\\t' read -r IDX ROLLOUT_ID INSTANCE_ID TASK_DIR WORKSPACE IMAGE MODEL LITELLM_MODEL API_KEY_ENV API_BASE EXTRA_BODY_JSON DIFFICULTY LANGUAGE STYLE REPO OUTSIDE_ORIGINAL_HIGH_QUALITY_SET <<<"$row"
  OUTSIDE_ORIGINAL_HIGH_QUALITY_SET="${{OUTSIDE_ORIGINAL_HIGH_QUALITY_SET:-false}}"

  local row_log_id="${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}_row${{row_number}}"
  local STDOUT_LOG="$LOG_DIR/{args.job_name}.${{row_log_id}}.out"
  local STDERR_LOG="$LOG_DIR/{args.job_name}.${{row_log_id}}.err"
  local PULL_LOCK="$LOG_DIR/{args.job_name}.${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.docker-pull.lock"

  {{
    set -u
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
    local BENCHMARK_PROFILE="$BENCHMARK_PROFILE_OVERRIDE"
    if [[ "$BENCHMARK_PROFILE" == "auto" ]]; then
      case "$STYLE" in
        original|swe_rebench)
          BENCHMARK_PROFILE="swebench-multilingual"
          ;;
        deepswe|rewritten|planned)
          BENCHMARK_PROFILE="deepswe"
          ;;
        *)
          BENCHMARK_PROFILE="datagen-strict"
          ;;
      esac
    fi
    local CONFIG_FILE
    if [[ -n "$CUSTOM_CONFIG_FILE" ]]; then
      CONFIG_FILE="$CUSTOM_CONFIG_FILE"
    else
      case "$BENCHMARK_PROFILE" in
        swebench-multilingual)
          CONFIG_FILE="$SWEBENCH_MULTILINGUAL_CONFIG"
          ;;
        deepswe)
          CONFIG_FILE="$DEEPSWE_CONFIG"
          ;;
        datagen-strict)
          CONFIG_FILE="$DATAGEN_STRICT_CONFIG"
          ;;
        *)
          echo "unknown benchmark profile: $BENCHMARK_PROFILE" >&2
          return 2
          ;;
      esac
    fi

    local PYXIS_IMAGE_REF="${{IMAGE#docker.io/}}"
    PYXIS_IMAGE_REF="${{PYXIS_IMAGE_REF#docker://}}"
    local DOCKER_PULL_REF="$PYXIS_IMAGE_REF"
    local ANON_PYXIS_IMAGE="registry-1.docker.io#${{PYXIS_IMAGE_REF}}"
    local AUTH_PYXIS_IMAGE="$ANON_PYXIS_IMAGE"
    if [[ -n "$DOCKER_USER" ]]; then
      AUTH_PYXIS_IMAGE="${{DOCKER_USER}}@registry-1.docker.io#${{PYXIS_IMAGE_REF}}"
    fi
    local DOCKERD_PYXIS_IMAGE="dockerd://${{PYXIS_IMAGE_REF}}"
    local LAST_PYXIS_IMAGE="$ANON_PYXIS_IMAGE"
    local ANON_ENROOT_CONFIG_PATH="$WORKSPACE/enroot-anonymous"

    export HOME="$WORKSPACE/home"
    export ENROOT_TEMP_PATH="$WORKSPACE/enroot-tmp"
    mkdir -p "$HOME" "$ENROOT_TEMP_PATH" "$ANON_ENROOT_CONFIG_PATH"

    set -a
    source "$ENV_FILE"
    set +a
    if [[ "$API_KEY_ENV" == "DEEPSEEK_API_KEY" && -n "${{DEEPSEEK_API_KEY:-}}" ]]; then
      export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
    fi
    if [[ -n "$API_BASE" ]]; then
      export OPENAI_BASE_URL="$API_BASE"
      export OPENAI_API_BASE="$API_BASE"
    fi

    local MOUNTS="$SHARED_ROOT:$SHARED_ROOT:rw,$WORKSPACE:/workspace:rw,$TASK_DIR/tests:/tests:ro"

    echo "row_number=$row_number"
    echo "node=$(hostname)"
    echo "instance_id=$INSTANCE_ID"
    echo "rollout_id=$ROLLOUT_ID"
    echo "model=$MODEL"
    echo "image=$IMAGE"
    echo "instruction_style=$STYLE"
    echo "benchmark_profile=$BENCHMARK_PROFILE"
    echo "mini_swe_agent_config=$CONFIG_FILE"
    echo "container_source=$CONTAINER_SOURCE"
    echo "workspace=$WORKSPACE"
    echo "ssl_cert_file=$SSL_CERT_FILE"

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
        local refreshed_auth=0
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
        if [[ "$attempt" -lt 6 ]] && grep -Eiq '429|502|Bad Gateway|Too Many Requests|rate limit|TLS handshake timeout|connection reset|unexpected EOF|EOF|failed to do request|temporarily unavailable|timeout' "$WORKSPACE/docker-pull.log"; then
          local sleep_seconds=$((20 * attempt + RANDOM % 20))
          echo "docker_pull_retry_sleep=$sleep_seconds" | tee -a "$WORKSPACE/docker-pull.log"
          sleep "$sleep_seconds"
          continue
        fi
        break
      done
      echo "docker_pull=failed status=$status"
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
      srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_ROW" \\
        --container-image="$attempt_image" \\
        --container-writable \\
        --container-remap-root \\
        --no-container-mount-home \\
        --container-mounts="$MOUNTS" \\
        --container-env=OPENAI_API_KEY,DEEPSEEK_API_KEY,OPENROUTER_API_KEY,OPENAI_BASE_URL,OPENAI_API_BASE,HF_HOME,XDG_CACHE_HOME,UV_CACHE_DIR,PIP_CACHE_DIR,PYTHONPATH,DATAGEN_CODE_COMMIT,SSL_CERT_FILE,REQUESTS_CA_BUNDLE,CURL_CA_BUNDLE,HOME,MSWEA_COST_TRACKING,MSWEA_SILENT_STARTUP,PYTHONDONTWRITEBYTECODE \\
        "$PYTHON_BIN" "$DRIVER" \\
          --task-dir "$TASK_DIR" \\
          --workspace /workspace \\
          --config-file "$CONFIG_FILE" \\
          --benchmark-profile "$BENCHMARK_PROFILE" \\
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
          --outside-original-high-quality-set "$OUTSIDE_ORIGINAL_HIGH_QUALITY_SET" \\
          --temperature {args.temperature} \\
          --max-tokens {args.max_tokens} \\
          --reasoning-effort {shell_quote(args.reasoning_effort)} \\
          --model-timeout {args.model_timeout} \\
          --agent-wall-time-limit {args.agent_wall_time_limit} \\
          --uses-updated-alignment "$USES_UPDATED_ALIGNMENT" \\
          --eligible-for-controlled-comparison "$ELIGIBLE_FOR_CONTROLLED_COMPARISON" \\
          --reason-excluded-from-comparison "$REASON_EXCLUDED_FROM_COMPARISON"{command_timeout_arg}
    }}

    set +e
    local STATUS=0
    if [[ "$CONTAINER_SOURCE" == "dockerd" ]]; then
      LAST_PYXIS_IMAGE="$DOCKERD_PYXIS_IMAGE"
      if command -v flock >/dev/null 2>&1; then
        (
          flock 9
          docker_pull_image
        ) 9>"$PULL_LOCK"
        STATUS=$?
      else
        docker_pull_image
        STATUS=$?
      fi
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
      "$PYTHON_BIN" "$FAILURE_WRITER" \\
        --workspace "$WORKSPACE" \\
        --instance-id "$INSTANCE_ID" \\
        --rollout-id "$ROLLOUT_ID" \\
        --model "$MODEL" \\
        --litellm-model "$LITELLM_MODEL" \\
        --instruction-style "$STYLE" \\
        --benchmark-profile "$BENCHMARK_PROFILE" \\
        --difficulty "$DIFFICULTY" \\
        --language "$LANGUAGE" \\
        --repo "$REPO" \\
        --outside-original-high-quality-set "$OUTSIDE_ORIGINAL_HIGH_QUALITY_SET" \\
        --mini-swe-agent-config-file "$CONFIG_FILE" \\
        --uses-updated-alignment "$USES_UPDATED_ALIGNMENT" \\
        --eligible-for-controlled-comparison "$ELIGIBLE_FOR_CONTROLLED_COMPARISON" \\
        --reason-excluded-from-comparison "$REASON_EXCLUDED_FROM_COMPARISON" \\
        --task-dir "$TASK_DIR" \\
        --image "$IMAGE" \\
        --pyxis-image "$LAST_PYXIS_IMAGE" \\
        --exit-status "$STATUS" \\
        --stdout-log "$STDOUT_LOG" \\
        --stderr-log "$STDERR_LOG"
    fi
    echo "row_status=$STATUS"
    return 0
  }} >"$STDOUT_LOG" 2>"$STDERR_LOG"
}}

start_row=$((SLURM_ARRAY_TASK_ID * ROWS_PER_JOB + 1))
end_row=$((start_row + ROWS_PER_JOB - 1))
if [[ "$end_row" -gt "$TOTAL_ROWS" ]]; then
  end_row="$TOTAL_ROWS"
fi

active=0
for row_number in $(seq "$start_row" "$end_row"); do
  run_row "$row_number" &
  active=$((active + 1))
  if [[ "$active" -ge "$PARALLEL_ROWS" ]]; then
    wait -n || true
    active=$((active - 1))
  elif [[ "$STAGGER_SECONDS" != "0" && "$STAGGER_SECONDS" != "0.0" && "$row_number" -lt "$end_row" ]]; then
    sleep "$STAGGER_SECONDS"
  fi
done
wait || true
echo "packed_array_task_done start_row=$start_row end_row=$end_row"
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
    print(f"rows_per_job={args.rows_per_job}")
    print(f"array_elements={math.ceil(n_rows / args.rows_per_job)}")
    print("command=" + " ".join(shlex.quote(x) for x in command))
    if args.dry_run:
        return
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            env=os.environ.copy(),
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stdout.strip() or str(exc)) from exc
    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise SystemExit("sbatch succeeded but did not print a job id")
    job_id = output_lines[-1]
    (args.run_root / "slurm" / f"{args.job_name}.jobid").write_text(job_id + "\n", encoding="utf-8")
    print(f"job_id={job_id}")


if __name__ == "__main__":
    main()
