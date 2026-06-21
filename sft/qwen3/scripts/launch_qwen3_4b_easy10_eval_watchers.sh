#!/usr/bin/env bash
# Launch tmux checkpoint watchers that submit Qwen3-4B easy-10 n-trial evals.
#
# Expected inputs:
#   RUN_V0_DIR or TRAIN_RUN_ROOT   ms-swift v0-* dir, or parent containing v0-*
#   CHECKPOINT_LABEL_PREFIX        e.g. v75-expanded-lr1e-5
#   RUN_STEM_PREFIX                e.g. qwen3-4b-v75-expanded-lr1e-5
# Optional:
#   STEPS                          checkpoint steps to watch (default: 50 100 150 196)
#   TRIALS                         eval trials per checkpoint (default: 3)
#   MAX_TOKENS                     eval output cap (default: 8192)
#   EXPECTED_SHARDS                non-empty checkpoint shard count (default: 4)
#   EXTRA_BODY_* / MSWEA_*          optional scalar eval env vars passed through
#   SESSION_PREFIX                 tmux session prefix (default derived from label)
#   EVAL_REPO_ROOT                 eval harness repo root
set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
DEFAULT_EVAL_REPO_ROOT="/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen"
DEFAULT_INSTANCE_IDS_PATH="$DEFAULT_EVAL_REPO_ROOT/eval/benchmarks/swebench_multilingual/swebench_multilingual_easy_10_instance_ids.txt"

EVAL_REPO_ROOT="${EVAL_REPO_ROOT:-$DEFAULT_EVAL_REPO_ROOT}"
INSTANCE_IDS_PATH="${INSTANCE_IDS_PATH:-$DEFAULT_INSTANCE_IDS_PATH}"
SLURM_WRAPPER="${SLURM_WRAPPER:-$EVAL_REPO_ROOT/eval/benchmarks/swebench_multilingual/slurm_qwen3_4b_swe260612_step50_l40s_4gpu.sbatch}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
TRIALS="${TRIALS:-3}"
TRIAL_START="${TRIAL_START:-1}"
STEPS="${STEPS:-50 100 150 196}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-4}"
TEMPERATURE="${TEMPERATURE:-0.6}"
GENERATION_STEP_LIMIT="${GENERATION_STEP_LIMIT:-250}"
EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-4}"
DEVICE_LABEL="${DEVICE_LABEL:-l40s}"
BENCHMARK_LABEL="${BENCHMARK_LABEL:-swebench-ml-easy10}"
RUN_SUFFIX_PREFIX="${RUN_SUFFIX_PREFIX:-easy-n${TRIALS}}"
STABLE_SLEEP="${STABLE_SLEEP:-180}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-/wbl-fast/usrs/ee/clean-20260619/deepswe-data-gen/sft/qwen3/logs}"
mkdir -p "$LOG_DIR"

if ! [[ "$TRIALS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRIALS must be a positive integer; got $TRIALS" >&2
  exit 1
fi
if ! [[ "$TRIAL_START" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRIAL_START must be a positive integer; got $TRIAL_START" >&2
  exit 1
fi
if ! [[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TOKENS must be a positive integer; got $MAX_TOKENS" >&2
  exit 1
fi
if ! [[ "$EXPECTED_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_SHARDS must be a positive integer; got $EXPECTED_SHARDS" >&2
  exit 1
fi
if [ ! -x "$SLURM_WRAPPER" ]; then
  echo "Missing executable eval wrapper: $SLURM_WRAPPER" >&2
  exit 1
fi

safe_name() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9_.=-' '_' | sed 's/^_*//; s/_*$//'
}

append_optional_export() {
  local export_spec="$1"
  local name
  shift
  for name in "$@"; do
    if [ -n "${!name:-}" ]; then
      export_spec="${export_spec},${name}=${!name}"
    fi
  done
  printf '%s' "$export_spec"
}

watch_one_step() {
  : "${WATCH_STEP:?WATCH_STEP is required in worker mode}"
  : "${CHECKPOINT_LABEL_PREFIX:?CHECKPOINT_LABEL_PREFIX is required}"
  : "${RUN_STEM_PREFIX:?RUN_STEM_PREFIX is required}"

  local run_v0_dir="${RUN_V0_DIR:-}"
  local log_name
  log_name="$(safe_name "${CHECKPOINT_LABEL_PREFIX}-s${WATCH_STEP}")"
  local log_path="$LOG_DIR/watch-${log_name}-easy10.log"

  {
    echo "[$(date -Is)] watcher start step=${WATCH_STEP} max_tokens=${MAX_TOKENS}"
    if [ -z "$run_v0_dir" ]; then
      : "${TRAIN_RUN_ROOT:?TRAIN_RUN_ROOT or RUN_V0_DIR is required}"
      echo "[$(date -Is)] waiting for v0 dir under ${TRAIN_RUN_ROOT}"
      while true; do
        if [ -d "$TRAIN_RUN_ROOT" ]; then
          run_v0_dir="$(find "$TRAIN_RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'v0-*' | sort | tail -1)"
          if [ -n "$run_v0_dir" ]; then
            break
          fi
        fi
        sleep "$POLL_SECONDS"
      done
    fi

    local checkpoint_dir="${run_v0_dir}/checkpoint-${WATCH_STEP}"
    echo "[$(date -Is)] waiting for ${checkpoint_dir}"
    while true; do
      local shard_count=0
      local has_empty_shard=false
      local shard
      for shard in "${checkpoint_dir}"/model-*-of-*.safetensors; do
        [ -e "$shard" ] || continue
        if [ ! -s "$shard" ]; then
          has_empty_shard=true
          break
        fi
        shard_count=$((shard_count + 1))
      done
      if [ "$shard_count" -ge "$EXPECTED_SHARDS" ] && [ "$has_empty_shard" = "false" ]; then
        break
      fi
      sleep "$POLL_SECONDS"
    done
    echo "[$(date -Is)] found checkpoint-${WATCH_STEP} with ${shard_count} non-empty shards; waiting ${STABLE_SLEEP}s for stable writes"
    sleep "$STABLE_SLEEP"

    cd "$EVAL_REPO_ROOT"
    local trial
    local trial_end=$((TRIAL_START + TRIALS - 1))
    for trial in $(seq "$TRIAL_START" "$trial_end"); do
      local export_spec
      export_spec="ALL,REPO_ROOT=${EVAL_REPO_ROOT},CHECKPOINT_STEP_DIR=${checkpoint_dir},CHECKPOINT_LABEL=${CHECKPOINT_LABEL_PREFIX}-s${WATCH_STEP},RUN_STEM=${RUN_STEM_PREFIX}-s${WATCH_STEP}-65k,RUN_SUFFIX=${RUN_SUFFIX_PREFIX}-t${trial},MODEL_NAME=${MODEL_NAME},BENCHMARK=multilingual,BENCHMARK_LABEL=${BENCHMARK_LABEL},INSTANCE_IDS_PATH=${INSTANCE_IDS_PATH},DEVICE_LABEL=${DEVICE_LABEL},EVAL_GPU_COUNT=${EVAL_GPU_COUNT},MAX_TOKENS=${MAX_TOKENS},TEMPERATURE=${TEMPERATURE},GENERATION_STEP_LIMIT=${GENERATION_STEP_LIMIT}"
      export_spec="$(append_optional_export "$export_spec" \
        EXTRA_BODY_TOP_P \
        EXTRA_BODY_TOP_K \
        EXTRA_BODY_MIN_P \
        EXTRA_BODY_PRESENCE_PENALTY \
        EXTRA_BODY_REPETITION_PENALTY \
        EXTRA_BODY_FREQUENCY_PENALTY \
        MSWEA_REPEAT_GUARD_THRESHOLD \
        MSWEA_REPEAT_GUARD_BLOCK_AFTER \
        MSWEA_CONTEXT_GUARD_MAX_CHARS \
        MSWEA_CONTEXT_GUARD_MIN_CALLS \
        MSWEA_OUTPUT_MAX_CHARS \
        MSWEA_OUTPUT_HEAD_CHARS \
        MSWEA_OUTPUT_TAIL_CHARS \
        MSWEA_HISTORY_COMPACT_MAX_CHARS \
        MSWEA_HISTORY_COMPACT_KEEP_RECENT \
        MSWEA_HISTORY_COMPACT_MSG_CHARS \
        MSWEA_HISTORY_COMPACT_HEAD_CHARS \
        MSWEA_HISTORY_COMPACT_TAIL_CHARS \
        MSWEA_NO_EDIT_WARNING_CALLS \
        MSWEA_NO_EDIT_WARNING_INTERVAL \
        MSWEA_FORCE_SUBMIT_AFTER_CALLS \
        MSWEA_VALIDATE_SUBMISSION \
        MSWEA_MIN_SUBMISSION_CHARS \
        MSWEA_REQUIRE_DIFF_SUBMISSION \
        MSWEA_VALIDATE_GIT_APPLY \
        MSWEA_VALIDATE_SMOKE \
        MSWEA_VALIDATE_SMOKE_TIMEOUT)"
      sbatch \
        --export="$export_spec" \
        "$SLURM_WRAPPER"
    done
    echo "[$(date -Is)] submitted checkpoint-${WATCH_STEP} easy10 n=${TRIALS} eval jobs"
  } >>"$log_path" 2>&1
}

if [ "${WATCHER_WORKER:-false}" = "true" ]; then
  watch_one_step
  exit 0
fi

: "${CHECKPOINT_LABEL_PREFIX:?CHECKPOINT_LABEL_PREFIX is required}"
: "${RUN_STEM_PREFIX:?RUN_STEM_PREFIX is required}"
if [ -z "${RUN_V0_DIR:-}" ] && [ -z "${TRAIN_RUN_ROOT:-}" ]; then
  echo "Set RUN_V0_DIR or TRAIN_RUN_ROOT" >&2
  exit 1
fi

SESSION_PREFIX="${SESSION_PREFIX:-$(safe_name "$CHECKPOINT_LABEL_PREFIX")}"
for step in $STEPS; do
  if ! [[ "$step" =~ ^[1-9][0-9]*$ ]]; then
    echo "Checkpoint steps must be positive integers; got $step" >&2
    exit 1
  fi
  session_name="$(safe_name "${SESSION_PREFIX}_s${step}_eval_watch")"
  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session already exists: $session_name"
    continue
  fi
  tmux new-session -d -s "$session_name" \
    "env WATCHER_WORKER=true WATCH_STEP='$step' RUN_V0_DIR='${RUN_V0_DIR:-}' TRAIN_RUN_ROOT='${TRAIN_RUN_ROOT:-}' CHECKPOINT_LABEL_PREFIX='$CHECKPOINT_LABEL_PREFIX' RUN_STEM_PREFIX='$RUN_STEM_PREFIX' EVAL_REPO_ROOT='$EVAL_REPO_ROOT' INSTANCE_IDS_PATH='$INSTANCE_IDS_PATH' SLURM_WRAPPER='$SLURM_WRAPPER' MODEL_NAME='$MODEL_NAME' TRIALS='$TRIALS' TRIAL_START='$TRIAL_START' MAX_TOKENS='$MAX_TOKENS' EXPECTED_SHARDS='$EXPECTED_SHARDS' TEMPERATURE='$TEMPERATURE' GENERATION_STEP_LIMIT='$GENERATION_STEP_LIMIT' EVAL_GPU_COUNT='$EVAL_GPU_COUNT' DEVICE_LABEL='$DEVICE_LABEL' BENCHMARK_LABEL='$BENCHMARK_LABEL' RUN_SUFFIX_PREFIX='$RUN_SUFFIX_PREFIX' STABLE_SLEEP='$STABLE_SLEEP' POLL_SECONDS='$POLL_SECONDS' LOG_DIR='$LOG_DIR' EXTRA_BODY_TOP_P='${EXTRA_BODY_TOP_P:-}' EXTRA_BODY_TOP_K='${EXTRA_BODY_TOP_K:-}' EXTRA_BODY_MIN_P='${EXTRA_BODY_MIN_P:-}' EXTRA_BODY_PRESENCE_PENALTY='${EXTRA_BODY_PRESENCE_PENALTY:-}' EXTRA_BODY_REPETITION_PENALTY='${EXTRA_BODY_REPETITION_PENALTY:-}' EXTRA_BODY_FREQUENCY_PENALTY='${EXTRA_BODY_FREQUENCY_PENALTY:-}' MSWEA_REPEAT_GUARD_THRESHOLD='${MSWEA_REPEAT_GUARD_THRESHOLD:-}' MSWEA_REPEAT_GUARD_BLOCK_AFTER='${MSWEA_REPEAT_GUARD_BLOCK_AFTER:-}' MSWEA_CONTEXT_GUARD_MAX_CHARS='${MSWEA_CONTEXT_GUARD_MAX_CHARS:-}' MSWEA_CONTEXT_GUARD_MIN_CALLS='${MSWEA_CONTEXT_GUARD_MIN_CALLS:-}' MSWEA_OUTPUT_MAX_CHARS='${MSWEA_OUTPUT_MAX_CHARS:-}' MSWEA_OUTPUT_HEAD_CHARS='${MSWEA_OUTPUT_HEAD_CHARS:-}' MSWEA_OUTPUT_TAIL_CHARS='${MSWEA_OUTPUT_TAIL_CHARS:-}' MSWEA_HISTORY_COMPACT_MAX_CHARS='${MSWEA_HISTORY_COMPACT_MAX_CHARS:-}' MSWEA_HISTORY_COMPACT_KEEP_RECENT='${MSWEA_HISTORY_COMPACT_KEEP_RECENT:-}' MSWEA_HISTORY_COMPACT_MSG_CHARS='${MSWEA_HISTORY_COMPACT_MSG_CHARS:-}' MSWEA_HISTORY_COMPACT_HEAD_CHARS='${MSWEA_HISTORY_COMPACT_HEAD_CHARS:-}' MSWEA_HISTORY_COMPACT_TAIL_CHARS='${MSWEA_HISTORY_COMPACT_TAIL_CHARS:-}' MSWEA_NO_EDIT_WARNING_CALLS='${MSWEA_NO_EDIT_WARNING_CALLS:-}' MSWEA_NO_EDIT_WARNING_INTERVAL='${MSWEA_NO_EDIT_WARNING_INTERVAL:-}' MSWEA_FORCE_SUBMIT_AFTER_CALLS='${MSWEA_FORCE_SUBMIT_AFTER_CALLS:-}' MSWEA_VALIDATE_SUBMISSION='${MSWEA_VALIDATE_SUBMISSION:-}' MSWEA_MIN_SUBMISSION_CHARS='${MSWEA_MIN_SUBMISSION_CHARS:-}' MSWEA_REQUIRE_DIFF_SUBMISSION='${MSWEA_REQUIRE_DIFF_SUBMISSION:-}' MSWEA_VALIDATE_GIT_APPLY='${MSWEA_VALIDATE_GIT_APPLY:-}' MSWEA_VALIDATE_SMOKE='${MSWEA_VALIDATE_SMOKE:-}' MSWEA_VALIDATE_SMOKE_TIMEOUT='${MSWEA_VALIDATE_SMOKE_TIMEOUT:-}' bash '$SCRIPT_PATH'"
  echo "started ${session_name}"
done
