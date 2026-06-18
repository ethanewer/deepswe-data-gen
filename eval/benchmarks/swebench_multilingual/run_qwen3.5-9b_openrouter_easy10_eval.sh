#!/usr/bin/env bash
#
# Reproducible eval: qwen/qwen3.5-9b on the SWE-bench Multilingual easy-10
# debugging subset via OpenRouter, N trials, mini-swe-agent harness + the
# official SWE-bench Docker evaluation.
#
# Concurrency: trials run in parallel (MAX_PARALLEL_TRIALS at a time) and each
# trial uses GEN_WORKERS generation workers + EVAL_WORKERS eval workers. The
# defaults below are the highest settings validated on a 10-core/32G host:
# 2 trials in parallel = 20 simultaneous agent containers, which left the box
# ~70% idle (generation is OpenRouter-latency-bound, not CPU-bound). GEN_WORKERS
# is capped at 10 because the subset has 10 tasks (more workers is a no-op).
#
# Requirements: Docker running; .venv-swe-uv set up; OPENROUTER_API_KEY exported
# in the environment. NO API KEY IS STORED IN THIS SCRIPT -- it is read from the
# environment by name only (--api-key-env OPENROUTER_API_KEY).
#
# Usage:
#   export OPENROUTER_API_KEY=sk-or-...        # never commit this value
#   TRIALS=5 ./run_qwen3.5-9b_openrouter_easy10_eval.sh
#
# Tunables (env overrides):
#   TRIALS=5 MAX_PARALLEL_TRIALS=2 GEN_WORKERS=10 EVAL_WORKERS=8 MAX_TOKENS=4096
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY must be set in the environment" >&2
  exit 1
fi

PY="${PY:-$REPO_ROOT/.venv-swe-uv/bin/python}"
INSTANCE_IDS="${INSTANCE_IDS:-$SCRIPT_DIR/swebench_multilingual_easy_10_instance_ids.txt}"

TRIALS="${TRIALS:-5}"
MAX_PARALLEL_TRIALS="${MAX_PARALLEL_TRIALS:-2}"
GEN_WORKERS="${GEN_WORKERS:-10}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
GENERATION_STEP_LIMIT="${GENERATION_STEP_LIMIT:-250}"
# Mirrors eval/configs/openrouter_qwen.example.json
EXTRA_BODY='{"provider":{"order":["venice/fp8","together","siliconflow/fp8"],"allow_fallbacks":true}}'

TS="$(date +%Y%m%d-%H%M%S)"
BATCH="$REPO_ROOT/runs/qwen3.5-9b_swebench-ml-easy10_n${TRIALS}_${TS}"
mkdir -p "$BATCH/logs"
echo "Batch output: $BATCH"

run_trial() {
  local t="$1"
  "$PY" -m eval.benchmarks.swebench_multilingual.run \
    --instance-ids "$INSTANCE_IDS" \
    --output "$BATCH/trial-$t" \
    --run-id "qwen3.5-9b-ml-easy10-t${t}-${TS}" \
    --harness mini-swe-agent \
    --generation-workers "$GEN_WORKERS" \
    --eval-workers "$EVAL_WORKERS" \
    --model qwen/qwen3.5-9b \
    --api-base https://openrouter.ai/api/v1 \
    --api-key-env OPENROUTER_API_KEY \
    --temperature 0 \
    --max-tokens "$MAX_TOKENS" \
    --extra-body-json "$EXTRA_BODY" \
    > "$BATCH/logs/trial-$t.log" 2>&1
  echo "trial-$t exit=$? $(date)"
}

# Launch trials in parallel, at most MAX_PARALLEL_TRIALS concurrently.
running=0
for t in $(seq 1 "$TRIALS"); do
  run_trial "$t" &
  running=$((running + 1))
  if (( running >= MAX_PARALLEL_TRIALS )); then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait
echo "All $TRIALS trials done. Reports: $REPO_ROOT/runs/results/ and per-trial report JSON in repo root."
echo "Aggregate with: python3 -c \"...\"  # see eval/results summary for the scoring snippet"
