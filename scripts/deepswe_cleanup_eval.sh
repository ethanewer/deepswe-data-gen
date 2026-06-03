#!/usr/bin/env bash
set -euo pipefail

# Stop the local DeepSWE eval stack used for Qwen/Qwen3.6-27B-FP8. This avoids
# touching unrelated GPU 0 work by matching only known local server commands and
# DeepSWE task container/network names.

TASK_NAME_REGEX='^(happy-dom|wazero|psd-tools|ytt-jsonpath|true-myth)'

echo "Stopping Pier jobs, local vLLM servers, and the round-robin proxy..."
while read -r pid; do
  [[ -n "$pid" ]] || continue
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  if [[ -n "$pgid" ]]; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
  fi
done < <(
  pgrep -af 'pier run|vllm serve Qwen/Qwen3\.6-27B-FP8|openai_round_robin_proxy.py' |
    awk '{print $1}' || true
)

sleep "${TERM_GRACE_SEC:-8}"

while read -r pid; do
  [[ -n "$pid" ]] || continue
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  if [[ -n "$pgid" ]]; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
done < <(
  pgrep -af 'pier run|vllm serve Qwen/Qwen3\.6-27B-FP8|openai_round_robin_proxy.py' |
    awk '{print $1}' || true
)

echo "Removing DeepSWE task containers..."
names="$(docker ps -a --format '{{.Names}}' | grep -E "$TASK_NAME_REGEX" || true)"
if [[ -n "$names" ]]; then
  printf '%s\n' "$names" | xargs -r docker rm -f
fi

echo "Removing DeepSWE task networks..."
nets="$(docker network ls --format '{{.Name}}' | grep -E "$TASK_NAME_REGEX" || true)"
if [[ -n "$nets" ]]; then
  printf '%s\n' "$nets" | xargs -r docker network rm || true
fi

echo "Remaining matched processes:"
pgrep -af 'pier run|vllm serve Qwen/Qwen3\.6-27B-FP8|openai_round_robin_proxy.py' || true

echo "Remaining matched Docker resources:"
docker ps -a --format '{{.Names}} {{.Status}}' | grep -E "$TASK_NAME_REGEX" || true
docker network ls --format '{{.Name}}' | grep -E "$TASK_NAME_REGEX" || true
