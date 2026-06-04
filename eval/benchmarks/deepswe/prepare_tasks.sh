#!/usr/bin/env bash
set -euo pipefail

# Fetch DeepSWE task definitions locally. The eval launch script expects
# TASKS_DIR to point at the tasks directory created here.

DEST="${DEST:-/tmp/deep-swe}"
REMOTE="${REMOTE:-https://github.com/datacurve-ai/deep-swe.git}"
REF="${REF:-main}"

if [[ -d "$DEST/.git" ]]; then
  git -C "$DEST" fetch --depth 1 origin "$REF"
  git -C "$DEST" checkout FETCH_HEAD
else
  git clone --filter=blob:none --sparse --no-checkout "$REMOTE" "$DEST"
  git -C "$DEST" sparse-checkout set tasks
  git -C "$DEST" fetch --depth 1 origin "$REF"
  git -C "$DEST" checkout FETCH_HEAD
fi

for task in \
  true-myth-iterable-collection-combinators \
  happy-dom-abort-pending-body-reads \
  wazero-multi-module-snapshots \
  psd-tools-blend-range-api \
  ytt-jsonpath-query-api
do
  test -d "$DEST/tasks/$task" || {
    echo "Missing task: $task" >&2
    exit 1
  }
done

echo "DeepSWE tasks are available at $DEST/tasks"
