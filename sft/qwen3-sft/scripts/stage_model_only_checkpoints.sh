#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 SRC_CHECKPOINT_DIR DST_CHECKPOINT_DIR STEP [STEP ...]" >&2
  exit 2
fi

SRC_DIR="$1"
DST_DIR="$2"
shift 2

SETTLE_SECONDS="${SETTLE_SECONDS:-45}"
MIN_SAFETENSORS="${MIN_SAFETENSORS:-8}"
POLL_SECONDS="${POLL_SECONDS:-20}"

mkdir -p "$DST_DIR"
echo "$(date -u) staging watcher start src=$SRC_DIR dst=$DST_DIR steps=$*"

for step in "$@"; do
  target="$DST_DIR/epoch_0_step_${step}"
  marker="$target/.staged_complete"
  if [ -f "$marker" ]; then
    echo "$(date -u) step $step already staged"
    continue
  fi

  source_step="$SRC_DIR/epoch_0_step_${step}"
  echo "$(date -u) waiting for $source_step/model"
  while true; do
    if [ -d "$source_step/model" ] && [ -e "$source_step/model/.hf_metadata" ]; then
      tensor_count="$(find "$source_step/model" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)"
      if [ "$tensor_count" -ge "$MIN_SAFETENSORS" ]; then
        break
      fi
    fi
    sleep "$POLL_SECONDS"
  done

  echo "$(date -u) source for step $step appears ready; settling ${SETTLE_SECONDS}s"
  sleep "$SETTLE_SECONDS"

  tmp="$DST_DIR/.epoch_0_step_${step}.tmp.$$"
  rm -rf "$tmp"
  mkdir -p "$tmp"
  rsync -a --delete "$source_step/model/" "$tmp/model/"
  for file in config.yaml losses.json step_scheduler.pt; do
    if [ -e "$source_step/$file" ]; then
      rsync -a "$source_step/$file" "$tmp/$file"
    fi
  done

  {
    echo "staged_from=$source_step"
    echo "staged_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$tmp/STAGED_MODEL_ONLY.txt"
  touch "$tmp/.staged_complete"

  rm -rf "$target"
  mv "$tmp" "$target"
  echo "$(date -u) staged model-only step $step to $target"
done

echo "$(date -u) staging watcher done"
