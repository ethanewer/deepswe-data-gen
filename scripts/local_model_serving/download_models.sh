#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

PYTHON_BIN="${PYTHON_BIN:-$LOCAL_MODEL_SERVING_ROOT/venv/bin/python}"

download_one() {
  local repo_id="$1"
  local stable_link="$2"
  "$PYTHON_BIN" "$SCRIPT_DIR/download_snapshot.py" "$repo_id" "$stable_link"
}

download_one "moonshotai/Kimi-K2.6" "$KIMI_MODEL_PATH"
download_one "XiaomiMiMo/MiMo-V2.5" "$MIMO_MODEL_PATH"
