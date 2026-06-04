#!/usr/bin/env bash
set -euo pipefail

JOBS_DIR="${JOBS_DIR:-runs/pier-jobs}"
JOB_NAME="${JOB_NAME:-$(cat /tmp/deepswe_active_job_name.txt)}"
RESULT_JSON="$JOBS_DIR/$JOB_NAME/result.json"
INTERVAL_SEC="${INTERVAL_SEC:-60}"
PYTHON_BIN="${PYTHON_BIN:-.venv-swe-uv/bin/python}"

if [[ ! -f "$RESULT_JSON" ]]; then
  echo "Missing result file: $RESULT_JSON" >&2
  exit 1
fi

while true; do
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  "$PYTHON_BIN" - "$RESULT_JSON" <<'PY'
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
s = d["stats"]
print("finished_at=", d.get("finished_at"))
print(
    "completed={n_completed_trials} errored={n_errored_trials} "
    "running={n_running_trials} pending={n_pending_trials} "
    "cancelled={n_cancelled_trials} retries={n_retries}".format(**s)
)
print("evals=", s.get("evals"))
PY
  if "$PYTHON_BIN" - "$RESULT_JSON" <<'PY'
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if d.get("finished_at") else 1)
PY
  then
    break
  fi
  sleep "$INTERVAL_SEC"
done
