#!/usr/bin/env bash
# Execution setting: LOCAL-GPU (DeepSWE), setup step (no GPU).
# Purpose: add port 8000 to Pier's Squid Safe_ports so task containers can reach
# the local round-robin proxy on host port 8000. Idempotent.
# Key env vars: PYTHON_BIN (used to locate the installed Pier), TARGET (override
# path to pier/environments/agent_setup.py if Pier is installed elsewhere).
# Prerequisites: Pier installed (so agent_setup.py can be found/patched).
set -euo pipefail

# Pier's task egress proxy only allowed HTTP ports 80 and 443 in the installed
# template on this node. Local model serving used host port 8000, so the task
# containers needed port 8000 added to Squid Safe_ports.

PYTHON_BIN="${PYTHON_BIN:-.venv-swe-uv/bin/python}"
TARGET="${TARGET:-}"

if [[ -z "$TARGET" ]]; then
  TARGET="$("$PYTHON_BIN" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("pier")
if spec is None or spec.origin is None:
    print("")
else:
    print(Path(spec.origin).resolve().parent / "environments" / "agent_setup.py")
PY
)"
fi

if [[ ! -f "$TARGET" ]]; then
  echo "Missing Pier agent setup file: ${TARGET:-<not found>}." >&2
  echo "Set TARGET=/path/to/pier/environments/agent_setup.py if Pier is installed outside $PYTHON_BIN." >&2
  exit 1
fi

if grep -q 'acl Safe_ports port 80 443 8000' "$TARGET"; then
  echo "Port 8000 is already allowed in $TARGET"
  exit 0
fi

if ! grep -q 'acl Safe_ports port 80 443' "$TARGET"; then
  echo "Did not find expected Safe_ports line in $TARGET" >&2
  exit 1
fi

sed -i 's/acl Safe_ports port 80 443/acl Safe_ports port 80 443 8000/' "$TARGET"
echo "Allowed port 8000 in $TARGET"
