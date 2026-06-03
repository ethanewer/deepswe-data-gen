#!/usr/bin/env bash
set -euo pipefail

# Pier's task egress proxy only allowed HTTP ports 80 and 443 in the installed
# template on this node. Local model serving used host port 8000, so the task
# containers needed port 8000 added to Squid Safe_ports.

TARGET="${TARGET:-.venv/lib/python3.12/site-packages/pier/environments/agent_setup.py}"

if [[ ! -f "$TARGET" ]]; then
  echo "Missing Pier agent setup file: $TARGET" >&2
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
