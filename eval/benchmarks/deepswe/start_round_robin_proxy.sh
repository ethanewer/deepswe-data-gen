#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
OPENAI_BACKENDS="${OPENAI_BACKENDS:-http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103,http://127.0.0.1:8104,http://127.0.0.1:8105,http://127.0.0.1:8106,http://127.0.0.1:8107}"
VENV_BIN="${VENV_BIN:-.venv-swe-uv/bin}"

export HOST PORT OPENAI_BACKENDS
"$VENV_BIN/python" -m eval.model.round_robin_proxy
