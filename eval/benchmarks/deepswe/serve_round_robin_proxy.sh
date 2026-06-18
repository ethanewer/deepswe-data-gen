#!/usr/bin/env bash
# Execution setting: LOCAL-GPU (DeepSWE).
# Purpose: start the OpenAI-compatible round-robin proxy on :8000 that fans
# requests across the per-GPU vLLM replicas started by serve_vllm_replicas.sh.
# Pier task containers reach it via http://172.17.0.1.nip.io:8000/v1.
# Key env vars: HOST, PORT (default 8000), OPENAI_BACKENDS (CSV of replica URLs,
# default :8101-:8107), VENV_BIN.
# Prerequisites: vLLM replicas already serving on the OPENAI_BACKENDS ports.
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
OPENAI_BACKENDS="${OPENAI_BACKENDS:-http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103,http://127.0.0.1:8104,http://127.0.0.1:8105,http://127.0.0.1:8106,http://127.0.0.1:8107}"
VENV_BIN="${VENV_BIN:-.venv-swe-uv/bin}"

export HOST PORT OPENAI_BACKENDS
"$VENV_BIN/python" -m eval.model.round_robin_proxy
