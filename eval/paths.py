"""Shared filesystem locations for the eval package."""

from __future__ import annotations

from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
RUNS_DIR = REPO_ROOT / "runs"
CACHE_DIR = REPO_ROOT / ".cache" / "eval"
DEFAULT_CA_BUNDLE_PATH = RUNS_DIR / "system-ca-bundle.pem"
PYTHON = REPO_ROOT / ".venv-swe-uv" / "bin" / "python"


def python_executable() -> str:
    if PYTHON.exists():
        return str(PYTHON)
    import sys

    return sys.executable


def configure_ca_bundle(env: dict[str, str]) -> None:
    if DEFAULT_CA_BUNDLE_PATH.exists():
        env.setdefault("REQUESTS_CA_BUNDLE", str(DEFAULT_CA_BUNDLE_PATH))
        env.setdefault("SSL_CERT_FILE", str(DEFAULT_CA_BUNDLE_PATH))
