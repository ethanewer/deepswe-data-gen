"""Shared filesystem locations for the eval package."""

from __future__ import annotations

import subprocess
import sys
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


def ensure_system_ca_bundle() -> Path | None:
    if DEFAULT_CA_BUNDLE_PATH.exists():
        return DEFAULT_CA_BUNDLE_PATH
    if sys.platform != "darwin":
        return None

    result = subprocess.run(
        ["security", "find-certificate", "-a", "-p"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0 or "BEGIN CERTIFICATE" not in result.stdout:
        return None

    DEFAULT_CA_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DEFAULT_CA_BUNDLE_PATH.with_suffix(".tmp")
    tmp_path.write_text(result.stdout)
    tmp_path.replace(DEFAULT_CA_BUNDLE_PATH)
    return DEFAULT_CA_BUNDLE_PATH


def configure_ca_bundle(env: dict[str, str]) -> None:
    ca_bundle = ensure_system_ca_bundle()
    if ca_bundle:
        env.setdefault("REQUESTS_CA_BUNDLE", str(ca_bundle))
        env.setdefault("SSL_CERT_FILE", str(ca_bundle))
        env.setdefault("CURL_CA_BUNDLE", str(ca_bundle))
