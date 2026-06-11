"""Pinned mini-swe-agent source used by eval and data-generation harnesses."""

from __future__ import annotations

import os
from pathlib import Path

from eval.paths import REPO_ROOT


MINI_SWE_AGENT_GIT_URL = "https://github.com/SWE-agent/mini-swe-agent.git"
MINI_SWE_AGENT_GIT_SHA = "a85bf5eedb3a557715038a00e43a92e7831462e3"
MINI_SWE_AGENT_REQUIREMENT = (
    f"mini-swe-agent @ git+{MINI_SWE_AGENT_GIT_URL}@{MINI_SWE_AGENT_GIT_SHA}"
)
MINI_SWE_AGENT_PIER_EXTRA_PACKAGES = [
    MINI_SWE_AGENT_REQUIREMENT,
    "click>=8.1,<9",
    "attrs>=23,<27",
    "pydantic>=2,<3",
    "idna>=3,<4",
    "typing_extensions>=4.12,<5",
]

MINI_SWE_AGENT_OVERLAY_ENV = "PYDEPS_OVERLAY"
MINI_SWE_AGENT_ALLOW_UNPINNED_ENV = "MINI_SWE_AGENT_ALLOW_UNPINNED"
MINI_SWE_AGENT_OVERLAY = (
    REPO_ROOT.parent
    / "runtime"
    / f"pydeps-miniswe-upstream-{MINI_SWE_AGENT_GIT_SHA[:7]}"
)
MINI_SWE_AGENT_CLEAN_OVERLAY = (
    REPO_ROOT.parent
    / "runtime"
    / f"pydeps-miniswe-upstream-{MINI_SWE_AGENT_GIT_SHA[:7]}-clean-20260610T2340"
)


def pinned_minisweagent_overlay() -> Path:
    """Return the mini-swe-agent overlay for future local/Pyxis runs."""
    override = os.environ.get(MINI_SWE_AGENT_OVERLAY_ENV)
    if override:
        return Path(override)
    if MINI_SWE_AGENT_CLEAN_OVERLAY.exists():
        return MINI_SWE_AGENT_CLEAN_OVERLAY
    return MINI_SWE_AGENT_OVERLAY


def require_pinned_minisweagent_overlay() -> Path:
    """Return the overlay path, failing unless an explicit override is enabled."""
    overlay = pinned_minisweagent_overlay()
    if overlay.exists():
        return overlay
    if os.environ.get(MINI_SWE_AGENT_ALLOW_UNPINNED_ENV):
        return overlay
    raise FileNotFoundError(
        f"pinned mini-swe-agent overlay does not exist: {overlay}. "
        f"Create it from {MINI_SWE_AGENT_REQUIREMENT}, set "
        f"{MINI_SWE_AGENT_OVERLAY_ENV}=... to override, or set "
        f"{MINI_SWE_AGENT_ALLOW_UNPINNED_ENV}=1 to use the ambient environment."
    )


def prepend_minisweagent_overlay(env: dict[str, str]) -> Path:
    """Put the pinned mini-swe-agent overlay first on PYTHONPATH."""
    overlay = require_pinned_minisweagent_overlay()
    if os.environ.get(MINI_SWE_AGENT_ALLOW_UNPINNED_ENV) and not overlay.exists():
        return overlay
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join([str(overlay), existing]) if existing else str(overlay)
    )
    return overlay
