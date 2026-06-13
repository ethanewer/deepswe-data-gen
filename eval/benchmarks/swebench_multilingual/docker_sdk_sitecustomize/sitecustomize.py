"""Runtime Docker SDK tweaks for SWE-bench scoring subprocesses."""

from __future__ import annotations

import os


try:
    import docker
except Exception:
    docker = None


if docker is not None:
    _from_env = docker.from_env

    def _from_env_with_timeout(**kwargs):
        if "timeout" not in kwargs:
            timeout = os.environ.get("SWEBENCH_DOCKER_TIMEOUT")
            if timeout:
                kwargs["timeout"] = int(timeout)
        if "max_pool_size" not in kwargs:
            max_pool_size = os.environ.get("SWEBENCH_DOCKER_MAX_POOL_SIZE")
            if max_pool_size:
                kwargs["max_pool_size"] = int(max_pool_size)
        return _from_env(**kwargs)

    docker.from_env = _from_env_with_timeout
