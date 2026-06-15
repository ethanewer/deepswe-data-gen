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

if os.environ.get("SWEBENCH_SKIP_CONTAINER_CLEANUP") == "1":
    try:
        import swebench.harness.docker_utils as _swebench_docker_utils
    except Exception:
        _swebench_docker_utils = None

    if _swebench_docker_utils is not None:

        def _skip_cleanup_container(client, container, logger):
            if container is None:
                return
            name = getattr(container, "name", "<unknown>")
            if logger and logger != "quiet":
                logger.info(
                    "Skipping cleanup for container %s because "
                    "SWEBENCH_SKIP_CONTAINER_CLEANUP=1",
                    name,
                )

        _swebench_docker_utils.cleanup_container = _skip_cleanup_container
