#!/usr/bin/env python3
"""Expose `docker system dial-stdio` as a user-owned Unix socket.

The cluster Docker CLI is setgid docker, but Python processes are not in the
docker group. The Docker SDK talks directly to /var/run/docker.sock and fails.
This proxy lets Docker SDK clients connect to a socket owned by the current
user while each connection is forwarded through the setgid Docker CLI.
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path


def _drain_stderr(pipe) -> None:
    try:
        for line in iter(pipe.readline, b""):
            if line:
                sys.stderr.buffer.write(line)
                sys.stderr.buffer.flush()
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def _bridge_connection(conn: socket.socket, docker_bin: str) -> None:
    env = os.environ.copy()
    env.pop("DOCKER_HOST", None)
    proc = subprocess.Popen(
        [docker_bin, "system", "dial-stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    threading.Thread(target=_drain_stderr, args=(proc.stderr,), daemon=True).start()

    conn_fd = conn.fileno()
    stdout_fd = proc.stdout.fileno()
    stdin_fd = proc.stdin.fileno()
    open_fds = {conn_fd, stdout_fd}
    try:
        while open_fds:
            readable, _, _ = select.select(list(open_fds), [], [], 30.0)
            if not readable:
                if proc.poll() is not None:
                    break
                continue
            for fd in readable:
                if fd == conn_fd:
                    try:
                        data = conn.recv(1024 * 1024)
                    except OSError:
                        data = b""
                    if not data:
                        open_fds.discard(conn_fd)
                        try:
                            proc.stdin.close()
                        except OSError:
                            pass
                    else:
                        try:
                            os.write(stdin_fd, data)
                        except OSError:
                            open_fds.discard(conn_fd)
                elif fd == stdout_fd:
                    try:
                        data = os.read(stdout_fd, 1024 * 1024)
                    except OSError:
                        data = b""
                    if not data:
                        open_fds.discard(stdout_fd)
                        try:
                            conn.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                    else:
                        try:
                            conn.sendall(data)
                        except OSError:
                            open_fds.discard(stdout_fd)
                            open_fds.discard(conn_fd)
    finally:
        try:
            conn.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def serve(socket_path: Path, docker_bin: str) -> None:
    if socket_path.exists():
        socket_path.unlink()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    server.listen(64)
    stop = False

    def _handle_signal(signum, frame) -> None:  # noqa: ARG001
        nonlocal stop
        stop = True
        try:
            server.close()
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while not stop:
            try:
                conn, _ = server.accept()
            except OSError:
                if stop:
                    break
                raise
            threading.Thread(
                target=_bridge_connection,
                args=(conn, docker_bin),
                daemon=True,
            ).start()
    finally:
        try:
            server.close()
        except OSError:
            pass
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("socket_path", type=Path)
    parser.add_argument("--docker-bin", default="/usr/bin/docker")
    args = parser.parse_args()
    serve(args.socket_path, args.docker_bin)


if __name__ == "__main__":
    main()
