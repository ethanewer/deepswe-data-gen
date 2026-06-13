#!/usr/bin/env python
"""Run a model on the predictive SWE-bench Multilingual subset."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.model.config import add_model_args, model_from_defaults
from eval.minisweagent_pin import prepend_minisweagent_overlay
from eval.paths import REPO_ROOT, configure_ca_bundle, python_executable


SUBSET_DIR = Path(__file__).resolve().parent
DEFAULTS_PATH = SUBSET_DIR / "defaults.json"
DEFAULT_INSTANCE_IDS_PATH = SUBSET_DIR / "predictive_30_instance_ids.txt"
DATASET_NAME = "SWE-bench/SWE-bench_Multilingual"
MINISWEAGENT_SUBSET = "multilingual"
BENCHMARK_DISPLAY_NAME = "SWE-bench Multilingual"
BENCHMARK_RUN_TAG = "multilingual-30"
SUPPORTED_HARNESSES = ("mini-swe-agent", "openhands-swe", "opencode", "terminus-2")
OPENHANDS_SETUP_FILES_TO_REMOVE = ("pyproject.toml", "tox.ini", "setup.py")
OPENCODE_DEFAULT_COMMAND = "npx --yes opencode-ai"
OPENCODE_DEFAULT_CONTEXT_LIMIT = 128000
OPENCODE_MIN_OUTPUT_LIMIT = 8192
OPENCODE_BENCHMARK_AGENT = "swebench"
DOCKER_CLI = os.environ.get("DOCKER_CLI", "/usr/bin/docker")
OPENCODE_OPTIONAL_SUBMODULES_TO_SKIP = {
    "sharkdp/bat": {
        # These syntax bundle repositories are unavailable at older bat base
        # commits, but are not required for source repair in this benchmark.
        "assets/syntaxes/TypeScript": (
            "https://github.com/Microsoft/TypeScript-Sublime-Plugin"
        ),
        "assets/syntaxes/02_Extra/LiveScript": (
            "https://github.com/paulmillr/LiveScript.tmbundle"
        ),
        "assets/syntaxes/02_Extra/Nginx": (
            "https://github.com/brandonwamboldt/sublime-nginx"
        ),
        "assets/syntaxes/hosts": (
            "https://github.com/brandonwamboldt/sublime-hosts"
        ),
    },
}
OPENCODE_BENCHMARK_AGENT_PROMPT = (
    "You are a benchmark repair agent. Work directly in the checked-out "
    "repository, make the smallest correct source and test changes needed for "
    "the issue, and do not commit. Search only enough to locate the relevant "
    "code path, avoid delegation/subagents, and keep visible output concise. "
    "After editing, inspect the diff and finish any referenced call sites, "
    "symbols, generated files, and tests that your change requires. "
    "Do not install language toolchains or debug local environment setup; "
    "if a local check is unavailable, stop after diff review. "
    "Before stopping, leave the worktree with a non-empty git diff unless the "
    "task is impossible."
)


def load_defaults(path: Path) -> dict:
    return json.loads(path.read_text())


def read_instance_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_cli_paths(args: argparse.Namespace) -> None:
    for attr in (
        "instance_ids",
        "output",
        "openhands_llm_config",
        "openhands_output_json",
        "openhands_command_cwd",
        "opencode_config",
        "opencode_workspace",
        "terminus_workspace",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(args, attr, value.expanduser().resolve())


def make_filter_regex(instance_ids: list[str]) -> str:
    return "^(" + "|".join(re.escape(instance_id) for instance_id in instance_ids) + ")$"


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def run_in_dir(
    cmd: list[str],
    env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    log_path: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout or "")
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)
    return result


def default_env() -> dict[str, str]:
    env = os.environ.copy()
    configure_ca_bundle(env)
    return env


def docker_sdk_usable(env: dict[str, str]) -> bool:
    try:
        import docker

        client = docker.from_env(timeout=3, environment=env)
        client.ping()
    except Exception:
        return False
    return True


class DockerStdioProxy:
    """Expose Docker CLI stdio transport through a user-owned Unix socket."""

    def __init__(self, socket_path: Path):
        self.socket_path = socket_path
        self._server: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connections: list[socket.socket] = []

    def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        server.listen()
        server.settimeout(0.5)
        self._server = server
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            with contextlib.suppress(OSError):
                self._server.close()
        for conn in self._connections:
            with contextlib.suppress(OSError):
                conn.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.socket_path.unlink(missing_ok=True)

    def _serve(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._connections.append(conn)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        env = os.environ.copy()
        env.pop("DOCKER_HOST", None)
        proc = subprocess.Popen(
            [DOCKER_CLI, "system", "dial-stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None
        to_proc = threading.Thread(
            target=self._socket_to_pipe, args=(conn, proc.stdin), daemon=True
        )
        from_proc = threading.Thread(
            target=self._pipe_to_socket, args=(proc.stdout, conn), daemon=True
        )
        stderr = threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True)
        to_proc.start()
        from_proc.start()
        stderr.start()
        to_proc.join()
        from_proc.join()
        with contextlib.suppress(Exception):
            proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=2)
        with contextlib.suppress(OSError):
            conn.close()

    @staticmethod
    def _socket_to_pipe(src: socket.socket, dst) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.write(data)
                dst.flush()
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                dst.close()

    @staticmethod
    def _pipe_to_socket(src, dst: socket.socket) -> None:
        try:
            while True:
                data = os.read(src.fileno(), 65536)
                if not data:
                    break
                dst.sendall(data)
        except Exception:
            pass
        finally:
            with contextlib.suppress(OSError):
                dst.shutdown(socket.SHUT_WR)

    @staticmethod
    def _drain_stderr(proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            text = line.decode(errors="replace").strip()
            if text:
                print(f"[docker-stdio-proxy] {text}", flush=True)


@contextlib.contextmanager
def docker_evaluation_env(env: dict[str, str], output_dir: Path):
    if env.get("SWEBENCH_DOCKER_STDIO_PROXY") == "0" or docker_sdk_usable(env):
        yield
        return

    proxy_dir = Path(
        tempfile.mkdtemp(
            prefix="swebench-docker-",
            dir=env.get("SWEBENCH_DOCKER_PROXY_DIR", os.environ.get("TMPDIR", "/tmp")),
        )
    )
    proxy = DockerStdioProxy(proxy_dir / "docker.sock")
    proxy.start()
    env["DOCKER_HOST"] = f"unix://{proxy.socket_path}"
    print(f"Started Docker stdio proxy at {proxy.socket_path}", flush=True)
    if not docker_sdk_usable(env):
        proxy.stop()
        shutil.rmtree(proxy_dir, ignore_errors=True)
        raise RuntimeError("Docker SDK could not access Docker directly or via stdio proxy")
    try:
        yield
    finally:
        proxy.stop()
        shutil.rmtree(proxy_dir, ignore_errors=True)


def run_capture_json(cmd: list[str], env: dict[str, str], cwd: Path = REPO_ROOT) -> dict[str, Any]:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def write_openhands_llm_config(path: Path, model_config) -> None:
    payload = {
        "model": model_config.litellm_name,
        "api_key": model_config.api_key(),
        "temperature": model_config.temperature,
        "max_output_tokens": model_config.max_tokens,
    }
    if model_config.api_base:
        payload["base_url"] = model_config.api_base
    if model_config.extra_body:
        payload["litellm_extra_body"] = model_config.extra_body
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(payload, indent=2) + "\n")


def remove_files_from_patch(git_patch: str, files: tuple[str, ...]) -> str:
    if not git_patch:
        return git_patch

    diff_matches = list(re.finditer(r"^diff --git [^\n]*\n", git_patch, flags=re.MULTILINE))
    if not diff_matches:
        return git_patch

    kept_diffs = []
    for index, match in enumerate(diff_matches):
        start = match.start()
        end = diff_matches[index + 1].start() if index + 1 < len(diff_matches) else len(git_patch)
        diff = git_patch[start:end]
        header = diff.split("\n", 1)[0]
        parsed = re.match(r"diff --git a/(.+) b/(.+)", header)
        if parsed and (parsed.group(1) in files or parsed.group(2) in files):
            continue
        kept_diffs.append(diff)

    return "".join(kept_diffs)


def convert_openhands_predictions(
    input_path: Path,
    output_path: Path,
    model_name: str,
    instance_ids: list[str] | None = None,
) -> None:
    predictions = []
    seen_instance_ids = set()
    with input_path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            if not instance_id:
                raise RuntimeError(f"{input_path}:{line_number} is missing instance_id")
            test_result = row.get("test_result") or {}
            git_patch = remove_files_from_patch(
                test_result.get("git_patch", ""), OPENHANDS_SETUP_FILES_TO_REMOVE
            )
            seen_instance_ids.add(instance_id)
            predictions.append(
                {
                    "instance_id": instance_id,
                    "model_patch": git_patch,
                    "model_name_or_path": model_name,
                }
            )
    if instance_ids:
        for instance_id in instance_ids:
            if instance_id in seen_instance_ids:
                continue
            predictions.append(
                {
                    "instance_id": instance_id,
                    "model_patch": "",
                    "model_name_or_path": model_name,
                }
            )
        predictions_by_id = {prediction["instance_id"]: prediction for prediction in predictions}
        predictions = [predictions_by_id[instance_id] for instance_id in instance_ids]
    if not predictions:
        raise RuntimeError(f"{input_path} did not contain any OpenHands predictions")
    output_path.write_text(json.dumps(predictions, indent=2) + "\n")


def latest_openhands_output(output_dir: Path) -> Path | None:
    candidates = sorted(
        output_dir.rglob("output.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def patch_openhands_checkout_for_docker_ca(
    command_cwd: Path,
    env: dict[str, str],
    *,
    forward_ca_bundle: bool = True,
) -> bool:
    ca_path = None
    if forward_ca_bundle:
        ca_bundle = env.get("REQUESTS_CA_BUNDLE") or env.get("SSL_CERT_FILE")
        ca_path = Path(ca_bundle) if ca_bundle else None
        if ca_path is not None and not ca_path.is_file():
            ca_path = None

    run_infer = command_cwd / "benchmarks" / "swebenchmultilingual" / "run_infer.py"
    if not run_infer.is_file():
        return False

    text = run_infer.read_text()
    ca_env_block = """                for env_name in (
                    "SSL_CERT_FILE",
                    "REQUESTS_CA_BUNDLE",
                    "CURL_CA_BUNDLE",
                ):
                    if env_name in os.environ and env_name not in docker_forward_env:
                        docker_forward_env.append(env_name)
"""
    ca_and_pager_env_block = """                for env_name in (
                    "SSL_CERT_FILE",
                    "REQUESTS_CA_BUNDLE",
                    "CURL_CA_BUNDLE",
                    "GIT_PAGER",
                    "PAGER",
                    "LESS",
                ):
                    if env_name in os.environ and env_name not in docker_forward_env:
                        docker_forward_env.append(env_name)
"""
    pager_env_block = """            for env_name in ("GIT_PAGER", "PAGER", "LESS"):
                if env_name in os.environ and env_name not in docker_forward_env:
                    docker_forward_env.append(env_name)
"""
    ca_and_pager_env_replacement = ca_env_block + pager_env_block
    if "OPENHANDS_DOCKER_CA_BUNDLE" in text:
        original_text = text
        text = text.replace(ca_and_pager_env_block, ca_and_pager_env_replacement)
        if "GIT_PAGER" not in text and ca_env_block in text:
            text = text.replace(ca_env_block, ca_and_pager_env_replacement)
        if "OPENHANDS_DOCKER_PYTHONPATH" not in text:
            text = text.replace(
                """            workspace = DockerWorkspace(
""",
                """            docker_pythonpath = os.getenv("OPENHANDS_DOCKER_PYTHONPATH")
            if docker_pythonpath:
                docker_volumes.append(f"{docker_pythonpath}:{docker_pythonpath}:ro")
                if "PYTHONPATH" not in docker_forward_env:
                    docker_forward_env.append("PYTHONPATH")
            workspace = DockerWorkspace(
""",
            )
        if text != original_text:
            run_infer.write_text(text)
        if ca_path is not None:
            env.setdefault("OPENHANDS_DOCKER_CA_BUNDLE", str(ca_path))
        return True

    old = """            workspace = DockerWorkspace(
                server_image=agent_server_image,
                working_dir="/workspace",
                forward_env=forward_env or [],
            )
"""
    new = """            docker_forward_env = list(forward_env or [])
            docker_volumes = []
            docker_ca_bundle = os.getenv("OPENHANDS_DOCKER_CA_BUNDLE")
            if docker_ca_bundle:
                docker_volumes.append(f"{docker_ca_bundle}:{docker_ca_bundle}:ro")
                for env_name in (
                    "SSL_CERT_FILE",
                    "REQUESTS_CA_BUNDLE",
                    "CURL_CA_BUNDLE",
                ):
                    if env_name in os.environ and env_name not in docker_forward_env:
                        docker_forward_env.append(env_name)
            for env_name in ("GIT_PAGER", "PAGER", "LESS"):
                if env_name in os.environ and env_name not in docker_forward_env:
                    docker_forward_env.append(env_name)
            docker_pythonpath = os.getenv("OPENHANDS_DOCKER_PYTHONPATH")
            if docker_pythonpath:
                docker_volumes.append(f"{docker_pythonpath}:{docker_pythonpath}:ro")
                if "PYTHONPATH" not in docker_forward_env:
                    docker_forward_env.append("PYTHONPATH")
            workspace = DockerWorkspace(
                server_image=agent_server_image,
                working_dir="/workspace",
                forward_env=docker_forward_env,
                volumes=docker_volumes,
            )
"""
    if old not in text:
        raise RuntimeError(
            f"{run_infer} does not match the expected OpenHands multilingual "
            "DockerWorkspace block; cannot patch CA forwarding"
        )
    run_infer.write_text(text.replace(old, new))
    if ca_path is not None:
        env.setdefault("OPENHANDS_DOCKER_CA_BUNDLE", str(ca_path))
    return True


def patch_openhands_checkout_for_testbed_copy(command_cwd: Path) -> bool:
    run_infer = command_cwd / "benchmarks" / "swebenchmultilingual" / "run_infer.py"
    if not run_infer.is_file():
        return False

    text = run_infer.read_text()
    if "OPENHANDS_TESTBED_COPY_LOCK_CLEANUP" in text:
        return True

    old = """        cp_testebed_repo = workspace.execute_command(
            (f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}")
        )
"""
    new = """        # OPENHANDS_TESTBED_COPY_LOCK_CLEANUP: Rust builds can leave unreadable
        # Cargo incremental lock files that make plain `cp -r /testbed/.` fail.
        cp_testebed_repo = workspace.execute_command(
            (
                "sudo find /testbed -path '*/target/*/incremental/*.lock' -delete "
                "2>/dev/null || true; "
                f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}"
            )
        )
"""
    if old not in text:
        raise RuntimeError(
            f"{run_infer} does not match the expected OpenHands testbed copy block; "
            "cannot patch Cargo lock cleanup"
        )
    run_infer.write_text(text.replace(old, new))
    return True


def write_openhands_docker_sitecustomize(output_dir: Path, env: dict[str, str]) -> Path:
    site_dir = output_dir / "openhands_docker_sitecustomize"
    site_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = site_dir / "sitecustomize.py"
    sitecustomize.write_text(
        "\n".join(
            [
                "import ssl",
                "",
                "_original_create_default_context = ssl.create_default_context",
                "",
                "",
                "def create_default_context(*args, **kwargs):",
                "    context = _original_create_default_context(*args, **kwargs)",
                "    if hasattr(ssl, 'VERIFY_X509_STRICT'):",
                "        context.verify_flags &= ~ssl.VERIFY_X509_STRICT",
                "    return context",
                "",
                "",
                "ssl.create_default_context = create_default_context",
                "",
            ]
        )
    )
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([str(site_dir), existing_pythonpath])
    else:
        env["PYTHONPATH"] = str(site_dir)
    env["OPENHANDS_DOCKER_PYTHONPATH"] = str(site_dir)
    return sitecustomize


def is_openhands_source_checkout(command_cwd: Path) -> bool:
    return (command_cwd / "benchmarks" / "swebenchmultilingual" / "run_infer.py").is_file()


def openhands_venv_python(command_cwd: Path) -> Path | None:
    candidates = [
        command_cwd / ".venv" / "bin" / "python",
        command_cwd / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def patch_openhands_checkout_for_docker_platform(command_cwd: Path) -> bool:
    workspace_py = (
        command_cwd
        / "vendor"
        / "software-agent-sdk"
        / "openhands-workspace"
        / "openhands"
        / "workspace"
        / "docker"
        / "workspace.py"
    )
    if not workspace_py.is_file():
        return False

    text = workspace_py.read_text()
    if "OPENHANDS_DOCKER_OMIT_PLATFORM" in text:
        return True

    old = """        run_cmd = [
            "docker",
            "run",
            "-d",
            "--platform",
            self.platform,
            "--rm",
"""
    new = """        platform_flags = []
        if os.getenv("OPENHANDS_DOCKER_OMIT_PLATFORM") != "1":
            platform_flags = ["--platform", self.platform]

        run_cmd = [
            "docker",
            "run",
            "-d",
            *platform_flags,
            "--rm",
"""
    if old not in text:
        raise RuntimeError(
            f"{workspace_py} does not match the expected Docker run block; "
            "cannot patch platform omission"
        )
    workspace_py.write_text(text.replace(old, new))
    return True


def patch_openhands_checkout_for_agent_server_platform(command_cwd: Path) -> bool:
    build_py = (
        command_cwd
        / "vendor"
        / "software-agent-sdk"
        / "openhands-agent-server"
        / "openhands"
        / "agent_server"
        / "docker"
        / "build.py"
    )
    if not build_py.is_file():
        return False

    text = build_py.read_text()
    if "OPENHANDS_AGENT_SERVER_PLATFORM" in text:
        return True

    old = """    if push:
        args += ["--platform", ",".join(opts.platforms), "--push"]
    else:
        args += ["--load"]
"""
    new = """    local_platform = os.getenv("OPENHANDS_AGENT_SERVER_PLATFORM")
    if push:
        args += ["--platform", ",".join(opts.platforms), "--push"]
    else:
        if local_platform:
            args += ["--platform", local_platform]
        args += ["--load"]
"""
    if old not in text:
        raise RuntimeError(
            f"{build_py} does not match the expected buildx platform block; "
            "cannot patch local agent-server platform"
        )

    text = text.replace(old, new)
    old_log = """        f"for platforms='{opts.platforms if push else 'local-arch'}'"
"""
    new_log = """        f"for platforms='{opts.platforms if push else (local_platform or 'local-arch')}'"
"""
    if old_log in text:
        text = text.replace(old_log, new_log)
    build_py.write_text(text)
    return True


def ensure_openhands_dataset_dependency(command_cwd: Path, env: dict[str, str]) -> bool:
    if not is_openhands_source_checkout(command_cwd):
        return False
    python = openhands_venv_python(command_cwd)
    if python is None:
        return False

    result = subprocess.run(
        [
            str(python),
            "-c",
            "import datasets; print(datasets.__version__)",
        ],
        cwd=command_cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    version = (result.stdout or "").strip().splitlines()[-1] if result.stdout else ""
    try:
        parts = tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        parts = ()
    if parts >= (4, 5):
        env.setdefault("UV_NO_SYNC", "1")
        return False

    run_in_dir(
        ["uv", "pip", "install", "--python", str(python), "datasets>=4.5.0"],
        env,
        command_cwd,
        timeout=600,
    )
    env.setdefault("UV_NO_SYNC", "1")
    return True


def render_template_command(template: str, values: dict[str, str]) -> list[str]:
    rendered = template.format(**{key: shlex.quote(value) for key, value in values.items()})
    return shlex.split(rendered)


def safe_instance_name(instance_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", instance_id).strip("-")


class LocalTmuxSession:
    def __init__(
        self,
        session_name: str,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
    ) -> None:
        self._session_name = session_name
        self._cwd = cwd
        self._env = env
        self._log_path = log_path
        self._socket_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", session_name)[:80]
        self._previous_buffer: str | None = None
        self._started_at = time.time()

    def _run_tmux(
        self, command: list[str], *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["tmux", "-L", self._socket_name, *command],
            env=self._env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, ["tmux", *command], output=result.stdout
            )
        return result

    def start(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_tmux(
            [
                "new-session",
                "-x",
                "160",
                "-y",
                "40",
                "-d",
                "-s",
                self._session_name,
                "-c",
                str(self._cwd),
                ";",
                "set-option",
                "-t",
                self._session_name,
                "history-limit",
                "50000",
                ";",
                "pipe-pane",
                "-t",
                self._session_name,
                f"cat > {shlex.quote(str(self._log_path))}",
            ]
        )

    def stop(self) -> None:
        self._run_tmux(["kill-session", "-t", self._session_name], check=False)
        self._run_tmux(["kill-server"], check=False)

    def is_session_alive(self) -> bool:
        result = self._run_tmux(["has-session", "-t", self._session_name], check=False)
        return result.returncode == 0

    def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> None:
        if isinstance(keys, str):
            keys = [keys]
        start_time = time.time()
        self._run_tmux(["send-keys", "-t", self._session_name, *keys])
        elapsed = time.time() - start_time
        if elapsed < min_timeout_sec:
            time.sleep(min_timeout_sec - elapsed)

    def capture_pane(self, capture_entire: bool = False) -> str:
        extra_args = ["-S", "-"] if capture_entire else []
        result = self._run_tmux(
            ["capture-pane", "-p", *extra_args, "-t", self._session_name],
            check=False,
        )
        if result.returncode != 0:
            return self._previous_buffer or ""
        return result.stdout

    def get_incremental_output(self) -> str:
        current_buffer = self.capture_pane(capture_entire=True)
        if self._previous_buffer is None:
            self._previous_buffer = current_buffer
            return f"Current Terminal Screen:\n{self._get_visible_screen()}"

        new_content = self._find_new_content(current_buffer)
        self._previous_buffer = current_buffer

        if new_content is not None:
            if new_content.strip():
                return f"New Terminal Output:\n{new_content}"
            return f"Current Terminal Screen:\n{self._get_visible_screen()}"
        return f"Current Terminal Screen:\n{self._get_visible_screen()}"

    def _find_new_content(self, current_buffer: str) -> str | None:
        if self._previous_buffer is None:
            return None

        previous_buffer = self._previous_buffer.strip()
        if previous_buffer in current_buffer:
            index = current_buffer.index(previous_buffer)
            return current_buffer[index + len(previous_buffer) :]
        return None

    def _get_visible_screen(self) -> str:
        return self.capture_pane(capture_entire=False)

    def get_asciinema_timestamp(self) -> float:
        return time.time() - self._started_at


def load_swebench_instances(instance_ids: list[str]) -> list[dict[str, Any]]:
    configure_ca_bundle(os.environ)
    from datasets import load_dataset

    wanted = set(instance_ids)
    rows = [
        dict(row)
        for row in load_dataset(DATASET_NAME, split="test")
        if row["instance_id"] in wanted
    ]
    found = {row["instance_id"] for row in rows}
    missing = [instance_id for instance_id in instance_ids if instance_id not in found]
    if missing:
        raise RuntimeError(f"{DATASET_NAME} test split is missing instance ids: {missing}")
    rows_by_id = {row["instance_id"]: row for row in rows}
    return [rows_by_id[instance_id] for instance_id in instance_ids]


def derive_opencode_model(model_config) -> str:
    model_name = model_config.openai_model
    if model_config.api_key_env == "DEEPSEEK_API_KEY":
        if model_name.startswith("deepseek/"):
            return model_name
        return f"deepseek/{model_name}"
    if model_config.api_base:
        return f"deepswe/{model_name}"
    if "/" in model_name:
        return model_name
    return f"openai/{model_name}"


def build_opencode_config_content(model_config, opencode_model: str) -> str:
    provider_id, _, model_id = opencode_model.partition("/")
    model_entry: dict[str, Any] = {
        "reasoning": False,
        "limit": {
            "context": OPENCODE_DEFAULT_CONTEXT_LIMIT,
            "output": max(model_config.max_tokens, OPENCODE_MIN_OUTPUT_LIMIT),
        },
    }
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "model": opencode_model,
        "small_model": opencode_model,
        "default_agent": OPENCODE_BENCHMARK_AGENT,
        "autoupdate": False,
        "share": "disabled",
        "permission": {
            "question": "deny",
            "task": "deny",
        },
        "agent": {
            OPENCODE_BENCHMARK_AGENT: {
                "mode": "primary",
                "model": opencode_model,
                "temperature": model_config.temperature,
                "description": "Solves SWE-bench tasks directly in the benchmark worktree.",
                "prompt": OPENCODE_BENCHMARK_AGENT_PROMPT,
                "permission": {
                    "question": "deny",
                    "task": "deny",
                },
            }
        },
    }
    provider: dict[str, Any] = {}
    if provider_id == "deepswe":
        provider[provider_id] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "DeepSWE OpenAI-Compatible",
            "options": {
                "baseURL": model_config.api_base,
                "apiKey": f"{{env:{model_config.api_key_env}}}",
            },
            "models": {
                model_id: {
                    "name": model_id,
                    **model_entry,
                }
            },
        }
    else:
        provider[provider_id] = {"models": {model_id: model_entry}}
        if model_config.api_key_env != "OPENAI_API_KEY":
            provider[provider_id]["options"] = {
                "apiKey": f"{{env:{model_config.api_key_env}}}"
            }
    if provider:
        config["provider"] = provider
    return json.dumps(config, separators=(",", ":"))


def opencode_prompt(row: dict[str, Any]) -> str:
    prompt_parts = [
        f"You are solving a {BENCHMARK_DISPLAY_NAME} task.",
        "The repository is already checked out at the base commit.",
        "Modify files in this worktree to fix the issue. Do not commit changes.",
        "Work directly with read/search/bash/edit/write tools; do not use subagents.",
        "Keep exploration brief, then make the required code change.",
        "After editing, inspect git diff and finish any incomplete references before stopping.",
        "Do not install toolchains or debug local environment setup; the benchmark harness will run tests.",
        "When you are finished, stop; the benchmark harness will run tests.",
        "Keep your output concise and do not include lengthy analysis.",
        "",
        "Problem statement:",
        row["problem_statement"],
    ]
    hints = (row.get("hints_text") or "").strip()
    if hints:
        prompt_parts.extend(["", "Hints:", hints])
    return "\n".join(prompt_parts)


def prepare_opencode_worktree(
    row: dict[str, Any], worktree: Path, env: dict[str, str]
) -> None:
    if worktree.exists():
        shutil.rmtree(worktree)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://github.com/{row['repo']}.git"
    run_in_dir(["git", "init", str(worktree)], env, REPO_ROOT)
    run_in_dir(["git", "remote", "add", "origin", repo_url], env, worktree)
    run_in_dir(
        ["git", "fetch", "--depth", "1", "origin", row["base_commit"]],
        env,
        worktree,
        timeout=1800,
    )
    run_in_dir(["git", "checkout", "--detach", row["base_commit"]], env, worktree)
    skip_optional_opencode_submodules(row, worktree, env)
    run_in_dir(
        ["git", "submodule", "update", "--init", "--recursive"],
        env,
        worktree,
        timeout=1800,
    )


def skip_optional_opencode_submodules(
    row: dict[str, Any], worktree: Path, env: dict[str, str]
) -> None:
    optional_submodules = OPENCODE_OPTIONAL_SUBMODULES_TO_SKIP.get(row["repo"], {})
    if not optional_submodules or not (worktree / ".gitmodules").is_file():
        return

    for submodule_name, expected_url in optional_submodules.items():
        result = run_in_dir(
            [
                "git",
                "config",
                "--file",
                ".gitmodules",
                "--get",
                f"submodule.{submodule_name}.url",
            ],
            env,
            worktree,
            check=False,
        )
        actual_url = (result.stdout or "").strip().rstrip("/")
        if result.returncode != 0 or actual_url.lower() != expected_url.lower():
            continue
        run_in_dir(
            ["git", "config", f"submodule.{submodule_name}.update", "none"],
            env,
            worktree,
        )
        print(
            "warning: skipping optional unavailable submodule "
            f"{submodule_name} for {row['instance_id']}",
            flush=True,
        )


def opencode_instance_env(
    base_env: dict[str, str],
    workspace_root: Path,
    instance_id: str,
    model_config,
    opencode_model: str,
    opencode_config: Path | None,
) -> dict[str, str]:
    env = base_env.copy()
    original_home = env.get("HOME")
    state_name = safe_instance_name(instance_id)
    env["HOME"] = str(workspace_root / "home" / state_name)
    env["XDG_DATA_HOME"] = str(workspace_root / "xdg-data" / state_name)
    env["XDG_CONFIG_HOME"] = str(workspace_root / "xdg-config" / state_name)
    env["XDG_CACHE_HOME"] = str(workspace_root / "xdg-cache" / state_name)
    if original_home:
        original_home_path = Path(original_home)
        for env_name, dirname in (
            ("CARGO_HOME", ".cargo"),
            ("RUSTUP_HOME", ".rustup"),
        ):
            tool_home = original_home_path / dirname
            if tool_home.exists():
                env.setdefault(env_name, str(tool_home))
    env.setdefault("OPENCODE_DISABLE_UPDATE", "1")
    if opencode_config is not None:
        env["OPENCODE_CONFIG"] = str(opencode_config)
        env.pop("OPENCODE_CONFIG_CONTENT", None)
    else:
        env["OPENCODE_CONFIG_CONTENT"] = build_opencode_config_content(
            model_config, opencode_model
        )
    return env


def collect_git_patch(worktree: Path, env: dict[str, str]) -> str:
    run_in_dir(["git", "add", "-A"], env, worktree)
    result = run_in_dir(
        ["git", "diff", "--cached", "--binary"],
        env,
        worktree,
        check=True,
    )
    return result.stdout


def run_opencode_instance(
    row: dict[str, Any],
    args: argparse.Namespace,
    model_config,
    workspace_root: Path,
    base_env: dict[str, str],
    opencode_model: str,
) -> dict[str, str]:
    instance_id = row["instance_id"]
    safe_name = safe_instance_name(instance_id)
    worktree = workspace_root / "worktrees" / safe_name
    log_path = workspace_root / "logs" / f"{safe_name}.log"
    (workspace_root / "home" / safe_name).mkdir(parents=True, exist_ok=True)
    instance_env = opencode_instance_env(
        base_env,
        workspace_root,
        instance_id,
        model_config,
        opencode_model,
        args.opencode_config,
    )
    prepare_opencode_worktree(row, worktree, instance_env)
    command = [
        *shlex.split(args.opencode_command),
        "run",
        "--model",
        opencode_model,
        "--dir",
        str(worktree),
        "--dangerously-skip-permissions",
    ]
    if args.opencode_agent:
        command.extend(["--agent", args.opencode_agent])
    elif args.opencode_config is None:
        command.extend(["--agent", OPENCODE_BENCHMARK_AGENT])
    if args.opencode_variant:
        command.extend(["--variant", args.opencode_variant])
    for extra_arg in args.opencode_extra_arg:
        command.extend(shlex.split(extra_arg))
    command.append(opencode_prompt(row))
    run_in_dir(
        command,
        instance_env,
        worktree,
        timeout=args.opencode_timeout,
        log_path=log_path,
        check=True,
    )
    return {
        "instance_id": instance_id,
        "model_patch": collect_git_patch(worktree, instance_env),
        "model_name_or_path": model_config.slug,
    }


def build_evaluation_command(
    python: str,
    predictions_path: Path,
    instance_ids: list[str],
    eval_workers: int,
    run_id: str,
    defaults: dict,
) -> list[str]:
    return [
        python,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        DATASET_NAME,
        "--split",
        "test",
        "--instance_ids",
        *instance_ids,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(eval_workers),
        "--run_id",
        run_id,
        "--cache_level",
        defaults["evaluation_cache_level"],
        "--clean",
        "False",
        "--timeout",
        str(defaults["evaluation_timeout_seconds"]),
    ]


def run_minisweagent_generation(
    args: argparse.Namespace,
    defaults: dict,
    model_config,
    output_dir: Path,
    filter_regex: str,
    python: str,
) -> dict[str, str]:
    generation_env = model_config.generation_env()
    prepend_minisweagent_overlay(generation_env)
    extra_body = json.dumps(model_config.extra_body, separators=(",", ":"))
    generation_cmd = [
        python,
        "-m",
        "minisweagent.run.benchmarks.swebench",
        "--subset",
        MINISWEAGENT_SUBSET,
        "--split",
        "test",
        "--filter",
        filter_regex,
        "--output",
        str(output_dir),
        "--workers",
        str(args.generation_workers),
        "--model",
        model_config.litellm_name,
        "--model-class",
        "litellm",
        "-c",
        "swebench.yaml",
        "-c",
        f"model.model_kwargs.temperature={model_config.temperature}",
        "-c",
        f"model.model_kwargs.max_tokens={model_config.max_tokens}",
        "-c",
        "environment.pull_timeout=1800",
        "-c",
        f"agent.step_limit={args.generation_step_limit}",
    ]
    if model_config.api_base:
        generation_cmd.extend(["-c", f"model.model_kwargs.api_base={model_config.api_base}"])
    if model_config.extra_body:
        generation_cmd.extend(["-c", f"model.model_kwargs.extra_body={extra_body}"])
    if args.generation_workers == 1:
        generation_cmd.extend(["-c", f"agent.output_path={output_dir / 'live.traj.json'}"])
    run(generation_cmd, generation_env)
    return {"predictions_path": str(output_dir / "preds.json"), "env": generation_env}


def run_openhands_generation(
    args: argparse.Namespace,
    model_config,
    output_dir: Path,
    instance_ids_path: Path,
) -> dict[str, str]:
    output_json = args.openhands_output_json
    if output_json is None:
        llm_config_path = args.openhands_llm_config or (output_dir / "openhands_llm_config.json")
        generated_llm_config = args.openhands_llm_config is None
        if generated_llm_config:
            generation_env = model_config.generation_env()
            write_openhands_llm_config(llm_config_path, model_config)
        else:
            generation_env = default_env()
        generation_env.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
        try:
            generation_cmd = [
                *shlex.split(args.openhands_infer_command),
                str(llm_config_path),
                "--dataset",
                DATASET_NAME,
                "--split",
                "test",
                "--select",
                str(instance_ids_path),
                "--workspace",
                args.openhands_workspace,
                "--num-workers",
                str(args.generation_workers),
                "--max-iterations",
                str(args.openhands_max_iterations),
                "--output-dir",
                str(output_dir / "openhands"),
                "--n-critic-runs",
                str(args.openhands_n_critic_runs),
                "--max-retries",
                str(args.openhands_max_retries),
                "--tool-preset",
                args.openhands_tool_preset,
            ]
            if args.openhands_enable_delegation:
                generation_cmd.append("--enable-delegation")
            for extra_arg in args.openhands_extra_arg:
                generation_cmd.extend(shlex.split(extra_arg))
            command_cwd = args.openhands_command_cwd or REPO_ROOT
            if getattr(args, "openhands_fix_datasets_dependency", True):
                ensure_openhands_dataset_dependency(command_cwd, generation_env)
            if args.openhands_workspace == "docker":
                generation_env.setdefault("GIT_PAGER", "cat")
                generation_env.setdefault("PAGER", "cat")
                generation_env.setdefault("LESS", "-F -X")
                write_openhands_docker_sitecustomize(output_dir, generation_env)
            if args.openhands_workspace == "docker":
                patch_openhands_checkout_for_docker_ca(
                    command_cwd,
                    generation_env,
                    forward_ca_bundle=getattr(args, "openhands_forward_ca_bundle", True),
                )
                patch_openhands_checkout_for_testbed_copy(command_cwd)
                patch_openhands_checkout_for_docker_platform(command_cwd)
                patch_openhands_checkout_for_agent_server_platform(command_cwd)
                generation_env.setdefault("OPENHANDS_AGENT_SERVER_PLATFORM", "linux/amd64")
            payload = run_capture_json(generation_cmd, generation_env, cwd=command_cwd)
            if payload.get("output_json"):
                output_json = Path(payload["output_json"])
            else:
                output_json = latest_openhands_output(output_dir / "openhands")
        finally:
            if generated_llm_config:
                llm_config_path.unlink(missing_ok=True)
    else:
        generation_env = default_env()
    if output_json is None:
        raise RuntimeError("OpenHands did not report output_json and no output.jsonl was found")
    predictions_path = output_dir / "preds.json"
    selected_instance_ids = (
        read_instance_ids(instance_ids_path) if instance_ids_path.exists() else None
    )
    convert_openhands_predictions(
        Path(output_json),
        predictions_path,
        model_config.slug,
        selected_instance_ids,
    )
    return {"predictions_path": str(predictions_path), "env": generation_env}


def run_opencode_generation(
    args: argparse.Namespace,
    model_config,
    output_dir: Path,
    instance_ids_path: Path,
    instance_ids: list[str],
    filter_regex: str,
) -> dict[str, str]:
    generation_env = model_config.generation_env()
    generation_env.setdefault(model_config.api_key_env, model_config.api_key())
    predictions_path = output_dir / "preds.json"
    predictions_path.unlink(missing_ok=True)
    if args.opencode_command_template:
        values = {
            "api_base": model_config.api_base or "",
            "api_key_env": model_config.api_key_env,
            "filter_regex": filter_regex,
            "instance_ids": ",".join(instance_ids),
            "instance_ids_path": str(instance_ids_path),
            "litellm_model": model_config.litellm_name,
            "max_tokens": str(model_config.max_tokens),
            "model": model_config.openai_model,
            "opencode_model": args.opencode_model or derive_opencode_model(model_config),
            "output_dir": str(output_dir),
            "predictions_path": str(predictions_path),
            "temperature": str(model_config.temperature),
            "workers": str(args.generation_workers),
        }
        run(render_template_command(args.opencode_command_template, values), generation_env)
        if not predictions_path.exists():
            raise RuntimeError(
                f"opencode command completed but did not create predictions at {predictions_path}"
            )
        return {"predictions_path": str(predictions_path), "env": generation_env}

    rows = load_swebench_instances(instance_ids)
    workspace_root = args.opencode_workspace or (output_dir / "opencode")
    opencode_model = args.opencode_model or derive_opencode_model(model_config)
    predictions: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.generation_workers) as executor:
        futures = [
            executor.submit(
                run_opencode_instance,
                row,
                args,
                model_config,
                workspace_root,
                generation_env,
                opencode_model,
            )
            for row in rows
        ]
        for future in concurrent.futures.as_completed(futures):
            predictions.append(future.result())
            predictions_path.write_text(json.dumps(predictions, indent=2) + "\n")

    predictions_by_id = {prediction["instance_id"]: prediction for prediction in predictions}
    ordered_predictions = [predictions_by_id[instance_id] for instance_id in instance_ids]
    predictions_path.write_text(json.dumps(ordered_predictions, indent=2) + "\n")
    return {"predictions_path": str(predictions_path), "env": generation_env}


def terminus_instruction(row: dict[str, Any]) -> str:
    prompt_parts = [
        f"You are solving a {BENCHMARK_DISPLAY_NAME} task.",
        "The repository is already checked out at the base commit in the current directory.",
        "Modify files in this worktree to fix the issue. Do not commit changes.",
        "Keep exploration brief, then make the required code change.",
        "After editing, inspect git diff and finish any incomplete references before marking the task complete.",
        "The benchmark harness will run tests after you stop.",
        "",
        "Problem statement:",
        row["problem_statement"],
    ]
    hints = (row.get("hints_text") or "").strip()
    if hints:
        prompt_parts.extend(["", "Hints:", hints])
    return "\n".join(prompt_parts)


def is_sensitive_env_name(env_name: str) -> bool:
    normalized = env_name.upper()
    sensitive_fragments = ("API_KEY", "ACCESS_KEY", "SECRET", "TOKEN", "PASSWORD")
    return any(fragment in normalized for fragment in sensitive_fragments)


def terminus_terminal_env(
    base_env: dict[str, str], workspace_root: Path, instance_id: str
) -> dict[str, str]:
    state_name = safe_instance_name(instance_id)
    home = workspace_root / "home" / state_name
    xdg_data = workspace_root / "xdg-data" / state_name
    xdg_config = workspace_root / "xdg-config" / state_name
    xdg_cache = workspace_root / "xdg-cache" / state_name
    tmp = workspace_root / "tmp" / state_name
    for path in (home, xdg_data, xdg_config, xdg_cache, tmp):
        path.mkdir(parents=True, exist_ok=True)

    allowlisted_env = {
        "CURL_CA_BUNDLE",
        "GIT_SSL_CAINFO",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NODE_EXTRA_CA_CERTS",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
    }
    env = {
        env_name: value
        for env_name, value in base_env.items()
        if env_name in allowlisted_env and not is_sensitive_env_name(env_name)
    }
    env["HOME"] = str(home)
    env["XDG_DATA_HOME"] = str(xdg_data)
    env["XDG_CONFIG_HOME"] = str(xdg_config)
    env["XDG_CACHE_HOME"] = str(xdg_cache)
    env["TMPDIR"] = str(tmp)
    env.setdefault("TERM", "screen-256color")
    return env


def terminus_api_key(args: argparse.Namespace, model_config) -> str:
    api_key_env = args.terminus_api_key_env or model_config.api_key_env
    if api_key_env == model_config.api_key_env:
        return model_config.api_key()
    value = os.environ.get(api_key_env)
    if not value:
        raise RuntimeError(f"{api_key_env} must be set for --terminus-api-key-env")
    return value


def terminus_model_label(args: argparse.Namespace, model_config) -> str:
    if args.terminus_model:
        return safe_instance_name(args.terminus_model).lower()
    return model_config.slug


def run_terminus_instance(
    row: dict[str, Any],
    args: argparse.Namespace,
    model_config,
    workspace_root: Path,
    base_env: dict[str, str],
) -> dict[str, str]:
    from eval.terminal_bench.agents.terminus_2.terminus_2 import Terminus2

    instance_id = row["instance_id"]
    safe_name = safe_instance_name(instance_id)
    worktree = workspace_root / "worktrees" / safe_name
    log_root = workspace_root / "logs" / safe_name
    terminal_log = log_root / "terminal.log"
    prepare_opencode_worktree(row, worktree, base_env)

    session_name = (
        f"deepswe-terminus-{safe_name[:40]}-{os.getpid()}-{int(time.time() * 1000)}"
    )
    session = LocalTmuxSession(
        session_name=session_name,
        cwd=worktree,
        env=terminus_terminal_env(base_env, workspace_root, instance_id),
        log_path=terminal_log,
    )
    agent = Terminus2(
        model_name=args.terminus_model or model_config.litellm_name,
        max_episodes=args.terminus_max_episodes,
        parser_name=args.terminus_parser,
        api_base=args.terminus_api_base or model_config.api_base,
        api_key=terminus_api_key(args, model_config),
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        extra_body=model_config.extra_body,
        request_timeout=args.terminus_request_timeout,
    )
    session.start()
    try:
        agent.perform_task(
            terminus_instruction(row),
            session,
            logging_dir=log_root / "episodes",
        )
    finally:
        session.stop()

    return {
        "instance_id": instance_id,
        "model_patch": collect_git_patch(worktree, base_env),
        "model_name_or_path": terminus_model_label(args, model_config),
    }


def run_terminus_generation(
    args: argparse.Namespace,
    model_config,
    output_dir: Path,
    instance_ids: list[str],
) -> dict[str, str]:
    generation_env = model_config.generation_env()
    predictions_path = output_dir / "preds.json"
    predictions_path.unlink(missing_ok=True)
    rows = load_swebench_instances(instance_ids)
    workspace_root = args.terminus_workspace or (output_dir / "terminus-2")
    predictions: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.generation_workers) as executor:
        futures = [
            executor.submit(
                run_terminus_instance,
                row,
                args,
                model_config,
                workspace_root,
                generation_env,
            )
            for row in rows
        ]
        for future in concurrent.futures.as_completed(futures):
            predictions.append(future.result())
            predictions_path.write_text(json.dumps(predictions, indent=2) + "\n")

    predictions_by_id = {prediction["instance_id"]: prediction for prediction in predictions}
    ordered_predictions = [predictions_by_id[instance_id] for instance_id in instance_ids]
    predictions_path.write_text(json.dumps(ordered_predictions, indent=2) + "\n")
    return {"predictions_path": str(predictions_path), "env": generation_env}


def main() -> None:
    defaults = load_defaults(DEFAULTS_PATH)
    parser = argparse.ArgumentParser(
        description=(
            "Run an OpenAI-compatible model on the "
            f"{BENCHMARK_DISPLAY_NAME} predictive subset."
        )
    )
    parser.add_argument(
        "--harness",
        choices=SUPPORTED_HARNESSES,
        default=defaults.get("harness", "mini-swe-agent"),
        help="Generation harness to use before official SWE-bench evaluation.",
    )
    parser.add_argument("--instance-ids", type=Path, default=DEFAULT_INSTANCE_IDS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--generation-workers",
        type=int,
        default=defaults["generation_workers"],
        help="mini-swe-agent generation workers.",
    )
    parser.add_argument(
        "--generation-step-limit",
        type=int,
        default=defaults["generation_step_limit"],
        help="Maximum mini-swe-agent steps per instance.",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=defaults["eval_workers"],
        help="SWE-bench evaluation workers.",
    )
    parser.add_argument(
        "--openhands-infer-command",
        default=defaults.get("openhands_infer_command", "swebenchmultilingual-infer"),
        help="OpenHands SWE-bench Multilingual inference command.",
    )
    parser.add_argument(
        "--openhands-command-cwd",
        type=Path,
        default=defaults.get("openhands_command_cwd"),
        help="Working directory for the OpenHands inference command.",
    )
    parser.add_argument(
        "--openhands-llm-config",
        type=Path,
        default=None,
        help="Existing OpenHands LLM config JSON. Defaults to a generated file in output dir.",
    )
    parser.add_argument(
        "--openhands-output-json",
        type=Path,
        default=None,
        help="Existing OpenHands output.jsonl to convert instead of running inference.",
    )
    parser.add_argument(
        "--openhands-workspace",
        choices=["docker", "remote", "apptainer"],
        default=defaults.get("openhands_workspace", "docker"),
    )
    parser.add_argument(
        "--openhands-max-iterations",
        type=int,
        default=defaults.get("openhands_max_iterations", defaults["generation_step_limit"]),
    )
    parser.add_argument(
        "--openhands-n-critic-runs",
        type=int,
        default=defaults.get("openhands_n_critic_runs", 1),
    )
    parser.add_argument(
        "--openhands-max-retries",
        type=int,
        default=defaults.get("openhands_max_retries", 3),
    )
    parser.add_argument(
        "--openhands-tool-preset",
        default=defaults.get("openhands_tool_preset", "default"),
        choices=["default", "gemini", "gpt5", "planning"],
    )
    parser.add_argument(
        "--openhands-forward-ca-bundle",
        dest="openhands_forward_ca_bundle",
        action="store_true",
        default=defaults.get("openhands_forward_ca_bundle", True),
        help="Mount and forward the configured CA bundle into OpenHands Docker workspaces.",
    )
    parser.add_argument(
        "--no-openhands-forward-ca-bundle",
        dest="openhands_forward_ca_bundle",
        action="store_false",
        help="Disable OpenHands Docker CA bundle forwarding.",
    )
    parser.add_argument(
        "--openhands-fix-datasets-dependency",
        dest="openhands_fix_datasets_dependency",
        action="store_true",
        default=defaults.get("openhands_fix_datasets_dependency", True),
        help="Upgrade OpenHands source-checkout datasets dependency when its lockfile is too old.",
    )
    parser.add_argument(
        "--no-openhands-fix-datasets-dependency",
        dest="openhands_fix_datasets_dependency",
        action="store_false",
        help="Do not modify the OpenHands source-checkout datasets dependency.",
    )
    parser.add_argument("--openhands-enable-delegation", action="store_true")
    parser.add_argument(
        "--openhands-extra-arg",
        action="append",
        default=[],
        help="Additional argument(s) to pass to OpenHands inference.",
    )
    parser.add_argument(
        "--opencode-command-template",
        default=defaults.get("opencode_command_template"),
        help=(
            "Optional custom opencode generation command. It must write "
            "SWE-bench predictions to {predictions_path}. If omitted, this "
            "runner executes opencode natively for each instance."
        ),
    )
    parser.add_argument(
        "--opencode-command",
        default=defaults.get("opencode_command", OPENCODE_DEFAULT_COMMAND),
        help="opencode executable command used by the native opencode harness.",
    )
    parser.add_argument(
        "--opencode-model",
        default=defaults.get("opencode_model"),
        help="opencode model id in provider/model format. Defaults are inferred from model config.",
    )
    parser.add_argument(
        "--opencode-config",
        type=Path,
        default=defaults.get("opencode_config"),
        help="Existing opencode config file. Defaults to generated OPENCODE_CONFIG_CONTENT.",
    )
    parser.add_argument(
        "--opencode-workspace",
        type=Path,
        default=defaults.get("opencode_workspace"),
        help="Directory for opencode worktrees, logs, and per-instance state.",
    )
    parser.add_argument(
        "--opencode-timeout",
        type=int,
        default=defaults.get("opencode_timeout", 3600),
        help="Per-instance opencode timeout in seconds.",
    )
    parser.add_argument(
        "--opencode-agent",
        default=defaults.get("opencode_agent"),
        help="Optional opencode agent to use.",
    )
    parser.add_argument(
        "--opencode-variant",
        default=defaults.get("opencode_variant"),
        help="Optional opencode model variant, such as high or minimal.",
    )
    parser.add_argument(
        "--opencode-extra-arg",
        action="append",
        default=[],
        help="Additional argument(s) to pass to opencode run.",
    )
    parser.add_argument(
        "--terminus-model",
        default=defaults.get("terminus_model"),
        help="LiteLLM model name for Terminus 2. Defaults to the benchmark model config.",
    )
    parser.add_argument(
        "--terminus-api-base",
        default=defaults.get("terminus_api_base"),
        help="Optional API base URL for --terminus-model. Defaults to the benchmark model config.",
    )
    parser.add_argument(
        "--terminus-api-key-env",
        default=defaults.get("terminus_api_key_env"),
        help="Optional API key environment variable for --terminus-model. Defaults to the benchmark model config.",
    )
    parser.add_argument(
        "--terminus-parser",
        choices=["json", "xml"],
        default=defaults.get("terminus_parser", "json"),
        help="Terminus 2 response parser format.",
    )
    parser.add_argument(
        "--terminus-max-episodes",
        type=int,
        default=defaults.get("terminus_max_episodes", defaults["generation_step_limit"]),
        help="Maximum Terminus 2 episodes per instance.",
    )
    parser.add_argument(
        "--terminus-request-timeout",
        type=float,
        default=defaults.get("terminus_request_timeout"),
        help="Optional LiteLLM request timeout in seconds for Terminus 2.",
    )
    parser.add_argument(
        "--terminus-workspace",
        type=Path,
        default=defaults.get("terminus_workspace"),
        help="Directory for Terminus 2 worktrees, logs, and per-instance state.",
    )
    add_model_args(parser)
    args = parser.parse_args()
    resolve_cli_paths(args)
    model_config = model_from_defaults(defaults, args)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output or (
        REPO_ROOT / "runs" / f"{model_config.slug}-{BENCHMARK_RUN_TAG}-{timestamp}"
    )
    run_id = args.run_id or f"{model_config.slug}-{BENCHMARK_RUN_TAG}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_ids = read_instance_ids(args.instance_ids)
    filter_regex = make_filter_regex(instance_ids)
    python = python_executable()

    if not args.skip_generation:
        if args.harness == "mini-swe-agent":
            generation_result = run_minisweagent_generation(
                args, defaults, model_config, output_dir, filter_regex, python
            )
        elif args.harness == "openhands-swe":
            generation_result = run_openhands_generation(
                args, model_config, output_dir, args.instance_ids
            )
        elif args.harness == "opencode":
            generation_result = run_opencode_generation(
                args, model_config, output_dir, args.instance_ids, instance_ids, filter_regex
            )
        elif args.harness == "terminus-2":
            generation_result = run_terminus_generation(
                args, model_config, output_dir, instance_ids
            )
        else:
            raise ValueError(args.harness)
        generation_env = generation_result["env"]
        predictions_path = Path(generation_result["predictions_path"])
    else:
        predictions_path = output_dir / "preds.json"

    if not args.skip_evaluation:
        evaluation_env = dict(generation_env if not args.skip_generation else {})
        if not evaluation_env:
            evaluation_env = default_env()
        evaluation_cmd = build_evaluation_command(
            python, predictions_path, instance_ids, args.eval_workers, run_id, defaults
        )
        with docker_evaluation_env(evaluation_env, output_dir):
            run(evaluation_cmd, evaluation_env)


if __name__ == "__main__":
    main()
