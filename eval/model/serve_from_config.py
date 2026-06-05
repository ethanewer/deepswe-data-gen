"""Start local vLLM replicas and the round-robin proxy from a JSON config."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.paths import REPO_ROOT, RUNS_DIR


def vllm_executable() -> str:
    candidate = Path(sys.executable).resolve().parent / "vllm"
    if candidate.exists():
        return str(candidate)
    return "vllm"


def repo_path(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(REPO_ROOT / candidate)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = json.load(f)
    if "model" not in config or "serve" not in config:
        raise ValueError("Config must contain top-level 'model' and 'serve' keys")
    return config


def backend_urls(serve: dict[str, Any]) -> list[str]:
    host = serve.get("backend_base_url_host", "127.0.0.1")
    return [f"http://{host}:{port}" for port in serve["backend_ports"]]


def build_vllm_command(config: dict[str, Any], gpu: int, port: int) -> list[str]:
    serve = config["serve"]
    command = [
        vllm_executable(),
        "serve",
        config["model"],
        "--host",
        serve.get("backend_host", "0.0.0.0"),
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(serve.get("tensor_parallel_size", 1)),
        "--served-model-name",
        serve.get("served_model_name", config["model"]),
        "--max-model-len",
        str(serve["max_model_len"]),
        "--gpu-memory-utilization",
        str(serve.get("gpu_memory_utilization", 0.9)),
    ]
    if serve.get("trust_remote_code", True):
        command.append("--trust-remote-code")
    chat_template = repo_path(serve.get("chat_template"))
    if chat_template:
        command += ["--chat-template", chat_template]
    if serve.get("chat_template_content_format"):
        command += ["--chat-template-content-format", serve["chat_template_content_format"]]
    command += list(serve.get("vllm_args", []))
    return command


def wait_health(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for {url}/health: {last_error}")


def start_process(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("wb")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    return process


def terminate(processes: list[subprocess.Popen[bytes]], timeout_s: float = 30) -> None:
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.time() + timeout_s
    while time.time() < deadline and any(process.poll() is None for process in processes):
        time.sleep(1)
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--health-timeout", type=float, default=1800)
    args = parser.parse_args()

    config = load_config(args.config)
    serve = config["serve"]
    gpus = serve["gpus"]
    ports = serve["backend_ports"]
    if len(gpus) != len(ports):
        raise ValueError("serve.gpus and serve.backend_ports must have the same length")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir or (RUNS_DIR / "serving" / f"{args.config.stem}-{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    processes: list[subprocess.Popen[bytes]] = []
    manifest: dict[str, Any] = {
        "config": str(args.config),
        "run_dir": str(run_dir),
        "model": config["model"],
        "backend_urls": backend_urls(serve),
        "processes": [],
    }

    try:
        for gpu, port in zip(gpus, ports, strict=True):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = build_vllm_command(config, gpu, port)
            process = start_process(
                command,
                env=env,
                log_path=run_dir / f"vllm-gpu{gpu}-port{port}.log",
            )
            processes.append(process)
            manifest["processes"].append(
                {
                    "kind": "vllm",
                    "gpu": gpu,
                    "port": port,
                    "pid": process.pid,
                    "command": command,
                    "log": str(run_dir / f"vllm-gpu{gpu}-port{port}.log"),
                }
            )
            print(f"Started vLLM GPU {gpu} on port {port} with pid {process.pid}", flush=True)

        if not args.skip_health:
            for url in backend_urls(serve):
                print(f"Waiting for {url}/health", flush=True)
                wait_health(url, args.health_timeout)

        if not args.no_proxy:
            proxy_env = os.environ.copy()
            proxy_env["OPENAI_BACKENDS"] = ",".join(backend_urls(serve))
            proxy_env["HOST"] = serve.get("proxy_host", "0.0.0.0")
            proxy_env["PORT"] = str(serve.get("proxy_port", 8000))
            proxy_command = [sys.executable, "-m", "eval.model.round_robin_proxy"]
            proxy_process = start_process(
                proxy_command,
                env=proxy_env,
                log_path=run_dir / "round-robin-proxy.log",
            )
            processes.append(proxy_process)
            manifest["proxy_url"] = f"http://127.0.0.1:{serve.get('proxy_port', 8000)}/v1"
            manifest["processes"].append(
                {
                    "kind": "proxy",
                    "pid": proxy_process.pid,
                    "command": proxy_command,
                    "log": str(run_dir / "round-robin-proxy.log"),
                    "backends": backend_urls(serve),
                }
            )
            print(f"Started proxy on port {serve.get('proxy_port', 8000)} with pid {proxy_process.pid}", flush=True)

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote manifest to {manifest_path}", flush=True)

        if args.background:
            return

        while True:
            exited = [process for process in processes if process.poll() is not None]
            if exited:
                raise SystemExit(exited[0].returncode)
            time.sleep(5)
    except KeyboardInterrupt:
        terminate(processes)
        raise
    except Exception:
        terminate(processes)
        raise


if __name__ == "__main__":
    main()
