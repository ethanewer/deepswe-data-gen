"""Helpers for local OpenAI-compatible serving."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from eval.paths import REPO_ROOT


DEFAULT_LOG_DIR = REPO_ROOT / "runs" / "local-vllm"


def vllm_executable() -> str:
    candidate = Path(sys.executable).resolve().parent / "vllm"
    if candidate.exists():
        return str(candidate)
    return "vllm"


def parse_gpu_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def start_vllm_replicas(
    *,
    model: str,
    gpus: list[int],
    served_model_name: str | None,
    base_port: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    background: bool,
    extra_args: list[str],
) -> list[subprocess.Popen | int]:
    if not background and len(gpus) > 1:
        raise ValueError(
            "Foreground serving can start only one blocking vLLM process. "
            "Use --background for multiple GPUs or pass a single GPU."
        )
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]

    handles = []
    log_dir = DEFAULT_LOG_DIR
    if background:
        log_dir.mkdir(parents=True, exist_ok=True)

    for gpu in gpus:
        port = base_port + gpu
        command = [
            vllm_executable(),
            "serve",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tensor-parallel-size",
            "1",
            "--served-model-name",
            served_model_name or model,
            "--trust-remote-code",
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            *extra_args,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"Starting {model} on GPU {gpu} at :{port}", flush=True)
        if background:
            log_path = log_dir / f"vllm-gpu{gpu}.log"
            log = log_path.open("w")
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            (log_dir / f"vllm-gpu{gpu}.pid").write_text(str(process.pid) + "\n")
            handles.append(process)
        else:
            completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
            handles.append(completed.returncode)
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(description="Start one vLLM OpenAI-compatible server per GPU.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    gpus = parse_gpu_csv(args.gpus)
    if not args.background and len(gpus) > 1:
        parser.error("multiple --gpus require --background; foreground mode supports one GPU")
    start_vllm_replicas(
        model=args.model,
        gpus=gpus,
        served_model_name=args.served_model_name,
        base_port=args.base_port,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        background=args.background,
        extra_args=args.extra_args,
    )


if __name__ == "__main__":
    main()
