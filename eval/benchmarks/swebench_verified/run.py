#!/usr/bin/env python
"""Run a model on the predictive SWE-bench Verified subset."""

from __future__ import annotations

from pathlib import Path

from eval.benchmarks.swebench_multilingual import run as shared_runner


SUBSET_DIR = Path(__file__).resolve().parent

VERIFIED_SETTINGS = {
    "SUBSET_DIR": SUBSET_DIR,
    "DEFAULTS_PATH": SUBSET_DIR / "defaults.json",
    "DEFAULT_INSTANCE_IDS_PATH": SUBSET_DIR / "predictive_20_instance_ids.txt",
    "DATASET_NAME": "SWE-bench/SWE-bench_Verified",
    "MINISWEAGENT_SUBSET": "verified",
    "BENCHMARK_DISPLAY_NAME": "SWE-bench Verified",
    "BENCHMARK_RUN_TAG": "verified-20",
}


def main() -> None:
    previous_settings = {
        name: getattr(shared_runner, name) for name in VERIFIED_SETTINGS
    }
    try:
        for name, value in VERIFIED_SETTINGS.items():
            setattr(shared_runner, name, value)
        shared_runner.main()
    finally:
        for name, value in previous_settings.items():
            setattr(shared_runner, name, value)


if __name__ == "__main__":
    main()
