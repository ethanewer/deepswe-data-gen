from argparse import Namespace
from pathlib import Path

from datagen.swerebench_v2 import submit_pyxis_datagen
from datagen.swerebench_v2.pyxis_miniswe_agent_driver import (
    resolve_benchmark_profile,
    resolve_model_class_and_name,
)


def test_instruction_style_routes_to_benchmark_profile():
    assert resolve_benchmark_profile(Namespace(benchmark_profile="auto", instruction_style="original")) == "swebench-multilingual"
    assert resolve_benchmark_profile(Namespace(benchmark_profile="auto", instruction_style="swe_rebench")) == "swebench-multilingual"
    assert resolve_benchmark_profile(Namespace(benchmark_profile="auto", instruction_style="deepswe")) == "deepswe"
    assert resolve_benchmark_profile(Namespace(benchmark_profile="auto", instruction_style="rewritten")) == "deepswe"
    assert resolve_benchmark_profile(Namespace(benchmark_profile="auto", instruction_style="planned")) == "deepswe"
    assert resolve_benchmark_profile(Namespace(benchmark_profile="auto", instruction_style="unknown")) == "datagen-strict"


def test_explicit_benchmark_profile_overrides_instruction_style():
    args = Namespace(benchmark_profile="swebench-multilingual", instruction_style="deepswe")
    assert resolve_benchmark_profile(args) == "swebench-multilingual"


def test_model_class_matches_benchmark_profile():
    assert resolve_model_class_and_name("swebench-multilingual", "openrouter/xiaomi/mimo-v2.5") == (
        "litellm",
        "openrouter/xiaomi/mimo-v2.5",
    )
    assert resolve_model_class_and_name("deepswe", "openai/gpt-5.4-mini") == (
        "litellm_response",
        "openai/gpt-5.4-mini",
    )
    assert resolve_model_class_and_name("deepswe", "openrouter/xiaomi/mimo-v2.5") == (
        "openrouter",
        "xiaomi/mimo-v2.5",
    )


def submit_script(tmp_path, monkeypatch, *, style="original", benchmark_profile="auto"):
    monkeypatch.setattr(
        submit_pyxis_datagen,
        "require_pinned_minisweagent_overlay",
        lambda: tmp_path / "pydeps",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "0\tr00\tinst\t/tasks/inst\t/tmp/ws\tdocker.io/test/image\tmodel\t"
        f"openai/model\tOPENAI_API_KEY\t-\t-\teasy\tpython\t{style}\trepo/name\n",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=dummy\n", encoding="utf-8")
    args = Namespace(
        job_name="smoke",
        partition="m7i-cpu2",
        cpus=4,
        mem="16G",
        tmp="",
        time="00:10:00",
        array_concurrency=1,
        manifest_tsv=manifest,
        run_root=tmp_path / "run",
        python=Path("/usr/bin/python3"),
        config_file=None,
        benchmark_profile=benchmark_profile,
        env_file=env_file,
        enroot_config_path=None,
        docker_user="",
        container_source="registry",
        auth_first=False,
        docker_login_from_enroot=False,
        temperature=0.0,
        max_tokens=4096,
        reasoning_effort="high",
        model_timeout=180,
        agent_wall_time_limit=2700,
        command_timeout=180,
    )
    return submit_pyxis_datagen.write_array_script(args, 1).read_text(encoding="utf-8")


def test_submit_script_selects_config_by_instruction_style(tmp_path, monkeypatch):
    script = submit_script(tmp_path, monkeypatch)

    assert "SWEBENCH_MULTILINGUAL_CONFIG=" in script
    assert 'BENCHMARK_PROFILE="swebench-multilingual"' in script
    assert '--config-file "$CONFIG_FILE"' in script
    assert '--benchmark-profile "$BENCHMARK_PROFILE"' in script


def test_submit_script_selects_config_from_explicit_profile(tmp_path, monkeypatch):
    script = submit_script(
        tmp_path,
        monkeypatch,
        style="deepswe",
        benchmark_profile="swebench-multilingual",
    )

    assert 'BENCHMARK_PROFILE="swebench-multilingual"' in script
    assert 'case "$BENCHMARK_PROFILE" in' in script
    assert 'CONFIG_FILE="$SWEBENCH_MULTILINGUAL_CONFIG"' in script
