from argparse import Namespace
from pathlib import Path

from datagen.swerebench_v2 import run_docker_datagen_packed
from datagen.swerebench_v2 import submit_pyxis_datagen
from datagen.swerebench_v2 import submit_pyxis_datagen_packed
from datagen.swerebench_v2 import pyxis_miniswe_agent_driver
from datagen.swerebench_v2.pyxis_miniswe_agent_driver import (
    build_model_kwargs,
    build_agent,
    ensure_testbed_alias,
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


def test_benchmark_profiles_use_benchmark_model_kwargs():
    args = Namespace(
        temperature=0.0,
        max_tokens=4096,
        model_timeout=180,
        api_base="",
        reasoning_effort="high",
    )
    extra_body = {"thinking": {"type": "enabled"}}

    swebench_kwargs = build_model_kwargs(
        args,
        benchmark_profile="swebench-multilingual",
        extra_body=extra_body,
        use_native_openrouter=False,
    )
    deepswe_kwargs = build_model_kwargs(
        args,
        benchmark_profile="deepswe",
        extra_body=extra_body,
        use_native_openrouter=False,
    )

    assert swebench_kwargs["temperature"] == 0.0
    assert swebench_kwargs["extra_body"] == extra_body
    assert "reasoning_effort" not in swebench_kwargs
    assert "timeout" not in swebench_kwargs
    assert "request_timeout" not in swebench_kwargs
    assert deepswe_kwargs["temperature"] == 0.0
    assert deepswe_kwargs["extra_body"] == extra_body
    assert "reasoning_effort" not in deepswe_kwargs
    assert "timeout" not in deepswe_kwargs
    assert "request_timeout" not in deepswe_kwargs


def test_datagen_strict_keeps_legacy_reasoning_kwargs():
    args = Namespace(
        temperature=0.0,
        max_tokens=4096,
        model_timeout=180,
        api_base="",
        reasoning_effort="high",
    )

    kwargs = build_model_kwargs(
        args,
        benchmark_profile="datagen-strict",
        extra_body={"thinking": {"type": "enabled"}},
        use_native_openrouter=False,
    )

    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["timeout"] == 180
    assert "temperature" not in kwargs


def test_ensure_testbed_alias_points_to_task_workdir(tmp_path):
    workdir = tmp_path / "repo"
    workdir.mkdir()
    alias = tmp_path / "testbed"

    record = ensure_testbed_alias(str(workdir), alias=alias)

    assert record["created"] is True
    assert record["usable"] is True
    assert alias.resolve() == workdir


def submit_script(tmp_path, monkeypatch, *, style="original", benchmark_profile="auto", command_timeout=180):
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
        command_timeout=command_timeout,
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


def test_submit_script_omits_command_timeout_when_not_overridden(tmp_path, monkeypatch):
    script = submit_script(tmp_path, monkeypatch, command_timeout=None)

    assert "--command-timeout None" not in script
    assert "--command-timeout" not in script


def test_packed_submit_script_selects_benchmark_profile_and_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        submit_pyxis_datagen_packed,
        "require_pinned_minisweagent_overlay",
        lambda: tmp_path / "pydeps",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "0\tr00\tinst\t/tasks/inst\t/tmp/ws\tdocker.io/test/image\tmodel\t"
        "openai/model\tOPENAI_API_KEY\t-\t-\teasy\tpython\tdeepswe\trepo/name\tfalse\n",
        encoding="utf-8",
    )
    args = Namespace(
        job_name="smoke",
        partition="m7i-cpu2",
        array_concurrency=1,
        rows_per_job=4,
        parallel_rows=2,
        cpus_per_row=4,
        mem="16G",
        tmp="",
        time="00:10:00",
        manifest_tsv=manifest,
        run_root=tmp_path / "run",
        python=Path("/usr/bin/python3"),
        config_file=None,
        benchmark_profile="auto",
        env_file=tmp_path / ".env",
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
        command_timeout=None,
    )
    args.env_file.write_text("OPENAI_API_KEY=dummy\n", encoding="utf-8")

    script = submit_pyxis_datagen_packed.write_array_script(args, 1).read_text(encoding="utf-8")

    assert 'BENCHMARK_PROFILE="deepswe"' in script
    assert 'CONFIG_FILE="$DEEPSWE_CONFIG"' in script
    assert '--benchmark-profile "$BENCHMARK_PROFILE"' in script
    assert "--command-timeout" not in script


def test_docker_packed_command_selects_benchmark_profile_and_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_docker_datagen_packed,
        "require_pinned_minisweagent_overlay",
        lambda: tmp_path / "pydeps",
    )
    row = run_docker_datagen_packed.ManifestRow(
        index="0",
        rollout_id="r00",
        instance_id="inst",
        task_dir=tmp_path / "task",
        workspace=tmp_path / "workspace",
        image="docker.io/test/image",
        model="model",
        litellm_model="openai/model",
        api_key_env="OPENAI_API_KEY",
        api_base="",
        extra_body_json="",
        difficulty="easy",
        language="python",
        instruction_style="original",
        repo="repo/name",
        outside_original_high_quality_set="false",
    )
    args = Namespace(
        job_name="smoke",
        cpus_per_worker=4,
        memory_per_worker="16g",
        python=Path("/usr/bin/python3"),
        config_file=None,
        benchmark_profile="auto",
        temperature=0.0,
        max_tokens=4096,
        reasoning_effort="high",
        model_timeout=180,
        agent_wall_time_limit=2700,
        command_timeout=None,
    )

    command = run_docker_datagen_packed.docker_run_command(args, row)

    assert "--benchmark-profile" in command
    assert command[command.index("--benchmark-profile") + 1] == "swebench-multilingual"
    assert "--command-timeout" not in command
    assert str(run_docker_datagen_packed.DEFAULT_SWEBENCH_MULTILINGUAL_CONFIG) in command


def test_driver_uses_selected_config_environment(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
agent:
  system_template: system
  instance_template: "{{task}}"
model:
  model_kwargs:
    drop_params: true
environment:
  timeout: 60
  env:
    PAGER: cat
    BASH_ENV: /root/.bashrc
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(pyxis_miniswe_agent_driver, "get_model", lambda config: object())
    monkeypatch.setattr(pyxis_miniswe_agent_driver, "prepare_agent_bin", lambda args: tmp_path / "agent-bin")
    args = Namespace(
        config_file=config_file,
        extra_body_json="",
        benchmark_profile="swebench-multilingual",
        instruction_style="original",
        litellm_model="openai/model",
        max_tokens=4096,
        model_timeout=180,
        reasoning_effort="high",
        temperature=0.0,
        api_base="",
        agent_wall_time_limit=2700,
        command_timeout=None,
    )

    agent = build_agent(args, "/testbed", tmp_path / "trajectory.json")

    assert agent.env.config.timeout == 60
    assert agent.env.config.env["BASH_ENV"] == "/root/.bashrc"
