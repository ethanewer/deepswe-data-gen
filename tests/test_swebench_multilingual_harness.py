import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from eval import run_all
from eval.benchmarks.swebench_multilingual import run


def require_terminus_optional_deps() -> None:
    for module in ("litellm", "pydantic", "tenacity"):
        pytest.importorskip(module)


def test_convert_openhands_predictions(tmp_path: Path):
    input_path = tmp_path / "output.jsonl"
    output_path = tmp_path / "preds.json"
    input_path.write_text(
        json.dumps(
            {
                "instance_id": "apache__druid-13704",
                "test_result": {
                    "git_patch": (
                        "diff --git a/pyproject.toml b/pyproject.toml\n"
                        "--- a/pyproject.toml\n"
                        "+++ b/pyproject.toml\n"
                        "@@ -1 +1 @@\n"
                        "-old\n"
                        "+new\n"
                        "diff --git a/a b/a\n"
                    )
                },
                "history": [],
            }
        )
        + "\n"
    )

    run.convert_openhands_predictions(input_path, output_path, "deepseek-v4-flash")

    assert json.loads(output_path.read_text()) == [
        {
            "instance_id": "apache__druid-13704",
            "model_patch": "diff --git a/a b/a\n",
            "model_name_or_path": "deepseek-v4-flash",
        }
    ]


def test_convert_openhands_predictions_fills_missing_selected_instances(tmp_path: Path):
    input_path = tmp_path / "output.jsonl"
    output_path = tmp_path / "preds.json"
    input_path.write_text("")

    run.convert_openhands_predictions(
        input_path,
        output_path,
        "deepseek-v4-flash",
        ["jqlang__jq-2750"],
    )

    assert json.loads(output_path.read_text()) == [
        {
            "instance_id": "jqlang__jq-2750",
            "model_patch": "",
            "model_name_or_path": "deepseek-v4-flash",
        }
    ]


def test_remove_files_from_patch_ignores_embedded_diff_text():
    patch = (
        "diff --git a/docs/example.txt b/docs/example.txt\n"
        "--- a/docs/example.txt\n"
        "+++ b/docs/example.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+literal text: diff --git a/setup.py b/setup.py\n"
        "diff --git a/setup.py b/setup.py\n"
        "--- a/setup.py\n"
        "+++ b/setup.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    filtered = run.remove_files_from_patch(patch, ("setup.py",))

    assert filtered == (
        "diff --git a/docs/example.txt b/docs/example.txt\n"
        "--- a/docs/example.txt\n"
        "+++ b/docs/example.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+literal text: diff --git a/setup.py b/setup.py\n"
    )


def test_render_template_command_splits_placeholders():
    command = run.render_template_command(
        "python tool.py --instances {instance_ids_path} --model {model}",
        {
            "instance_ids_path": "/tmp/ids.txt",
            "model": "deepseek-v4-flash",
        },
    )

    assert command == [
        "python",
        "tool.py",
        "--instances",
        "/tmp/ids.txt",
        "--model",
        "deepseek-v4-flash",
    ]


def test_build_evaluation_command_uses_multilingual_dataset(tmp_path: Path):
    command = run.build_evaluation_command(
        "python",
        tmp_path / "preds.json",
        ["apache__druid-13704"],
        2,
        "run-1",
        {
            "evaluation_cache_level": "instance",
            "evaluation_timeout_seconds": 1800,
        },
    )

    assert "swebench.harness.run_evaluation" in command
    assert command[command.index("--dataset_name") + 1] == "SWE-bench/SWE-bench_Multilingual"
    assert command[command.index("--predictions_path") + 1] == str(tmp_path / "preds.json")


def test_resolve_cli_paths_anchors_relative_paths_to_invocation_cwd(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    args = Namespace(
        instance_ids=Path("ids.txt"),
        output=Path("out"),
        openhands_command_cwd=Path("openhands-checkout"),
        openhands_llm_config=Path("llm.json"),
        openhands_output_json=Path("openhands/output.jsonl"),
        opencode_config=Path("opencode.json"),
        opencode_workspace=Path("opencode-workspace"),
        terminus_workspace=Path("terminus-workspace"),
    )

    run.resolve_cli_paths(args)

    assert args.instance_ids == tmp_path / "ids.txt"
    assert args.output == tmp_path / "out"
    assert args.openhands_command_cwd == tmp_path / "openhands-checkout"
    assert args.openhands_llm_config == tmp_path / "llm.json"
    assert args.openhands_output_json == tmp_path / "openhands" / "output.jsonl"
    assert args.opencode_config == tmp_path / "opencode.json"
    assert args.opencode_workspace == tmp_path / "opencode-workspace"
    assert args.terminus_workspace == tmp_path / "terminus-workspace"


def test_openhands_existing_output_does_not_require_api_key(tmp_path: Path):
    input_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "instance_id": "apache__druid-13704",
                "test_result": {"git_patch": "diff --git a/a b/a\n"},
            }
        )
        + "\n"
    )

    class ModelConfigWithoutCredentials:
        slug = "deepseek-v4-flash"

        def generation_env(self):
            raise AssertionError("generation_env should not be needed for existing output")

        def api_key(self):
            raise AssertionError("api_key should not be needed for existing output")

    result = run.run_openhands_generation(
        Namespace(openhands_output_json=input_path, openhands_llm_config=None),
        ModelConfigWithoutCredentials(),
        tmp_path,
        tmp_path / "ids.txt",
    )

    assert Path(result["predictions_path"]).exists()


def test_patch_openhands_checkout_for_testbed_copy_removes_cargo_locks(tmp_path: Path):
    run_infer = tmp_path / "benchmarks" / "swebenchmultilingual" / "run_infer.py"
    run_infer.parent.mkdir(parents=True)
    run_infer.write_text(
        """
        cp_testebed_repo = workspace.execute_command(
            (f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}")
        )
"""
    )

    assert run.patch_openhands_checkout_for_testbed_copy(tmp_path)
    patched = run_infer.read_text()

    assert "OPENHANDS_TESTBED_COPY_LOCK_CLEANUP" in patched
    assert "sudo find /testbed -path '*/target/*/incremental/*.lock' -delete" in patched
    assert "cp -r /testbed/. {repo_path}" in patched
    assert run.patch_openhands_checkout_for_testbed_copy(tmp_path)
    assert run_infer.read_text() == patched


def test_openhands_generated_llm_config_is_temporary_and_preserves_options(
    tmp_path: Path, monkeypatch
):
    output_json = tmp_path / "openhands" / "output.jsonl"
    output_json.parent.mkdir()
    output_json.write_text(
        json.dumps(
            {
                "instance_id": "apache__druid-13704",
                "test_result": {"git_patch": "diff --git a/a b/a\n"},
            }
        )
        + "\n"
    )
    captured = {}

    def fake_run_capture_json(command, env, cwd=run.REPO_ROOT):
        config_path = Path(command[1])
        captured["config_path"] = config_path
        captured["config"] = json.loads(config_path.read_text())
        captured["cwd"] = cwd
        captured["env"] = env
        captured["mode"] = config_path.stat().st_mode & 0o777
        return {"output_json": str(output_json)}

    monkeypatch.setattr(run, "run_capture_json", fake_run_capture_json)

    class ModelConfig:
        litellm_name = "openai/deepseek-v4-flash"
        api_base = "https://api.deepseek.com"
        temperature = 0
        max_tokens = 4096
        extra_body = {"thinking": {"type": "disabled"}}
        slug = "deepseek-v4-flash"

        def api_key(self):
            return "secret"

        def generation_env(self):
            return {"OPENAI_API_KEY": "secret"}

    result = run.run_openhands_generation(
        Namespace(
            generation_workers=1,
            openhands_enable_delegation=False,
            openhands_command_cwd=tmp_path / "openhands-checkout",
            openhands_extra_arg=[],
            openhands_infer_command="swebenchmultilingual-infer",
            openhands_llm_config=None,
            openhands_max_iterations=250,
            openhands_max_retries=3,
            openhands_n_critic_runs=1,
            openhands_output_json=None,
            openhands_forward_ca_bundle=True,
            openhands_tool_preset="default",
            openhands_workspace="docker",
        ),
        ModelConfig(),
        tmp_path,
        tmp_path / "ids.txt",
    )

    assert Path(result["predictions_path"]).exists()
    assert not captured["config_path"].exists()
    assert captured["cwd"] == tmp_path / "openhands-checkout"
    assert "OPENHANDS_DOCKER_OMIT_PLATFORM" not in captured["env"]
    assert captured["env"]["OPENHANDS_AGENT_SERVER_PLATFORM"] == "linux/amd64"
    assert captured["mode"] == 0o600
    assert captured["config"] == {
        "model": "openai/deepseek-v4-flash",
        "api_key": "secret",
        "temperature": 0,
        "max_output_tokens": 4096,
        "base_url": "https://api.deepseek.com",
        "litellm_extra_body": {"thinking": {"type": "disabled"}},
    }


def test_openhands_infer_command_can_be_multi_word(tmp_path: Path, monkeypatch):
    output_json = tmp_path / "openhands" / "output.jsonl"
    output_json.parent.mkdir()
    output_json.write_text(
        json.dumps(
            {
                "instance_id": "apache__druid-13704",
                "test_result": {"git_patch": "diff --git a/a b/a\n"},
            }
        )
        + "\n"
    )
    captured = {}

    def fake_run_capture_json(command, env, cwd=run.REPO_ROOT):
        captured["command"] = command
        return {"output_json": str(output_json)}

    monkeypatch.setattr(run, "run_capture_json", fake_run_capture_json)

    class ModelConfig:
        litellm_name = "openai/deepseek-v4-flash"
        api_base = None
        temperature = 0
        max_tokens = 4096
        extra_body = {}
        slug = "deepseek-v4-flash"

        def api_key(self):
            return "secret"

        def generation_env(self):
            return {"OPENAI_API_KEY": "secret"}

    run.run_openhands_generation(
        Namespace(
            generation_workers=1,
            openhands_enable_delegation=False,
            openhands_command_cwd=None,
            openhands_extra_arg=[],
            openhands_infer_command="uv run swebenchmultilingual-infer",
            openhands_llm_config=None,
            openhands_max_iterations=250,
            openhands_max_retries=3,
            openhands_n_critic_runs=1,
            openhands_output_json=None,
            openhands_forward_ca_bundle=True,
            openhands_tool_preset="default",
            openhands_workspace="docker",
        ),
        ModelConfig(),
        tmp_path,
        tmp_path / "ids.txt",
    )

    assert captured["command"][:3] == ["uv", "run", "swebenchmultilingual-infer"]
    assert Path(captured["command"][3]).name == "openhands_llm_config.json"


def test_openhands_generated_llm_config_replaces_loose_existing_file(tmp_path: Path):
    config_path = tmp_path / "openhands_llm_config.json"
    config_path.write_text("old\n")
    config_path.chmod(0o644)

    class ModelConfig:
        litellm_name = "openai/deepseek-v4-flash"
        api_base = None
        temperature = 0
        max_tokens = 4096
        extra_body = {}

        def api_key(self):
            return "secret"

    run.write_openhands_llm_config(config_path, ModelConfig())

    assert config_path.stat().st_mode & 0o777 == 0o600
    assert json.loads(config_path.read_text())["api_key"] == "secret"


def test_openhands_supplied_llm_config_does_not_require_model_credentials(
    tmp_path: Path, monkeypatch
):
    llm_config = tmp_path / "openhands_llm_config.json"
    llm_config.write_text("{}\n")
    output_json = tmp_path / "openhands" / "output.jsonl"
    output_json.parent.mkdir()
    output_json.write_text(
        json.dumps(
            {
                "instance_id": "apache__druid-13704",
                "test_result": {"git_patch": "diff --git a/a b/a\n"},
            }
        )
        + "\n"
    )

    def fake_run_capture_json(command, env, cwd=run.REPO_ROOT):
        assert Path(command[1]) == llm_config
        assert cwd == run.REPO_ROOT
        return {"output_json": str(output_json)}

    monkeypatch.setattr(run, "run_capture_json", fake_run_capture_json)

    class ModelConfigWithoutCredentials:
        slug = "deepseek-v4-flash"

        def generation_env(self):
            raise AssertionError("supplied OpenHands config should own generation credentials")

        def api_key(self):
            raise AssertionError("supplied OpenHands config should own API credentials")

    result = run.run_openhands_generation(
        Namespace(
            generation_workers=1,
            openhands_enable_delegation=False,
            openhands_command_cwd=None,
            openhands_extra_arg=[],
            openhands_infer_command="swebenchmultilingual-infer",
            openhands_llm_config=llm_config,
            openhands_max_iterations=250,
            openhands_max_retries=3,
            openhands_n_critic_runs=1,
            openhands_output_json=None,
            openhands_forward_ca_bundle=True,
            openhands_tool_preset="default",
            openhands_workspace="docker",
        ),
        ModelConfigWithoutCredentials(),
        tmp_path,
        tmp_path / "ids.txt",
    )

    assert Path(result["predictions_path"]).exists()


def test_openhands_checkout_patch_forwards_ca_bundle(tmp_path: Path):
    run_infer = tmp_path / "benchmarks" / "swebenchmultilingual" / "run_infer.py"
    run_infer.parent.mkdir(parents=True)
    run_infer.write_text(
        "import os\n"
        "def prepare():\n"
        "            workspace = DockerWorkspace(\n"
        "                server_image=agent_server_image,\n"
        "                working_dir=\"/workspace\",\n"
        "                forward_env=forward_env or [],\n"
        "            )\n"
    )
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----\n")
    env = {"REQUESTS_CA_BUNDLE": str(ca_bundle)}

    assert run.patch_openhands_checkout_for_docker_ca(tmp_path, env) is True
    patched = run_infer.read_text()

    assert "OPENHANDS_DOCKER_CA_BUNDLE" in patched
    assert "OPENHANDS_DOCKER_PYTHONPATH" in patched
    assert 'for env_name in ("GIT_PAGER", "PAGER", "LESS")' in patched
    assert '"CURL_CA_BUNDLE",\n                    "GIT_PAGER"' not in patched
    assert "volumes=docker_volumes" in patched
    assert env["OPENHANDS_DOCKER_CA_BUNDLE"] == str(ca_bundle)
    assert run.patch_openhands_checkout_for_docker_ca(tmp_path, env) is True


def test_openhands_checkout_patch_persists_pager_upgrade(tmp_path: Path):
    run_infer = tmp_path / "benchmarks" / "swebenchmultilingual" / "run_infer.py"
    run_infer.parent.mkdir(parents=True)
    run_infer.write_text(
        "docker_ca_bundle = os.getenv(\"OPENHANDS_DOCKER_CA_BUNDLE\")\n"
        "docker_pythonpath = os.getenv(\"OPENHANDS_DOCKER_PYTHONPATH\")\n"
        "                for env_name in (\n"
        "                    \"SSL_CERT_FILE\",\n"
        "                    \"REQUESTS_CA_BUNDLE\",\n"
        "                    \"CURL_CA_BUNDLE\",\n"
        "                ):\n"
        "                    if env_name in os.environ and env_name not in docker_forward_env:\n"
        "                        docker_forward_env.append(env_name)\n"
    )
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----\n")

    assert run.patch_openhands_checkout_for_docker_ca(
        tmp_path, {"REQUESTS_CA_BUNDLE": str(ca_bundle)}
    ) is True

    patched = run_infer.read_text()
    assert 'for env_name in ("GIT_PAGER", "PAGER", "LESS")' in patched
    assert '"CURL_CA_BUNDLE",\n                    "GIT_PAGER"' not in patched


def test_openhands_checkout_patch_repairs_pager_inside_ca_guard(tmp_path: Path):
    run_infer = tmp_path / "benchmarks" / "swebenchmultilingual" / "run_infer.py"
    run_infer.parent.mkdir(parents=True)
    run_infer.write_text(
        "docker_ca_bundle = os.getenv(\"OPENHANDS_DOCKER_CA_BUNDLE\")\n"
        "if docker_ca_bundle:\n"
        "    docker_volumes.append(f\"{docker_ca_bundle}:{docker_ca_bundle}:ro\")\n"
        "                for env_name in (\n"
        "                    \"SSL_CERT_FILE\",\n"
        "                    \"REQUESTS_CA_BUNDLE\",\n"
        "                    \"CURL_CA_BUNDLE\",\n"
        "                    \"GIT_PAGER\",\n"
        "                    \"PAGER\",\n"
        "                    \"LESS\",\n"
        "                ):\n"
        "                    if env_name in os.environ and env_name not in docker_forward_env:\n"
        "                        docker_forward_env.append(env_name)\n"
        "docker_pythonpath = os.getenv(\"OPENHANDS_DOCKER_PYTHONPATH\")\n"
    )

    assert run.patch_openhands_checkout_for_docker_ca(tmp_path, {}) is True
    patched = run_infer.read_text()

    assert 'for env_name in ("GIT_PAGER", "PAGER", "LESS")' in patched
    assert '"CURL_CA_BUNDLE",\n                    "GIT_PAGER"' not in patched


def test_openhands_checkout_patch_skips_without_source_checkout(tmp_path: Path):
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----\n")

    assert (
        run.patch_openhands_checkout_for_docker_ca(
            tmp_path, {"REQUESTS_CA_BUNDLE": str(ca_bundle)}
        )
        is False
    )


def test_openhands_checkout_patch_forwards_pager_without_ca_bundle(tmp_path: Path):
    run_infer = tmp_path / "benchmarks" / "swebenchmultilingual" / "run_infer.py"
    run_infer.parent.mkdir(parents=True)
    run_infer.write_text(
        "            workspace = DockerWorkspace(\n"
        "                server_image=agent_server_image,\n"
        "                working_dir=\"/workspace\",\n"
        "                forward_env=forward_env or [],\n"
        "            )\n"
    )
    env = {}

    assert run.patch_openhands_checkout_for_docker_ca(tmp_path, env) is True
    patched = run_infer.read_text()

    assert 'for env_name in ("GIT_PAGER", "PAGER", "LESS")' in patched
    assert '"CURL_CA_BUNDLE",\n                    "GIT_PAGER"' not in patched
    assert "OPENHANDS_DOCKER_PYTHONPATH" in patched
    assert "OPENHANDS_DOCKER_CA_BUNDLE" not in env


def test_openhands_checkout_patch_can_disable_ca_forwarding_only(tmp_path: Path):
    run_infer = tmp_path / "benchmarks" / "swebenchmultilingual" / "run_infer.py"
    run_infer.parent.mkdir(parents=True)
    run_infer.write_text(
        "            workspace = DockerWorkspace(\n"
        "                server_image=agent_server_image,\n"
        "                working_dir=\"/workspace\",\n"
        "                forward_env=forward_env or [],\n"
        "            )\n"
    )
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----\n")
    env = {"REQUESTS_CA_BUNDLE": str(ca_bundle)}

    assert (
        run.patch_openhands_checkout_for_docker_ca(
            tmp_path, env, forward_ca_bundle=False
        )
        is True
    )
    patched = run_infer.read_text()

    assert "OPENHANDS_DOCKER_CA_BUNDLE" in patched
    assert 'for env_name in ("GIT_PAGER", "PAGER", "LESS")' in patched
    assert "OPENHANDS_DOCKER_CA_BUNDLE" not in env


def test_openhands_docker_sitecustomize_sets_pythonpath(tmp_path: Path):
    env = {"PYTHONPATH": "/existing/path"}

    sitecustomize = run.write_openhands_docker_sitecustomize(tmp_path, env)

    assert sitecustomize == tmp_path / "openhands_docker_sitecustomize" / "sitecustomize.py"
    assert env["PYTHONPATH"] == f"{sitecustomize.parent}:/existing/path"
    assert env["OPENHANDS_DOCKER_PYTHONPATH"] == str(sitecustomize.parent)
    content = sitecustomize.read_text()
    assert "VERIFY_X509_STRICT" in content
    assert "ssl.create_default_context = create_default_context" in content


def test_openhands_dataset_dependency_guard_upgrades_old_venv(
    tmp_path: Path, monkeypatch
):
    run_infer = tmp_path / "benchmarks" / "swebenchmultilingual" / "run_infer.py"
    run_infer.parent.mkdir(parents=True)
    run_infer.write_text("")
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    calls = []

    class Completed:
        stdout = "3.0.1\n"

    monkeypatch.setattr(run.subprocess, "run", lambda *args, **kwargs: Completed())

    def fake_run_in_dir(command, env, cwd, **kwargs):
        calls.append((command, env, cwd, kwargs))

    monkeypatch.setattr(run, "run_in_dir", fake_run_in_dir)
    env = {}

    assert run.ensure_openhands_dataset_dependency(tmp_path, env) is True
    assert calls[0][0] == [
        "uv",
        "pip",
        "install",
        "--python",
        str(venv_python),
        "datasets>=4.5.0",
    ]
    assert env["UV_NO_SYNC"] == "1"


def test_openhands_checkout_patch_omits_docker_platform(tmp_path: Path):
    workspace_py = (
        tmp_path
        / "vendor"
        / "software-agent-sdk"
        / "openhands-workspace"
        / "openhands"
        / "workspace"
        / "docker"
        / "workspace.py"
    )
    workspace_py.parent.mkdir(parents=True)
    workspace_py.write_text(
        "        run_cmd = [\n"
        "            \"docker\",\n"
        "            \"run\",\n"
        "            \"-d\",\n"
        "            \"--platform\",\n"
        "            self.platform,\n"
        "            \"--rm\",\n"
    )

    assert run.patch_openhands_checkout_for_docker_platform(tmp_path) is True
    patched = workspace_py.read_text()

    assert "OPENHANDS_DOCKER_OMIT_PLATFORM" in patched
    assert "*platform_flags" in patched
    assert run.patch_openhands_checkout_for_docker_platform(tmp_path) is True


def test_openhands_checkout_patch_sets_local_agent_server_platform(
    tmp_path: Path,
):
    build_py = (
        tmp_path
        / "vendor"
        / "software-agent-sdk"
        / "openhands-agent-server"
        / "openhands"
        / "agent_server"
        / "docker"
        / "build.py"
    )
    build_py.parent.mkdir(parents=True)
    build_py.write_text(
        "    if push:\n"
        "        args += [\"--platform\", \",\".join(opts.platforms), \"--push\"]\n"
        "    else:\n"
        "        args += [\"--load\"]\n"
        "\n"
        "    logger.info(\n"
        "        f\"for platforms='{opts.platforms if push else 'local-arch'}'\"\n"
        "    )\n"
    )

    assert run.patch_openhands_checkout_for_agent_server_platform(tmp_path) is True
    patched = build_py.read_text()

    assert "OPENHANDS_AGENT_SERVER_PLATFORM" in patched
    assert "args += [\"--platform\", local_platform]" in patched
    assert "local_platform or 'local-arch'" in patched
    assert run.patch_openhands_checkout_for_agent_server_platform(tmp_path) is True


def test_openhands_dataset_dependency_guard_keeps_new_venv(
    tmp_path: Path, monkeypatch
):
    run_infer = tmp_path / "benchmarks" / "swebenchmultilingual" / "run_infer.py"
    run_infer.parent.mkdir(parents=True)
    run_infer.write_text("")
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")

    class Completed:
        stdout = "4.8.5\n"

    monkeypatch.setattr(run.subprocess, "run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr(
        run,
        "run_in_dir",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no install")),
    )
    env = {}

    assert run.ensure_openhands_dataset_dependency(tmp_path, env) is False
    assert env["UV_NO_SYNC"] == "1"


def test_openhands_dataset_dependency_guard_skips_non_source_checkout(
    tmp_path: Path, monkeypatch
):
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    monkeypatch.setattr(
        run.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no probe")),
    )

    assert run.ensure_openhands_dataset_dependency(tmp_path, {}) is False


def test_opencode_rejects_stale_predictions(tmp_path: Path):
    predictions_path = tmp_path / "preds.json"
    predictions_path.write_text("[]\n")

    class ModelConfig:
        api_base = None
        api_key_env = "DEEPSEEK_API_KEY"
        litellm_name = "openai/deepseek-v4-flash"
        max_tokens = 4096
        openai_model = "deepseek-v4-flash"
        temperature = 0

        def api_key(self):
            return "secret"

        def generation_env(self):
            return {}

    try:
        run.run_opencode_generation(
            Namespace(
                generation_workers=1,
                opencode_command_template="true",
                opencode_model=None,
            ),
            ModelConfig(),
            tmp_path,
            tmp_path / "ids.txt",
            ["apache__druid-13704"],
            "^apache__druid-13704$",
        )
    except RuntimeError as exc:
        assert "did not create predictions" in str(exc)
    else:
        raise AssertionError("stale preds.json should not be accepted")
    assert not predictions_path.exists()


def test_opencode_derives_deepseek_model_id():
    class ModelConfig:
        api_base = "https://api.deepseek.com"
        api_key_env = "DEEPSEEK_API_KEY"
        openai_model = "deepseek-v4-flash"

    assert run.derive_opencode_model(ModelConfig()) == "deepseek/deepseek-v4-flash"


def test_opencode_derives_custom_provider_for_api_base_with_slash_model():
    class ModelConfig:
        api_base = "https://openrouter.ai/api/v1"
        api_key_env = "OPENROUTER_API_KEY"
        openai_model = "qwen/qwen3.5-9b"

    assert run.derive_opencode_model(ModelConfig()) == "deepswe/qwen/qwen3.5-9b"


def test_opencode_generated_config_uses_env_reference():
    class ModelConfig:
        api_base = "https://api.example.com/v1"
        api_key_env = "EXAMPLE_API_KEY"
        max_tokens = 8192
        temperature = 0

    config = json.loads(run.build_opencode_config_content(ModelConfig(), "deepswe/model-x"))

    assert config["model"] == "deepswe/model-x"
    assert config["small_model"] == "deepswe/model-x"
    assert config["default_agent"] == run.OPENCODE_BENCHMARK_AGENT
    assert config["permission"]["task"] == "deny"
    assert config["agent"][run.OPENCODE_BENCHMARK_AGENT]["model"] == "deepswe/model-x"
    assert config["agent"][run.OPENCODE_BENCHMARK_AGENT]["permission"]["task"] == "deny"
    assert config["provider"]["deepswe"]["options"] == {
        "baseURL": "https://api.example.com/v1",
        "apiKey": "{env:EXAMPLE_API_KEY}",
    }
    assert "EXAMPLE_API_KEY" in config["provider"]["deepswe"]["options"]["apiKey"]
    assert config["provider"]["deepswe"]["models"]["model-x"]["limit"] == {
        "context": run.OPENCODE_DEFAULT_CONTEXT_LIMIT,
        "output": 8192,
    }
    assert config["provider"]["deepswe"]["models"]["model-x"]["reasoning"] is False


def test_opencode_generated_config_sets_builtin_provider_model_limit():
    class ModelConfig:
        api_base = None
        api_key_env = "DEEPSEEK_API_KEY"
        max_tokens = 4096
        temperature = 0

    config = json.loads(
        run.build_opencode_config_content(ModelConfig(), "deepseek/deepseek-v4-flash")
    )

    assert config["provider"]["deepseek"]["options"] == {
        "apiKey": "{env:DEEPSEEK_API_KEY}",
    }
    assert config["provider"]["deepseek"]["models"]["deepseek-v4-flash"]["limit"] == {
        "context": run.OPENCODE_DEFAULT_CONTEXT_LIMIT,
        "output": run.OPENCODE_MIN_OUTPUT_LIMIT,
    }
    assert (
        config["provider"]["deepseek"]["models"]["deepseek-v4-flash"]["reasoning"]
        is False
    )


def test_opencode_instance_env_isolates_home_and_config(tmp_path: Path):
    class ModelConfig:
        api_base = None
        api_key_env = "DEEPSEEK_API_KEY"
        max_tokens = 4096
        temperature = 0

    env = run.opencode_instance_env(
        {"HOME": "/Users/example", "DEEPSEEK_API_KEY": "secret"},
        tmp_path,
        "repo__issue-1",
        ModelConfig(),
        "deepseek/deepseek-v4-flash",
        None,
    )

    assert env["HOME"] == str(tmp_path / "home" / "repo__issue-1")
    assert env["XDG_CONFIG_HOME"] == str(tmp_path / "xdg-config" / "repo__issue-1")
    assert env["OPENCODE_DISABLE_UPDATE"] == "1"
    assert "OPENCODE_CONFIG_CONTENT" in env
    generated_config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert generated_config["default_agent"] == run.OPENCODE_BENCHMARK_AGENT


def test_opencode_instance_env_forwards_rust_toolchain_homes(tmp_path: Path):
    real_home = tmp_path / "real-home"
    (real_home / ".cargo").mkdir(parents=True)
    (real_home / ".rustup").mkdir()

    class ModelConfig:
        api_base = None
        api_key_env = "DEEPSEEK_API_KEY"
        max_tokens = 4096
        temperature = 0

    env = run.opencode_instance_env(
        {"HOME": str(real_home), "DEEPSEEK_API_KEY": "secret"},
        tmp_path / "workspace",
        "repo__issue-1",
        ModelConfig(),
        "deepseek/deepseek-v4-flash",
        None,
    )

    assert env["HOME"] == str(tmp_path / "workspace" / "home" / "repo__issue-1")
    assert env["CARGO_HOME"] == str(real_home / ".cargo")
    assert env["RUSTUP_HOME"] == str(real_home / ".rustup")


def test_opencode_prompt_forbids_subagents():
    prompt = run.opencode_prompt(
        {
            "problem_statement": "fix the parser",
            "hints_text": "",
        }
    )

    assert "do not use subagents" in prompt
    assert "Keep exploration brief" in prompt
    assert "inspect git diff" in prompt
    assert "Do not install toolchains" in prompt


def test_prepare_opencode_worktree_skips_known_unavailable_bat_submodules(
    tmp_path: Path, monkeypatch
):
    row = {
        "instance_id": "sharkdp__bat-3108",
        "repo": "sharkdp/bat",
        "base_commit": "abc123",
    }
    calls = []
    submodule_urls = {
        "submodule.assets/syntaxes/TypeScript.url": (
            "https://github.com/Microsoft/TypeScript-Sublime-Plugin"
        ),
        "submodule.assets/syntaxes/02_Extra/LiveScript.url": (
            "https://github.com/paulmillr/LiveScript.tmbundle"
        ),
        "submodule.assets/syntaxes/02_Extra/Nginx.url": (
            "https://github.com/brandonwamboldt/sublime-nginx"
        ),
        "submodule.assets/syntaxes/hosts.url": (
            "https://github.com/brandonwamboldt/sublime-hosts"
        ),
    }

    def fake_run_in_dir(command, env, cwd, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["git", "init"]:
            Path(command[-1]).mkdir(parents=True)
        if command[:2] == ["git", "checkout"]:
            (cwd / ".gitmodules").write_text(
                """
[submodule "assets/syntaxes/TypeScript"]
\tpath = assets/syntaxes/02_Extra/TypeScript
\turl = https://github.com/Microsoft/TypeScript-Sublime-Plugin
[submodule "assets/syntaxes/02_Extra/LiveScript"]
\tpath = assets/syntaxes/02_Extra/LiveScript
\turl = https://github.com/paulmillr/LiveScript.tmbundle
[submodule "assets/syntaxes/02_Extra/Nginx"]
\tpath = assets/syntaxes/02_Extra/Nginx
\turl = https://github.com/brandonwamboldt/sublime-nginx
[submodule "assets/syntaxes/hosts"]
\tpath = assets/syntaxes/02_Extra/hosts
\turl = https://github.com/brandonwamboldt/sublime-hosts
"""
            )
        if command[:4] == ["git", "config", "--file", ".gitmodules"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=submodule_urls[command[-1]]
            )
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(run, "run_in_dir", fake_run_in_dir)

    run.prepare_opencode_worktree(row, tmp_path / "worktree", {})

    git_config_commands = [
        command
        for command, _ in calls
        if command[:2] == ["git", "config"] and "--file" not in command
    ]
    assert git_config_commands == [
        ["git", "config", "submodule.assets/syntaxes/TypeScript.update", "none"],
        [
            "git",
            "config",
            "submodule.assets/syntaxes/02_Extra/LiveScript.update",
            "none",
        ],
        [
            "git",
            "config",
            "submodule.assets/syntaxes/02_Extra/Nginx.update",
            "none",
        ],
        [
            "git",
            "config",
            "submodule.assets/syntaxes/hosts.update",
            "none",
        ],
    ]
    submodule_update = [
        (command, kwargs)
        for command, kwargs in calls
        if command[:2] == ["git", "submodule"]
    ]
    assert submodule_update == [
        (["git", "submodule", "update", "--init", "--recursive"], {"timeout": 1800})
    ]


def test_opencode_instance_uses_generated_benchmark_agent(tmp_path: Path, monkeypatch):
    row = {
        "instance_id": "repo__issue-1",
        "problem_statement": "fix a bug",
        "repo": "owner/repo",
        "base_commit": "abc123",
    }
    captured = {}

    monkeypatch.setattr(run, "prepare_opencode_worktree", lambda *args: None)
    monkeypatch.setattr(run, "collect_git_patch", lambda *args: "diff --git a/x b/x\n")

    def fake_run_in_dir(command, env, cwd, **kwargs):
        captured["command"] = command
        captured["env"] = env

    monkeypatch.setattr(run, "run_in_dir", fake_run_in_dir)

    class ModelConfig:
        api_base = "https://api.deepseek.com"
        api_key_env = "DEEPSEEK_API_KEY"
        max_tokens = 4096
        slug = "deepseek-v4-flash"
        temperature = 0

    prediction = run.run_opencode_instance(
        row,
        Namespace(
            opencode_agent=None,
            opencode_command="npx --yes opencode-ai",
            opencode_config=None,
            opencode_extra_arg=[],
            opencode_timeout=60,
            opencode_variant=None,
        ),
        ModelConfig(),
        tmp_path,
        {"DEEPSEEK_API_KEY": "secret"},
        "deepseek/deepseek-v4-flash",
    )

    assert prediction["model_patch"] == "diff --git a/x b/x\n"
    assert captured["command"][
        captured["command"].index("--agent") + 1
    ] == run.OPENCODE_BENCHMARK_AGENT
    assert "OPENCODE_CONFIG_CONTENT" in captured["env"]


def test_opencode_instance_respects_explicit_agent_and_config(
    tmp_path: Path, monkeypatch
):
    row = {
        "instance_id": "repo__issue-1",
        "problem_statement": "fix a bug",
        "repo": "owner/repo",
        "base_commit": "abc123",
    }
    captured = {}

    monkeypatch.setattr(run, "prepare_opencode_worktree", lambda *args: None)
    monkeypatch.setattr(run, "collect_git_patch", lambda *args: "")
    monkeypatch.setattr(
        run,
        "run_in_dir",
        lambda command, env, cwd, **kwargs: captured.update(command=command, env=env),
    )

    class ModelConfig:
        api_base = "https://api.deepseek.com"
        api_key_env = "DEEPSEEK_API_KEY"
        max_tokens = 4096
        slug = "deepseek-v4-flash"
        temperature = 0

    run.run_opencode_instance(
        row,
        Namespace(
            opencode_agent="custom",
            opencode_command="npx --yes opencode-ai",
            opencode_config=tmp_path / "opencode.json",
            opencode_extra_arg=[],
            opencode_timeout=60,
            opencode_variant=None,
        ),
        ModelConfig(),
        tmp_path,
        {"DEEPSEEK_API_KEY": "secret"},
        "deepseek/deepseek-v4-flash",
    )

    assert captured["command"][captured["command"].index("--agent") + 1] == "custom"
    assert "OPENCODE_CONFIG" in captured["env"]
    assert "OPENCODE_CONFIG_CONTENT" not in captured["env"]


def test_native_opencode_generation_writes_ordered_predictions(tmp_path: Path, monkeypatch):
    rows = [
        {"instance_id": "repo__b-2", "problem_statement": "fix b"},
        {"instance_id": "repo__a-1", "problem_statement": "fix a"},
    ]

    monkeypatch.setattr(run, "load_swebench_instances", lambda instance_ids: rows)

    def fake_run_opencode_instance(
        row, args, model_config, workspace_root, base_env, opencode_model
    ):
        return {
            "instance_id": row["instance_id"],
            "model_patch": f"diff --git a/{row['instance_id']} b/{row['instance_id']}\n",
            "model_name_or_path": model_config.slug,
        }

    monkeypatch.setattr(run, "run_opencode_instance", fake_run_opencode_instance)

    class ModelConfig:
        api_base = "https://api.deepseek.com"
        api_key_env = "DEEPSEEK_API_KEY"
        litellm_name = "openai/deepseek-v4-flash"
        max_tokens = 4096
        openai_model = "deepseek-v4-flash"
        slug = "deepseek-v4-flash"
        temperature = 0

        def api_key(self):
            return "secret"

        def generation_env(self):
            return {"DEEPSEEK_API_KEY": "secret", "OPENAI_API_KEY": "secret"}

    result = run.run_opencode_generation(
        Namespace(
            generation_workers=2,
            opencode_agent=None,
            opencode_command="npx --yes opencode-ai",
            opencode_command_template=None,
            opencode_config=None,
            opencode_extra_arg=[],
            opencode_model=None,
            opencode_timeout=60,
            opencode_variant=None,
            opencode_workspace=tmp_path / "workspace",
        ),
        ModelConfig(),
        tmp_path,
        tmp_path / "ids.txt",
        ["repo__b-2", "repo__a-1"],
        "^(repo__b\\-2|repo__a\\-1)$",
    )

    assert Path(result["predictions_path"]) == tmp_path / "preds.json"
    predictions = json.loads((tmp_path / "preds.json").read_text())
    assert [prediction["instance_id"] for prediction in predictions] == [
        "repo__b-2",
        "repo__a-1",
    ]


def test_terminus_instruction_uses_current_directory_and_diff():
    prompt = run.terminus_instruction(
        {
            "problem_statement": "fix the parser",
            "hints_text": "look at parse.rs",
        }
    )

    assert "current directory" in prompt
    assert "git diff" in prompt
    assert "fix the parser" in prompt
    assert "look at parse.rs" in prompt


def test_terminus_terminal_env_strips_llm_credentials(tmp_path: Path):
    env = run.terminus_terminal_env(
        {
            "OPENAI_API_KEY": "openai-secret",
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "OPENAI_API_BASE": "https://api.deepseek.com",
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "GITHUB_TOKEN": "github-token",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/google.json",
            "PATH": "/bin",
            "SSH_AUTH_SOCK": "/tmp/ssh-agent",
        },
        tmp_path,
        "repo__issue-1",
    )

    assert "OPENAI_API_KEY" not in env
    assert "OPENAI_BASE_URL" not in env
    assert "OPENAI_API_BASE" not in env
    assert "DEEPSEEK_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env
    assert "SSH_AUTH_SOCK" not in env
    assert env["PATH"] == "/bin"
    assert env["HOME"] == str(tmp_path / "home" / "repo__issue-1")
    assert env["XDG_CONFIG_HOME"] == str(tmp_path / "xdg-config" / "repo__issue-1")


def test_local_tmux_session_incremental_output_excludes_previous_buffer(tmp_path: Path):
    session = run.LocalTmuxSession(
        "test-session",
        tmp_path,
        {},
        tmp_path / "terminal.log",
    )
    session._previous_buffer = "line1\nline2"

    assert session._find_new_content("line1\nline2\nline3") == "\nline3"


def test_local_tmux_session_uses_isolated_socket(tmp_path: Path, monkeypatch):
    commands = []

    def fake_subprocess_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    session = run.LocalTmuxSession(
        "test session",
        tmp_path,
        {"PATH": "/bin"},
        tmp_path / "terminal.log",
    )

    session.start()

    assert commands[0][:3] == ["tmux", "-L", "test-session"]
    assert commands[0][3] == "new-session"


def test_litellm_logger_redacts_nested_api_keys():
    require_terminus_optional_deps()

    from eval.terminal_bench.llms.lite_llm import LiteLLM

    llm = LiteLLM.__new__(LiteLLM)
    payload = llm._redact_secret_values(
        {
            "api_key": "root-secret",
            "nested": {"x-api-key": "nested-secret"},
            "items": [{"authorization": "Bearer token"}],
            "apiKey": "camel-secret",
            "bearer_token": "bearer-secret",
            "x-goog-api-key": "google-secret",
            "proxy_authorization": "proxy-secret",
            "password": "password-secret",
            "access_key": "access-secret",
        }
    )

    assert "api_key" not in payload
    assert payload["api_key_sha256"]
    assert "x-api-key" not in payload["nested"]
    assert payload["nested"]["x-api-key_sha256"]
    assert "authorization" not in payload["items"][0]
    assert payload["items"][0]["authorization_sha256"]
    assert "apiKey" not in payload
    assert payload["apiKey_sha256"]
    assert "bearer_token" not in payload
    assert payload["bearer_token_sha256"]
    assert "x-goog-api-key" not in payload
    assert payload["x-goog-api-key_sha256"]
    assert "proxy_authorization" not in payload
    assert payload["proxy_authorization_sha256"]
    assert "password" not in payload
    assert payload["password_sha256"]
    assert "access_key" not in payload
    assert payload["access_key_sha256"]


def test_terminus_litellm_forwards_model_limits_and_extra_body(monkeypatch):
    require_terminus_optional_deps()

    from eval.terminal_bench.agents.terminus_2.terminus_2 import Terminus2
    from eval.terminal_bench.llms import lite_llm

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "ok"},
                }
            ]
        }

    monkeypatch.setattr(lite_llm, "get_supported_openai_params", lambda _: [])
    monkeypatch.setattr(lite_llm.litellm, "completion", fake_completion)
    monkeypatch.setattr(lite_llm, "add_anthropic_caching", lambda messages, _: messages)

    agent = Terminus2(
        model_name="openai/deepseek-v4-flash",
        temperature=0,
        api_base="https://api.deepseek.com",
        api_key="secret",
        max_tokens=4096,
        extra_body={"thinking": {"type": "disabled"}},
    )

    assert agent._llm.call("hello", timeout=120) == "ok"
    assert captured["max_tokens"] == 4096
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["timeout"] == 120
    assert captured["api_base"] == "https://api.deepseek.com"
    assert captured["api_key"] == "secret"


def test_terminus_provider_override_uses_selected_key_and_label(monkeypatch):
    class ModelConfig:
        api_key_env = "DEEPSEEK_API_KEY"
        slug = "deepseek-v4-flash"

        def api_key(self):
            return "deepseek-secret"

    args = Namespace(
        terminus_model="openrouter/qwen/qwen3.5-9b",
        terminus_api_key_env="OPENROUTER_API_KEY",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")

    assert run.terminus_api_key(args, ModelConfig()) == "openrouter-secret"
    assert (
        run.terminus_model_label(args, ModelConfig())
        == "openrouter-qwen-qwen3.5-9b"
    )


def test_native_terminus_generation_writes_ordered_predictions(tmp_path: Path, monkeypatch):
    rows = [
        {"instance_id": "repo__b-2", "problem_statement": "fix b"},
        {"instance_id": "repo__a-1", "problem_statement": "fix a"},
    ]

    monkeypatch.setattr(run, "load_swebench_instances", lambda instance_ids: rows)

    def fake_run_terminus_instance(row, args, model_config, workspace_root, base_env):
        return {
            "instance_id": row["instance_id"],
            "model_patch": f"diff --git a/{row['instance_id']} b/{row['instance_id']}\n",
            "model_name_or_path": model_config.slug,
        }

    monkeypatch.setattr(run, "run_terminus_instance", fake_run_terminus_instance)

    class ModelConfig:
        api_base = "https://api.deepseek.com"
        api_key_env = "DEEPSEEK_API_KEY"
        litellm_name = "openai/deepseek-v4-flash"
        slug = "deepseek-v4-flash"
        temperature = 0

        def api_key(self):
            return "secret"

        def generation_env(self):
            return {"DEEPSEEK_API_KEY": "secret", "OPENAI_API_KEY": "secret"}

    result = run.run_terminus_generation(
        Namespace(
            generation_workers=2,
            terminus_max_episodes=2,
            terminus_model=None,
            terminus_api_base=None,
            terminus_api_key_env=None,
            terminus_parser="json",
            terminus_request_timeout=120,
            terminus_workspace=tmp_path / "workspace",
        ),
        ModelConfig(),
        tmp_path,
        ["repo__b-2", "repo__a-1"],
    )

    assert Path(result["predictions_path"]) == tmp_path / "preds.json"
    predictions = json.loads((tmp_path / "preds.json").read_text())
    assert [prediction["instance_id"] for prediction in predictions] == [
        "repo__b-2",
        "repo__a-1",
    ]


def test_run_all_forwards_dash_prefixed_openhands_extra_args():
    args = run_all.benchmark_args(
        "swebench_multilingual",
        {
            "harness": "openhands-swe",
            "openhands_command_cwd": "/tmp/openhands-benchmarks",
            "openhands_forward_ca_bundle": False,
            "openhands_fix_datasets_dependency": False,
            "openhands_extra_arg": ["--modal", "--note smoke"],
        },
    )

    assert "--openhands-command-cwd" in args
    assert "/tmp/openhands-benchmarks" in args
    assert "--openhands-extra-arg=--modal" in args
    assert "--openhands-extra-arg=--note smoke" in args
    assert "--no-openhands-forward-ca-bundle" in args
    assert "--no-openhands-fix-datasets-dependency" in args


def test_run_all_forwards_opencode_options():
    args = run_all.benchmark_args(
        "swebench_multilingual",
        {
            "harness": "opencode",
            "opencode_command": "npx --yes opencode-ai",
            "opencode_model": "deepseek/deepseek-v4-flash",
            "opencode_config": "/tmp/opencode.json",
            "opencode_workspace": "/tmp/opencode-workspace",
            "opencode_extra_arg": ["--print-logs", "--log-level DEBUG"],
        },
    )

    assert "--opencode-command" in args
    assert "npx --yes opencode-ai" in args
    assert "--opencode-model" in args
    assert "deepseek/deepseek-v4-flash" in args
    assert "--opencode-config" in args
    assert "/tmp/opencode.json" in args
    assert "--opencode-workspace" in args
    assert "/tmp/opencode-workspace" in args
    assert "--opencode-extra-arg=--print-logs" in args
    assert "--opencode-extra-arg=--log-level DEBUG" in args


def test_run_all_forwards_terminus_options():
    args = run_all.benchmark_args(
        "swebench_multilingual",
        {
            "harness": "terminus-2",
            "terminus_model": "openai/deepseek-v4-flash",
            "terminus_api_base": "https://api.deepseek.com",
            "terminus_api_key_env": "DEEPSEEK_API_KEY",
            "terminus_parser": "xml",
            "terminus_max_episodes": 12,
            "terminus_request_timeout": 120,
            "terminus_workspace": "/tmp/terminus-workspace",
        },
    )

    assert "--harness" in args
    assert "terminus-2" in args
    assert "--terminus-model" in args
    assert "openai/deepseek-v4-flash" in args
    assert "--terminus-api-base" in args
    assert "https://api.deepseek.com" in args
    assert "--terminus-api-key-env" in args
    assert "DEEPSEEK_API_KEY" in args
    assert "--terminus-parser" in args
    assert "xml" in args
    assert "--terminus-max-episodes" in args
    assert "12" in args
    assert "--terminus-request-timeout" in args
    assert "120" in args
    assert "--terminus-workspace" in args
    assert "/tmp/terminus-workspace" in args
