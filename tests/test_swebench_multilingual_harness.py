import json
from argparse import Namespace
from pathlib import Path

from eval import run_all
from eval.benchmarks.swebench_multilingual import run


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
    )

    run.resolve_cli_paths(args)

    assert args.instance_ids == tmp_path / "ids.txt"
    assert args.output == tmp_path / "out"
    assert args.openhands_command_cwd == tmp_path / "openhands-checkout"
    assert args.openhands_llm_config == tmp_path / "llm.json"
    assert args.openhands_output_json == tmp_path / "openhands" / "output.jsonl"
    assert args.opencode_config == tmp_path / "opencode.json"
    assert args.opencode_workspace == tmp_path / "opencode-workspace"


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
            openhands_tool_preset="default",
            openhands_workspace="docker",
        ),
        ModelConfigWithoutCredentials(),
        tmp_path,
        tmp_path / "ids.txt",
    )

    assert Path(result["predictions_path"]).exists()


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

    config = json.loads(run.build_opencode_config_content(ModelConfig(), "deepswe/model-x"))

    assert config["model"] == "deepswe/model-x"
    assert config["provider"]["deepswe"]["options"] == {
        "baseURL": "https://api.example.com/v1",
        "apiKey": "{env:EXAMPLE_API_KEY}",
    }
    assert "EXAMPLE_API_KEY" in config["provider"]["deepswe"]["options"]["apiKey"]
    assert config["provider"]["deepswe"]["models"]["model-x"]["limit"] == {
        "context": run.OPENCODE_DEFAULT_CONTEXT_LIMIT,
        "output": 8192,
    }


def test_opencode_generated_config_sets_builtin_provider_model_limit():
    class ModelConfig:
        api_base = None
        api_key_env = "DEEPSEEK_API_KEY"
        max_tokens = 4096

    config = json.loads(
        run.build_opencode_config_content(ModelConfig(), "deepseek/deepseek-v4-flash")
    )

    assert config["provider"]["deepseek"]["options"] == {
        "apiKey": "{env:DEEPSEEK_API_KEY}",
    }
    assert config["provider"]["deepseek"]["models"]["deepseek-v4-flash"] == {
        "limit": {"context": run.OPENCODE_DEFAULT_CONTEXT_LIMIT, "output": 4096},
    }


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


def test_run_all_forwards_dash_prefixed_openhands_extra_args():
    args = run_all.benchmark_args(
        "swebench_multilingual",
        {
            "harness": "openhands-swe",
            "openhands_command_cwd": "/tmp/openhands-benchmarks",
            "openhands_extra_arg": ["--modal", "--note smoke"],
        },
    )

    assert "--openhands-command-cwd" in args
    assert "/tmp/openhands-benchmarks" in args
    assert "--openhands-extra-arg=--modal" in args
    assert "--openhands-extra-arg=--note smoke" in args


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
