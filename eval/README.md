# Eval Module

This module contains benchmark code in this repo.

## Layout

- `eval/model/`: shared OpenAI-compatible model configuration and local serving helpers.
- `eval/run_all.py`: config-driven runner for all benchmarks.
- `eval/benchmarks/deepswe/`: DeepSWE easiest-5 subset benchmark through Pier/mini-swe-agent.
- `eval/benchmarks/swebench_multilingual/`: 30-task predictive SWE-bench Multilingual subset.
- `eval/benchmarks/livecodebench_v6/`: LiveCodeBench v6 50-task predictive subset and 175-task full slice.

Run outputs go under `runs/` and are gitignored. API keys are read from
environment variables. The OpenHands adapter writes a temporary LLM config under
the run output directory unless you pass `--openhands-llm-config`.

## Model Config

All benchmark runners accept the same OpenAI-compatible model fields:

```json
{
  "model": {
    "name": "deepseek-v4-flash",
    "litellm_model": "openai/deepseek-v4-flash",
    "api_base": "https://api.deepseek.com",
    "api_key_env": "DEEPSEEK_API_KEY",
    "temperature": 0,
    "max_tokens": 4096,
    "extra_body": {
      "thinking": {"type": "disabled"}
    }
  }
}
```

- `name`: model name sent to direct OpenAI-compatible clients.
- `litellm_model`: model name for Pier/mini-swe-agent and mini-swe-agent LiteLLM flows.
- `api_base`: OpenAI-compatible base URL. Omit for the default OpenAI API.
- `api_key_env`: environment variable containing the API key.
- `extra_body`: provider-specific request body, such as DeepSeek thinking mode or OpenRouter routing.

For local vLLM/SGLang OpenAI-compatible servers, set `api_base` to the local
server URL and either set `OPENAI_API_KEY=dummy` or set `"require_api_key": false`.

## Run All Benchmarks

Dry run the configured commands:

```bash
.venv-swe-uv/bin/python -m eval.run_all \
  --config eval/configs/all_benchmarks.example.json \
  --dry-run
```

Run the configured benchmarks:

```bash
.venv-swe-uv/bin/python -m eval.run_all \
  --config eval/configs/all_benchmarks.example.json
```

Skip a benchmark:

```bash
.venv-swe-uv/bin/python -m eval.run_all --skip livecodebench_v6
```

Run only one benchmark:

```bash
.venv-swe-uv/bin/python -m eval.run_all --only swebench_multilingual
```

Override config values without editing the file:

```bash
.venv-swe-uv/bin/python -m eval.run_all \
  --set model.name='"Qwen/Qwen3-8B"' \
  --set model.api_base='"http://172.17.0.1.nip.io:8000/v1"' \
  --set model.api_key_env='"OPENAI_API_KEY"' \
  --set model.extra_body='{}' \
  --set model.require_api_key=false \
  --set benchmarks.livecodebench_v6.generation_workers=525
```

## Provider Examples

DeepSeek API:

```bash
export DEEPSEEK_API_KEY=...
.venv-swe-uv/bin/python -m eval.run_all \
  --set model.name='"deepseek-v4-flash"' \
  --set model.litellm_model='"openai/deepseek-v4-flash"' \
  --set model.api_base='"https://api.deepseek.com"' \
  --set model.api_key_env='"DEEPSEEK_API_KEY"' \
  --set model.extra_body='{"thinking":{"type":"disabled"}}'
```

OpenRouter API:

```bash
export OPENROUTER_API_KEY=...
.venv-swe-uv/bin/python -m eval.run_all \
  --set model.name='"qwen/qwen3.5-9b"' \
  --set model.litellm_model='"openai/qwen/qwen3.5-9b"' \
  --set model.api_base='"https://openrouter.ai/api/v1"' \
  --set model.api_key_env='"OPENROUTER_API_KEY"' \
  --set model.extra_body='{"provider":{"order":["venice/fp8","together","siliconflow/fp8"],"allow_fallbacks":true}}'
```

Local OpenAI-compatible server:

```bash
export OPENAI_API_KEY=dummy
.venv-swe-uv/bin/python -m eval.run_all \
  --set model.name='"Qwen/Qwen3-8B"' \
  --set model.api_base='"http://172.17.0.1.nip.io:8000/v1"' \
  --set model.extra_body='{}' \
  --set model.require_api_key=false
```

The `172.17.0.1.nip.io` address is intentional for DeepSWE: Pier runs the
agent inside Docker task containers, where `127.0.0.1` would mean the task
container itself rather than the host proxy/server.

Equivalent ready-to-edit configs are available in:

- `eval/configs/all_benchmarks.example.json`
- `eval/configs/openrouter_qwen.example.json`
- `eval/configs/local_openai.example.json`

## Local Serving

Start one vLLM server per GPU:

```bash
.venv-swe-uv/bin/python -m eval.model.serving \
  --model Qwen/Qwen3-8B \
  --served-model-name Qwen/Qwen3-8B \
  --gpus 0 \
  --base-port 8100 \
  --max-model-len 32768 \
  --background
```

For multiple local replicas, start a round-robin OpenAI-compatible proxy:

```bash
OPENAI_BACKENDS=http://127.0.0.1:8100,http://127.0.0.1:8101 \
.venv-swe-uv/bin/python -m eval.model.round_robin_proxy
```

Then point benchmark config at `http://172.17.0.1.nip.io:8000/v1` when running
DeepSWE, or `http://127.0.0.1:8000/v1` for host-only benchmarks.

## Individual Benchmarks

DeepSWE easiest-5 subset:

```bash
.venv-swe-uv/bin/python -m eval.benchmarks.deepswe.run \
  --model deepseek-v4-flash \
  --litellm-model openai/deepseek-v4-flash \
  --api-base https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY
```

SWE-bench Multilingual predictive 30:

```bash
.venv-swe-uv/bin/python -m eval.benchmarks.swebench_multilingual.run \
  --model deepseek-v4-flash \
  --litellm-model openai/deepseek-v4-flash \
  --api-base https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY
```

SWE-bench Multilingual supports three generation harness values:

- `mini-swe-agent`: the existing default. It writes `preds.json` directly and
  then runs `swebench.harness.run_evaluation`.
- `openhands-swe`: invokes the OpenHands benchmark command
  `swebenchmultilingual-infer`, converts its `output.jsonl` to `preds.json`,
  fills missing failed instances with empty patches, and then runs the same
  official evaluation command. This harness is experimental: it now runs end to
  end in the local smoke setup, but agent quality is not validated and failures
  may evaluate as empty patches.
- `opencode`: checks out each selected repository at `base_commit`, runs
  `opencode run` in that worktree, collects the git diff, writes `preds.json`,
  and then runs the same official evaluation command. A command-template
  compatibility hook is still available for custom wrappers. This harness is
  experimental: it now runs end to end in the local smoke setup, but agent
  quality is not validated beyond that smoke coverage.

Keep `mini-swe-agent` as the default harness for benchmark runs until
`openhands-swe` and `opencode` are validated beyond smoke coverage.

OpenHands example:

```bash
.venv-swe-uv/bin/python -m eval.benchmarks.swebench_multilingual.run \
  --harness openhands-swe \
  --model deepseek-v4-flash \
  --litellm-model openai/deepseek-v4-flash \
  --api-base https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY \
  --openhands-workspace docker
```

The OpenHands command comes from the `OpenHands/benchmarks` project. Install it
so `swebenchmultilingual-infer` is on `PATH`, or pass
`--openhands-infer-command`. If you run the command from a source checkout, set
`--openhands-command-cwd` to that checkout path. For source checkouts, the
wrapper can patch the local checkout at runtime to forward CA/pager environment
into Docker, omit Docker `--platform` on local images, and keep `datasets`
compatible with the SWE-bench Multilingual cache. Those defaults can be disabled
with `--no-openhands-forward-ca-bundle` and
`--no-openhands-fix-datasets-dependency`.

opencode example:

```bash
.venv-swe-uv/bin/python -m eval.benchmarks.swebench_multilingual.run \
  --harness opencode \
  --model deepseek-v4-flash \
  --api-base https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY \
  --opencode-model deepseek/deepseek-v4-flash
```

The default opencode command is `npx --yes opencode-ai`. Override it with
`--opencode-command` if `opencode` is installed another way. If
`--opencode-model` is omitted, the runner infers `deepseek/<model>` for
`DEEPSEEK_API_KEY`, `openai/<model>` for OpenAI defaults, and
`deepswe/<model>` for other OpenAI-compatible `--api-base` values.

Unless `--opencode-config` is provided, the runner supplies
`OPENCODE_CONFIG_CONTENT` with the selected model and provider. That generated
config references API keys as `{env:...}` and does not write secrets to disk.
Use `--opencode-workspace` to choose where per-instance worktrees, logs, and
opencode state are stored. The native runner isolates `HOME` and XDG state per
instance so user-level opencode config/plugins do not affect benchmark runs.

opencode example with a local wrapper:

```bash
.venv-swe-uv/bin/python -m eval.benchmarks.swebench_multilingual.run \
  --harness opencode \
  --model deepseek-v4-flash \
  --api-base https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY \
  --opencode-command-template 'python scripts/opencode_swebench.py --instances {instance_ids_path} --output {predictions_path} --model {model} --api-base {api_base}'
```

Available opencode template fields are `{instance_ids_path}`,
`{predictions_path}`, `{output_dir}`, `{instance_ids}`, `{filter_regex}`,
`{model}`, `{opencode_model}`, `{litellm_model}`, `{api_base}`,
`{api_key_env}`, `{temperature}`, `{max_tokens}`, and `{workers}`. Use doubled
braces for literal JSON braces in the template.

LiveCodeBench v6 predictive 50:

```bash
.venv-swe-uv/bin/python -m eval.benchmarks.livecodebench_v6.run \
  --model deepseek-v4-flash \
  --api-base https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY
```

LiveCodeBench v6 full 175-task slice:

```bash
.venv-swe-uv/bin/python -m eval.benchmarks.livecodebench_v6.run --all-tasks
```
