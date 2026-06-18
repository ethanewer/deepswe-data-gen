# Eval Module

This module contains benchmark code in this repo.

## Layout

- `eval/model/`: shared OpenAI-compatible model configuration and local serving helpers.
- `eval/run_all.py`: config-driven runner for all benchmarks.
- `eval/serving/configs/`: recorded local vLLM serving configs (see that dir's README).
- `eval/benchmarks/deepswe/`: DeepSWE easiest-5 subset benchmark through Pier/mini-swe-agent.
- `eval/benchmarks/swebench_multilingual/`: 30-task predictive SWE-bench Multilingual subset.
- `eval/benchmarks/swebench_verified/`: 20-task predictive SWE-bench Verified subset.
- `eval/benchmarks/livecodebench_v6/`: LiveCodeBench v6 50-task predictive subset and 175-task full slice.
- `eval/terminal_bench/`: vendored Terminus-2 agent library (see Harnesses below).

Run outputs go under `runs/` and are gitignored. API keys are read from
environment variables. The OpenHands adapter writes a temporary LLM config under
the run output directory unless you pass `--openhands-llm-config`.

## Serving vs Runners

There are two distinct layers, and they are kept separate on purpose:

- **Serving plumbing (`eval/model/`)**: brings up an OpenAI-compatible endpoint.
  `eval.model.serving` starts one vLLM process per GPU; `eval.model.round_robin_proxy`
  fans requests across replicas; `eval.model.serve_from_config` does both from a
  recorded JSON config in `eval/serving/configs/`. This layer only *serves a
  model*; it knows nothing about benchmarks.
- **Per-benchmark runners**: each `eval/benchmarks/<name>/run.py` is the
  OpenAI-compatible runner that drives an agent harness and scores results. It
  talks to *any* OpenAI-compatible `--api-base` — a hosted API (DeepSeek /
  OpenRouter) or a locally served endpoint from the serving layer. The shell /
  sbatch scripts next to each `run.py` are convenience wrappers that wire a
  specific execution setting (local GPUs, SLURM GPUs, or OpenRouter) to that
  `run.py`. The serving config / replica setup and the runner are composed, not
  coupled.

## Benchmarks x execution settings

Which wrapper scripts exist per benchmark, by where the model runs. Every
benchmark's `run.py` additionally works directly against any hosted API. "serve
config" is the recorded `eval/serving/configs/` config most associated with the
benchmark (the local/SLURM serving setups; OpenRouter needs none).

| Benchmark | LOCAL-GPU | SLURM-GPU | OPENROUTER-API | serve config |
| --- | --- | --- | --- | --- |
| deepswe | `serve_vllm_replicas.sh` + `serve_round_robin_proxy.sh` + `run_eval_local.sh` (Pier) | — | via `run.py --api-base` | per-GPU vLLM replicas (no JSON config) |
| swebench_multilingual | via `run.py --api-base` (e.g. against a `serve_from_config` endpoint) | `slurm_qwen3_4b_swe260612_step50_l40s_{8,4}gpu.sbatch`, `..._warm_wait_...8gpu.sbatch`, `slurm_qwen3vl2b_text_8gpu.sbatch`, `slurm_qwen3vl2b_text_h200_1gpu.sbatch` | `run_qwen3.5-9b_openrouter_easy10_eval.sh` | `qwen3_vl_2b_text_*`, `qwen3_5_9b_8gpu_131k_tools.json` |
| swebench_verified | via `run.py --api-base` | — | via `run.py --api-base` | (reuses SWE-bench ML serve configs) |
| livecodebench_v6 | via `run.py --api-base` (against a `serve_from_config` endpoint) | — | via `run.py --api-base` | `qwen3_4b_thinking_2507_8gpu_65k.json`, `qwen3_5_4b_8gpu_262k_mtp_tools.json` |

Per-benchmark setup details: `eval/benchmarks/deepswe/README.md`,
`eval/benchmarks/swebench_multilingual/README.md`,
`eval/benchmarks/livecodebench_v6/README.md`.

## Results

Committed result write-ups live in `eval/results/` (indexed by
`eval/results/README.md`). Full artifacts (traces, predictions, official report
JSONs, logs) live under `runs/`, which is **gitignored**. The committed
`predictive_*` files and `defaults.json` in each benchmark directory are
subset-definition **inputs** written by `build_predictive_subset.py`, not
results.

## Harnesses

`mini-swe-agent` is the default generation harness for the SWE-bench benchmarks
(it writes `preds.json` and runs the official `swebench.harness.run_evaluation`).
The SWE-bench Multilingual runner also supports `openhands-swe`, `opencode`, and
`terminus-2`. The `terminus-2` harness uses the vendored agent library in
`eval/terminal_bench/` (a Terminus-2 agent implementation, not a standalone /
orphan benchmark — it has no `run.py`); see `eval/terminal_bench/README.md`.

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

SWE-bench Verified predictive 20:

```bash
.venv-swe-uv/bin/python -m eval.benchmarks.swebench_verified.run \
  --model deepseek-v4-flash \
  --litellm-model openai/deepseek-v4-flash \
  --api-base https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY
```

This is the 20-task predictive subset (`predictive_20_instance_ids.txt`, built
by `build_predictive_subset.py`). Like SWE-bench Multilingual, it defaults to the
`mini-swe-agent` harness and the official `swebench.harness.run_evaluation`.

LiveCodeBench v6 full 175-task slice:

```bash
.venv-swe-uv/bin/python -m eval.benchmarks.livecodebench_v6.run --all-tasks
```
