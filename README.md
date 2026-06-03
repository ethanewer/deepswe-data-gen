# DeepSWE Data Generation

Utilities for generating SWE-rebench V2 task subsets, converting them into
Pier/Harbor task directories, and running them with the same agent harness used
by DeepSWE: `pier run --agent mini-swe-agent`.

DeepSWE quality principles to keep in mind:

- Prompts should be short, natural, and behavior-focused.
- The harness should be fixed across models.
- Verifiers should check observable behavior and include regression coverage.
- Environment failures and flaky tests should not be counted as model failures.

The source dataset here is still SWE-rebench V2, so it is not contamination-free
in the way DeepSWE is. The scripts below move the harness and prompt packaging
closer to DeepSWE, but they do not turn public PR-derived tasks into original
DeepSWE tasks.

## Setup

```bash
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY=...
```

Or install Pier as a tool:

```bash
uv tool install git+https://github.com/datacurve-ai/pier
```

## Regenerate Compact Data

```bash
python3 scripts/run_data_generation.py
```

This regenerates the high-quality SWE-rebench V2 subset in `swerebench-v2/`.

## Analyze Prompt Style

```bash
python3 scripts/analyze_prompts.py
```

This writes:

- `swerebench-v2/prompt_analysis.csv`
- `swerebench-v2/prompt_style_analysis.md`

The current analysis concludes that most selected SWE-rebench prompts should be
reviewed or rewritten before treating them as DeepSWE-style prompts. The
generated Harbor tasks default to `--instruction-style deepswe`, which removes
issue-template checkboxes, generated interface blocks, and external URLs from
the agent-facing prompt.

## Generate Pier/Harbor Tasks

Generate one easy task for a smoke test:

```bash
python3 scripts/generate_harbor_tasks.py --clean --limit 1 --difficulty easy
```

Generate a larger filtered set:

```bash
python3 scripts/generate_harbor_tasks.py --clean --difficulty hard --language python
```

Generated task directories are written to `swerebench-v2/harbor-tasks/` and are
ignored by Git.

## Run With DeepSWE Harness

```bash
python3 scripts/run_deepswe.py --model openai/gpt-5.4-mini --limit 1
```

This shells out to:

```bash
pier run -p swerebench-v2/harbor-tasks --agent mini-swe-agent --model openai/gpt-5.4-mini
```

For a fast harness-only smoke test, skip verifier execution:

```bash
python3 scripts/run_deepswe.py --model openai/gpt-5.4-mini --limit 1 --disable-verification
```

Pier outputs are written under `runs/pier-jobs/` and ignored by Git.

If Docker build fails while installing `mini-swe-agent` with a `curl: (60) SSL
certificate problem`, Pier has parsed the task and started the correct harness,
but the local Docker build environment does not trust the certificate chain used
for outbound HTTPS. Fix the Docker/container CA configuration, then rerun the
same command.

## Run Data Generation And Smoke Test

```bash
python3 scripts/run_all.py --model openai/gpt-5.4-mini --limit 1 --difficulty easy --disable-verification
```
