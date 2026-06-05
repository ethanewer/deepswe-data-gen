# Data Generation

This package contains data generation and data preparation utilities that are
not benchmark runners.

## Layout

- `datagen/swerebench_v2/`: SWE-rebench V2 filtering, prompt rewriting, and
  Harbor task generation.

Generated outputs should go under `runs/` unless they are committed source data.

## SWE-rebench V2

Regenerate the confidence-filtered SWE-rebench V2 subset:

```bash
.venv-swe-uv/bin/python -m datagen.swerebench_v2.run_data_generation
```

Analyze prompt style:

```bash
.venv-swe-uv/bin/python -m datagen.swerebench_v2.analyze_prompts
```

Rewrite prompts with an OpenAI-compatible API:

```bash
export OPENAI_API_KEY=...
.venv-swe-uv/bin/python -m datagen.swerebench_v2.rewrite_prompts \
  --model gpt-5.4-mini \
  --limit 10
```

Generate Harbor task directories:

```bash
.venv-swe-uv/bin/python -m datagen.swerebench_v2.generate_harbor_tasks \
  --clean \
  --limit 10 \
  --instruction-style deepswe
```

The generated Harbor tasks are written under `runs/swerebench-v2/harbor-tasks`
by default.
