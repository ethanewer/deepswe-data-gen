# SWE-bench Verified predictive subset

This module builds and runs a 20-task predictive subset of the 500-task
SWE-bench Verified benchmark.

## Subset

`build_predictive_subset.py` selects 20 instances whose pass/fail pattern best
predicts full-benchmark scores, validated against known full runs (e.g.
`claude-opus-4.7`, `deepseek-v3.2`, `deepseek-v4-flash` from SWE-Router).
Generated subset artifacts (committed; do not hand-edit):

- `predictive_20_instance_ids.txt`
- `predictive_20_tasks.csv`
- `predictive_20_model_score_comparison.csv`
- `predictive_20_summary.json`

## Run

```bash
.venv-swe-uv/bin/python -m eval.benchmarks.swebench_verified.run \
  --model deepseek-v4-flash \
  --litellm-model openai/deepseek-v4-flash \
  --api-base https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY
```

Like SWE-bench Multilingual, it defaults to the `mini-swe-agent` harness and the
official `swebench.harness.run_evaluation`, reads `predictive_20_instance_ids.txt`
+ `defaults.json` at runtime, and writes results under `runs/swebench_verified/`
(gitignored). For a locally/slurm-served checkpoint, the swebench_multilingual
step50 driver exposes this benchmark via `BENCHMARK=verified` (it reuses the
SWE-bench ML serve configs). See [../../README.md](../../README.md) for the
benchmark × execution-setting map.
