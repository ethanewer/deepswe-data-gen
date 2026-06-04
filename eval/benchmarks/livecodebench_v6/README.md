# LiveCodeBench v6 predictive subset

This module builds and runs a 50-problem predictive subset of the 175-problem
`test6.jsonl` LiveCodeBench v6 slice.

## Subset

The subset is selected by `build_predictive_subset.py` from:

- Dataset: `livecodebench/code_generation_lite`, file `test6.jsonl`
- Public per-problem leaderboard data:
  `https://livecodebench.github.io/performances_generation.json`

The selected 50 tasks preserve full-v6 public model scores with:

- 28 complete public model vectors
- 0.606 percentage-point RMSE against full-v6 scores
- 1.143 percentage-point max absolute error

Generated subset artifacts:

- `predictive_50_question_ids.txt`
- `predictive_50_tasks.csv`
- `predictive_50_model_score_comparison.csv`
- `predictive_50_summary.json`

## DeepSeek run

The default config uses the DeepSeek OpenAI-compatible API and reads credentials
from `DEEPSEEK_API_KEY`.

```bash
REQUESTS_CA_BUNDLE="$PWD/runs/system-ca-bundle.pem" \
SSL_CERT_FILE="$PWD/runs/system-ca-bundle.pem" \
.venv-swe-uv/bin/python -m eval.benchmarks.livecodebench_v6.run
```

Current defaults are `n=3`, `generation_workers=525`, and
`evaluation_workers=64`. Add `--all-tasks` to run the full 175-task v6 slice
instead of the predictive 50-task subset.

Outputs are written under `runs/livecodebench-v6/<model>-<timestamp>/`, which is
gitignored.
