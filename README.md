# DeepSWE Eval Utilities

This repository is organized around the `eval/` module.

It contains:

- DeepSWE subset benchmark
- SWE-bench Multilingual predictive subset benchmark
- LiveCodeBench v6 predictive/full-slice benchmark
- SWE-rebench V2 synthetic data and Harbor task generation utilities

All benchmark runners use OpenAI-compatible model configuration, so the same
setup works for hosted APIs such as DeepSeek/OpenRouter and local vLLM/SGLang
servers.

Start with:

```bash
.venv-swe-uv/bin/python -m eval.run_all --config eval/configs/all_benchmarks.example.json --dry-run
```

Detailed setup, model configuration, and per-benchmark commands are documented
in [eval/README.md](eval/README.md).
