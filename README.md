# DeepSWE Utilities

This repository is organized around model SFT, benchmark evaluation, and data
generation.

It contains:

- Qwen3 agentic SFT (ms-swift) — `sft/qwen3/`
- DeepSWE subset benchmark
- SWE-bench Multilingual predictive subset benchmark
- LiveCodeBench v6 predictive/full-slice benchmark
- SWE-rebench V2 data generation and Harbor task utilities

All benchmark runners use OpenAI-compatible model configuration, so the same
setup works for hosted APIs such as DeepSeek/OpenRouter and local vLLM/SGLang
servers.

Start with:

```bash
.venv-swe-uv/bin/python -m eval.run_all --config eval/configs/all_benchmarks.example.json --dry-run
```

Detailed setup, model configuration, and per-benchmark commands are documented
in [eval/README.md](eval/README.md). Data generation commands are documented in
[datagen/README.md](datagen/README.md).

The Qwen3 SFT recipe (4B + 8B FSDP data-parallel ms-swift training, the
Qwen3-VL-text → Qwen3 checkpoint conversion for the 8B model, and the HF-dataset
preprocessing) is documented in [sft/qwen3/README.md](sft/qwen3/README.md).

Local Kimi K2.6 and MiMo V2.5 serving setup for SWE-RBench synthetic data is
documented in [docs/local_model_serving.md](docs/local_model_serving.md).
