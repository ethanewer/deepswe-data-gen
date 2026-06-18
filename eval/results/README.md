# Eval Results

Committed benchmark result write-ups. Each file summarizes a run; the full
artifacts (traces, predictions, official report JSONs, logs) live under `runs/`,
which is **gitignored**. The committed `predictive_*` files and `defaults.json`
in each benchmark directory are subset-definition **inputs** written by
`build_predictive_subset.py`, not results.

Setting legend: **local** = local-GPU vLLM on one host; **slurm** = vLLM on a
SLURM GPU allocation; **openrouter** = OpenRouter hosted API.

| Benchmark | Model | Setting | Date | Score | File |
| --- | --- | --- | --- | --- | --- |
| LiveCodeBench v6 + SWE-bench ML | 5 models (Olmo3-7B, Qwen3.5-9B/4B, Qwen3-4B-Thinking-2507, Qwen3-VL-8B) | local (8-GPU) | 2026-06-06 | per-model (LCB pass@1 18-54%; SWE-ML 3-20% / 30 tasks) | `2026-06-06_local-8gpu_livecodebench-swebench-ml.md` |
| SWE-bench ML easy-10 (n=5) | qwen/qwen3.5-9b | openrouter | 2026-06-18 | mean 52.0%; pass@5 80% | `qwen3.5-9b_swebench_ml_easy10_n5_20260618.md` |
| SWE-bench ML easy-10 (n=5) | Qwen/Qwen3-VL-8B-Thinking (base) | local (H200) | 2026-06-18 | mean 12.0% (stopped-partial) | `qwen3_vl_8b_thinking_base_swebench_ml_easy10_n5_20260618.md` |
| SWE-bench ML easy-10 (n=5) | Qwen/Qwen3-4B-Thinking-2507 (base) | slurm (H200, 2-GPU) | 2026-06-17 | mean 6.0% | `qwen3_4b_thinking_2507_base_swebench_ml_easy10_n5_20260617.md` |
| SWE-bench ML predictive-30 | Qwen3-VL-2B text SFT (step99-step399) | slurm/local (H200) | 2026-06-12+ | N/A (invalid harness use; degenerate traces) | `qwen3_vl_2b_text_sft_swebench_training.md` |

> Note: the easy-10 subset is a deliberately easier debugging slice, not the
> reportable SWE-bench Multilingual score (which uses predictive-30).
