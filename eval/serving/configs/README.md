# Local Serving Configs

Each JSON here records the exact local vLLM settings (model, GPUs, context
length, proxy port, vLLM args, chat template) for a benchmark run. Launch one
with `eval.model.serve_from_config` (see `../README.md`). The launcher starts one
vLLM backend per listed GPU, waits for health, starts the round-robin proxy, and
writes a manifest under `runs/serving/`.

`MODEL_CONFIG` on the SWE-bench Multilingual `slurm_qwen3vl2b_text_*` jobs selects
which Qwen3-VL-2B config to serve; the served-SFT-checkpoint L40S driver builds
its serving config inline rather than reading one of these files.

## Active vs historical

"Active" = wired into a runner default or referenced by a benchmark README.
"Historical" = a point-in-time snapshot kept for provenance (a specific SFT
checkpoint step, or a model/run captured in `eval/results/`); not currently a
runner default, but still loadable via `serve_from_config` / `MODEL_CONFIG`.

| Config | Model | Context | GPUs | Benchmark | Status |
| --- | --- | ---: | ---: | --- | --- |
| `qwen3_4b_thinking_2507_8gpu_65k.json` | Qwen/Qwen3-4B-Thinking-2507 | 65k | 8 | LiveCodeBench v6 (175) | Active (example in `eval/serving/README.md`) |
| `qwen3_5_9b_8gpu_131k_tools.json` | Qwen/Qwen3.5-9B | 131k | 8 | SWE-bench Multilingual p30 / LCB | Historical (2026-06-06 local 8-GPU run) |
| `qwen3_5_4b_8gpu_262k_mtp_tools.json` | Qwen/Qwen3.5-4B | 262k | 8 | LiveCodeBench v6 p50 | Historical (2026-06-06 local 8-GPU run) |
| `qwen3_vl_8b_thinking_8gpu_65k_text.json` | Qwen/Qwen3-VL-8B-Thinking | 65k | 8 | LCB v6 175 / SWE-bench ML p30 | Historical (2026-06-06 local 8-GPU run) |
| `olmo3_7b_think_tools.json` | allenai/Olmo-3-7B-Think | 32k | 1 | (diagnostic, abandoned) | Historical (weaker than Qwen; not a scored run) |
| `qwen3_vl_2b_text_start_8gpu_40k.json` | Qwen/Qwen3-VL-2B-Thinking | 40k | 8 | SWE-bench ML p30 | Active (default of `slurm_qwen3vl2b_text_8gpu.sbatch`) |
| `qwen3_vl_2b_text_start_8gpu_131k.json` | Qwen/Qwen3-VL-2B-Thinking | 131k | 8 | SWE-bench ML p30 | Active (selectable via `MODEL_CONFIG`) |
| `qwen3_vl_2b_text_start_h200_1gpu_40k.json` | Qwen/Qwen3-VL-2B-Thinking | 40k | 1 | SWE-bench ML p30 | Active (default of `slurm_qwen3vl2b_text_h200_1gpu.sbatch`) |
| `qwen3_vl_2b_text_start_h200_1gpu_131k.json` | Qwen/Qwen3-VL-2B-Thinking | 131k | 1 | SWE-bench ML p30 | Active (selectable via `MODEL_CONFIG`) |
| `qwen3_vl_2b_text_step99_h200_1gpu_40k.json` | VL-2B text SFT step99 | 40k | 1 | SWE-bench ML p30 | Historical (SFT step snapshot) |
| `qwen3_vl_2b_text_step99_h200_1gpu_131k.json` | VL-2B text SFT step99 | 131k | 1 | SWE-bench ML p30 | Historical (SFT step snapshot) |
| `qwen3_vl_2b_text_step199_h200_1gpu_40k.json` | VL-2B text SFT step199 | 40k | 1 | SWE-bench ML p30 | Historical (SFT step snapshot) |
| `qwen3_vl_2b_text_step199_h200_1gpu_131k.json` | VL-2B text SFT step199 | 131k | 1 | SWE-bench ML p30 | Historical (SFT step snapshot) |
| `qwen3_vl_2b_text_step299_h200_1gpu_40k.json` | VL-2B text SFT step299 | 40k | 1 | SWE-bench ML p30 | Historical (SFT step snapshot) |
| `qwen3_vl_2b_text_final_h200_1gpu_40k.json` | VL-2B text SFT step399 (final) | 40k | 1 | SWE-bench ML p30 | Historical (SFT final snapshot) |
| `qwen3_vl_2b_text_final_h200_1gpu_131k.json` | VL-2B text SFT step399 (final) | 131k | 1 | SWE-bench ML p30 | Historical (SFT final snapshot) |

The VL-2B SFT-checkpoint configs point at `sft/qwen3/checkpoints/...` overlays
(see `eval/results/qwen3_vl_2b_text_sft_swebench_training.md`). Configs are kept,
not deleted, so past runs remain reproducible.
