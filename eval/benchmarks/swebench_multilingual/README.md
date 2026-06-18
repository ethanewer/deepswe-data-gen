# SWE-bench Multilingual benchmark

`run.py` is the OpenAI-compatible runner (used by `eval.run_all` and directly).
It supports four generation harnesses — `mini-swe-agent` (default),
`openhands-swe`, `opencode`, and `terminus-2` — then runs the official
`swebench.harness.run_evaluation` in Docker. See the top-level `eval/README.md`
for harness details and direct-API examples.

## Subsets / instance-id files

- `predictive_30_instance_ids.txt` — the reportable 30-task predictive subset
  (built by `build_predictive_subset.py`; see the committed `predictive_30_*`
  inputs). This is the default for the served-checkpoint flow.
- `swebench_multilingual_easy_10_instance_ids.txt` — a fixed 10-task **debugging
  subset** for fast SFT iteration. NOT the reportable score.
- `diagnostic_8_instance_ids.txt` — an 8-task diagnostic slice.

## Runner families by execution setting

### SLURM-GPU — served SFT checkpoint (Qwen3-4B-Thinking swe260612)

Serves a ready Hugging Face SFT checkpoint with local vLLM on the allocation,
then runs the predictive-30 (or easy-10) subset.

- `run_qwen3_4b_swe260612_step50_l40s_eval.sh` — **inner driver**. Serves
  `CHECKPOINT_STEP_DIR` directly (or the base model with `BASELINE_MODEL=true`)
  and runs the eval. The retired NeMo DCP-consolidation path has been removed;
  it now expects a servable HF checkpoint (`*.safetensors`).
- `slurm_qwen3_4b_swe260612_step50_l40s_8gpu.sbatch` /
  `..._4gpu.sbatch` — SLURM wrappers that set `EVAL_GPU_COUNT` and `exec` the
  driver. `scripts/watch_submit_qwen3_4b_swe260612_step50_eval.sh` submits the
  8- or 4-GPU wrapper depending on idle nodes.
- `slurm_qwen3_4b_swe260612_warm_wait_step50_l40s_8gpu.sbatch` — grabs the L40S
  allocation early, runs a base-model smoke serve to warm caches, **waits** for
  the training checkpoint to land, then execs the driver.
- `run_qwen3_4b_swebench_multilingual_easy_n_trials_eval.sh` — loops the driver
  over N trials on the easy-10 subset.

### SLURM-GPU — Qwen3-VL-2B text checkpoints

- `slurm_qwen3vl2b_text_8gpu.sbatch` (L40S, 8 GPUs) and
  `slurm_qwen3vl2b_text_h200_1gpu.sbatch` (H200, 1 GPU) — serve a Qwen3-VL-2B
  text checkpoint from an `eval/serving/configs/qwen3_vl_2b_text_*.json` config
  (`MODEL_CONFIG`), then run predictive-30 and write `timing.json`. (These were
  formerly named `run_local_served_*`; they are SLURM jobs, hence `slurm_`.)

### OPENROUTER-API

- `run_qwen3.5-9b_openrouter_easy10_eval.sh` — runs `qwen/qwen3.5-9b` over the
  easy-10 subset via OpenRouter (no local serving). Trials run in parallel;
  reads `OPENROUTER_API_KEY` from the environment by name only.

## Support files

- `docker_stdio_proxy.py` — exposes `docker system dial-stdio` as a user-owned
  Unix socket so the Docker SDK (used by the agent / scoring) can reach a setgid
  Docker CLI without docker-group membership. Started by the L40S driver when
  `ENABLE_DOCKER_STDIO_PROXY=true`.
- `docker_sdk_sitecustomize/` — a `sitecustomize.py` placed on `PYTHONPATH` for
  swebench scoring subprocesses; applies Docker SDK timeout / pool-size tweaks.
- `export_dcp_torchsave_to_hf.py` — legacy NeMo DCP/torch.save → HF exporter.
  **Now orphaned**: the served-checkpoint flow serves ready HF checkpoints and
  no longer calls it. Kept for reference.
- `aggregate_easy10_reports.py` — aggregates official report JSONs across trials
  for the easy-10 subset (mean, pass@N, pass^N, per-instance solve counts).
- `build_predictive_subset.py` — builds the predictive-30 subset and its
  committed `predictive_30_*` definition inputs.
