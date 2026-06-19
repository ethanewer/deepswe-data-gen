# Data Generation

This package contains data generation and data preparation utilities that are
not benchmark runners.

## Layout

- `datagen/swerebench_v2/` — the SWE-rebench V2 synthetic-trace pipeline (flat
  modules, grouped by stage in the module map below), its `monitoring/` helpers,
  and committed `data/`, `examples/`, `reports/`, and `minisweagent_*.yaml`
  profile configs.
- `datagen/dataset_builders/` — standalone dataset-construction CLIs
  (other-source ingestion, append-only raw builds, trace compaction, quality
  audits). Previously under the top-level `scripts/` directory; several import
  each other by bare module name, so they live together in one directory.
  `compact_long_passed_traces.py` imports `litellm`, which is in
  `requirements-terminus.txt` (the terminus extras), not the base requirements.

All modules run via `python -m datagen.<pkg>.<module>` from the repo root.
Generated outputs go under `runs/` unless they are committed source data.

See `docs/swerebench_datagen_harness.md` for the full datagen harness — it spans
both `datagen/swerebench_v2/` and `datagen/dataset_builders/` — covering
audit/export, compaction, other-source, and raw source dataset tools.

## Module map (`datagen/swerebench_v2/`)

The flat modules form a pipeline; flow is generate → select → manifest → submit →
run (drivers) → monitor → audit/report:

| Stage | Modules |
| --- | --- |
| generate | `generate_high_quality_subset`, `run_data_generation`, `run_all` |
| prompts | `analyze_prompts`, `rewrite_prompts` |
| selection | `select_supplemental_tasks`, `select_openrouter_original_wave`, `prepare_openrouter_comparison`, `build_mimo_easy_assignments` |
| manifests | `prepare_pyxis_manifest`, `build_direct_retry_manifest` |
| tasks | `generate_harbor_tasks` |
| runners | `pyxis_miniswe_agent_driver`, `run_docker_datagen_packed`, `write_pyxis_failure_result` |
| submit | `submit_pyxis_datagen`, `submit_pyxis_datagen_packed`, `submit_docker_datagen_packed` |
| monitoring (in `monitoring/`) | `monitor_l40s_direct_qwen`, `monitor`, `active_hourly_check` |
| reporting | `audit_export_clean_trajectory`, `audit_export_clean_run`, `summarize_pier_runs` |

Supporting assets at the package root:

- `data/` — committed source artifacts; `high_quality_conf_ge_0.95_tasks.csv` is
  the canonical input consumed by the analyze/rewrite/generate/select modules.
- `examples/` — rewritten-prompt samples; `rewrites.jsonl` in each
  `rewritten-prompts*` dir is a live input to `generate_harbor_tasks`, not just a
  sample.
- `reports/` — committed markdown analyses, including the DeepSWE datagen
  progress log (`progress-report-deepswe-datagen.md`).
- `minisweagent_*.yaml` — prompt/submission profiles (datagen-strict,
  swebench-multilingual, deepswe-pier, and the mimo system/observation/combined
  variants) selected by the submit/runner modules.

## SWE-rebench V2

The default generated SWE-rebench V2 subset is confidence-filtered and limited
to languages represented in either the SWE-bench Multilingual predictive
30-task subset or the DeepSWE easiest 5-task subset:

```text
c, cpp, go, java, js, php, python, ruby, rust, ts
```

With the current `nebius/SWE-rebench-V2` train split and `confidence >= 0.95`,
that default produces 15,296 tasks: 5,518 easy, 9,443 medium, and 335 hard.
Ruby is included in the default split, but currently contributes zero matching
SWE-rebench V2 rows after the other quality filters.

Regenerate the default subset:

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

Pyxis mini-swe-agent trace generation now selects benchmark-matching prompt
configs from `instruction_style` by default:

- `original`/`swe_rebench` -> SWE-bench Multilingual `swebench.yaml`-style
  prompt and submission flow.
- `deepswe`/`rewritten`/`planned` -> DeepSWE/Pier `mini.yaml`-style prompt
  and submission flow.

Pass `--config-file` or `--benchmark-profile` to
`datagen.swerebench_v2.submit_pyxis_datagen` only when intentionally running a
custom or legacy prompt contract.
