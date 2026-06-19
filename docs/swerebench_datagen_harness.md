# SWE-rebench Datagen Harness

This document covers the datagen-only harness and source-dataset tooling used
for SWE-rebench style synthetic trace generation.

## Goals

- Keep the default benchmark harness available and benchmark-shaped.
- Allow a specialized datagen harness for controlled pilots without making the
  exported trace format depend on datagen-only prompt artifacts.
- Save accepted, rejected, raw, and converted traces so downstream SFT builders
  can filter, cap, mask, or reweight instead of relying on generation-time
  deletion.
- Prefer high-quality, short, passing trajectories, but preserve all generated
  source rows with metadata when building raw source datasets.

## Benchmark Profiles

`datagen/swerebench_v2/pyxis_miniswe_agent_driver.py` and the packed Docker
runner select benchmark profiles from `instruction_style` unless explicitly
overridden:

- `original` and `swe_rebench` use the SWE-bench Multilingual style config.
- `deepswe`, `rewritten`, and `planned` use the DeepSWE/Pier style config.
- unknown/custom styles fall back to `datagen-strict`.

Use `--benchmark-profile` or `--config-file` only for controlled experiments.
For benchmark-shaped data generation, keep the benchmark YAML selected by the
profile.

## Mimo Clean Harness Variants

The Mimo pilot configs live in `datagen/swerebench_v2/`:

- `minisweagent_mimo_clean_system.yaml`: benchmark-like prompt plus additional
  system guidance about inspecting source, making small production edits, using
  `git diff`, and verifying a non-empty `patch.txt` preview.
- `minisweagent_mimo_clean_observation.yaml`: benchmark-like prompt plus
  observation reminders.
- `minisweagent_mimo_clean_combined.yaml`: both system guidance and
  observation reminders.

Observation reminders are enabled by:

```yaml
datagen_harness:
  observation_reminders: true
```

The runner inserts reminders only in `output.extra.reminder`; the YAML renders
them as `<blocks_reminder>...</blocks_reminder>`. The exporter strips these
blocks when converting accepted traces back to the benchmark-shaped target
config.

The current reminder triggers are:

- empty visible `git diff` output;
- reading an empty `patch.txt`;
- weak patch checks such as `head`, `tail`, `grep`, or `wc` on `patch.txt`
  without a full patch preview.

## Runner Reliability

The Docker packed runner now supports:

- `--max-concurrent-pulls` to avoid Docker daemon and registry storms on packed
  CPU nodes;
- unique container names based on rollout id, manifest index, pid, and monotonic
  timestamp;
- early success when `result.json` is written, so container shutdown races do
  not lose completed rows;
- `DATAGEN_HOST_WORKSPACE`, allowing container-written result metadata to point
  back to the host workspace;
- sanitized task environments that remove runner `PYTHONPATH`, `PYTHONHOME`,
  and `PYDEPS_OVERLAY` from commands executed inside the task repository.

The Pyxis packed submitter also supports local Docker builds for other-source
tasks with `environment/Dockerfile.prepared` when `CONTAINER_SOURCE=dockerd`.

## Trace Audit And Export

Single trace:

```bash
python -m datagen.swerebench_v2.audit_export_clean_trajectory \
  --result-json /path/to/result.json \
  --output-trajectory /path/to/exported.trajectory.json \
  --audit-json /path/to/exported.audit.json
```

Full run:

```bash
python -m datagen.swerebench_v2.audit_export_clean_run \
  --result-index /path/to/result_index.jsonl \
  --manifest-tsv /path/to/manifest.tsv \
  --output-dir /path/to/audit-export
```

The full-run exporter writes:

- `accepted/`: benchmark-shaped accepted traces;
- `accepted_raw/`: original accepted traces before conversion;
- `rejected/`: rejected raw traces;
- `accepted_patches/` and `rejected_patches/`;
- `audits/`: row-level audit JSON with accept/reject reasons.

The audit rejects traces with empty or invalid patches, manual patch
construction, patch-fragment assembly, missing submit mechanics, non-source
patches, and obvious test/config/generated-file targets. Rejections are saved
instead of deleted.

## Task Selection

`build_mimo_easy_assignments.py` creates easy-only Mimo assignment CSVs,
preferring smaller source changes and fewer failing tests.

`generate_harbor_tasks.py` accepts repeated `--instance-id` and repeated
`--instance-id-file` arguments, which makes targeted reruns and C/C++ coverage
top-ups auditable.

Targeted limitation selectors:

- `datagen/dataset_builders/select_initial_limitations_wave.py`
- `datagen/dataset_builders/select_limitations_target_tasks.py`
- `datagen/dataset_builders/research_task_sources.py`
- `datagen/dataset_builders/prepare_other_source_datagen.py`
- `datagen/dataset_builders/assert_other_source_task_quality.py`

These scripts prioritize underrepresented languages, small diffs, strict
metadata, and runnable/verifiable task sources.

## Compaction

`datagen/dataset_builders/compact_long_passed_traces.py` creates compacted rows from long passed
traces. It keeps:

- original row identifiers and source paths;
- cut/boundary metadata;
- generated prompt metadata;
- compaction model responses and reasoning;
- compacted raw records and `compaction_index.jsonl`.

Audit compaction outputs with:

```bash
python datagen/dataset_builders/audit_compaction_outputs.py \
  --compaction-output /path/to/compaction-output \
  --output-dir /path/to/compaction-audit

python datagen/dataset_builders/audit_compaction_prompt_boundaries.py \
  --dataset /path/to/raw-source-dataset \
  --output-dir /path/to/boundary-audit
```

Compacted rows should keep enough metadata to let a downstream builder skip the
original row when training on its compacted descendant.

## Raw Source Dataset Assembly

Raw source dataset builders intentionally include generated traces without
enforcing final SFT filters:

- `datagen/dataset_builders/build_raw_all_generated_dataset_20260616.py`
- `datagen/dataset_builders/build_raw_all_generated_dataset_fast_20260616.py`
- `datagen/dataset_builders/build_raw_all_generated_append_only_20260616.py`
- `datagen/dataset_builders/build_limitations_raw_dataset.py`
- `datagen/dataset_builders/build_combined_other_sources_dataset.py`
- `datagen/dataset_builders/build_c_cpp_only_dataset.py`

These scripts attach metadata for source paths, source kind, pass/fail state,
reasoning fraction, token-length signals where available, audit status,
compaction lineage, and rejection reasons.

## Local Teacher Serving

`docs/local_model_serving.md` and `scripts/local_model_serving/` document and
launch local OpenAI-compatible teacher endpoints. The Kimi K2.7-Code vLLM and
SGLang launchers are included for controlled future data-generation runs.
