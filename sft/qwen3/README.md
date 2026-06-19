# Qwen3 Thinking Agentic SFT (ms-swift)

Full-parameter supervised fine-tuning of Qwen3 thinking models on agentic
SWE traces, using [ms-swift](https://github.com/modelscope/ms-swift) inside the
ModelScope Swift Docker image. Three recipes are supported:

- **2B** — `eewer/qwen3-vl-2b-thinking-text`, single 8×H200 node.
- **4B** — `Qwen/Qwen3-4B-Thinking-2507`, single 8×H200 node.
- **8B** — `eewer/qwen3-vl-8b-thinking-text`, 2×8×H200 nodes.

The 2B and 8B bases are **text-only Qwen3 checkpoints derived from the Qwen3-VL
text towers** (see "Base checkpoints" below); the 4B base is a native Qwen3
release. All recipes are **FSDP fully-sharded data-parallel with no tensor parallelism**:
every rank is a data-parallel replica whose parameters, gradients, and optimizer
state are sharded across ranks (ZeRO-3 equivalent). There is intentionally no
DeepSpeed or tensor-parallel code path.

## Layout

```
configs/qwen3_swift_fsdp_65k_memory_first.json   shared FSDP config (wraps Qwen3DecoderLayer)
scripts/                                         the recipe (run these)
  run_qwen3_swift_inside_container.sh            the `swift sft` invocation; runs INSIDE the container
  run_qwen3_2b_swift_local_h200.sh               2B launcher: 1 node × 8 H200 (docker)
  run_qwen3_4b_swift_local_h200.sh               4B launcher: 1 node × 8 H200 (docker)
  slurm_qwen3_8b_swift_2node_h200.sbatch         8B launcher: 2 nodes × 8 H200 (slurm + docker)
  materialize_swift_messages_dataset.py          data prep: HF dataset -> swift messages train.jsonl
  prepare_qwen3_vl_text_checkpoint.py            VL->text: build a text-only view of a Qwen3-VL model
  save_qwen3_vl_text_as_qwen3_checkpoint.py      VL->text: convert that view to a normal Qwen3 checkpoint
dataset_reproduce/                               provenance: how the published v75 HF dataset was built
src/qwen_agentic_sft/                            normalization + loss-policy lib used by dataset_reproduce/
data/                                            gitignored; recipe inputs/outputs land here
```

## Prerequisites

- The ModelScope Swift image (pulled by the launchers via `DOCKER_IMAGE`):
  `modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1`
- A `zstd`/`zstdcat` CLI on the host (for data prep, which reads `*.jsonl.zst`).
- 8 visible H200 GPUs per node (the launchers assert this).

The launchers default `ROOT_DIR` and HF cache paths to a specific cluster
(`/wbl-fast/...`); override `ROOT_DIR`, `HF_HOME`, `DOCKER_IMAGE`, `MODEL`, etc.
to run elsewhere.

## Base checkpoints

The 2B and 8B recipes train **text-only Qwen3 checkpoints** extracted from the
Qwen3-VL text towers (vision/aligner modules dropped, `model.language_model.*`
renamed to `model.*`, `model_type: qwen3`). All four are published (private)
under the `eewer` account and were built with
`prepare_qwen3_vl_text_checkpoint.py` + `save_qwen3_vl_text_as_qwen3_checkpoint.py`:

| Base (HF repo)                     | Source VL model              | Used by |
| ---------------------------------- | ---------------------------- | ------- |
| `eewer/qwen3-vl-2b-thinking-text`  | `Qwen/Qwen3-VL-2B-Thinking`  | 2B recipe (default) |
| `eewer/qwen3-vl-2b-instruct-text`  | `Qwen/Qwen3-VL-2B-Instruct`  | 2B recipe (`MODEL=` override) |
| `eewer/qwen3-vl-8b-thinking-text`  | `Qwen/Qwen3-VL-8B-Thinking`  | 8B recipe (default) |
| `eewer/qwen3-vl-8b-instruct-text`  | `Qwen/Qwen3-VL-8B-Instruct`  | 8B recipe (`MODEL=` override) |

The 4B recipe uses the native `Qwen/Qwen3-4B-Thinking-2507` release (no
conversion). To rebuild any text checkpoint locally (e.g. to re-upload):

```bash
./scripts/prepare_qwen3_vl_text_checkpoint.py --repo-id Qwen/Qwen3-VL-8B-Thinking \
  --output-dir data/qwen3_vl_8b_thinking_view
./scripts/save_qwen3_vl_text_as_qwen3_checkpoint.py \
  --input-dir data/qwen3_vl_8b_thinking_view \
  --output-dir data/qwen3_vl_8b_thinking_text --copy-files
```

## 1. Prepare the training data

`materialize_swift_messages_dataset.py` turns the published, processed v75 HF
dataset into the swift messages `train.jsonl` the recipe consumes. The dataset
stores assistant tool calls in `message.tool_calls` and shell observations as
`role=tool`; ms-swift's messages preprocessor would drop both, so this step
serializes tool calls into assistant `content` as `<tool_call>...</tool_call>`
and renders observations as user-side `<tool_response>...</tool_response>`.

Download the dataset from the Hub and materialize it (default dataset is
`eewer/qwen3-4b-thinking-sft-v75-verification-strictpassed-cap4-processed`):

```bash
python3 scripts/materialize_swift_messages_dataset.py \
  --output-dir data/qwen3_v75_swift_messages
```

Or point at an existing local snapshot (a dir containing `data/*.jsonl.zst`):

```bash
python3 scripts/materialize_swift_messages_dataset.py \
  --input-root /path/to/local/snapshot \
  --output-dir data/qwen3_v75_swift_messages
```

This writes `data/qwen3_v75_swift_messages/train.jsonl` (+ a `manifest.json`).

## 2a. Train the 4B recipe (1 node × 8 H200)

```bash
./scripts/run_qwen3_4b_swift_local_h200.sh
```

Defaults: `MODEL=Qwen/Qwen3-4B-Thinking-2507`, `PACKING_LENGTH=65536`,
`PER_DEVICE_BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=2` → 16 packed sequences /
1,048,576 tokens per optimizer update. The wrapper sets env defaults and execs
`run_qwen3_swift_inside_container.sh` inside the Swift container.

## 2b. Train the 2B recipe (1 node × 8 H200)

```bash
./scripts/run_qwen3_2b_swift_local_h200.sh
```

Defaults: `MODEL=eewer/qwen3-vl-2b-thinking-text` (downloaded from the Hub),
`GRAD_ACCUM_STEPS=2` → 16 packed sequences / 1,048,576 tokens per optimizer
update. This launcher mirrors the 4B one apart from the base model. Use the
instruct base with `MODEL=eewer/qwen3-vl-2b-instruct-text`.

## 2c. Train the 8B recipe (2 nodes × 8 H200)

```bash
sbatch scripts/slurm_qwen3_8b_swift_2node_h200.sbatch
```

Defaults: `MODEL=eewer/qwen3-vl-8b-thinking-text` (downloaded from the Hub),
2 × 8 H200, `PACKING_LENGTH=65536`, `PER_DEVICE_BATCH_SIZE=1`,
`GRAD_ACCUM_STEPS=1` → 16 packed sequences / 1,048,576 tokens per update. Use the
instruct base with `MODEL=eewer/qwen3-vl-8b-instruct-text`.

LR-sweep example:

```bash
sbatch \
  --export=ALL,LR=1e-6,RUN_NAME=qwen3_8b_swift_lr1e6_s100_2node_h200 \
  scripts/slurm_qwen3_8b_swift_2node_h200.sbatch
```

## Shared training arguments

All recipes route through `run_qwen3_swift_inside_container.sh`, which calls
`swift sft` with `--model_type qwen3 --template qwen3 --train_type full
--packing --use_liger_kernel --attn_impl flash_attn` and the FSDP config
`configs/qwen3_swift_fsdp_65k_memory_first.json` (wraps `Qwen3DecoderLayer`,
activation checkpointing on, `limit_all_gathers=true`,
`backward_prefetch=backward_post`). Common overrides: `LR`, `MAX_STEPS`,
`SAVE_STEPS`, `PACKING_LENGTH`, `PER_DEVICE_BATCH_SIZE`, `GRAD_ACCUM_STEPS`,
`OUTPUT_DIR`, `RUN_NAME`.

## Reproducing the v75 dataset (provenance)

`dataset_reproduce/` documents how the published HF dataset
(`eewer/qwen3-4b-thinking-sft-v75-verification-strictpassed-cap4-processed`) was
built from raw verification-enhanced traces. It is kept for provenance and is
not needed for normal training (start from step 1 above instead). The pipeline:

1. `build_swerebench_verification_enhanced_strict_pass_allowlist.py` — strict-passed cap-4 allowlist
2. `build_swe260612_miniswe_raw.py` — apply allowlist, normalize to mini-swe-agent format
   (uses `build_swebench_ml_sft_mix.py` and `src/qwen_agentic_sft`)
3. `spread_miniswe_rows_by_task.py` → `reorder_miniswe_shards_preserve_prefix.py` — shard for training
4. `materialize_qwen3_4b_v75_hf_dataset.py` — pack/upload the processed HF dataset

`src/qwen_agentic_sft` (raw-data normalization + assistant loss policy) is
exercised by `tests/test_qwen_agentic_sft_data.py` at the repo root.
