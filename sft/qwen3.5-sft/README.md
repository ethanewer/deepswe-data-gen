# Qwen3.5/Qwen3.6 Single-GPU SFT Recipes

This workspace sets up NeMo AutoModel for single-GPU text SFT on:

`Qwen/Qwen3.5-4B`
`Qwen/Qwen3.5-9B`
`Qwen/Qwen3.6-27B-FP8`

The runnable script uses GPU 0 only and defaults to LoRA/PEFT, sequence packing, and an 8k packed sequence size.

## Setup

```bash
./scripts/setup_nemo_automodel_env.sh
```

This creates `.venv`, clones NeMo AutoModel under `third_party/Automodel`, installs the base CUDA/VLM stack plus the FLA kernels used by Qwen3.5/Qwen3.6 hybrid attention, and installs a prebuilt FA3 interface for packed sequence training.

Optional optimized attention extras:

```bash
INSTALL_FLASH_ATTN3=1 ./scripts/setup_nemo_automodel_env.sh
```

`INSTALL_FLASH_ATTN3=1` is the default. It uses the prebuilt `kernels-community/vllm-flash-attn3` artifact for CUDA 12.8 instead of compiling `flash-attn` from source.

## Run

Smoke run on the included tiny JSONL files:

```bash
./scripts/run_qwen36_27b_fp8_lora_sft_gpu0.sh
```

Run on your own OpenAI-style chat JSONL:

```bash
TRAIN_DATA=/path/to/train.jsonl \
VAL_DATA=/path/to/validation.jsonl \
PACK_SIZE=8192 \
MAX_STEPS=100 \
./scripts/run_qwen36_27b_fp8_lora_sft_gpu0.sh
```

Each JSONL row should look like:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

## Full FP8 SFT on One H200

Full-parameter SFT of a 27B model is not realistically possible on a single 143 GB H200 with AdamW, even with FP8 training enabled.

Approximate lower bound before activations:

- BF16 trainable weights: `27B * 2 bytes = 54 GB`
- BF16 gradients: `54 GB`
- AdamW FP32 states: `27B * 8 bytes = 216 GB`
- Total before activations, logits, temporary buffers: about `324 GB`

NeMo AutoModel's TorchAO FP8 path accelerates GEMMs, but its own docs say FP8 training has memory usage on par with the BF16 baseline. The FP8 Hugging Face repo is an inference-oriented fine-grained FP8 checkpoint, not a magic way to avoid optimizer state for full fine-tuning.

For actual full FP8 training, use the BF16 checkpoint (`Qwen/Qwen3.6-27B`) with NeMo FP8 enabled and multiple GPUs. I included `configs/qwen36_27b_full_fp8_training_single_gpu_attempt.yaml` plus a guarded launcher for reference, but it is expected to OOM on one GPU.

## Sequence Length

For full-parameter training on one H200: no useful sequence length exists because the optimizer state already exceeds memory.

For the runnable LoRA path, start with:

- `PACK_SIZE=8192`: conservative default.
- `PACK_SIZE=16384`: reasonable next test on an H200.
- `PACK_SIZE=32768`: possible only if the run stays text-only, fused CE works, and activation memory remains controlled.

Packing improves token utilization, but it does not remove activation memory. Increase `PACK_SIZE` only after a short `MAX_STEPS=2` smoke run.

## Qwen3.5-4B BF16 Packed SFT

The 4B recipe mirrors the 9B full-parameter BF16 setup, but the best measured
single-H200 throughput came from shorter packed sequences with a larger local
batch:

```bash
./scripts/run_qwen35_4b_bf16_packed_sft_gpu0.sh
```

Default measured recipe:

- Model: `Qwen/Qwen3.5-4B`
- Packed sequence length: `4096`
- Local/global batch: `32`
- Activation checkpointing: enabled
- Per-layer compile: enabled
- Online tokenization and THD sequence packing from JSONL

Use OpenAI-style chat JSONL via:

```bash
TRAIN_DATA=/path/to/train.jsonl \
VAL_DATA=/path/to/val.jsonl \
MAX_STEPS=100 \
./scripts/run_qwen35_4b_bf16_packed_sft_gpu0.sh
```

Throughput was measured on one local H200 with generated benchmark JSONL,
online tokenization, checkpoint writes disabled, and validation enabled:

| pack | local batch | compile | notes | label TPS |
| ---: | ---: | :---: | --- | ---: |
| 4096 | 32 | on | default, validation passed, 73.9 GiB | 11469 |
| 4096 | 32 | off | stable, 82.2 GiB | 10642 |
| 4096 | 64 | off | stable but memory-heavy, 133.8 GiB | 10007 |
| 8192 | 16 | off | stable, 85.7 GiB | 6718 |
| 16384 | 8 | off | stable, 88.7 GiB | 4774 |
| 32768 | 4 | off | stable, 94.7 GiB | 2544 |
| 65536 | 2 | off | stable, 106.7 GiB | 1424 |
| 131072 | 1 | off | loss/grad became `nan`, 130.7 GiB | 764 |

The Qwen3.5 hybrid-attention stack is much faster with shorter packs. Use
larger `PACK_SIZE` only when a longer packed sequence is more important than raw
tokens/sec. `PACK_SIZE=4096` still uses sequence packing, but it is also the
configured per-sample `SEQ_LENGTH`, so it is the shortest legal packed length
for this recipe.

The following fit tests failed and are intentionally not defaults:

- `PACK_SIZE=4096 LOCAL_BATCH_SIZE=48/64 ENABLE_COMPILE=true`: Inductor/Triton resource failure.
- `ACTIVATION_CHECKPOINTING=false` at `PACK_SIZE=4096` with local batch `16` or `32`: OOM.
- `PACK_SIZE=16384 LOCAL_BATCH_SIZE=8 ACTIVATION_CHECKPOINTING=false`: OOM.

## Qwen3.5-9B BF16 Packed SFT

The current full-parameter path uses BF16 and sequence packing with FA3:

```bash
./scripts/run_qwen35_9b_bf16_packed_sft_gpu0.sh
```

Start conservatively, then scale after short smoke runs:

```bash
PACK_SIZE=4096 MAX_STEPS=2 ./scripts/run_qwen35_9b_bf16_packed_sft_gpu0.sh
PACK_SIZE=16384 MAX_STEPS=2 ./scripts/run_qwen35_9b_bf16_packed_sft_gpu0.sh
PACK_SIZE=32768 MAX_STEPS=2 ./scripts/run_qwen35_9b_bf16_packed_sft_gpu0.sh
```

Use OpenAI-style chat JSONL via:

```bash
TRAIN_DATA=/path/to/train.jsonl \
VAL_DATA=/path/to/val.jsonl \
PACK_SIZE=32768 \
MAX_STEPS=100 \
./scripts/run_qwen35_9b_bf16_packed_sft_gpu0.sh
```

The tested packed path starts from `Qwen/Qwen3.5-9B`, sets `attn_implementation: flash_attention_3`, uses `packed_sequence.packing_strategy: thd`, and keeps all `8,953,803,264` parameters trainable.

Measured compile-enabled default on one H200:

- Packed sequence length: `32768`
- Local/global batch: `2`
- Activation checkpointing: enabled
- Post-warmup label TPS: `2291`
- Peak memory during the measured run: `97.5 GiB`
