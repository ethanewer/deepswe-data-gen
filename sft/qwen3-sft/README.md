# Qwen3 Thinking Agentic SFT

This recipe runs full-parameter SFT for:

- `Qwen/Qwen3-4B-Thinking-2507`
- the text tower from `Qwen/Qwen3-VL-8B-Thinking`

It is built for the local 8 x H200 node and refuses to launch unless exactly
eight visible GPUs are selected.

## What This Uses

- NeMo AutoModel full-parameter BF16 training with FSDP2.
- FlashAttention 3, fused linear cross entropy, model-specific compile
  defaults, and FSDP2 prefetch.
- Online raw-data normalization, Qwen chat-template rendering, tokenization, and THD sequence packing.
- The required chat template:
  `/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2`
- Raw datasets under:
  `/wbl-fast/usrs/ee/code-swe-data/data/code-swe-terminal-agentic-sft`

The online dataset yields fixed-length packed samples with `input_ids`,
`labels`, `position_ids`, `seq_lens`, and `seq_lens_padded`. The NeMo dataloader
uses `packed_sequence_thd_collater` to keep document boundaries for packed
attention. Do not enable NeMo's offline `packed_sequence` wrapper for this
recipe; that path materializes packs up front.

## Setup

```bash
cd /wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3-sft
./scripts/setup_nemo_automodel_env.sh
```

This creates `.venv`, clones NeMo AutoModel into `third_party/Automodel`,
applies the local Qwen3-VL text-tower patch, and installs CUDA 12.8 Torch,
FlashAttention 3, Qwen kernel dependencies, pyarrow, and the local recipe
package.

## Build Compact Smoke Data

```bash
./scripts/build_smoke_raw_dataset.sh
```

The smoke builder mirrors a few rows from every top-level dataset without
loading full datasets. JSONL files are read from the start only. Parquet files
use small `pyarrow` batches.

Validate template rendering, tokenization, loss masks, and fixed-length packing:

```bash
./scripts/inspect_online_packer.sh
```

## Run The 4B Throughput Recipe

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PACK_SIZE=131072 \
LOCAL_BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=1 \
ENABLE_COMPILE=true \
ENABLE_FSDP2_PREFETCH=true \
FSDP2_BACKWARD_PREFETCH_DEPTH=3 \
FSDP2_FORWARD_PREFETCH_DEPTH=1 \
MAX_STEPS=1000 \
./scripts/run_qwen3_4b_thinking_sft_8gpu.sh
```

Default global token batch:

`8 packed sequences * 131,072 tokens = 1,048,576 tokens/update`

That is inside the requested 1M-5M token range.

## Run The 4B L40S Slurm Recipe

The L40S recipe keeps the newer 8B text-tower data path but uses
`Qwen/Qwen3-4B-Thinking-2507`: 32,768-token packs, local batch 1, gradient
accumulation 4, online tokenization/packing, FSDP2, activation checkpointing,
compile disabled, FSDP2 prefetch disabled, and the shared Qwen3 thinking chat
template. This keeps the global token batch at 1,048,576 tokens/update while
remaining stable on 46 GB L40S GPUs.

Submit on an 8-GPU L40S node:

```bash
sbatch scripts/slurm_qwen3_4b_thinking_l40s_sft_8gpu.sbatch
```

Smoke run on the local raw smoke data:

```bash
TRAIN_RAW_ROOT=$PWD/data/smoke_raw \
MAX_STEPS=25 \
RUN_NAME=qwen3_4b_thinking_l40s_32k_smoke25 \
sbatch scripts/slurm_qwen3_4b_thinking_l40s_sft_8gpu.sbatch
```

Use any dataset in the existing local raw format by overriding
`TRAIN_RAW_ROOT=/path/to/raw/root`. The launcher defaults to
`PACK_SIZE=32768`, rejects values below 32,768, and uses the chat template at
`/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/qwen3_thinking_acc.jinja2`.

Measured unstable L40S variants during smoke testing:

- `PACK_SIZE=65536`, local batch 1: OOM with compile on, compile off, and prefetch off.
- `PACK_SIZE=49152`, local batch 1, compile on: OOM.
- `PACK_SIZE=32768`, local batch 2, compile off: OOM.
- `PACK_SIZE=32768`, local batch 1, compile on: OOM.

`PACK_SIZE=49152`, local batch 1, gradient accumulation 3, compile off, and
prefetch off is a higher-context override, but it is slower and closer to the
memory limit than the default 32k recipe.

For a quick smoke training run:

```bash
TRAIN_RAW_ROOT=$PWD/data/smoke_raw \
VAL_RAW_ROOT=$PWD/data/smoke_raw \
MAX_STEPS=3 \
./scripts/run_qwen3_4b_thinking_sft_8gpu.sh
```

## Run The Qwen3-VL 2B Text L40S Recipe

`Qwen/Qwen3-VL-2B-Thinking` is trained through a text-only safetensors view, so
the vision tensors are not loaded by the training model. The measured L40S
default is `PACK_SIZE=40960`, `LOCAL_BATCH_SIZE=4`, `GRAD_ACCUM_STEPS=1`,
`ENABLE_COMPILE=false`, and `ENABLE_FSDP2_PREFETCH=false`, for 1,310,720
tokens/update. It uses the shared Qwen3 thinking ACC chat template:

```bash
sbatch scripts/slurm_qwen3_vl_2b_text_l40s_sft_8gpu.sbatch
```

Measured L40S smoke results on 8 GPUs:

| Shape | Status | Steps | Avg TPS, excluding step 0 | Max Mem/GPU | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| 32k, local batch 2, GA 2, no compile | completed | 25 | 32,160 | 24.91 GiB | Conservative baseline. |
| 32k, local batch 4, GA 1, no compile | partial | 16 | 33,087 | 31.50 GiB | Stable, but only modestly faster than the baseline. |
| 40k, local batch 4, GA 1, no compile | completed | 25 | 28,968 | 36.89 GiB | Recommended long-context default. |
| 40k, local batch 4, GA 1, no compile, FSDP2 prefetch | partial | 7 | 29,314 | 36.98 GiB | No clear gain over prefetch off. |
| 49k, local batch 4, GA 1, no compile | completed | 25 | 26,038 | 43.42 GiB | Works, but leaves little L40S memory headroom. |
| 49k, local batch 2, GA 2, no compile | partial | 9 | 26,980 | 30.86 GiB | Safer memory, slower than 40k-lb4. |

`Qwen/Qwen3.5-2B` was also tested with the existing
`/wbl-fast/usrs/ee/code-swe-data/sft/qwen3.5-sft` pattern
(`NeMoAutoModelForCausalLM`, FA3, BF16, FSDP2, activation checkpointing, and the
tokenizer's original chat template). It did not fit the 32k minimum on 8 x L40S
for full-parameter SFT: local batch 4 with compile, local batch 2 with no
compile, and local batch 1 with no compile all OOMed before completing step 0.

## Run The 8B Text-Tower Recipe

The first run builds `data/qwen3_vl_8b_text_checkpoint`, a filtered local view
of `Qwen/Qwen3-VL-8B-Thinking` that contains config/tokenizer files and only
the language-model safetensors keys.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PACK_SIZE=65536 \
LOCAL_BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=2 \
MAX_STEPS=1000 \
./scripts/run_qwen3_8b_thinking_sft_8gpu.sh
```

The 8B wrapper defaults to `ENABLE_COMPILE=true`, 65k packs, gradient
accumulation 2, and FSDP2 prefetch `B2/F1`.  It also enables the measured
lower-memory text path for Qwen3-VL text: native RMSNorm and eager MLP with a
65,536-token MLP chunk cap.  Compiled 131k packs still OOM for the 8B text tower
on the measured 8 x H200 node.

## Throughput Sweep

```bash
TRAIN_RAW_ROOT=$PWD/data/smoke_raw MAX_STEPS=24 MODEL_SIZE=4b ./scripts/sweep_throughput.sh
TRAIN_RAW_ROOT=$PWD/data/smoke_raw MAX_STEPS=24 MODEL_SIZE=8b ./scripts/sweep_throughput.sh
```

The sweep tries:

- `PACK_SIZE=131072`, `LOCAL_BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=1`, `ENABLE_COMPILE=true`, `ENABLE_FSDP2_PREFETCH=true`
- `PACK_SIZE=131072`, `LOCAL_BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=1`, `ENABLE_COMPILE=true`, `ENABLE_FSDP2_PREFETCH=false`
- `PACK_SIZE=131072`, `LOCAL_BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=1`, `ENABLE_COMPILE=false`
- `PACK_SIZE=262144`, `LOCAL_BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=1`, `ENABLE_COMPILE=false`
- `PACK_SIZE=65536`, `LOCAL_BATCH_SIZE=4`, `GRAD_ACCUM_STEPS=1`, `ENABLE_COMPILE=false`
- `PACK_SIZE=65536`, `LOCAL_BATCH_SIZE=2`, `GRAD_ACCUM_STEPS=1`, `ENABLE_COMPILE=false`

Compiled 262k packs OOM on this 8 x H200 node for 4B, so the 4B default keeps
131k packs with compile enabled. For 8B, compiled 131k packs still OOM, but the
reduced-memory 65k compiled recipe is faster than the older 131k non-compiled
baseline.

## Runtime Overrides

Common overrides:

```bash
TRAIN_RAW_ROOT=/path/to/raw/root
VAL_RAW_ROOT=/path/to/smoke/root
PACK_SIZE=131072
LOCAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=1
ENABLE_COMPILE=true
ENABLE_FSDP2_PREFETCH=true
FSDP2_BACKWARD_PREFETCH_DEPTH=3
FSDP2_FORWARD_PREFETCH_DEPTH=1
MAX_STEPS=1000
NUM_WORKERS=2
CHECKPOINT_ENABLED=true
CHECKPOINT_DIR=checkpoints/qwen3_4b_thinking_online_packed_sft/
VALIDATION_ENABLED=false
```

For `MODEL_SIZE=8b` or `run_qwen3_8b_thinking_sft_8gpu.sh`, the measured
defaults are `PACK_SIZE=65536`, `GRAD_ACCUM_STEPS=2`, `ENABLE_COMPILE=true`,
`FSDP2_BACKWARD_PREFETCH_DEPTH=2`, `FSDP2_FORWARD_PREFETCH_DEPTH=1`,
`QWEN3_VL_TEXT_USE_NATIVE_RMSNORM=1`,
`QWEN3_VL_TEXT_DISABLE_MLP_COMPILE=1`, and
`QWEN3_VL_TEXT_MLP_CHUNK_TOKENS=65536`.

The launcher computes and validates the global token batch:

`GLOBAL_BATCH_SIZE * PACK_SIZE`

It exits unless that value is between 1M and 5M tokens.

## Dataset Compatibility

The normalizer supports the local dataset variants observed in
`code-swe-terminal-agentic-sft`:

- OpenAI-style `messages`
- stringified `messages`
- `conversations`, `conversation`, or `trajectory`
- prompt/response rows
- JSONL event-log sessions
- parquet and JSONL inputs
- assistant reasoning and tool-calling fields

Overlength examples are split by default so very long trajectories do not crash
the packer. Chunks with no assistant loss tokens are skipped.
