# Local Model Serving Handoff

This documents the local model setup for SWE-RBench synthetic data generation. The reusable code lives in this repo under `scripts/local_model_serving/`. Large model snapshots, caches, logs, and virtual environments live outside the repo under `/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving`.

## Runtime Root

Default runtime root:

```bash
/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving
```

The scripts set these paths so downloads and caches stay under `/wbl-fast` and do not fall back to `/home`:

- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `HF_XET_CACHE`
- `TRANSFORMERS_CACHE`
- `XDG_CACHE_HOME`
- `PIP_CACHE_DIR`
- `TORCH_HOME`
- `TRITON_CACHE_DIR`
- `TMPDIR`

Run:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/status.sh
```

## Downloaded Checkpoints

Kimi K2.6:

- Hub repo: `moonshotai/Kimi-K2.6`
- Quantization: official Moonshot native INT4/QAT checkpoint
- Snapshot revision: `7eb5002f6aadc958aed6a9177b7ed26bb94011bb`
- Stable local path: `/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/models/moonshotai_Kimi-K2.6.snapshot`
- Snapshot target: `/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/hf_cache/models--moonshotai--Kimi-K2.6/snapshots/7eb5002f6aadc958aed6a9177b7ed26bb94011bb`

MiMo V2.5:

- Hub repo: `XiaomiMiMo/MiMo-V2.5`
- Quantization: official FP8 checkpoint, recommended for 8x H200 serving
- Snapshot revision: `2fd4f899a491de2fb0beeafe32b5d700b251f593`
- Stable local path: `/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/models/XiaomiMiMo_MiMo-V2.5.snapshot`
- Snapshot target: `/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/hf_cache/models--XiaomiMiMo--MiMo-V2.5/snapshots/2fd4f899a491de2fb0beeafe32b5d700b251f593`

Resume or repair downloads with:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/download_models.sh
```

The downloader uses `huggingface_hub.snapshot_download` and refreshes stable symlinks only after a snapshot completes.

## Python Environments

The runtime uses separate venvs to avoid vLLM/SGLang dependency conflicts:

- Download helper env: `/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/venv`
- Kimi/vLLM env: `/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/venv-vllm`
- MiMo/SGLang env: `/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/venv-sglang`

Install or repair them with:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/install_serving_envs.sh
```

Verified versions after setup:

- vLLM env: `vllm==0.19.1`, `torch==2.10.0+cu128`, `transformers==4.57.6`
- SGLang env: `sglang==0.5.12.post1`, `torch==2.11.0+cu128`, `transformers==5.6.0`, `kernels==0.14.1`

Important SGLang notes:

- The default SGLang package pulled CUDA 13 packages, which cannot run on this node's CUDA 12.8 driver. The installer force-reinstalls CUDA 12.8 PyTorch wheels.
- `kernels>=0.15` breaks this SGLang/Transformers combo with `Either a revision or a version must be specified`, so the installer pins `kernels<0.15`.
- `deep_ep` failed to build without a working NVSHMEM stack. The MiMo launcher defaults to no DeepEP; set `ENABLE_DEEPEP=1` only after NVSHMEM/DeepEP is repaired.

## Serving

Run one 8-GPU model server at a time unless GPUs are intentionally partitioned.

Kimi K2.6 via vLLM:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/serve_kimi_vllm.sh
```

Default OpenAI-compatible base URL: `http://localhost:18000/v1`

MiMo V2.5 via SGLang:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/serve_mimo_sglang.sh
```

Default OpenAI-compatible base URL: `http://localhost:19001/v1`

## Datagen Smoke Commands

Kimi:

```bash
cd /wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen
python -m datagen.swerebench_v2.run_all \
  --model kimi-k2.6 \
  --litellm-model openai/kimi-k2.6 \
  --api-base http://127.0.0.1:18000/v1 \
  --no-require-api-key \
  --max-tokens 16384 \
  --extra-body-json '{"chat_template_kwargs":{"thinking":false}}' \
  --limit 1 \
  --disable-verification
```

MiMo:

```bash
cd /wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen
python -m datagen.swerebench_v2.run_all \
  --model mimo-v2.5 \
  --litellm-model openai/mimo-v2.5 \
  --api-base http://127.0.0.1:19001/v1 \
  --no-require-api-key \
  --max-tokens 16384 \
  --extra-body-json '{"chat_template_kwargs":{"thinking":true}}' \
  --limit 1 \
  --disable-verification
```

## Throughput Notes

- For real throughput measurements, wait until all 8 H200s are free.
- Kimi defaults: TP8, 256K context, prefix caching, Kimi reasoning/tool parsers, data-parallel multimodal encoder.
- Kimi startup with `GPU_MEMORY_UTILIZATION=0.92` requires roughly 129 GiB free per GPU at initialization. If other processes occupy GPU memory, vLLM will fail before loading.
- MiMo defaults: TP8/DP2, FP8, 256K context, DP attention, multi-threaded loading, FA3 attention, and multi-layer EAGLE speculative decoding.
- Start synthetic data clients around 64-128 concurrent requests and tune from generated-token throughput, queueing, and TTFT.
- For shared-GPU smoke tests only, lower Kimi cache reservation, for example `GPU_MEMORY_UTILIZATION=0.70 scripts/local_model_serving/serve_kimi_vllm.sh`. Do not use reduced reservation for final throughput runs.

## Last Verification

The setup scripts passed shell syntax checks. The runtime imports were verified:

```text
vllm_check 2.10.0+cu128 True 8 4.57.6 0.19.1
sglang_check 2.11.0+cu128 True 8 5.6.0 0.14.1 0.5.12.post1
```

Kimi server launch was attempted while another user's retrieval servers occupied about 34 GiB on every H200, so vLLM refused startup because only about 105 GiB/GPU was free. No local model server was left running after the attempt.
