# Local Model Serving Handoff

This documents the verified local model setup for SWE-RBench synthetic data
generation on the 8x H200 node. Reusable scripts live under
`scripts/local_model_serving/`. Large model snapshots, caches, logs, and the
serving venv live outside the repo under `/scratch/local_model_serving`.

## Runtime Root

Default runtime root:

```bash
/scratch/local_model_serving
```

The scripts set Hugging Face, PyTorch, Triton, FlashInfer, TVM, CUDA, temp, and
Python bytecode cache paths under this root. Keeping these caches off `/home`
avoids slow NFS-backed kernel compilation during SGLang startup.

Slurm compute nodes should use the shared runtime root instead:

```bash
/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving
```

Run:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/status.sh
```

## Downloaded Checkpoints

Kimi K2.6:

- Hub repo: `moonshotai/Kimi-K2.6`
- Quantization: official Moonshot native INT4/QAT checkpoint
- Snapshot revision: `7eb5002f6aadc958aed6a9177b7ed26bb94011bb`
- Local path: `/scratch/local_model_serving/models/moonshotai_Kimi-K2.6.snapshot`

MiMo V2.5:

- Hub repo: `XiaomiMiMo/MiMo-V2.5`
- Quantization: official FP8 checkpoint
- Snapshot revision: `2fd4f899a491de2fb0beeafe32b5d700b251f593`
- Local path: `/scratch/local_model_serving/models/XiaomiMiMo_MiMo-V2.5.snapshot`

Qwen3.6-27B:

- Hub repo: `Qwen/Qwen3.6-27B`
- Quantization: official BF16 checkpoint
- Snapshot revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- Local path: `/scratch/local_model_serving/models/Qwen_Qwen3.6-27B.snapshot`

Resume or repair downloads with:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/download_models.sh
```

## Python Environment

Verified serving env:

```text
/scratch/local_model_serving/venvs/venv-sglang
```

Verified package pins:

- `sglang==0.5.12.post1`
- `torch==2.9.1+cu128`
- `torchvision==0.24.1+cu128`
- `torchaudio==2.9.1+cu128`
- `sgl-kernel==0.3.21`
- `flashinfer-python==0.6.11.post1`
- `transformers==5.6.0`
- `kernels==0.14.1`
- `compressed-tensors==0.17.1a20260604`

Install or repair the env with:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/install_serving_envs.sh
```

SGLang notes:

- `sglang-kernel==0.4.2.post2` pulls CUDA 13 runtime libraries and does not run
  on this node's CUDA 12.8/driver 570 stack.
- `sgl-kernel==0.3.21` works with `torch==2.9.1+cu128`, but SGLang's kernel
  version check must be skipped. `env.sh` exports
  `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`.
- `deep_gemm` imports a CUDA 13 linked dependency in this environment, so
  DeepGEMM JIT/precompile is disabled by default.

## Serving

Run one 8-GPU model server at a time unless GPUs are intentionally partitioned.

Kimi K2.6 via SGLang:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/serve_kimi_sglang.sh
```

Default OpenAI-compatible base URL: `http://localhost:18000/v1`

Kimi uses TP8, `compressed-tensors`, Kimi reasoning/tool parsers, 128K context,
FlashInfer attention, and disables FlashInfer all-reduce fusion/autotune.

MiMo V2.5 via SGLang:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/serve_mimo_sglang.sh
```

Default OpenAI-compatible base URL: `http://localhost:19001/v1`

MiMo uses TP8/DP2, DP attention, FP8, 256K context, DP LM head, and Triton
attention. FlashInfer attention failed for MiMo because the local backend does
not accept MiMo's attention `sinks` argument; FA3 in this SGLang build requires
newer CUDA-13-linked kernels on this node.

MiMo V2.5 on Slurm:

```bash
sbatch scripts/local_model_serving/serve_mimo_slurm.sbatch
```

The Slurm script defaults to 4x RTX PRO 6000 with TP4/DP1 and a 32K context.
For an 8x L40S node, override the Slurm resources and SGLang layout:

```bash
sbatch \
  --partition=l40s-8gpu \
  --gres=gpu:l40s:8 \
  --cpus-per-task=96 \
  --mem=1200G \
  --job-name=serve-mimo-v2.5-l40s-32k \
  --output=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/logs/slurm-mimo-l40s-32k-%j.out \
  --error=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/logs/slurm-mimo-l40s-32k-%j.err \
  --export=ALL,TP_SIZE=8,DP_SIZE=2,ENABLE_DP_ATTENTION=1,MEM_FRACTION_STATIC=0.94,CONTEXT_LENGTH=32768,MAX_RUNNING_REQUESTS=8,CHUNKED_PREFILL_SIZE=4096,PORT=19001 \
  scripts/local_model_serving/serve_mimo_slurm.sbatch
```

The Slurm launcher uses the shared model and venv under
`/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving`, while keeping
mutable caches on node-local `/scratch/${USER}/local_model_serving`.
The verified 8x L40S setup has a KV pool of about 47.8K total tokens with
about 2.37 GB GPU memory left after CUDA graph capture, so 32K context fits.

Qwen3.6-27B via SGLang:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/serve_qwen36_sglang.sh
```

Default OpenAI-compatible base URL: `http://localhost:20000/v1`

Qwen3.6-27B uses TP8/DP1, 131,072 context, Qwen3 reasoning, Qwen3 Coder
tool parsing, FlashInfer full attention, Triton linear attention, and an 800K
token pool for multiple long sequences. DP attention is off by default:
TP8/DP2 loaded, but the first generation crashed in FlashInfer paged prefill on
the hybrid-attention path.

Qwen3.6-27B on Slurm:

```bash
sbatch scripts/local_model_serving/serve_qwen36_slurm.sbatch
```

The Slurm script defaults to 8x L40S with TP8/DP1. For more memory per GPU,
override the Slurm resources to an 8x RTX PRO 6000 node:

```bash
sbatch \
  --partition=rtx6000pro-8gpu \
  --gres=gpu:rtxproserver6000:8 \
  --cpus-per-task=96 \
  --mem=1200G \
  scripts/local_model_serving/serve_qwen36_slurm.sbatch
```

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

Qwen3.6:

```bash
cd /wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen
python -m datagen.swerebench_v2.run_all \
  --model qwen3.6-27b \
  --litellm-model openai/qwen3.6-27b \
  --api-base http://127.0.0.1:20000/v1 \
  --no-require-api-key \
  --max-tokens 16384 \
  --extra-body-json '{"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}' \
  --limit 1 \
  --disable-verification
```

## Throughput Results

Benchmark shape: OpenAI `/v1/chat/completions`, 64 concurrent clients, 128 total
requests, `max_tokens=1024`, real SWE-RBench task instructions from this repo,
and a bash tool schema matching SWE-agent style data generation.

Kimi K2.6:

```text
success=128/128
output_tps=629.2214841355657
completion_tokens=5000
prompt_tokens=107420
elapsed_s=7.946327527053654
tool_call_responses=128
finish_reasons={"tool_calls": 128}
latency_s={"mean": 3.7851073426827497, "p50": 3.843026545830071, "p95": 6.093388921115547, "max": 7.567593679297715}
```

MiMo V2.5:

```text
success=128/128
output_tps=600.5453881237946
completion_tokens=10876
prompt_tokens=138123
elapsed_s=18.110204848926514
tool_call_responses=127
finish_reasons={"length": 1, "tool_calls": 127}
latency_s={"mean": 4.645947013108525, "p50": 4.367954423185438, "p95": 8.722209536936134, "max": 18.085793481674045}
```

Qwen3.6-27B:

```text
success=128/128
output_tps=37.35589745655711
completion_tokens=6830
prompt_tokens=166076
elapsed_s=182.83592324191704
tool_call_responses=128
finish_reasons={"tool_calls": 128}
latency_s={"mean": 77.64255335546841, "p50": 85.20868428656831, "p95": 106.3105297437869, "max": 115.62520491797477}
```

## Last Verification

- Kimi `/v1/models` served `kimi-k2.6` with `max_model_len=128000`.
- Kimi smoke chat returned `local kimi ok`.
- Kimi high-concurrency benchmark completed with zero errors.
- MiMo `/v1/models` served `mimo-v2.5` with `max_model_len=262144`.
- MiMo smoke chat returned `local mimo ok` with reasoning content.
- MiMo high-concurrency benchmark completed with zero errors.
- Slurm MiMo on 8x L40S served `mimo-v2.5` with `max_model_len=32768`.
- Slurm MiMo smoke chat returned `local slurm mimo 32k ok`.
- Slurm MiMo accepted a 12,036-token prompt and returned `long context ok`.
- Slurm Qwen3.6-27B on 8x L40S served `qwen3.6-27b` with
  `max_model_len=131072`.
- Slurm Qwen3.6-27B TP8/DP1 allocated `max_total_num_tokens=800000` with
  about 8.57 GB GPU memory left after CUDA graph capture.
- Slurm Qwen3.6-27B smoke chat returned `local slurm qwen ok`.
- Slurm Qwen3.6-27B high-concurrency benchmark completed with zero errors.
- Slurm Qwen3.6-27B accepted four parallel 130,804-token prompts and all four
  returned `long context ok` in 178.83 s.
- `bash -n` passed for the local model serving scripts after the launcher
  updates.

Kimi vLLM was not the working path. It loaded far enough to expose a WNA16
MoE/Marlin group-size kernel incompatibility for the official Kimi INT4/QAT
checkpoint. Use the SGLang launcher above for local Kimi serving.
