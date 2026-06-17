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

Kimi K2.7-Code:

- Hub repo: `moonshotai/Kimi-K2.7-Code`
- Quantization: official Moonshot compressed-tensors/native INT4 checkpoint
- Snapshot revision: `74797c9c62378b951a1f6fcf5c4631024e9b8bef`
- Shared local path:
  `/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving/models/moonshotai_Kimi-K2.7-Code.snapshot`

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

Qwen3.6-35B-A3B-FP8:

- Hub repo: `Qwen/Qwen3.6-35B-A3B-FP8`
- Quantization: official fine-grained FP8 checkpoint, 128x128 block size
- Snapshot revision: `95a723d08a9490559dae23d0cff1d9466213d989`
- Local path: `/scratch/local_model_serving/models/Qwen_Qwen3.6-35B-A3B-FP8.snapshot`

Resume or repair downloads with:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/download_models.sh
```

## Official Guide Notes

- The Qwen3.6-35B-A3B-FP8 model card documents 128-block FP8 weights and
  lists SGLang and vLLM as supported serving stacks:
  <https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8>
- Qwen's recommended SGLang command uses TP8, 262K context,
  `--reasoning-parser qwen3`, and `--tool-call-parser qwen3_coder`. On L40S,
  TP8 is not usable with this FP8 checkpoint in the installed SGLang because
  one shared-expert projection is partitioned to width 64, below the 128-wide
  FP8 block size.
- vLLM's Qwen3.5/Qwen3.6 recipe recommends `--language-model-only` for maximum
  text throughput under high concurrency because it skips the vision encoder
  and frees memory for KV cache:
  <https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html>

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

Kimi K2.7-Code via vLLM:

```bash
LOCAL_MODEL_SERVING_ROOT=/wbl-fast/usrs/ee/code-swe-data/runtime/local_model_serving \
  /wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/serve_kimi27_code_vllm.sh
```

Default OpenAI-compatible base URL: `http://localhost:18010/v1`

Prepared H200 Slurm launcher:

```bash
sbatch /wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/serve_kimi27_code_h200_slurm.sbatch
```

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

Qwen3.6-35B-A3B-FP8 via SGLang:

```bash
/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/scripts/local_model_serving/serve_qwen36_moe_fp8_sglang.sh
```

Default OpenAI-compatible base URL: `http://localhost:20010/v1`

The verified 8x L40S config uses TP4/DP2, 131,072 context, Qwen3 reasoning,
Qwen3 Coder tool parsing, FlashInfer full attention, Triton linear attention,
and the Triton MoE runner. The server allocates 600K cache tokens per DP replica,
or about 1.2M aggregate scheduled tokens across the two replicas.

Qwen3.6-35B-A3B-FP8 on Slurm:

```bash
sbatch scripts/local_model_serving/serve_qwen36_moe_fp8_slurm.sbatch
```

The Slurm script defaults to 8x L40S with TP4/DP2 and port `20010`.

Saved but not yet verified: vLLM text-only launcher with the vision encoder
disabled:

```bash
sbatch scripts/local_model_serving/serve_qwen36_moe_fp8_vllm_slurm.sbatch
```

The vLLM script uses `--language-model-only`, `--reasoning-parser qwen3`,
`--tool-call-parser qwen3_coder`, prefix caching, TP8, and port `20011`.
Two Slurm attempts on idle L40S nodes stayed in `CONFIGUR` without starting the
batch script, so the vLLM path was not benchmarked.

L40S SGLang backend notes for Qwen3.6-35B-A3B-FP8:

- `TP_SIZE=8` fails during model construction because the official FP8 block
  size is 128 and one shared-expert partition becomes width 64.
- `MOE_RUNNER_BACKEND=flashinfer_trtllm` loads weights but fails during CUDA
  graph capture on L40S by trying to build an unsupported SM100/SM120 path.
- `MOE_RUNNER_BACKEND=cutlass` asserts that FP8 MoE requires SM90/SM100/SM120;
  L40S is SM89.
- `MOE_RUNNER_BACKEND=triton` is the verified L40S backend.

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

Qwen3.6 MoE FP8:

```bash
cd /wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen
python -m datagen.swerebench_v2.run_all \
  --model qwen3.6-35b-a3b-fp8 \
  --litellm-model openai/qwen3.6-35b-a3b-fp8 \
  --api-base http://l40s-8gpu-dy-l40s-8gpu-cr-0-2.integrated.pcluster:20010/v1 \
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

Qwen3.6-35B-A3B-FP8, SGLang TP4/DP2 on 8x L40S:

```text
success=128/128
output_tps=268.7941482152726
completion_tokens=5789
prompt_tokens=201146
elapsed_s=21.536927192937583
tool_call_responses=128
finish_reasons={"tool_calls": 128}
latency_s={"mean": 8.652216484995733, "p50": 9.171103690285236, "p95": 17.412013965751974, "max": 20.552074189763516}
```

The first post-start benchmark pass was also clean but included more warmup/JIT
overhead:

```text
success=128/128
output_tps=214.87024914585265
completion_tokens=5780
prompt_tokens=201146
elapsed_s=26.899954847060144
tool_call_responses=128
finish_reasons={"tool_calls": 128}
```

Long-context Qwen3.6-35B-A3B-FP8 check:

```text
parallel_requests=8
prompt_tokens_each=130800
aggregate_prompt_tokens=1046400
success=8/8
elapsed_s=48.84757383586839
responses=["long context ok", "long context ok", "long context ok", "long context ok", "long context ok", "long context ok", "long context ok", "long context ok"]
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
- Slurm Qwen3.6-35B-A3B-FP8 SGLang on 8x L40S served
  `qwen3.6-35b-a3b-fp8` with `max_model_len=131072`.
- Slurm Qwen3.6-35B-A3B-FP8 TP4/DP2 allocated
  `max_total_num_tokens=600000` per DP replica with about 14.01 GB GPU memory
  left after CUDA graph capture.
- Slurm Qwen3.6-35B-A3B-FP8 smoke chat returned
  `local slurm qwen moe ok`.
- Slurm Qwen3.6-35B-A3B-FP8 emitted valid `bash` tool calls.
- Slurm Qwen3.6-35B-A3B-FP8 high-concurrency benchmark completed with zero
  errors at 268.79 output TPS on the warmed pass.
- Slurm Qwen3.6-35B-A3B-FP8 accepted eight parallel 130,800-token prompts and
  all eight returned `long context ok` in 48.85 s.
- `bash -n` passed for the local model serving scripts after the launcher
  updates.

Kimi vLLM was not the working path. It loaded far enough to expose a WNA16
MoE/Marlin group-size kernel incompatibility for the official Kimi INT4/QAT
checkpoint. Use the SGLang launcher above for local Kimi serving.
