# Throughput Recipe

Default 4B target:

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

Default 8B text-tower target uses the reduced-memory compiled path:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PACK_SIZE=65536 \
LOCAL_BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=2 \
ENABLE_COMPILE=true \
ENABLE_FSDP2_PREFETCH=true \
FSDP2_BACKWARD_PREFETCH_DEPTH=2 \
FSDP2_FORWARD_PREFETCH_DEPTH=1 \
QWEN3_VL_TEXT_USE_NATIVE_RMSNORM=1 \
QWEN3_VL_TEXT_DISABLE_MLP_COMPILE=1 \
QWEN3_VL_TEXT_MLP_CHUNK_TOKENS=65536 \
MAX_STEPS=1000 \
./scripts/run_qwen3_8b_thinking_sft_8gpu.sh
```

Token math:

| Pack length | Local batch/GPU | Grad accum | Global packed seqs | Global tokens |
| --- | ---: | ---: | ---: | ---: |
| 131,072 | 1 | 1 | 8 | 1,048,576 |
| 262,144 | 1 | 1 | 8 | 2,097,152 |
| 65,536 | 1 | 2 | 16 | 1,048,576 |
| 65,536 | 4 | 1 | 32 | 2,097,152 |
| 65,536 | 2 | 1 | 16 | 1,048,576 |

Measured with online tokenization and packing:

| Dataset root | Setting | Compile | FSDP2 prefetch | Best TPS | Steady avg TPS | Max GPU memory |
| --- | --- | --- | --- | ---: | ---: | ---: |
| full raw root | 131k/LBS1/GA1 | true | B3/F1 | 49,538.22 | 42,105.02 | 111.40 GiB |
| smoke raw | 131k/LBS1/GA1 | true | B3/F1 | 35,966.50 | 32,205.05 | 111.40 GiB |
| smoke raw | 131k/LBS1/GA1 | true | B2/F1 | 36,037.00 | 32,199.99 | 111.21 GiB |
| smoke raw | 131k/LBS1/GA1 | true | off | 35,958.72 | 31,792.41 | 111.02 GiB |
| smoke raw | 262k/LBS1/GA1 | false | off | 34,358.73 | 31,001.89 | 106.60 GiB |
| smoke raw | 131k/LBS1/GA1 | false | off | 33,989.61 | 30,322.75 | 56.62 GiB |
| smoke raw | 65k/LBS4/GA1 | false | off | 27,351.42 | 26,466.49 | 106.60 GiB |
| smoke raw | 65k/LBS2/GA1 | false | off | 27,865.26 | 25,909.49 | 56.62 GiB |
| smoke raw | 131k/LBS2/GA1 | false | off | 18,960.82 | 17,062.09 | 106.60 GiB |

Measured 8B text tower from `Qwen/Qwen3-VL-8B-Thinking`:

| Dataset root | Setting | Compile | FSDP2 prefetch | Best TPS | Steady avg TPS | Max GPU memory |
| --- | --- | --- | --- | ---: | ---: | ---: |
| full raw root | 65k/LBS1/GA2/native RMS/eager MLP 65k chunk | true | B2/F1 | 44,288.36 | 39,600.60 | 123.00 GiB |
| full raw root | 131k/LBS1/GA1 | false | B2/F1 | 35,485.42 | 31,250.79 | 95.94 GiB |
| full raw root | 72k/LBS1/GA2/native RMS/eager MLP 16k chunk | true | off | 38,571.90 | 35,255.60 | 129.93 GiB |
| smoke raw | 131k/LBS1/GA1 | false | B2/F1 | 27,434.04 | 26,715.72 | 95.94 GiB |
| smoke raw | 131k/LBS1/GA1 | false | B1/F1 | 27,456.32 | 26,665.96 | 95.94 GiB |
| smoke raw | 131k/LBS1/GA1 | false | off | 27,358.31 | 26,616.86 | 83.67 GiB |
| smoke raw | 65k/LBS1/GA2 | false | B3/F1 | 21,574.23 | 21,574.23 | 83.13 GiB |

Selection rule:

- Keep 131k/LBS1/GA1 with compile enabled and FSDP2 prefetch B3/F1 as the
  default 4B throughput recipe.
- Keep 65k/LBS1/GA2 with compile enabled, FSDP2 prefetch B2/F1, native RMSNorm,
  and eager MLP with a 65,536-token chunk cap as the default 8B text-tower
  throughput recipe.
- Use 262k/LBS1/GA1 without compile for 4B only if the longer pack is worth the lower
  measured peak and higher memory.
- Do not use compiled 262k for 4B on this node; it OOMs in the compiled MLP allocation.
- Do not use compiled 131k for 8B on this node; it OOMs even with the reduced-memory
  text path. 80k/LBS1/GA2 also OOMs during backward recompute.
- Full MLP compile at 65k measured 39,661.49 steady TPS, but it produced an
  abnormal first-step grad norm and worse losses on the same deterministic packs,
  so the stable eager-MLP recipe is selected.
- Do not disable activation checkpointing at 131k; it OOMs before the first
  step. Do not use plain SDPA at 131k; it OOMs in attention. FlashAttention 2
  is not installed in this environment.

For throughput measurement, ignore the first logged step and compare total
tokens per second across all eight GPUs.
