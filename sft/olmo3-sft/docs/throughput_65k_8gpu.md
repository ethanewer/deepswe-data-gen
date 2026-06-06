# OLMo-3 7B Think 65k SFT Throughput

Benchmarks were run on 8 GPUs with packed fixed-sequence-length training, BF16, FlashAttention2, full activation checkpointing, model compile enabled, and gradient clipping left on. Throughput below is `throughput/device/TPS`, averaged after the first two reported throughput points unless noted.

| Recipe | Global tokens | TPS/GPU | Total TPS | Max reserved/GPU | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| FSDP, LBS=4, GBS=64, GA=2, fused AdamW | 4,194,304 | 7,690.03 | 61,520.26 | 126.16 GiB | Best measured default |
| FSDP, LBS=4, GBS=64, GA=2, fused AdamW, optimizer compile | 4,194,304 | 7,688.08 | 61,504.62 | 126.16 GiB | No material gain |
| FSDP, LBS=4, GBS=32, GA=1, fused AdamW | 2,097,152 | 7,661.43 | 61,291.45 | 125.96 GiB | Best strict sub-4M-token GBS |
| FSDP, LBS=4, GBS=32, GA=1, SkipStepAdamW | 2,097,152 | 7,653.47 | 61,227.80 | 125.96 GiB | Slightly slower |
| DDP, LBS=2, GBS=64, GA=4 | 4,194,304 | 7,509.45 | 60,075.62 | 123.59 GiB | Slower despite lower memory |
| HSDP, LBS=4, GBS=32, GA=1 | 2,097,152 | 7,023.33 | 56,186.62 | 136.86 GiB | Slower |

Recommended default:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SEQUENCE_LENGTH=65536 \
LOCAL_BATCH_SEQS=4 \
GRAD_ACCUM_STEPS=2 \
DATA_PARALLEL=fsdp \
OPTIM=adamw \
ADAMW_FUSED=true \
COMPILE_OPTIM=false \
ACTIVATION_CHECKPOINTING=full \
COMPILE_MODEL=true \
ATTN_BACKEND=flash_2 \
olmo3-sft/scripts/run_olmo3_7b_think_sft_ddp.sh
```

Notes:

- The earlier `compileoptim` result file was produced before `--compile-optim` was wired through to `AdamWConfig`, so it is effectively the standard fused-AdamW run. The later `realcompileoptim` run showed no meaningful throughput improvement and keeps extra startup overhead, so optimizer compile is off by default.
- LBS=5 did not complete reliably. FSDP saved enough memory to run the faster LBS=4 recipe cleanly at 65k.
- Context parallelism is not part of the default because the current packed intra-document masking path is not compatible with the CP experiments we tried.
