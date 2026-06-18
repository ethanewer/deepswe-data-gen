# Qwen3-VL-2B Text SFT SWE-bench Tracking

Run root: `/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/sft/qwen3/checkpoints/qwen3_vl_2b_text_h200_4gpu_40k_hq3x_reasoning90_lr5e6_s400_418791`

Training setup:
- Base model: `Qwen/Qwen3-VL-2B-Thinking`
- Training view: text-only checkpoint prepared from the base model
- Sequence length: 40,960
- Hardware: 4 H200 GPUs on `h200-st-h200-cr-0-16`
- Max steps: 400
- Checkpoint interval: 100 optimizer steps
- Dataset: `/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260611/highquality-3x-duplicate-reasoning-90pct`
- Loss mask requirements: train only assistant turns with reasoning and valid tool calls; invalid assistant turns remain in history but are not labeled.

## SWE-bench Multilingual Results

Benchmark subset: 30-task predictive subset used by `eval/benchmarks/swebench_multilingual`.

| Checkpoint | Context | Status | Score | Runtime | Notes |
| --- | ---: | --- | ---: | ---: | --- |
| Starting model | 40k | Invalid harness use | N/A | 9m23s before cancellation | First task made 9 API calls; all ended at 16,384 generated tokens with no content and no tool calls. |
| Starting model | 131k | Invalid harness use | N/A | 9m23s before cancellation | Same trace pattern as 40k: repeated length-capped reasoning and zero tool calls. |
| epoch_0_step_99 | 40k | Invalid harness use | N/A | 3m43s before cancellation | First response ended at 16,384 generated tokens with no content and no tool calls; reasoning text was repetitive. |
| epoch_0_step_199 | 40k | Invalid harness use | N/A | 18m before cancellation | First task made 168 API calls; no tool calls were produced. The first response hit the 16,384-token generation cap with no content, followed by repeated empty stop responses and later reasoning loops. |
| epoch_0_step_299 | 40k | Invalid harness use | N/A | 5m47s before cancellation | First task made 3 API calls; no tool calls were produced. The first response stopped after 10,686 reasoning tokens with no content; later responses remained reasoning-only. |
| epoch_0_step_399 | 40k | Invalid harness use | N/A | 6m23s before cancellation | First task made 6 API calls; no tool calls were produced. Responses alternated between reasoning-only stops and length-capped loops of whitespace/backticks. |
| epoch_0_step_399 | 131k | Invalid harness use | N/A | 6m23s before cancellation | Same trace pattern as final 40k; increasing serving context to 131,072 did not change the failure mode. |

## Checkpoint Notes

### epoch_0_step_99

Consolidation succeeded:
- Text checkpoint: `epoch_0_step_99/model/consolidated`
- Full VL overlay: `epoch_0_step_99/qwen3_vl_2b_full_overlay`
- Trained text tensors replaced: 310
- Missing trained keys in base checkpoint: 0

Live validation failed on the first SWE-bench instance, so the full step-99 benchmark was not run.

### epoch_0_step_199

Consolidation succeeded:
- Text checkpoint: `epoch_0_step_199/model/consolidated`
- Full VL overlay: `epoch_0_step_199/qwen3_vl_2b_full_overlay`
- Trained text tensors replaced: 310
- Missing trained keys in base checkpoint: 0

Live validation failed on the first SWE-bench instance after 168 model calls without a valid tool call, so the full step-199 benchmark was cancelled.

### epoch_0_step_299

Consolidation succeeded:
- Text checkpoint: `epoch_0_step_299/model/consolidated`
- Full VL overlay: `epoch_0_step_299/qwen3_vl_2b_full_overlay`
- Trained text tensors replaced: 310
- Missing trained keys in base checkpoint: 0

Live validation failed on the first SWE-bench instance after 3 model calls without a valid tool call, so the step-299 benchmark was cancelled.

### epoch_0_step_399

Consolidation succeeded:
- Text checkpoint: `epoch_0_step_399/model/consolidated`
- Full VL overlay: `epoch_0_step_399/qwen3_vl_2b_full_overlay`
- Trained text tensors replaced: 310
- Missing trained keys in base checkpoint: 0

Both final 40k and 131k evaluations failed on the first SWE-bench instance with reasoning-only responses and zero tool calls, so neither final benchmark was allowed to run to completion.

## Training Timeline

- Initial training job `418791`: `2026-06-12T00:21:38` to `2026-06-12T08:21:56`, timed out after 8h00m18s at logged step 281. Latest saved checkpoint was `epoch_0_step_199`.
- Naive resume job `425066`: cancelled after 21m30s because `StatefulDataLoader` replayed 200 online-packed iterable batches and logged no optimizer step.
- Resume job `426501`: `2026-06-12T08:44:08` to `2026-06-12T14:34:48`, completed from checkpoint step 199 to step 399 after setting `NEMO_AUTOMODEL_SKIP_DATALOADER_RESTORE=true`.
- Total elapsed wall time from first training start to final checkpoint: about 14h13m. Slurm GPU allocation time for the three training jobs was 14h12m28s, including the cancelled dataloader-replay resume.

## Current State

Training completed at 400/400 steps. No SWE-bench run produced valid tool calls, so there are no successful SWE-bench scores for this SFT run.
