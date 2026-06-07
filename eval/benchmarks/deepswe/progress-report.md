# SWE-rebench V2 High-Quality Datagen Progress

Last updated: 2026-06-07 03:30 UTC

## 2026-06-07 03:30 UTC

Additional throughput increase using only allowed non-Kimi models:

- Submitted `225319` / `swere-mnk-mix1` under `datagen-20260607-pyxis-local-medium-nonkimi-scale1`.
- Manifest has 780 medium trials using local `.sqsh` images only, so there are no Docker pulls in this batch.
- Model mix: 480 `deepseek-v4-flash`, 120 `deepseek-v4-pro`, and 180 `xiaomi/mimo-v2.5-pro`.
- Prompt style mix: 390 original and 390 DeepSWE.
- Task/language mix: 220 JS trials on `serverless__serverless-6545`, 280 JS trials on `serverless__serverless-6417`, and 280 Rust trials on `swc-project__swc-4250`.
- This does not add any new duplicate easy-task submissions; it uses cached medium images to keep throughput high while uncached image imports are blocked.

Startup check:

- `583 swere-mnk-mix1 RUNNING`, `17 CONFIGURING`, and `1 PENDING` at first queue check.
- Sampled stderr files were empty, so local-image startup is clean so far.
- Active datagen remains CPU-only on `m7i-cpu2`; no Kimi jobs are active.

## 2026-06-07 03:28 UTC

Clarification applied: deduplication is for new submissions going forward. Already-started duplicate easy rows were left running so their traces are preserved.

Current active Slurm state:

- `240 swere-hnk-mix1 RUNNING` on `m7i-cpu2`.
- `13 swere-mdvr-mimo RUNNING` on `m7i-cpu2`.
- `4 swere-uef-ewe-o CONFIGURING` on `m7i-cpu2`.
- No active Kimi jobs.
- No active H200/GPU data-generation jobs.

New non-Kimi hard submission:

- Submitted `225075` / `swere-hnk-mix1` under `datagen-20260607-pyxis-local-hard-nonkimi-scale1`.
- Manifest has 240 hard trials using local `.sqsh` images only, so there are no Docker pulls in this batch.
- Model mix: 180 `xiaomi/mimo-v2.5-pro` and 60 `deepseek-v4-pro`.
- Prompt style mix: 120 original and 120 DeepSWE.
- Task/language mix: 80 Python trials on `getmoto__moto-6391`, 80 Rust trials on `swc-project__swc-2598`, and 80 Rust trials on `swc-project__swc-3163`.
- These cached hard tasks previously had Kimi coverage only; this adds non-Kimi traces now that Kimi is disabled.

Unique easy status:

- The 2,998-row unique easy DeepSeek Flash batch mostly failed at Pyxis image import due Docker Hub 429s despite authenticated credentials.
- `sacct` now shows 2,994 failed-start rows and 4 still configuring.
- Failure records/logs are preserved; these are not useful agent trajectories because the containers did not start.

Image/import status:

- Manual local import of one unused hard image downloaded Docker layers but failed during Enroot extraction on root-owned Go module-cache paths. Retrying with scratch extraction and root remapping did not fix it.
- I am not submitting more broad uncached-image arrays until imports are reliable; new useful throughput is coming from local `.sqsh` images.

## 2026-06-07 03:16 UTC

New action after user request to use cheap DeepSeek Flash easy tasks while avoiding duplicates:

- Submitted an initial cached-image DeepSeek Flash easy batch, then canceled 4,055 pending elements after the duplicate-risk correction. Already-started cached-image elements were left running/configuring so partial traces are preserved.
- Submitted `datagen-20260607-pyxis-unique-easy-flash-scale1`: 2,998 easy `deepseek-v4-flash` trials with 2,998 unique task IDs and zero duplicate `instance_id`s.
- Unique easy batch style mix: 1,500 original and 1,498 DeepSWE.
- Unique easy batch language mix: 1,096 Python, 575 Go, 523 JS, 473 TS, 201 Rust, 79 Java, 50 PHP, and 1 C task.
- The unique batch uses the pre-materialized high-quality easy task directories from `datagen-20260606-easy-scale3` and Docker-authenticated Pyxis image pulls split across the three Docker credential directories.

Status at submission check:

- No active Kimi jobs remained.
- Active `swere*` queue: 2,826 rows visible, all on `m7i-cpu2`.
- New unique easy arrays accounted for 2,114 visible rows at scan time; the rest were already-started cached Flake8 Flash rows plus remaining MiMo medium rows.
- Immediate stderr check showed Docker Hub 429s on most unique easy image imports despite authenticated enroot credentials. These failed-start traces are preserved through Slurm logs and the Pyxis failure-result writer where the wrapper regains control; the currently viable easy work is mostly from images already cached/imported.
- Going forward, new submissions are restricted to `deepseek-v4-flash`, `deepseek-v4-pro`, and `xiaomi/mimo-v2.5-pro`; Kimi is disabled for new work.

## 2026-06-07 02:21 UTC

New action after user request to replace unstarted Kimi medium jobs with cheaper MiMo while preserving diversity:

- Canceled 121 pending Kimi medium Slurm array elements from `datagen-20260607-pyxis-local-medium-serverless-scale3`; running Kimi elements were left alone so in-flight traces are preserved.
- Submitted replacement array `218290` / `swere-mdvr-mimo` under `datagen-20260607-pyxis-local-medium-mimo-diverse-replace1`.
- Replacement manifest has 240 medium MiMo (`xiaomi/mimo-v2.5-pro`) trials using local `.sqsh` images: 120 on `serverless__serverless-6545`, 60 on `serverless__serverless-6417`, and 60 on `swc-project__swc-4250`.
- Style mix is 140 original prompt trials and 100 DeepSWE prompt trials; language mix is 180 JS and 60 Rust.

Status at submission check:

- The Kimi medium arrays had no pending elements remaining; only running elements remained active.
- The replacement MiMo array was accepted on `m7i-cpu2`; all visible `swere*` datagen jobs were on `m7i-cpu2`, with no H200/GPU rows.

## 2026-06-07 02:10 UTC

New submission after user request to bias toward cheaper MiMo:

- Submitted `datagen-20260607-pyxis-local-medium-mimo-scale1`: 3,000 medium JS MiMo (`xiaomi/mimo-v2.5-pro`) trials on `serverless__serverless-6545`.
- Style mix is weighted to the stronger observed slice: 2,400 original prompt trials and 600 DeepSWE prompt trials.
- The first attempt to submit one 2,400-row original array was rejected by Slurm, so the original prompt manifest was split into four 600-row arrays. All four original arrays plus the 600-row DeepSWE array were accepted.

Status at submission check:

- New MiMo arrays are on `m7i-cpu2`; no H200/GPU rows.
- No non-empty stderr files yet for the new MiMo scale run.
- This makes the newly submitted medium workload MiMo-heavy relative to the Kimi-heavy hard and previous medium scale jobs.

## 2026-06-07 02:03 UTC

New submissions after user request to increase hard and medium trials:

- Submitted `datagen-20260607-pyxis-local-hard-kimi-scale1`: 600 hard Kimi (`moonshotai/kimi-k2.6`) trials across `swc-project__swc-3163`, `getmoto__moto-6391`, and `swc-project__swc-2598`, split original/DeepSWE 50/50. These use local `.sqsh` images and run on `m7i-cpu2`.
- Submitted `datagen-20260607-pyxis-local-medium-serverless-scale3`: 1,000 medium JS trials on the high-pass `serverless__serverless-6545` task: 400 Kimi original, 300 Kimi DeepSWE, and 300 MiMo original. MiMo DeepSWE remains excluded because its observed pass rate was weaker.

Status at submission check:

- Active/pending `swere*` queue: 902 Slurm array rows visible immediately after submission.
- All visible `swere*` jobs are on `m7i-cpu2`; no H200/GPU rows.
- No non-empty stderr files yet for the new hard Kimi scale or medium scale3 runs.

Recent completed quality signal used for the new mix:

- Hard Kimi smokes completed with 0/30 reward-pass, but hard traces are being scaled anyway per request because failed hard traces are still useful.
- Medium `serverless__serverless-6545` remains the strongest non-easy task source: Kimi original and MiMo original are high-pass, Kimi DeepSWE is good, and MiMo DeepSWE is substantially weaker.

## 2026-06-07 00:36 UTC

Checkpoint:

- Committed and pushed the Pyxis/mini-swe-agent datagen harness changes: `2ecc6eb Add Pyxis SWE-rebench datagen harness`.
- Current active `swere*` jobs are all on CPU-only `m7i-cpu2`; no `swere*` jobs are on H200/GPU partitions.

Audit findings:

- Harness alignment is good for new Pyxis jobs: they run mini-swe-agent v2 through the committed Pyxis driver, use the Harbor task verifier, preserve trajectories and patches, and write `result.json` even for container/startup failures.
- Task filtering is still the intended high-quality SWE-rebench V2 subset (`confidence >= 0.95` plus existing metadata filters).
- Main weakness in generated data is duplicate concentration: `martinthoma__flake8-simplify-124` dominates completed results. This is high reward but lower diversity.
- Second weakness is the JS/TS easy scale wave: it added diversity but had much lower reward than flake8. Latest audit showed `datagen-20260607-pyxis-local-diverse-scale1` at 416/1,124 reward-pass when scanned.
- Strongest new quality signal is medium serverless with Kimi/MiMo. The initial 20-rollout serverless medium smoke passed 18/20; in the larger scale audit, Kimi original was 44/45, Kimi DeepSWE 50/60, and MiMo original 26/27. MiMo DeepSWE was weaker at 19/40, so it is no longer being scaled.
- SWC medium is slow and has not shown reward yet; it is still smoke-only and is not being scaled.

Actions taken after the audit:

- Submitted `datagen-20260607-pyxis-local-medium-serverless-scale1`: 450 medium JS rollouts on `serverless__serverless-6545`, weighted toward Kimi and MiMo original.
- Submitted `datagen-20260607-pyxis-local-hard-smoke1`: 20 hard Kimi rollouts on `swc-project__swc-3163` and `getmoto__moto-6391`, original and DeepSWE.
- Submitted `datagen-20260607-pyxis-local-medium-serverless-6417-smoke`: 20 medium JS smoke rollouts on a second serverless medium task.
- Submitted `datagen-20260607-pyxis-local-hard-swc2598-smoke`: 10 hard Kimi rollouts on a second SWC hard task.
- Submitted `datagen-20260607-pyxis-local-medium-serverless-scale2`: 500 additional medium JS rollouts using only the strong slices: Kimi original, Kimi DeepSWE, and MiMo original.

Current active queue at 00:36 UTC:

- Medium serverless scale1/scale2: 594 running rows plus a few pending, all `m7i-cpu2`.
- Hard smokes: 30 running rows, all `m7i-cpu2`.
- New serverless 6417 medium smoke: 20 running rows, all `m7i-cpu2`.
- Remaining SWC medium smoke: 19 running rows, all `m7i-cpu2`.


Smoke run root: `/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260606-smoke`

Batch 1 run root: `/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260606-batch1`

## Current Phase

Small targeted quality smoke is running on `m7i-cpu2` through the DeepSWE Pier/mini-swe-agent harness. No agent generation is running locally. The working submission (`208359`-`208364`) forces mini-swe-agent's `model_class=litellm` chat-completions path. Batch 1 has been sampled from the high-quality subset and the first original-style easy/medium agent jobs are running on m7i nodes.

## Smoke Jobs

| Slurm job | Model | API route | Difficulty | Language | Instance | Instruction style | State |
|---:|---|---|---|---|---|---|---|
| 208359 | `deepseek-v4-flash` | DeepSeek API | easy | go | `99designs__aws-vault-1178` | original | pass |
| 208360 | `deepseek-v4-flash` | DeepSeek API | easy | go | `99designs__aws-vault-1178` | rewritten | pass |
| 208361 | `xiaomi/mimo-v2.5-pro` | OpenRouter | medium | python | `3yourmind__django-migration-linter-186` | original | pass |
| 208362 | `xiaomi/mimo-v2.5-pro` | OpenRouter | medium | python | `3yourmind__django-migration-linter-186` | rewritten | pass, quality reject |
| 208363 | `moonshotai/kimi-k2.6` | OpenRouter | hard | python | `axelrod-python__axelrod-975` | original | cancelled, runaway loop |
| 208364 | `moonshotai/kimi-k2.6` | OpenRouter | hard | python | `axelrod-python__axelrod-975` | rewritten | cancelled, runaway loop |
| 208397 | `moonshotai/kimi-k2.6` | OpenRouter | hard | go | `brimsec__zq-2173` | original, capped | running |
| 208398 | `moonshotai/kimi-k2.6` | OpenRouter | hard | go | `brimsec__zq-2173` | rewritten, capped | running |

## Generated Data Counts

| Model | Completed smoke traces | Verified pass | Clean training-quality pass | Reward pass rate |
|---|---:|---:|---:|---:|
| `deepseek-v4-flash` | 2 | 2 | 2 | 100.0% |
| `xiaomi/mimo-v2.5-pro` | 2 | 2 | 1 | 100.0% |
| `moonshotai/kimi-k2.6` | 0 | 0 | 0 | rerunning capped hard smoke |

## Coverage So Far

| Difficulty | Planned smoke traces | Completed traces |
|---|---:|---:|
| easy | 2 | 2 |
| medium | 2 | 2 reward passes, 1 clean |
| hard | 2 | 0 |

| Language | Planned smoke traces | Completed traces |
|---|---:|---:|
| go | 2 | 2 |
| python | 4 | 2 reward passes, 1 clean |

## Batch 1

Batch 1 samples 81 new tasks, excluding the three smoke instances, uniformly by language:

| Difficulty | Tasks | Instruction mix | Model |
|---|---:|---|---|
| easy | 45 | 22 original / 23 rewritten | `deepseek-v4-flash` |
| medium | 27 | 18 original / 9 rewritten | `xiaomi/mimo-v2.5-pro` |
| hard | 9 | 5 original / 4 rewritten | `moonshotai/kimi-k2.6` |

Each language with high-quality coverage has 9 selected tasks total: c, cpp, go, java, js, php, python, rust, ts. Ruby has no high-quality tasks in this subset.

Active Batch 1 jobs:

| Slurm job | Purpose | State |
|---:|---|---|
| 208370-208373 | Initial Kimi prompt rewrites | completed but clipped by default rewrite `--limit=10`; 40 rewrites produced |
| 208387-208390 | Fixed Kimi prompt rewrites with JSON mode and reasoning disabled | completed; 40 rewrites produced |
| 208394-208395 | Fixed Kimi prompt rewrites for the remaining 41 selected tasks | completed |
| 208378 | DeepSeek easy original traces, 22 tasks, `n_concurrent=8` | running |
| 208379 | MiMo medium original traces, 18 tasks, `n_concurrent=4` | running |
| 208404 | DeepSeek easy clean-rewritten traces, 22 tasks, `n_concurrent=8` | running |
| 208405 | MiMo medium clean-rewritten traces, 7 tasks, `n_concurrent=3` | running |
| 208406 | DeepSeek easy original-fallback trace, 1 task | running |
| 208407 | MiMo medium original-fallback traces, 2 tasks | running |

Current Batch 1 generated-trace summary from `summary.json`:

| Model | Completed trials | Reward pass | Clean quality pass | Notes |
|---|---:|---:|---:|---|
| `deepseek-v4-flash` | 9 | 2 | 2 | original/easy only so far; 3 test-file-edit rejects, 1 oversized patch reject |
| `xiaomi/mimo-v2.5-pro` | 1 | 0 | 0 | original/medium trajectories active |
| `moonshotai/kimi-k2.6` | 0 | 0 | 0 | hard batch still gated on smoke |

## Notes

- Harness path: `eval.benchmarks.deepswe.run` -> Pier -> `--agent mini-swe-agent`; installed mini-swe-agent is v2.3.0.
- m7i preflight confirmed Docker 29.5.3 on `m7i-cpu2`; the smoke jobs run on m7i nodes, not locally.
- API keys are loaded from `/wbl-fast/usrs/ee/code-swe-data/.env` inside Slurm jobs and are not written into the report.
- Rewritten prompts for this smoke batch came from the existing validated rewrite JSONLs under `datagen/swerebench_v2/examples/`.
- DeepSeek easy original and rewritten styles are tied so far, so the next easy batch should use approximately 50% rewritten instructions unless later evidence changes.
- MiMo reward pass rates are tied, but the rewritten smoke patch edited a repository test file. For quality, Batch 1 weights medium toward original prompts and the summarization step will treat test-file edits as quality rejects.
- Added `datagen.swerebench_v2.summarize_pier_runs`, which counts reward passes separately from clean training-quality passes. A clean pass requires a reward, a trajectory, a nonempty model patch, no exception, and no test-file edits.
- OpenRouter Kimi rewrite calls require chat JSON mode plus `reasoning.effort=none`; without that, several responses exhausted output on reasoning and returned no parseable JSON. The rewriter now supports `--response-format-json`, provider `--extra-body-json`, retries, and unclipped explicit `--instance-id` lists.
- Batch 1 rewrites are complete: 81/81 prompts generated. Three planned rewritten tasks had rewrite-quality warnings and were switched to original-fallback generation instead of using the warninged rewritten prompt.
- The first Kimi hard smoke on `axelrod-python__axelrod-975` was stopped after repeated command loops. The replacement hard smoke uses `brimsec__zq-2173` with a mini-swe-agent cap config (`step_limit=80`, `cost_limit=3.0`) while preserving the same Pier/mini-swe-agent harness.

## Automated Update 2026-06-06 19:30 UTC

- Batch 1 completed trials: 13
- Reward passes: 2
- Clean quality passes: 2
- Pending assignments: 68
- Quality rejects: {"empty_or_missing_patch": 1, "exceptions": 0, "missing_trajectory": 0, "patch_too_large": 1, "test_file_edits": 3}
- Kimi smoke2: original: api_calls=63, exit=running; rewritten: api_calls=80, exit=EOFError

Active jobs:
- `            208407    swere-b1-m-ofb-mimo-v CONFIGUR       1:38 m7i-cpu2-dy-m7i-cpu-cr-0-34`
- `            208408         swere-b1-monitor  RUNNING       0:02 m7i-cpu2-dy-m7i-cpu-cr-0-5`
- `            208406    swere-b1-e-ofb-deepse  RUNNING       1:39 m7i-cpu2-dy-m7i-cpu-cr-0-2`
- `            208405    swere-b1-m-r-mimo-v2.  RUNNING       3:32 m7i-cpu2-dy-m7i-cpu-cr-0-5`
- `            208404    swere-b1-e-r-deepseek  RUNNING       3:33 m7i-cpu2-dy-m7i-cpu-cr-0-3`
- `            208398                  kimi2-r  RUNNING       5:02 m7i-cpu2-dy-m7i-cpu-cr-0-31`
- `            208397                  kimi2-o  RUNNING       5:03 m7i-cpu2-dy-m7i-cpu-cr-0-31`
- `            208379    swere-b1-m-o-mimo-v2.  RUNNING      15:14 m7i-cpu2-dy-m7i-cpu-cr-0-4`
- `            208378    swere-b1-e-o-deepseek  RUNNING      18:56 m7i-cpu2-dy-m7i-cpu-cr-0-32`

By model/style:
- `deepseek-v4-flash/original`: total=11, reward=2, clean=2
- `xiaomi/mimo-v2.5-pro/original`: total=2, reward=0, clean=0

## Manual Update 2026-06-06 19:40 UTC

- Batch 1 completed trials: 32
- Reward passes: 12
- Clean training-quality passes: 8
- Pending assignments: 49
- Quality rejects: {"empty_or_missing_patch": 1, "exceptions": 0, "missing_trajectory": 0, "patch_too_large": 2, "test_file_edits": 8}
- Hard generation remains gated. Kimi native OpenRouter smoke3 (`208409`) is running on `m7i-cpu2`, has a live mini-swe-agent trajectory, and is at 49 API calls / $0.413 tracked cost with no verifier result yet.

By model/style:
- `deepseek-v4-flash/original`: total=20, reward=6, clean=4
- `deepseek-v4-flash/original-fallback`: total=1, reward=1, clean=1
- `deepseek-v4-flash/rewritten`: total=7, reward=4, clean=3
- `xiaomi/mimo-v2.5-pro/original`: total=3, reward=1, clean=0
- `xiaomi/mimo-v2.5-pro/rewritten`: total=1, reward=0, clean=0

By difficulty:
- easy: total=28, reward=11, clean=8
- medium: total=4, reward=1, clean=0
- hard: total=0, reward=0, clean=0

By language:
- c: total=7, reward=3, clean=3
- cpp: total=2, reward=0, clean=0
- go: total=7, reward=4, clean=2
- java: total=3, reward=0, clean=0
- js: total=2, reward=0, clean=0
- php: total=3, reward=1, clean=0
- python: total=3, reward=1, clean=1
- rust: total=3, reward=1, clean=1
- ts: total=2, reward=2, clean=1

Active jobs:
- `208409 kimi3-openr RUNNING 7:18`
- `208407 swere-b1-m-ofb-mimo-v RUNNING 7:25`
- `208408 swere-b1-monitor RUNNING 9:43`
- `208405 swere-b1-m-r-mimo-v2. RUNNING 13:13`
- `208404 swere-b1-e-r-deepseek RUNNING 13:14`
- `208379 swere-b1-m-o-mimo-v2. RUNNING 24:55`
- `208378 swere-b1-e-o-deepseek RUNNING 28:37`

Notes:
- Added `datagen/swerebench_v2/minisweagent_datagen_strict.yaml` for future batches. It keeps the mini-swe-agent v2 harness but adds benchmark-style no-test/no-artifact edit boundaries and a step/cost cap. Batch 1 is unchanged and continues with the original smoke-compatible `mini.yaml` path.
- On the current small sample, DeepSeek rewritten prompts are ahead of original prompts. If this holds through Batch 1, the next easy batch should weight rewritten prompts above 50%, but not switch entirely because we still need original-style coverage.

## Automated Update 2026-06-06 19:48 UTC

- Batch 1 completed trials: 45
- Reward passes: 14
- Clean quality passes: 9
- Pending assignments: 36
- Quality rejects: {"empty_or_missing_patch": 1, "exceptions": 0, "missing_trajectory": 0, "patch_too_large": 2, "test_file_edits": 16}
- Kimi smoke2: original: api_calls=80, exit=EOFError; rewritten: api_calls=80, exit=EOFError
- Kimi smoke3: original: api_calls=65, cost=0.61925956, exit=running
- Strict smoke completed trials: 0, reward=0, clean=0, pending=8
- Strict smoke quality rejects: {"empty_or_missing_patch": 0, "exceptions": 0, "missing_trajectory": 0, "patch_too_large": 0, "test_file_edits": 0}

Active jobs:
- `            208419        swere-strict-e-r2  RUNNING       2:26 m7i-cpu2-dy-m7i-cpu-cr-0-31`
- `            208418        swere-strict-e-o2  RUNNING       2:27 m7i-cpu2-dy-m7i-cpu-cr-0-2`
- `            208416         swere-strict-m-o  RUNNING       2:52 m7i-cpu2-dy-m7i-cpu-cr-0-31`
- `            208417         swere-strict-m-r  RUNNING       2:52 m7i-cpu2-dy-m7i-cpu-cr-0-34`
- `            208409              kimi3-openr  RUNNING      15:30 m7i-cpu2-dy-m7i-cpu-cr-0-4`
- `            208408         swere-b1-monitor  RUNNING      17:55 m7i-cpu2-dy-m7i-cpu-cr-0-5`
- `            208405    swere-b1-m-r-mimo-v2.  RUNNING      21:25 m7i-cpu2-dy-m7i-cpu-cr-0-5`
- `            208404    swere-b1-e-r-deepseek  RUNNING      21:26 m7i-cpu2-dy-m7i-cpu-cr-0-3`
- `            208379    swere-b1-m-o-mimo-v2.  RUNNING      33:07 m7i-cpu2-dy-m7i-cpu-cr-0-4`
- `            208378    swere-b1-e-o-deepseek  RUNNING      36:49 m7i-cpu2-dy-m7i-cpu-cr-0-32`

By model/style:
- `deepseek-v4-flash/original`: total=21, reward=6, clean=4
- `deepseek-v4-flash/original-fallback`: total=1, reward=1, clean=1
- `deepseek-v4-flash/rewritten`: total=14, reward=6, clean=4
- `xiaomi/mimo-v2.5-pro/original`: total=4, reward=1, clean=0
- `xiaomi/mimo-v2.5-pro/original-fallback`: total=2, reward=0, clean=0
- `xiaomi/mimo-v2.5-pro/rewritten`: total=3, reward=0, clean=0

## Automated Update 2026-06-06 19:55 UTC

- Batch 1 completed trials: 54
- Reward passes: 17
- Clean quality passes: 12
- Pending assignments: 27
- Quality rejects: {"empty_or_missing_patch": 1, "exceptions": 0, "missing_trajectory": 0, "patch_too_large": 5, "test_file_edits": 19}
- Kimi smoke2: original: api_calls=80, exit=EOFError; rewritten: api_calls=80, exit=EOFError
- Kimi smoke3: original: api_calls=77, cost=0.9181754199999999, exit=running
- Strict smoke completed trials: 7, reward=3, clean=3, pending=1
- Strict smoke quality rejects: {"empty_or_missing_patch": 0, "exceptions": 0, "missing_trajectory": 0, "patch_too_large": 0, "test_file_edits": 0}
Strict smoke by model/style:
- `deepseek-v4-flash/original`: total=2, reward=2, clean=2
- `deepseek-v4-flash/rewritten`: total=2, reward=1, clean=1
- `xiaomi/mimo-v2.5-pro/original`: total=2, reward=0, clean=0
- `xiaomi/mimo-v2.5-pro/rewritten`: total=1, reward=0, clean=0
- Batch 2 strict-easy completed trials: 0, reward=0, clean=0, pending=90
- Batch 2 strict-easy quality rejects: {"empty_or_missing_patch": 0, "exceptions": 0, "missing_trajectory": 0, "patch_too_large": 0, "test_file_edits": 0}

Active jobs:
- `            208422           swere-b2-rw-00  RUNNING       2:40 m7i-cpu2-dy-m7i-cpu-cr-0-31`
- `            208423           swere-b2-rw-01  RUNNING       2:40 m7i-cpu2-dy-m7i-cpu-cr-0-31`
- `            208424           swere-b2-rw-02  RUNNING       2:40 m7i-cpu2-dy-m7i-cpu-cr-0-34`
- `            208421          swere-b2-e-o-ds  RUNNING       2:53 m7i-cpu2-dy-m7i-cpu-cr-0-2`
- `            208417         swere-strict-m-r  RUNNING       9:11 m7i-cpu2-dy-m7i-cpu-cr-0-34`
- `            208409              kimi3-openr  RUNNING      21:49 m7i-cpu2-dy-m7i-cpu-cr-0-4`
- `            208408         swere-b1-monitor  RUNNING      24:14 m7i-cpu2-dy-m7i-cpu-cr-0-5`
- `            208405    swere-b1-m-r-mimo-v2.  RUNNING      27:44 m7i-cpu2-dy-m7i-cpu-cr-0-5`
- `            208404    swere-b1-e-r-deepseek  RUNNING      27:45 m7i-cpu2-dy-m7i-cpu-cr-0-3`
- `            208379    swere-b1-m-o-mimo-v2.  RUNNING      39:26 m7i-cpu2-dy-m7i-cpu-cr-0-4`
- `            208378    swere-b1-e-o-deepseek  RUNNING      43:08 m7i-cpu2-dy-m7i-cpu-cr-0-32`

## Manual Update 2026-06-06 22:58 UTC

CPU/GPU placement:
- All `swere*` and `kimi*` datagen jobs submitted for this effort have run on `m7i-cpu2`; no datagen jobs were submitted to H200.
- H200 jobs visible in `squeue` are unrelated non-datagen jobs and were not touched.
- New Slurm/Pyxis submitter refuses partitions whose names look GPU/H200-like and pins datagen arrays to CPU partitions.

Completed useful data:
- Clean training-quality passes: 30
- Reward passes: 40
- Successful/Pier result records summarized: 125
- `deepseek-v4-flash`: 36 reward, 28 clean
- `xiaomi/mimo-v2.5-pro`: 4 reward, 2 clean
- `moonshotai/kimi-k2.6`: 0 reward, 0 clean so far

Failed traces now preserved:
- Easy scale2 produced 605 result records, but all failed before agent execution because task images hit Docker Hub pull failures; these records remain under `/wbl-fast`.
- Kimi medium scale1 produced 75 result records, also failed before agent execution for the same image-pull reason; these records remain under `/wbl-fast`.
- Pyxis smoke attempts have 6 saved `result.json` startup-failure records across the original and fixed image-name runs.

Current blocker:
- The original high-concurrency Pier jobs failed at environment setup with Docker Hub 429s while pulling `docker.io/swerebenchv2/*` images.
- The Pyxis-native CPU-only smoke runner now normalizes image names correctly, but Pyxis imports are still blocked by Docker Hub 429s for `swerebenchv2/*`.
- No Docker Hub/enroot credentials are configured for this user (`.env` has no Docker/registry key; user Docker/enroot credential files are absent).
- I am not launching the 4,123-task scale3 array until image access is fixed, because it would create thousands of zero-call startup failures rather than spending API budget on useful traces.

New durable runner work:
- Added a CPU-only Slurm/Pyxis mini-swe-agent runner that runs inside each SWE-rebench task image and writes `metadata.json`, `result.json`, `model.patch`, verifier logs, and `agent/mini-swe-agent.trajectory.json` when the container starts.
- Added host-side `result.json` writing for Pyxis container-start failures so failed traces are still represented.
- Copied a Python 3.12 runtime to `/wbl-fast/usrs/ee/code-swe-data/runtime` and configured new jobs to use `/wbl-fast/usrs/ee/code-swe-data/cache` for HF, XDG, uv, and pip caches.

By model/style:
- `deepseek-v4-flash/original`: total=21, reward=6, clean=4
- `deepseek-v4-flash/original-fallback`: total=1, reward=1, clean=1
- `deepseek-v4-flash/rewritten`: total=19, reward=9, clean=7
- `xiaomi/mimo-v2.5-pro/original`: total=6, reward=1, clean=0
- `xiaomi/mimo-v2.5-pro/original-fallback`: total=2, reward=0, clean=0
- `xiaomi/mimo-v2.5-pro/rewritten`: total=5, reward=0, clean=0

## Automated Update 2026-06-06 20:31 UTC

- Batch 1 completed trials: 69
- Reward passes: 21
- Clean quality passes: 15
- Pending assignments: 12
- Quality rejects: {"empty_or_missing_patch": 2, "exceptions": 0, "missing_trajectory": 0, "patch_too_large": 6, "test_file_edits": 24}
- Kimi smoke2: original: api_calls=80, exit=EOFError; rewritten: api_calls=80, exit=EOFError
- Kimi smoke3: original: api_calls=80, cost=1.0098095399999998, exit=EOFError
- Strict smoke completed trials: 8, reward=3, clean=3, pending=0
- Strict smoke quality rejects: {"empty_or_missing_patch": 0, "exceptions": 0, "missing_trajectory": 0, "patch_too_large": 1, "test_file_edits": 0}
Strict smoke by model/style:
- `deepseek-v4-flash/original`: total=2, reward=2, clean=2
- `deepseek-v4-flash/rewritten`: total=2, reward=1, clean=1
- `xiaomi/mimo-v2.5-pro/original`: total=2, reward=0, clean=0
- `xiaomi/mimo-v2.5-pro/rewritten`: total=2, reward=0, clean=0
- Batch 2 strict-easy completed trials: 33, reward=13, clean=9, pending=57
- Batch 2 strict-easy quality rejects: {"empty_or_missing_patch": 1, "exceptions": 1, "missing_trajectory": 0, "patch_too_large": 3, "test_file_edits": 5}
Batch 2 strict-easy by model/style:
- `deepseek-v4-flash/original`: total=33, reward=13, clean=9

Active jobs:
- `            208421          swere-b2-e-o-ds  RUNNING      38:50 m7i-cpu2-dy-m7i-cpu-cr-0-2`
- `            208408         swere-b1-monitor  RUNNING    1:00:11 m7i-cpu2-dy-m7i-cpu-cr-0-5`
- `            208379    swere-b1-m-o-mimo-v2.  RUNNING    1:15:23 m7i-cpu2-dy-m7i-cpu-cr-0-4`
- `            208378    swere-b1-e-o-deepseek  RUNNING    1:19:05 m7i-cpu2-dy-m7i-cpu-cr-0-32`

By model/style:
- `deepseek-v4-flash/original`: total=21, reward=6, clean=4
- `deepseek-v4-flash/original-fallback`: total=1, reward=1, clean=1
- `deepseek-v4-flash/rewritten`: total=22, reward=10, clean=8
- `xiaomi/mimo-v2.5-pro/original`: total=16, reward=4, clean=2
- `xiaomi/mimo-v2.5-pro/original-fallback`: total=2, reward=0, clean=0
- `xiaomi/mimo-v2.5-pro/rewritten`: total=7, reward=0, clean=0

## Automated Update 2026-06-06 21:31 UTC

- Batch 1 completed trials: 72
- Reward passes: 21
- Clean quality passes: 15
- Pending assignments: 9
- Quality rejects: {"empty_or_missing_patch": 2, "exceptions": 1, "missing_trajectory": 0, "patch_too_large": 7, "test_file_edits": 25}
- Kimi smoke2: original: api_calls=80, exit=EOFError; rewritten: api_calls=80, exit=EOFError
- Kimi smoke3: original: api_calls=80, cost=1.0098095399999998, exit=EOFError
- Strict smoke completed trials: 8, reward=3, clean=3, pending=0
- Strict smoke quality rejects: {"empty_or_missing_patch": 0, "exceptions": 0, "missing_trajectory": 0, "patch_too_large": 1, "test_file_edits": 0}
Strict smoke by model/style:
- `deepseek-v4-flash/original`: total=2, reward=2, clean=2
- `deepseek-v4-flash/rewritten`: total=2, reward=1, clean=1
- `xiaomi/mimo-v2.5-pro/original`: total=2, reward=0, clean=0
- `xiaomi/mimo-v2.5-pro/rewritten`: total=2, reward=0, clean=0
- Batch 2 strict-easy completed trials: 45, reward=16, clean=12, pending=45
- Batch 2 strict-easy quality rejects: {"empty_or_missing_patch": 1, "exceptions": 1, "missing_trajectory": 0, "patch_too_large": 4, "test_file_edits": 5}
Batch 2 strict-easy by model/style:
- `deepseek-v4-flash/original`: total=45, reward=16, clean=12

Active jobs:
- `            208408         swere-b1-monitor  RUNNING    2:00:15 m7i-cpu2-dy-m7i-cpu-cr-0-5`

By model/style:
- `deepseek-v4-flash/original`: total=22, reward=6, clean=4
- `deepseek-v4-flash/original-fallback`: total=1, reward=1, clean=1
- `deepseek-v4-flash/rewritten`: total=22, reward=10, clean=8
- `xiaomi/mimo-v2.5-pro/original`: total=18, reward=4, clean=2
- `xiaomi/mimo-v2.5-pro/original-fallback`: total=2, reward=0, clean=0
- `xiaomi/mimo-v2.5-pro/rewritten`: total=7, reward=0, clean=0

## Current Status 2026-06-06 23:00 UTC

CPU/GPU placement:
- Datagen jobs named `swere*` and `kimi*` ran on `m7i-cpu2`; none were submitted to H200.
- The visible H200 jobs in `squeue` are unrelated non-datagen jobs.
- The stale `swere-b1-monitor` job was canceled to stop stale hourly appends.

Completed useful data:
- Clean training-quality passes: 30
- Reward passes: 40
- Successful/Pier result records summarized: 125
- `deepseek-v4-flash`: 36 reward, 28 clean
- `xiaomi/mimo-v2.5-pro`: 4 reward, 2 clean
- `moonshotai/kimi-k2.6`: 0 reward, 0 clean so far

Failed trace records preserved:
- Easy scale2: 605 saved result records, all failed before agent execution because Docker Hub blocked `swerebenchv2/*` image pulls.
- Kimi medium scale1: 75 saved result records, all failed before agent execution for the same image-pull issue.
- Pyxis smoke: 6 saved startup-failure `result.json` records across the initial and fixed image-name runs.

Current blocker:
- Docker Hub is returning 429 for `swerebenchv2/*` task images from both Pier/Docker Compose and Slurm/Pyxis.
- This user has no Docker Hub/enroot credentials configured; `.env` exposes no Docker/registry key names.
- The 4,123-task easy scale3 manifest is materialized under `/wbl-fast`, but I am not launching it until image access is fixed because it would produce thousands of zero-call startup failures.

Infrastructure changes:
- Added a CPU-only Slurm/Pyxis mini-swe-agent runner that writes `metadata.json`, `result.json`, `model.patch`, verifier logs, and `agent/mini-swe-agent.trajectory.json` for container-starting tasks.
- Added host-side result writing for Pyxis container-start failures.
- Copied Python runtime to `/wbl-fast/usrs/ee/code-swe-data/runtime` and configured future jobs to use `/wbl-fast/usrs/ee/code-swe-data/cache` for HF/XDG/uv/pip caches.

## Automated Update 2026-06-06 22:31 UTC

- Batch 1 completed trials: 72
- Reward passes: 21
- Clean quality passes: 15
- Pending assignments: 9
- Quality rejects: {"empty_or_missing_patch": 2, "exceptions": 1, "missing_trajectory": 0, "patch_too_large": 7, "test_file_edits": 25}
- Kimi smoke2: original: api_calls=80, exit=EOFError; rewritten: api_calls=80, exit=EOFError
- Kimi smoke3: original: api_calls=80, cost=1.0098095399999998, exit=EOFError
- Strict smoke completed trials: 8, reward=3, clean=3, pending=0
- Strict smoke quality rejects: {"empty_or_missing_patch": 0, "exceptions": 0, "missing_trajectory": 0, "patch_too_large": 1, "test_file_edits": 0}
Strict smoke by model/style:
- `deepseek-v4-flash/original`: total=2, reward=2, clean=2
- `deepseek-v4-flash/rewritten`: total=2, reward=1, clean=1
- `xiaomi/mimo-v2.5-pro/original`: total=2, reward=0, clean=0
- `xiaomi/mimo-v2.5-pro/rewritten`: total=2, reward=0, clean=0
- Batch 2 strict-easy completed trials: 45, reward=16, clean=12, pending=45
- Batch 2 strict-easy quality rejects: {"empty_or_missing_patch": 1, "exceptions": 1, "missing_trajectory": 0, "patch_too_large": 4, "test_file_edits": 5}
Batch 2 strict-easy by model/style:
- `deepseek-v4-flash/original`: total=45, reward=16, clean=12

Active jobs:
- `            208408         swere-b1-monitor  RUNNING    3:00:20 m7i-cpu2-dy-m7i-cpu-cr-0-5`

By model/style:
- `deepseek-v4-flash/original`: total=22, reward=6, clean=4
- `deepseek-v4-flash/original-fallback`: total=1, reward=1, clean=1
- `deepseek-v4-flash/rewritten`: total=22, reward=10, clean=8
- `xiaomi/mimo-v2.5-pro/original`: total=18, reward=4, clean=2
- `xiaomi/mimo-v2.5-pro/original-fallback`: total=2, reward=0, clean=0
- `xiaomi/mimo-v2.5-pro/rewritten`: total=7, reward=0, clean=0

## Final Current Status 2026-06-06 23:00 UTC

- No `swere*` or `kimi*` datagen jobs are currently running.
- All datagen jobs submitted for this effort ran on `m7i-cpu2`; no datagen job was submitted to H200.
- The old `swere-b1-monitor` job was canceled because it did not know about the Pyxis/image-access blocker.
- Clean training-quality passes remain 30: 28 from `deepseek-v4-flash`, 2 from `xiaomi/mimo-v2.5-pro`, 0 from `moonshotai/kimi-k2.6`.
- Failed traces are preserved: 605 easy scale2 environment failures, 75 Kimi-medium environment failures, and 6 Pyxis startup-failure records.
- The 4,123-task easy scale3 manifest and task directories are ready under `/wbl-fast`, but large submission is blocked by Docker Hub 429s on `swerebenchv2/*` task images.
- `.env` and user config have no Docker Hub/enroot registry credentials; image access needs credentials, a mirror, or preconverted `.sqsh` task images before high-throughput API generation can proceed.

## Automated Update 2026-06-06 23:29 UTC

CPU/GPU placement:
- Active datagen jobs are on `m7i-cpu2`; no datagen job has been submitted to H200.
- R2 pending/configuring unique-image pulls were canceled after Docker Hub 429/401 errors; running tasks that had already started were left to save results.

Current Pyxis saved artifacts:
- `result.json` records: 740
- metadata records: 331
- mini-swe-agent trajectories: 298
- model patches: 117
- status counts: {'completed_reward0': 60, 'reward_pass': 38, 'pyxis_start_error': 642}

Pass rates by model (Pyxis-era results only):
- `deepseek-v4-flash`: total=661, reward=33, passrate=4.99%
- `deepseek-v4-pro`: total=78, reward=5, passrate=6.41%
- `xiaomi/mimo-v2.5-pro`: total=1, reward=0, passrate=0.00%

By model and prompt style:
- `deepseek-v4-flash/deepswe`: total=423, reward=7, passrate=1.65%
- `deepseek-v4-flash/original`: total=238, reward=26, passrate=10.92%
- `deepseek-v4-pro/deepswe`: total=1, reward=0, passrate=0.00%
- `deepseek-v4-pro/original`: total=77, reward=5, passrate=6.49%
- `xiaomi/mimo-v2.5-pro/original`: total=1, reward=0, passrate=0.00%

Language and difficulty stats for saved Pyxis results:
- difficulty: {'easy': 740}
- language: {'python': 328, 'js': 136, 'go': 121, 'ts': 104, 'rust': 32, 'java': 10, 'php': 9}

Throughput actions since last update:
- Submitted 3,703 additional easy high-quality tasks split between original and deepswe prompts; canceled 3,238 non-started R2 array elements after registry errors to avoid wasting CPU and Docker quota.
- Submitted 620 duplicate rollouts against a local `/wbl-fast` SquashFS task image: 400 DeepSeek flash, 120 DeepSeek pro, 60 MiMo, 40 Kimi; split evenly original/deepswe and run without Docker Hub access.
- Local duplicate rollout jobs are using the mini-swe-agent-v2 Pyxis harness and preserve successes and failures.

Active Slurm datagen state:
- `177 swere-lsq-dsf-o RUNNING`
- `81 swere-r1-flash-eth RUNNING`
- `72 swere-lsq-dsf-d RUNNING`
- `60 swere-r2-fd-ewe RUNNING`
- `58 swere-lsq-dsp-o RUNNING`
- `56 swere-lsq-dsp-d RUNNING`
- `44 swere-r1-pro-eth RUNNING`
- `38 swere-r2-fd-och RUNNING`
- `30 swere-lsq-mimo-d CONFIGURING`
- `29 swere-lsq-mimo-o CONFIGURING`
- `20 swere-lsq-kimi-d RUNNING`
- `11 swere-lsq-kimi-o RUNNING`
- `9 swere-lsq-kimi-o CONFIGURING`
- `2 swere-r2-fd-oew RUNNING`
- `1 swere-r2-fo-och RUNNING`
- `1 swere-lsq-dsf-o COMPLETING`

## Automated Update 2026-06-06 23:51 UTC

CPU/GPU placement:
- Active datagen jobs remain on `m7i-cpu2`; no datagen jobs were submitted to H200.

Runtime/infrastructure changes:
- Docker Hub remains the limiter for distinct `swerebenchv2/*` images: authenticated pulls still return 429/401 on many tags.
- Repaired runtime dependency isolation by adding `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-overlay` ahead of the damaged shared `.venv` in new Pyxis jobs.
- Verified the local SquashFS task image path: flash original and deepswe smoke both produced trajectories and reward=1.

Current Pyxis saved artifacts across active Pyxis roots:
- `result.json` records: 1020
- metadata records: 855
- mini-swe-agent trajectories: 549
- model patches: 464
- status counts: {'completed_reward0': 222, 'reward_pass': 241, 'pyxis_start_error': 557}

Pass rates by model (Pyxis-era results only):
- `deepseek-v4-flash`: total=880, reward=202, passrate=22.95%
- `deepseek-v4-pro`: total=131, reward=30, passrate=22.90%
- `moonshotai/kimi-k2.6`: total=7, reward=7, passrate=100.00%
- `xiaomi/mimo-v2.5-pro`: total=2, reward=2, passrate=100.00%

By model and prompt style:
- `deepseek-v4-flash/deepswe`: total=442, reward=33, passrate=7.47%
- `deepseek-v4-flash/original`: total=438, reward=169, passrate=38.58%
- `deepseek-v4-pro/deepswe`: total=6, reward=6, passrate=100.00%
- `deepseek-v4-pro/original`: total=125, reward=24, passrate=19.20%
- `moonshotai/kimi-k2.6/deepswe`: total=1, reward=1, passrate=100.00%
- `moonshotai/kimi-k2.6/original`: total=6, reward=6, passrate=100.00%
- `xiaomi/mimo-v2.5-pro/deepswe`: total=1, reward=1, passrate=100.00%
- `xiaomi/mimo-v2.5-pro/original`: total=1, reward=1, passrate=100.00%

Language and difficulty stats for saved Pyxis results:
- difficulty: {'easy': 1020}
- language: {'python': 482, 'js': 194, 'ts': 146, 'go': 136, 'rust': 41, 'php': 11, 'java': 10}

Current high-throughput local-image batch:
- Flash scale1 submitted 400 local-image rollouts; at report time the original side has begun completing with reward=1 on all completed records.
- Model scale1 submitted 220 local-image rollouts across DeepSeek pro, MiMo, and Kimi after all model-route smoke records that completed passed.
- These local-image runs use saved `.sqsh` under `/wbl-fast` and do not hit Docker Hub.

Active Slurm datagen state:
- `200 swere-lf1-dsf-d RUNNING`
- `82 swere-lf1-dsf-o RUNNING`
- `56 swere-lm1-dsp-o RUNNING`
- `27 swere-lm1-dsp-d RUNNING`
- `27 swere-lm1-dsp-d CONFIGURING`
- `23 swere-lm1-mimo-o RUNNING`
- `20 swere-lm1-kimi-d CONFIGURING`
- `16 swere-lm1-mimo-d PENDING`
- `14 swere-lm1-mimo-d RUNNING`
- `10 swere-lm1-kimi-o CONFIGURING`
- `7 swere-lm1-mimo-o CONFIGURING`
- `5 swere-lm1-kimi-o RUNNING`
- `4 swere-lf1-dsf-o CONFIGURING`
- `1 swere-r2-fd-och RUNNING`
- `1 swere-r2-fd-ewe RUNNING`
- `1 swere-r1-pro-eth RUNNING`
- `1 swere-lmod-dsp-d RUNNING`
## 2026-06-07 00:12 UTC

Throughput has been increased and all new generation is on CPU-only `m7i-cpu2`; no `swere*` data-generation jobs are on H200/GPU partitions.

Saved Pyxis result records from the manifest-based scan at ~00:05 UTC:

- Total saved result records: 2,532.
- Reward-passing records: 1,746.
- Saved trajectories in active Pyxis manifests: at least 2,498 at scan time; trajectories are being written before verifier results, so failures/non-passes are preserved too.
- Status counts: `reward_pass=1,746`, `completed_reward0=786`.

Pass rates by model/style from saved result records:

- `deepseek-v4-flash`, original: 660/929 reward-pass, 23,989 API calls.
- `deepseek-v4-flash`, DeepSWE: 672/1,086 reward-pass, 25,438 API calls.
- `deepseek-v4-pro`, original: 232/334 reward-pass, 8,114 API calls.
- `deepseek-v4-pro`, DeepSWE: 81/82 reward-pass, 2,257 API calls.
- `moonshotai/kimi-k2.6`, original: 21/21 reward-pass, 614 API calls.
- `moonshotai/kimi-k2.6`, DeepSWE: 16/16 reward-pass, 560 API calls.
- `xiaomi/mimo-v2.5-pro`, original: 33/33 reward-pass, 793 API calls.
- `xiaomi/mimo-v2.5-pro`, DeepSWE: 31/31 reward-pass, 748 API calls.

Language/difficulty stats for saved result records:

- Difficulty: `easy=2,532` in completed result records so far.
- Languages: `python=1,991`, `js=194`, `ts=148`, `go=137`, `rust=41`, `php=11`, `java=10`.
- Prompt styles: `original=1,317`, `deepswe=1,215`.

New submissions since the prior report:

- `datagen-20260606-pyxis-local-flake8-scale2`: 1,700 local-image rollouts; at scan time 1,031 results and 1,521 trajectories were saved. Scale2 pass rate at scan time was 1,028/1,031.
- `datagen-20260607-pyxis-local-diverse-smoke1`: 20 JS/TS easy smoke rollouts on local serverless/FHIR images.
- `datagen-20260607-pyxis-local-medium-serverless-smoke`: 20 JS medium rollouts with Kimi and MiMo, original and DeepSWE.
- `datagen-20260607-pyxis-local-diverse-scale1`: 1,200 JS/TS easy rollouts: 800 DeepSeek flash and 400 DeepSeek pro, split 50/50 original vs DeepSWE.
- `datagen-20260607-pyxis-local-moto-smoke`: 20 Python easy smoke rollouts on a local Moto image.
- `datagen-20260607-pyxis-local-medium-swc-smoke`: 20 Rust medium smoke rollouts with Kimi and MiMo, original and DeepSWE.

Docker/image status:

- New Docker credentials are installed as Enroot credential files under `/wbl-fast/usrs/ee/code-swe-data/credentials/dockerhub`; tokens are not printed in logs.
- Converted local `.sqsh` images under `/wbl-fast/usrs/ee/code-swe-data/tmp`: flake8-simplify easy, serverless easy, FHIR easy, serverless medium, Moto easy, and SWC medium.
- Broad unique-image Pyxis pulls are still risky due Docker Hub 429s, but authenticated local conversion is now working for multiple image families.

Current active queue at 00:12 UTC is CPU-only on `m7i-cpu2`. The largest active groups are the diverse JS/TS scale jobs: 800 DeepSeek flash rows active/running-or-pending and 400 DeepSeek pro rows active/running-or-pending, plus the remaining scale2 and medium smoke jobs.

Quality notes:

- Scale2 local flake8 duplicate rollouts are high quality by reward so far: 1,028/1,031 at the latest scan.
- Original and DeepSWE are effectively tied on the high-pass local-image duplicate runs, so the new larger diverse wave remains split 50/50.
- Medium Kimi/MiMo coverage is now running on two local medium images: `serverless__serverless-6545` and `swc-project__swc-4250`.

## 2026-06-07 03:58 UTC

Kimi duplicate-spend audit:

- No new Kimi jobs are being submitted. New submissions are limited to DeepSeek and MiMo.
- Submitted-manifest audit found 2,198 Kimi rows total: 1,382 medium, 630 hard, 186 easy.
- The worst concentration is `serverless__serverless-6545`: 1,360 Kimi medium rows on one task. This was caused by scaling against locally cached task images without enforcing a per-task/model cap.
- Other Kimi concentrations: `swc-project__swc-3163`, `getmoto__moto-6391`, and `swc-project__swc-2598` each have 210 hard rows; `martinthoma__flake8-simplify-124` has 186 easy rows.
- Going forward, new medium submissions use a no-prior-manifest-overlap check and one rollout per task before any duplicates.

Diverse medium corrective wave:

- Prepared a 240-task medium batch from the high-quality subset with 240 unique instances and 240 unique repos, after replacing the one previously seen task (`ruby__bigdecimal-302`) with `unidata__netcdf-c-1464`.
- Language split: `go=45`, `python=45`, `rust=40`, `ts=40`, `js=35`, `java=15`, `php=15`, `c=3`, `cpp=2`.
- Model split: `xiaomi/mimo-v2.5-pro=160`, `deepseek-v4-flash=60`, `deepseek-v4-pro=20`; Kimi is 0.
- Prompt split: `original=120`, `deepswe=120`.
- Submitted as CPU-only `m7i-cpu2` Slurm/Pyxis arrays:
  - `226118` / `swere-mdu1-och`: 80 rows, Docker-authenticated, max 25 concurrent.
  - `226119` / `swere-mdu1-ewe`: 80 rows, Docker-authenticated, max 25 concurrent.
  - `226120` / `swere-mdu1-oew`: 80 rows, Docker-authenticated, max 25 concurrent.
- Initial status: `226119` hit Docker Hub 429s after successful auth; Pyxis start-failure `result.json` records are being written with zero API calls/cost. The other two shards started running, with 50 array tasks active at the first queue check.

Current quality/throughput note:

- The immediate failure mode for the diverse medium wave is registry import pressure, not model quality. Failed container starts are recorded as failed trials, and successful starts will save full mini-swe-agent trajectories plus verifier results.
- All datagen submissions in this wave are CPU-only and write under `/wbl-fast`.

## 2026-06-07 04:52 UTC

DeepSeek non-duplicate submission:

- Selected and materialized 360 new high-quality tasks with no overlap against any prior submitted manifest and no duplicate repos within the wave.
- Difficulty split: `easy=201`, `medium=120`, `hard=39`.
- Language split: `python=80`, `go=68`, `ts=59`, `rust=55`, `js=52`, `java=18`, `php=17`, `c=7`, `cpp=4`.
- Model split: `deepseek-v4-flash=281`, `deepseek-v4-pro=79`.
- Prompt split: `original=180`, `deepswe=180`.
- Submitted as CPU-only `m7i-cpu2` Slurm/Pyxis arrays:
  - `226390` / `swere-dsu1-och`: 180 rows, Docker-authenticated, max 15 concurrent.
  - `226391` / `swere-dsu1-oew`: 180 rows, Docker-authenticated, max 15 concurrent.

Initial status:

- `swere-dsu1-och` hit Docker Hub 429 on all 180 rows after successful auth; failure `result.json` records are present with zero API calls/cost.
- `swere-dsu1-oew` is still active: 15 running and 157 pending at the latest check; 8 rows have already written Pyxis-start failure records.
- Active older datagen rows: 2 MiMo rows from `swere-mnk-mix1` remain running.

Operational note:

- New DeepSeek jobs are task-deduplicated. The bottleneck is unique Docker image imports, not DeepSeek API capacity.

## 2026-06-07 05:33 UTC

Docker reset wait / DeepSeek-only delayed wave:

- OpenRouter key is deactivated. No OpenRouter/Kimi/MiMo jobs were submitted or scheduled in this wave.
- Prepared `datagen-20260607-pyxis-deepseek-afterreset1` under `/wbl-fast`: 300 high-quality SWE-rebench V2 tasks, 300 unique instances, 300 unique repos, one rollout per task, and zero overlap with all prior submitted manifests.
- Submitted three CPU-only `m7i-cpu2` Pyxis arrays with `--begin=2026-06-07T11:32:20Z`, so no Docker pulls or DeepSeek API calls should occur until after the Docker reset window:
  - `226804` / `swere-dsr1-ewe`: 100 rows, Docker auth shard `ethanewer`, throttle 100.
  - `226805` / `swere-dsr1-och`: 100 rows, Docker auth shard `ethanoch`, throttle 100.
  - `226806` / `swere-dsr1-oew`: 100 rows, Docker auth shard `ethanoewer`, throttle 100.
- Queue status at submission: all three arrays are `PENDING` for `BeginTime` on `m7i-cpu2`; no datagen job is on H200.

Scheduled split:

- Difficulty: `easy=166`, `medium=105`, `hard=29`.
- Model: `deepseek-v4-flash=218`, `deepseek-v4-pro=82`.
- Difficulty/model: `easy/deepseek-v4-flash=166`, `medium/deepseek-v4-flash=52`, `medium/deepseek-v4-pro=53`, `hard/deepseek-v4-pro=29`.
- Prompt style: `original=150`, `deepswe=150`.
- Language: `python=65`, `go=58`, `ts=51`, `rust=46`, `js=43`, `java=15`, `php=14`, `c=5`, `cpp=3`.

Pass-rate status:

- This delayed wave has `0` completed and `0` passed so far because it has not started yet.
- Failed container starts will still write `result.json` records; successful starts will save mini-swe-agent trajectories, verifier logs, and model patches under the run root.

## 2026-06-07 05:58 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1200` across waves 1-4.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300}`.
- Scheduled by model: `{'deepseek-v4-flash': 696, 'deepseek-v4-pro': 504}`.
- Scheduled by difficulty: `{'easy': 271, 'hard': 78, 'medium': 851}`.
- Scheduled prompt styles: `{'deepswe': 600, 'original': 600}`.
- Completed result records so far: `0`; reward-pass: `0`; saved trajectories: `0`; Pyxis start failures: `0`.
- Results by model: total `{}`, reward `{}`.
- Results by difficulty: total `{}`, reward `{}`.
- Results by style: `{}`.
- Probe success state: `{}`.
- Release state: `{}`.

Queue snapshot:

```
226823_[0-99%100]  m7i-cpu2               swere-dsr3-ewe    PENDING       0:00 (BeginTime)
 226824_[0-99%100]  m7i-cpu2               swere-dsr3-och    PENDING       0:00 (BeginTime)
 226825_[0-99%100]  m7i-cpu2               swere-dsr3-oew    PENDING       0:00 (BeginTime)
 226826_[0-99%100]  m7i-cpu2               swere-dsr4-ewe    PENDING       0:00 (BeginTime)
 226827_[0-99%100]  m7i-cpu2               swere-dsr4-och    PENDING       0:00 (BeginTime)
 226828_[0-99%100]  m7i-cpu2               swere-dsr4-oew    PENDING       0:00 (BeginTime)
 226804_[0-99%100]  m7i-cpu2               swere-dsr1-ewe    PENDING       0:00 (BeginTime)
 226805_[0-99%100]  m7i-cpu2               swere-dsr1-och    PENDING       0:00 (BeginTime)
 226806_[0-99%100]  m7i-cpu2               swere-dsr1-oew    PENDING       0:00 (BeginTime)
 226820_[0-99%100]  m7i-cpu2               swere-dsr2-ewe    PENDING       0:00 (BeginTime)
 226821_[0-99%100]  m7i-cpu2               swere-dsr2-och    PENDING       0:00 (BeginTime)
 226822_[0-99%100]  m7i-cpu2               swere-dsr2-oew    PENDING       0:00 (BeginTime)
```

## 2026-06-07 06:07 UTC

DeepSeek monitor correction / early launch:

- The 12-hour monitor is running under `/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/monitor-deepseek-20260607`.
- Registry probes initially used the wrong repository path and then were corrected. The corrected probe saw authenticated Docker responses for all three credentials and released waves 1-2 at `2026-06-07 06:05:13 UTC`.
- Important caveat: the Docker remaining headers were low, not fully reset: `ethanewer=28/200`, `ethanoch=24/200`, `ethanoewer=17/200`. This means waves 1-2 may produce many Pyxis start-failure records before the full reset window.
- I tightened the monitor after that release: future early releases now require at least `120` remaining pulls per credential. Waves 3-4 remain scheduled for `2026-06-07T17:32:20Z` unless a substantial reset-sized budget appears earlier.
- Queue status shortly after release: `CONFIGURING=582`, `RUNNING=13`, `PENDING=6` among the DeepSeek `swere-dsr*` arrays.
- Result records at this check: `5` completed, `0` reward-pass, all `5` are `PyxisContainerStartError` records. These are saved under the wave run roots.

## 2026-06-07 06:16 UTC

Final strict unique-container DeepSeek wave:

- Selected the remaining high-quality tasks not already present in any submitted manifest by task/image: `73` total, all `medium/go`, with `73` unique Docker images and `42` repos.
- Submitted as CPU-only `m7i-cpu2` arrays for `2026-06-07T11:32:20Z`:
  - `227452` / `swere-dsr5-ewe`: `25` rows, Docker auth shard `ethanewer`.
  - `227453` / `swere-dsr5-och`: `24` rows, Docker auth shard `ethanoch`.
  - `227454` / `swere-dsr5-oew`: `24` rows, Docker auth shard `ethanoewer`.
- Model split: `deepseek-v4-pro=37`, `deepseek-v4-flash=36`; prompt split: `original=37`, `deepswe=36`.
- Cross-wave validation: waves 1-5 now contain `1,273` rows, `1,273` unique tasks, and `1,273` unique Docker images.
- Active foreground hourly loop is running with `timeout 3660 sleep 3600` between checks; detached backup monitor now also includes wave 5 in queue/result accounting.

## 2026-06-07 07:16 UTC

First-reset capacity fill:

- Corrected the remaining-task policy to match the latest instruction: new work must be unique by task/container image, but does not need to be unique by repo.
- Available high-quality unique-container tasks after waves 1-5: `9,294`.
- Selected and submitted `datagen-20260607-pyxis-deepseek-afterreset6`: `527` rows, sized to fill the first reset window after wave 5 (`175/176/176` rows across the three Docker credentials).
- Wave 6 split: `hard=80`, `medium=320`, `easy=127`; `deepseek-v4-pro=240`, `deepseek-v4-flash=287`; `original=264`, `deepswe=263`.
- Language split: `c=50`, `cpp=52`, `go=61`, `java=60`, `js=60`, `php=61`, `python=61`, `rust=61`, `ts=61`.
- Submitted as CPU-only `m7i-cpu2` arrays for `2026-06-07T11:32:20Z`:
  - `227504` / `swere-dsr6-ewe`: `175` rows.
  - `227505` / `swere-dsr6-och`: `176` rows.
  - `227506` / `swere-dsr6-oew`: `176` rows.
- Cross-wave validation: waves 1-6 now contain `1,800` rows, `1,800` unique tasks, and `1,800` unique Docker images.
- The foreground hourly loop remains active and sleeping one hour at a time; the backup monitor now includes waves 1-6.

## 2026-06-07 09:16 UTC

Active monitor correction:

- The foreground loop caught that the backup monitor was not probing the new wave-5/6 first-reset window because the monitor still had a hardcoded window list.
- Patched and restarted the backup monitor so it now tracks `window1b = waves 5-6`.
- Corrected window1b registry probe results at `09:15 UTC`: `ethanewer=91/200`, `ethanoch=95/200`, `ethanoewer=49/200`.
- Since those are below the `120` remaining-pull threshold, waves 5-6 were not released early and remain pending for `2026-06-07T11:32:20Z`.

## 2026-06-07 05:59 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1200` across waves 1-4.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300}`.
- Scheduled by model: `{'deepseek-v4-flash': 696, 'deepseek-v4-pro': 504}`.
- Scheduled by difficulty: `{'easy': 271, 'hard': 78, 'medium': 851}`.
- Scheduled prompt styles: `{'deepswe': 600, 'original': 600}`.
- Completed result records so far: `0`; reward-pass: `0`; saved trajectories: `0`; Pyxis start failures: `0`.
- Results by model: total `{}`, reward `{}`.
- Results by difficulty: total `{}`, reward `{}`.
- Results by style: `{}`.
- Probe success state: `{}`.
- Release state: `{}`.

Queue snapshot:

```
226823_[0-99%100]  m7i-cpu2               swere-dsr3-ewe    PENDING       0:00 (BeginTime)
 226824_[0-99%100]  m7i-cpu2               swere-dsr3-och    PENDING       0:00 (BeginTime)
 226825_[0-99%100]  m7i-cpu2               swere-dsr3-oew    PENDING       0:00 (BeginTime)
 226826_[0-99%100]  m7i-cpu2               swere-dsr4-ewe    PENDING       0:00 (BeginTime)
 226827_[0-99%100]  m7i-cpu2               swere-dsr4-och    PENDING       0:00 (BeginTime)
 226828_[0-99%100]  m7i-cpu2               swere-dsr4-oew    PENDING       0:00 (BeginTime)
 226804_[0-99%100]  m7i-cpu2               swere-dsr1-ewe    PENDING       0:00 (BeginTime)
 226805_[0-99%100]  m7i-cpu2               swere-dsr1-och    PENDING       0:00 (BeginTime)
 226806_[0-99%100]  m7i-cpu2               swere-dsr1-oew    PENDING       0:00 (BeginTime)
 226820_[0-99%100]  m7i-cpu2               swere-dsr2-ewe    PENDING       0:00 (BeginTime)
 226821_[0-99%100]  m7i-cpu2               swere-dsr2-och    PENDING       0:00 (BeginTime)
 226822_[0-99%100]  m7i-cpu2               swere-dsr2-oew    PENDING       0:00 (BeginTime)
```

## 2026-06-07 06:01 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1200` across waves 1-4.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300}`.
- Scheduled by model: `{'deepseek-v4-flash': 696, 'deepseek-v4-pro': 504}`.
- Scheduled by difficulty: `{'easy': 271, 'hard': 78, 'medium': 851}`.
- Scheduled prompt styles: `{'deepswe': 600, 'original': 600}`.
- Completed result records so far: `0`; reward-pass: `0`; saved trajectories: `0`; Pyxis start failures: `0`.
- Results by model: total `{}`, reward `{}`.
- Results by difficulty: total `{}`, reward `{}`.
- Results by style: `{}`.
- Probe success state: `{}`.
- Release state: `{}`.

Queue snapshot:

```
226823_[0-99%100]  m7i-cpu2               swere-dsr3-ewe    PENDING       0:00 (BeginTime)
 226824_[0-99%100]  m7i-cpu2               swere-dsr3-och    PENDING       0:00 (BeginTime)
 226825_[0-99%100]  m7i-cpu2               swere-dsr3-oew    PENDING       0:00 (BeginTime)
 226826_[0-99%100]  m7i-cpu2               swere-dsr4-ewe    PENDING       0:00 (BeginTime)
 226827_[0-99%100]  m7i-cpu2               swere-dsr4-och    PENDING       0:00 (BeginTime)
 226828_[0-99%100]  m7i-cpu2               swere-dsr4-oew    PENDING       0:00 (BeginTime)
 226804_[0-99%100]  m7i-cpu2               swere-dsr1-ewe    PENDING       0:00 (BeginTime)
 226805_[0-99%100]  m7i-cpu2               swere-dsr1-och    PENDING       0:00 (BeginTime)
 226806_[0-99%100]  m7i-cpu2               swere-dsr1-oew    PENDING       0:00 (BeginTime)
 226820_[0-99%100]  m7i-cpu2               swere-dsr2-ewe    PENDING       0:00 (BeginTime)
 226821_[0-99%100]  m7i-cpu2               swere-dsr2-och    PENDING       0:00 (BeginTime)
 226822_[0-99%100]  m7i-cpu2               swere-dsr2-oew    PENDING       0:00 (BeginTime)
```

## 2026-06-07 06:04 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1200` across waves 1-4.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300}`.
- Scheduled by model: `{'deepseek-v4-flash': 696, 'deepseek-v4-pro': 504}`.
- Scheduled by difficulty: `{'easy': 271, 'hard': 78, 'medium': 851}`.
- Scheduled prompt styles: `{'deepswe': 600, 'original': 600}`.
- Completed result records so far: `0`; reward-pass: `0`; saved trajectories: `0`; Pyxis start failures: `0`.
- Results by model: total `{}`, reward `{}`.
- Results by difficulty: total `{}`, reward `{}`.
- Results by style: `{}`.
- Probe success state: `{}`.
- Release state: `{}`.

Queue snapshot:

```
226823_[0-99%100]  m7i-cpu2               swere-dsr3-ewe    PENDING       0:00 (BeginTime)
 226824_[0-99%100]  m7i-cpu2               swere-dsr3-och    PENDING       0:00 (BeginTime)
 226825_[0-99%100]  m7i-cpu2               swere-dsr3-oew    PENDING       0:00 (BeginTime)
 226826_[0-99%100]  m7i-cpu2               swere-dsr4-ewe    PENDING       0:00 (BeginTime)
 226827_[0-99%100]  m7i-cpu2               swere-dsr4-och    PENDING       0:00 (BeginTime)
 226828_[0-99%100]  m7i-cpu2               swere-dsr4-oew    PENDING       0:00 (BeginTime)
 226804_[0-99%100]  m7i-cpu2               swere-dsr1-ewe    PENDING       0:00 (BeginTime)
 226805_[0-99%100]  m7i-cpu2               swere-dsr1-och    PENDING       0:00 (BeginTime)
 226806_[0-99%100]  m7i-cpu2               swere-dsr1-oew    PENDING       0:00 (BeginTime)
 226820_[0-99%100]  m7i-cpu2               swere-dsr2-ewe    PENDING       0:00 (BeginTime)
 226821_[0-99%100]  m7i-cpu2               swere-dsr2-och    PENDING       0:00 (BeginTime)
 226822_[0-99%100]  m7i-cpu2               swere-dsr2-oew    PENDING       0:00 (BeginTime)
```

## 2026-06-07 07:06 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1273` across waves 1-4.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73}`.
- Scheduled by model: `{'deepseek-v4-flash': 732, 'deepseek-v4-pro': 541}`.
- Scheduled by difficulty: `{'easy': 271, 'hard': 78, 'medium': 924}`.
- Scheduled prompt styles: `{'deepswe': 636, 'original': 637}`.
- Completed result records so far: `600`; reward-pass: `7`; saved trajectories: `0`; Pyxis start failures: `576`.
- Results by model: total `{'deepseek-v4-flash': 383, 'deepseek-v4-pro': 217}`, reward `{'deepseek-v4-flash': 5, 'deepseek-v4-pro': 2}`.
- Results by difficulty: total `{'easy': 216, 'hard': 49, 'medium': 335}`, reward `{'easy': 3, 'hard': 2, 'medium': 2}`.
- Results by style: `{'deepswe': 300, 'original': 300}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC'}`.

Queue snapshot:

```
226823_[0-99%100]  m7i-cpu2               swere-dsr3-ewe    PENDING       0:00 (BeginTime)
 226824_[0-99%100]  m7i-cpu2               swere-dsr3-och    PENDING       0:00 (BeginTime)
 226825_[0-99%100]  m7i-cpu2               swere-dsr3-oew    PENDING       0:00 (BeginTime)
 226826_[0-99%100]  m7i-cpu2               swere-dsr4-ewe    PENDING       0:00 (BeginTime)
 226827_[0-99%100]  m7i-cpu2               swere-dsr4-och    PENDING       0:00 (BeginTime)
 226828_[0-99%100]  m7i-cpu2               swere-dsr4-oew    PENDING       0:00 (BeginTime)
  227452_[0-24%25]  m7i-cpu2               swere-dsr5-ewe    PENDING       0:00 (BeginTime)
  227453_[0-23%24]  m7i-cpu2               swere-dsr5-och    PENDING       0:00 (BeginTime)
  227454_[0-23%24]  m7i-cpu2               swere-dsr5-oew    PENDING       0:00 (BeginTime)
```

## 2026-06-07 08:11 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1800` across waves 1-4.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527}`.
- Scheduled by model: `{'deepseek-v4-flash': 1019, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 398, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 899, 'original': 901}`.
- Completed result records so far: `600`; reward-pass: `7`; saved trajectories: `0`; Pyxis start failures: `576`.
- Results by model: total `{'deepseek-v4-flash': 383, 'deepseek-v4-pro': 217}`, reward `{'deepseek-v4-flash': 5, 'deepseek-v4-pro': 2}`.
- Results by difficulty: total `{'easy': 216, 'hard': 49, 'medium': 335}`, reward `{'easy': 3, 'hard': 2, 'medium': 2}`.
- Results by style: `{'deepswe': 300, 'original': 300}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC'}`.

Queue snapshot:

```
226823_[0-99%100]  m7i-cpu2               swere-dsr3-ewe    PENDING       0:00 (BeginTime)
 226824_[0-99%100]  m7i-cpu2               swere-dsr3-och    PENDING       0:00 (BeginTime)
 226825_[0-99%100]  m7i-cpu2               swere-dsr3-oew    PENDING       0:00 (BeginTime)
 226826_[0-99%100]  m7i-cpu2               swere-dsr4-ewe    PENDING       0:00 (BeginTime)
 226827_[0-99%100]  m7i-cpu2               swere-dsr4-och    PENDING       0:00 (BeginTime)
 226828_[0-99%100]  m7i-cpu2               swere-dsr4-oew    PENDING       0:00 (BeginTime)
  227452_[0-24%25]  m7i-cpu2               swere-dsr5-ewe    PENDING       0:00 (BeginTime)
  227453_[0-23%24]  m7i-cpu2               swere-dsr5-och    PENDING       0:00 (BeginTime)
  227454_[0-23%24]  m7i-cpu2               swere-dsr5-oew    PENDING       0:00 (BeginTime)
227504_[0-174%175]  m7i-cpu2               swere-dsr6-ewe    PENDING       0:00 (BeginTime)
227505_[0-175%176]  m7i-cpu2               swere-dsr6-och    PENDING       0:00 (BeginTime)
227506_[0-175%176]  m7i-cpu2               swere-dsr6-oew    PENDING       0:00 (BeginTime)
```

## 2026-06-07 09:11 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1800` across waves 1-4.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527}`.
- Scheduled by model: `{'deepseek-v4-flash': 1019, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 398, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 899, 'original': 901}`.
- Completed result records so far: `600`; reward-pass: `7`; saved trajectories: `0`; Pyxis start failures: `576`.
- Results by model: total `{'deepseek-v4-flash': 383, 'deepseek-v4-pro': 217}`, reward `{'deepseek-v4-flash': 5, 'deepseek-v4-pro': 2}`.
- Results by difficulty: total `{'easy': 216, 'hard': 49, 'medium': 335}`, reward `{'easy': 3, 'hard': 2, 'medium': 2}`.
- Results by style: `{'deepswe': 300, 'original': 300}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC'}`.

Queue snapshot:

```
226823_[0-99%100]  m7i-cpu2               swere-dsr3-ewe    PENDING       0:00 (BeginTime)
 226824_[0-99%100]  m7i-cpu2               swere-dsr3-och    PENDING       0:00 (BeginTime)
 226825_[0-99%100]  m7i-cpu2               swere-dsr3-oew    PENDING       0:00 (BeginTime)
 226826_[0-99%100]  m7i-cpu2               swere-dsr4-ewe    PENDING       0:00 (BeginTime)
 226827_[0-99%100]  m7i-cpu2               swere-dsr4-och    PENDING       0:00 (BeginTime)
 226828_[0-99%100]  m7i-cpu2               swere-dsr4-oew    PENDING       0:00 (BeginTime)
  227452_[0-24%25]  m7i-cpu2               swere-dsr5-ewe    PENDING       0:00 (BeginTime)
  227453_[0-23%24]  m7i-cpu2               swere-dsr5-och    PENDING       0:00 (BeginTime)
  227454_[0-23%24]  m7i-cpu2               swere-dsr5-oew    PENDING       0:00 (BeginTime)
227504_[0-174%175]  m7i-cpu2               swere-dsr6-ewe    PENDING       0:00 (BeginTime)
227505_[0-175%176]  m7i-cpu2               swere-dsr6-och    PENDING       0:00 (BeginTime)
227506_[0-175%176]  m7i-cpu2               swere-dsr6-oew    PENDING       0:00 (BeginTime)
```

## 2026-06-07 10:15 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1800` across waves 1-4.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527}`.
- Scheduled by model: `{'deepseek-v4-flash': 1019, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 398, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 899, 'original': 901}`.
- Completed result records so far: `600`; reward-pass: `7`; saved trajectories: `0`; Pyxis start failures: `576`.
- Results by model: total `{'deepseek-v4-flash': 383, 'deepseek-v4-pro': 217}`, reward `{'deepseek-v4-flash': 5, 'deepseek-v4-pro': 2}`.
- Results by difficulty: total `{'easy': 216, 'hard': 49, 'medium': 335}`, reward `{'easy': 3, 'hard': 2, 'medium': 2}`.
- Results by style: `{'deepswe': 300, 'original': 300}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC'}`.

Queue snapshot:

```
226823_[0-99%100]  m7i-cpu2               swere-dsr3-ewe    PENDING       0:00 (BeginTime)
 226824_[0-99%100]  m7i-cpu2               swere-dsr3-och    PENDING       0:00 (BeginTime)
 226825_[0-99%100]  m7i-cpu2               swere-dsr3-oew    PENDING       0:00 (BeginTime)
 226826_[0-99%100]  m7i-cpu2               swere-dsr4-ewe    PENDING       0:00 (BeginTime)
 226827_[0-99%100]  m7i-cpu2               swere-dsr4-och    PENDING       0:00 (BeginTime)
 226828_[0-99%100]  m7i-cpu2               swere-dsr4-oew    PENDING       0:00 (BeginTime)
  227452_[0-24%25]  m7i-cpu2               swere-dsr5-ewe    PENDING       0:00 (BeginTime)
  227453_[0-23%24]  m7i-cpu2               swere-dsr5-och    PENDING       0:00 (BeginTime)
  227454_[0-23%24]  m7i-cpu2               swere-dsr5-oew    PENDING       0:00 (BeginTime)
227504_[0-174%175]  m7i-cpu2               swere-dsr6-ewe    PENDING       0:00 (BeginTime)
227505_[0-175%176]  m7i-cpu2               swere-dsr6-och    PENDING       0:00 (BeginTime)
227506_[0-175%176]  m7i-cpu2               swere-dsr6-oew    PENDING       0:00 (BeginTime)
```

## 2026-06-07 11:15 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1800` across waves 1-4.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527}`.
- Scheduled by model: `{'deepseek-v4-flash': 1019, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 398, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 899, 'original': 901}`.
- Completed result records so far: `600`; reward-pass: `7`; saved trajectories: `0`; Pyxis start failures: `576`.
- Results by model: total `{'deepseek-v4-flash': 383, 'deepseek-v4-pro': 217}`, reward `{'deepseek-v4-flash': 5, 'deepseek-v4-pro': 2}`.
- Results by difficulty: total `{'easy': 216, 'hard': 49, 'medium': 335}`, reward `{'easy': 3, 'hard': 2, 'medium': 2}`.
- Results by style: `{'deepswe': 300, 'original': 300}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC'}`.

Queue snapshot:

```
226823_[0-99%100]  m7i-cpu2               swere-dsr3-ewe    PENDING       0:00 (BeginTime)
 226824_[0-99%100]  m7i-cpu2               swere-dsr3-och    PENDING       0:00 (BeginTime)
 226825_[0-99%100]  m7i-cpu2               swere-dsr3-oew    PENDING       0:00 (BeginTime)
 226826_[0-99%100]  m7i-cpu2               swere-dsr4-ewe    PENDING       0:00 (BeginTime)
 226827_[0-99%100]  m7i-cpu2               swere-dsr4-och    PENDING       0:00 (BeginTime)
 226828_[0-99%100]  m7i-cpu2               swere-dsr4-oew    PENDING       0:00 (BeginTime)
  227452_[0-24%25]  m7i-cpu2               swere-dsr5-ewe    PENDING       0:00 (BeginTime)
  227453_[0-23%24]  m7i-cpu2               swere-dsr5-och    PENDING       0:00 (BeginTime)
  227454_[0-23%24]  m7i-cpu2               swere-dsr5-oew    PENDING       0:00 (BeginTime)
227504_[0-174%175]  m7i-cpu2               swere-dsr6-ewe    PENDING       0:00 (BeginTime)
227505_[0-175%176]  m7i-cpu2               swere-dsr6-och    PENDING       0:00 (BeginTime)
227506_[0-175%176]  m7i-cpu2               swere-dsr6-oew    PENDING       0:00 (BeginTime)
```

## 2026-06-07 12:17 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1800` across waves 1-4.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527}`.
- Scheduled by model: `{'deepseek-v4-flash': 1019, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 398, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 899, 'original': 901}`.
- Completed result records so far: `1197`; reward-pass: `16`; saved trajectories: `105`; Pyxis start failures: `1092`.
- Results by model: total `{'deepseek-v4-flash': 703, 'deepseek-v4-pro': 494}`, reward `{'deepseek-v4-flash': 11, 'deepseek-v4-pro': 5}`.
- Results by difficulty: total `{'easy': 342, 'hard': 129, 'medium': 726}`, reward `{'easy': 6, 'hard': 3, 'medium': 7}`.
- Results by style: `{'deepswe': 597, 'original': 600}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': False}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC'}`.

Queue snapshot:

```
226823_[0-99%100]  m7i-cpu2               swere-dsr3-ewe    PENDING       0:00 (BeginTime)
 226824_[0-99%100]  m7i-cpu2               swere-dsr3-och    PENDING       0:00 (BeginTime)
 226825_[0-99%100]  m7i-cpu2               swere-dsr3-oew    PENDING       0:00 (BeginTime)
 226826_[0-99%100]  m7i-cpu2               swere-dsr4-ewe    PENDING       0:00 (BeginTime)
 226827_[0-99%100]  m7i-cpu2               swere-dsr4-och    PENDING       0:00 (BeginTime)
 226828_[0-99%100]  m7i-cpu2               swere-dsr4-oew    PENDING       0:00 (BeginTime)
         227504_31  m7i-cpu2               swere-dsr6-ewe    RUNNING      40:34 m7i-cpu2-dy-m7i-cpu-cr-0-246
        227504_142  m7i-cpu2               swere-dsr6-ewe    RUNNING      40:34 m7i-cpu2-dy-m7i-cpu-cr-0-300
         227506_63  m7i-cpu2               swere-dsr6-oew    RUNNING      40:34 m7i-cpu2-dy-m7i-cpu-cr-0-239
```

## 2026-06-07 13:03 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1800` across waves 1-6.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527}`.
- Scheduled by model: `{'deepseek-v4-flash': 1019, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 398, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 899, 'original': 901}`.
- Completed result records so far: `1200`; reward-pass: `17`; saved trajectories: `108`; Pyxis start failures: `1092`.
- Results by model: total `{'deepseek-v4-flash': 706, 'deepseek-v4-pro': 494}`, reward `{'deepseek-v4-flash': 12, 'deepseek-v4-pro': 5}`.
- Results by difficulty: total `{'easy': 343, 'hard': 129, 'medium': 728}`, reward `{'easy': 6, 'hard': 3, 'medium': 8}`.
- Results by style: `{'deepswe': 599, 'original': 601}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': False}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC'}`.

Queue snapshot:

```
226823_[0-99%100]  m7i-cpu2               swere-dsr3-ewe    PENDING       0:00 (BeginTime)
 226824_[0-99%100]  m7i-cpu2               swere-dsr3-och    PENDING       0:00 (BeginTime)
 226825_[0-99%100]  m7i-cpu2               swere-dsr3-oew    PENDING       0:00 (BeginTime)
 226826_[0-99%100]  m7i-cpu2               swere-dsr4-ewe    PENDING       0:00 (BeginTime)
 226827_[0-99%100]  m7i-cpu2               swere-dsr4-och    PENDING       0:00 (BeginTime)
 226828_[0-99%100]  m7i-cpu2               swere-dsr4-oew    PENDING       0:00 (BeginTime)
```

## 2026-06-07 13:18 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1800` across waves 1-6.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527}`.
- Scheduled by model: `{'deepseek-v4-flash': 1019, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 398, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 899, 'original': 901}`.
- Completed result records so far: `1200`; reward-pass: `17`; saved trajectories: `108`; Pyxis start failures: `1092`.
- Results by model: total `{'deepseek-v4-flash': 706, 'deepseek-v4-pro': 494}`, reward `{'deepseek-v4-flash': 12, 'deepseek-v4-pro': 5}`.
- Results by difficulty: total `{'easy': 343, 'hard': 129, 'medium': 728}`, reward `{'easy': 6, 'hard': 3, 'medium': 8}`.
- Results by style: `{'deepswe': 599, 'original': 601}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': False}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC'}`.

Queue snapshot:

```
226823_[0-99%100]  m7i-cpu2               swere-dsr3-ewe    PENDING       0:00 (BeginTime)
 226824_[0-99%100]  m7i-cpu2               swere-dsr3-och    PENDING       0:00 (BeginTime)
 226825_[0-99%100]  m7i-cpu2               swere-dsr3-oew    PENDING       0:00 (BeginTime)
 226826_[0-99%100]  m7i-cpu2               swere-dsr4-ewe    PENDING       0:00 (BeginTime)
 226827_[0-99%100]  m7i-cpu2               swere-dsr4-och    PENDING       0:00 (BeginTime)
 226828_[0-99%100]  m7i-cpu2               swere-dsr4-oew    PENDING       0:00 (BeginTime)
```

## 2026-06-07 14:18 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1800` across waves 1-6.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527}`.
- Scheduled by model: `{'deepseek-v4-flash': 1019, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 398, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 899, 'original': 901}`.
- Completed result records so far: `1707`; reward-pass: `18`; saved trajectories: `117`; Pyxis start failures: `1590`.
- Results by model: total `{'deepseek-v4-flash': 966, 'deepseek-v4-pro': 741}`, reward `{'deepseek-v4-flash': 12, 'deepseek-v4-pro': 6}`.
- Results by difficulty: total `{'easy': 386, 'hard': 153, 'medium': 1168}`, reward `{'easy': 6, 'hard': 3, 'medium': 9}`.
- Results by style: `{'deepswe': 859, 'original': 848}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC', 'window2': '2026-06-07 14:08:23 UTC'}`.

Queue snapshot:

```
226827_0  m7i-cpu2               swere-dsr4-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-124
          226823_5  m7i-cpu2               swere-dsr3-ewe    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-83
         226823_51  m7i-cpu2               swere-dsr3-ewe    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-77
         226823_78  m7i-cpu2               swere-dsr3-ewe    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-105
         226823_80  m7i-cpu2               swere-dsr3-ewe    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-105
         226823_81  m7i-cpu2               swere-dsr3-ewe    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-105
         226823_95  m7i-cpu2               swere-dsr3-ewe    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-92
         226823_96  m7i-cpu2               swere-dsr3-ewe    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-92
         226824_15  m7i-cpu2               swere-dsr3-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-97
         226824_27  m7i-cpu2               swere-dsr3-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-100
         226824_29  m7i-cpu2               swere-dsr3-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-100
         226824_46  m7i-cpu2               swere-dsr3-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-63
         226824_48  m7i-cpu2               swere-dsr3-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-63
         226824_49  m7i-cpu2               swere-dsr3-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-63
         226824_87  m7i-cpu2               swere-dsr3-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-35
         226824_88  m7i-cpu2               swere-dsr3-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-35
         226825_14  m7i-cpu2               swere-dsr3-oew    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-42
         226825_43  m7i-cpu2               swere-dsr3-oew    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-285
         226825_45  m7i-cpu2               swere-dsr3-oew    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-285
         226825_96  m7i-cpu2               swere-dsr3-oew    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-298
         226825_97  m7i-cpu2               swere-dsr3-oew    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-298
         226826_74  m7i-cpu2               swere-dsr4-ewe    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-118
         226826_75  m7i-cpu2               swere-dsr4-ewe    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-118
         226826_80  m7i-cpu2               swere-dsr4-ewe    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-119
         226826_92  m7i-cpu2               swere-dsr4-ewe    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-122
         226826_98  m7i-cpu2               swere-dsr4-ewe    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-124
          226827_1  m7i-cpu2               swere-dsr4-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-124
         226827_51  m7i-cpu2               swere-dsr4-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-137
         226827_57  m7i-cpu2               swere-dsr4-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-138
         226827_62  m7i-cpu2               swere-dsr4-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-6
         226827_65  m7i-cpu2               swere-dsr4-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-6
         226827_81  m7i-cpu2               swere-dsr4-och    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-10
         226828_97  m7i-cpu2               swere-dsr4-oew    RUNNING       5:48 m7i-cpu2-dy-m7i-cpu-cr-0-250
         226823_10  m7i-cpu2               swere-dsr3-ewe    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-116
         226823_20  m7i-cpu2               swere-dsr3-ewe    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-3
         226823_27  m7i-cpu2               swere-dsr3-ewe    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-87
         226823_62  m7i-cpu2               swere-dsr3-ewe    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-80
         226823_70  m7i-cpu2               swere-dsr3-ewe    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-103
         226823_76  m7i-cpu2               swere-dsr3-ewe    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-104
         226823_98  m7i-cpu2               swere-dsr3-ewe    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-93
         226823_99  m7i-cpu2               swere-dsr3-ewe    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-93
          226824_3  m7i-cpu2               swere-dsr3-och    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-94
          226824_9  m7i-cpu2               swere-dsr3-och    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-95
         226824_21  m7i-cpu2               swere-dsr3-och    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-98
         226824_42  m7i-cpu2               swere-dsr3-och    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-62
         226824_44  m7i-cpu2               swere-dsr3-och    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-62
         226824_56  m7i-cpu2               swere-dsr3-och    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-65
         226824_66  m7i-cpu2               swere-dsr3-och    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-68
         226824_72  m7i-cpu2               swere-dsr3-och    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-69
         226824_73  m7i-cpu2               swere-dsr3-och    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-69
         226824_74  m7i-cpu2               swere-dsr3-och    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-70
         226824_79  m7i-cpu2               swere-dsr3-och    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-71
         226824_81  m7i-cpu2               swere-dsr3-och    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-71
         226825_27  m7i-cpu2               swere-dsr3-oew    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-45
         226825_33  m7i-cpu2               swere-dsr3-oew    RUNNING       6:18 m7i-cpu2-dy-m7i-cpu-cr-0-46
         226825_37  m7i-cpu2               swere-ds
```

## 2026-06-07 15:18 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1800` across waves 1-6.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527}`.
- Scheduled by model: `{'deepseek-v4-flash': 1019, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 398, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 899, 'original': 901}`.
- Completed result records so far: `1799`; reward-pass: `32`; saved trajectories: `209`; Pyxis start failures: `1590`.
- Results by model: total `{'deepseek-v4-flash': 1018, 'deepseek-v4-pro': 781}`, reward `{'deepseek-v4-flash': 18, 'deepseek-v4-pro': 14}`.
- Results by difficulty: total `{'easy': 398, 'hard': 158, 'medium': 1243}`, reward `{'easy': 9, 'hard': 4, 'medium': 19}`.
- Results by style: `{'deepswe': 898, 'original': 901}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC', 'window2': '2026-06-07 14:08:23 UTC'}`.

Queue snapshot:

```
226824_44  m7i-cpu2               swere-dsr3-och    RUNNING    1:06:23 m7i-cpu2-dy-m7i-cpu-cr-0-62
```

## 2026-06-07 15:37 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1920` across waves 1-7.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527, 'wave7': 120}`.
- Scheduled by model: `{'deepseek-v4-flash': 1139, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 518, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 959, 'original': 961}`.
- Completed result records so far: `1801`; reward-pass: `32`; saved trajectories: `210`; Pyxis start failures: `1591`.
- Results by model: total `{'deepseek-v4-flash': 1020, 'deepseek-v4-pro': 781}`, reward `{'deepseek-v4-flash': 18, 'deepseek-v4-pro': 14}`.
- Results by difficulty: total `{'easy': 399, 'hard': 158, 'medium': 1244}`, reward `{'easy': 9, 'hard': 4, 'medium': 19}`.
- Results by style: `{'deepswe': 900, 'original': 901}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC', 'window2': '2026-06-07 14:08:23 UTC'}`.

Queue snapshot:

```
228976_0  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-9
          228976_1  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-10
          228976_2  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-10
          228976_3  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-10
          228976_4  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-10
          228976_5  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-11
          228976_6  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-11
          228976_7  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-11
          228976_8  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-11
          228976_9  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-12
         228976_10  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-12
         228976_11  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-12
         228976_12  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-12
         228976_13  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-74
         228976_14  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-74
         228976_15  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-74
         228976_16  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-74
         228976_17  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-75
         228976_18  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-75
         228976_19  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-75
         228976_20  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-75
         228976_21  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-76
         228976_22  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-76
         228976_23  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-76
         228976_24  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-76
         228976_25  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-77
         228976_26  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-77
         228976_27  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-77
         228976_28  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-77
         228976_29  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-78
         228976_30  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-78
         228976_31  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-78
         228976_32  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-78
         228976_33  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-79
         228976_34  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-79
         228976_35  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-79
         228976_36  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-79
         228976_37  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-80
         228976_38  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-80
         228976_39  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-80
         228976_40  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-80
         228976_41  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-102
         228976_42  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-102
         228976_43  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-102
         228976_44  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-102
         228976_45  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-103
         228976_46  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-103
         228976_47  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-103
         228976_48  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-103
         228976_49  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-104
         228976_50  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-104
         228976_51  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-104
         228976_52  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-104
         228976_53  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-105
         228976_54  m7i-cpu2               swere-dsr7-oew CONFIGURIN       1:05 m7i-cpu2-dy-m7i-cpu-cr-0-105
         228976_55  m7i-cpu2               swere-dsr7-oew
```

## 2026-06-07 16:31 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1995` across waves 1-8.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527, 'wave7': 120, 'wave8': 75}`.
- Scheduled by model: `{'deepseek-v4-flash': 1214, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 593, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 997, 'original': 998}`.
- Completed result records so far: `1920`; reward-pass: `55`; saved trajectories: `270`; Pyxis start failures: `1650`.
- Results by model: total `{'deepseek-v4-flash': 1139, 'deepseek-v4-pro': 781}`, reward `{'deepseek-v4-flash': 41, 'deepseek-v4-pro': 14}`.
- Results by difficulty: total `{'easy': 518, 'hard': 158, 'medium': 1244}`, reward `{'easy': 32, 'hard': 4, 'medium': 19}`.
- Results by style: `{'deepswe': 959, 'original': 961}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC', 'window2': '2026-06-07 14:08:23 UTC'}`.

Queue snapshot:

```
229071_0  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229072_0  m7i-cpu2               swere-dsr8-ewe CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-78
         229070_16  m7i-cpu2               swere-dsr8-och CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-74
         229070_17  m7i-cpu2               swere-dsr8-och CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-74
         229070_18  m7i-cpu2               swere-dsr8-och CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-74
         229070_19  m7i-cpu2               swere-dsr8-och CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-74
          229071_1  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229071_2  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229071_3  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229071_4  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-116
          229071_5  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-116
          229071_6  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-116
          229071_7  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-116
          229071_8  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-2
          229071_9  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-2
         229071_10  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-2
         229071_11  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-2
         229071_12  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-3
         229071_13  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-3
         229071_14  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-3
         229071_15  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-3
         229071_16  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-4
         229071_17  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-4
         229071_18  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-4
         229071_19  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-4
         229071_20  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-87
         229071_21  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-87
         229071_22  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-87
         229071_23  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-87
         229071_24  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-88
         229071_25  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-88
         229071_26  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-88
         229071_27  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-88
         229071_28  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-89
         229071_29  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-89
         229071_30  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-89
         229071_31  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-89
         229071_32  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-90
         229071_33  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-90
         229071_34  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-90
         229071_35  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-90
         229071_36  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-76
         229071_37  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-76
         229071_38  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-76
         229071_39  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-76
         229071_40  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-77
         229071_41  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-77
         229071_42  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-77
         229071_43  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-77
         229071_44  m7i-cpu2               swere-dsr8-oew CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-78
          229072_1  m7i-cpu2               swere-dsr8-ewe CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-78
          229072_2  m7i-cpu2               swere-dsr8-ewe CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-78
          229072_3  m7i-cpu2               swere-dsr8-ewe CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-79
          229072_4  m7i-cpu2               swere-dsr8-ewe CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-79
          229072_5  m7i-cpu2               swere-dsr8-ewe CONFIGURIN       0:32 m7i-cpu2-dy-m7i-cpu-cr-0-79
          229072_6  m7i-cpu2               swere-dsr8-ewe CONFIGURIN      
```

## 2026-06-07 17:01 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `1995` across waves 1-8.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527, 'wave7': 120, 'wave8': 75}`.
- Scheduled by model: `{'deepseek-v4-flash': 1214, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 593, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 997, 'original': 998}`.
- Completed result records so far: `1995`; reward-pass: `74`; saved trajectories: `323`; Pyxis start failures: `1672`.
- Results by model: total `{'deepseek-v4-flash': 1214, 'deepseek-v4-pro': 781}`, reward `{'deepseek-v4-flash': 60, 'deepseek-v4-pro': 14}`.
- Results by difficulty: total `{'easy': 593, 'hard': 158, 'medium': 1244}`, reward `{'easy': 51, 'hard': 4, 'medium': 19}`.
- Results by style: `{'deepswe': 997, 'original': 998}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC', 'window2': '2026-06-07 14:08:23 UTC'}`.

Queue snapshot:

```
No scheduled DeepSeek datagen jobs currently visible in squeue.
```

## 2026-06-07 17:04 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `2037` across waves 1-9.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527, 'wave7': 120, 'wave8': 75, 'wave9': 42}`.
- Scheduled by model: `{'deepseek-v4-flash': 1256, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 635, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 1018, 'original': 1019}`.
- Completed result records so far: `1996`; reward-pass: `74`; saved trajectories: `323`; Pyxis start failures: `1673`.
- Results by model: total `{'deepseek-v4-flash': 1215, 'deepseek-v4-pro': 781}`, reward `{'deepseek-v4-flash': 60, 'deepseek-v4-pro': 14}`.
- Results by difficulty: total `{'easy': 594, 'hard': 158, 'medium': 1244}`, reward `{'easy': 51, 'hard': 4, 'medium': 19}`.
- Results by style: `{'deepswe': 998, 'original': 998}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC', 'window2': '2026-06-07 14:08:23 UTC'}`.

Queue snapshot:

```
229166_0  m7i-cpu2               swere-dsr9-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-90
         229165_14  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-74
         229165_15  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-74
         229165_16  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-74
         229165_17  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-74
         229165_18  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-87
         229165_19  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-87
         229165_20  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-87
         229165_21  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-87
         229165_22  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-89
         229165_23  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-89
         229165_24  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-89
         229165_25  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-89
         229165_26  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-90
         229165_27  m7i-cpu2               swere-dsr9-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-90
          229166_1  m7i-cpu2               swere-dsr9-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-90
          229166_2  m7i-cpu2               swere-dsr9-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229166_3  m7i-cpu2               swere-dsr9-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229166_4  m7i-cpu2               swere-dsr9-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229166_5  m7i-cpu2               swere-dsr9-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229165_0  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-88
          229164_1  m7i-cpu2               swere-dsr9-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-91
          229164_2  m7i-cpu2               swere-dsr9-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-13
          229164_3  m7i-cpu2               swere-dsr9-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-13
          229164_4  m7i-cpu2               swere-dsr9-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-75
          229164_5  m7i-cpu2               swere-dsr9-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-75
          229164_6  m7i-cpu2               swere-dsr9-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-75
          229164_7  m7i-cpu2               swere-dsr9-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-88
          229165_1  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-88
          229165_2  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-88
          229165_3  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-112
          229165_4  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-112
          229165_5  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-111
          229165_6  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-111
          229165_7  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-111
          229165_8  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-81
          229165_9  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-82
         229165_10  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-84
         229165_11  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-83
         229165_12  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-83
         229165_13  m7i-cpu2               swere-dsr9-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-83
```

## 2026-06-07 17:38 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `2037` across waves 1-9.
- Scheduled by wave: `{'wave1': 300, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527, 'wave7': 120, 'wave8': 75, 'wave9': 42}`.
- Scheduled by model: `{'deepseek-v4-flash': 1256, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 635, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 1018, 'original': 1019}`.
- Completed result records so far: `2037`; reward-pass: `90`; saved trajectories: `359`; Pyxis start failures: `1678`.
- Results by model: total `{'deepseek-v4-flash': 1256, 'deepseek-v4-pro': 781}`, reward `{'deepseek-v4-flash': 76, 'deepseek-v4-pro': 14}`.
- Results by difficulty: total `{'easy': 635, 'hard': 158, 'medium': 1244}`, reward `{'easy': 67, 'hard': 4, 'medium': 19}`.
- Results by style: `{'deepswe': 1018, 'original': 1019}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC', 'window2': '2026-06-07 14:08:23 UTC'}`.

Queue snapshot:

```
No scheduled DeepSeek datagen jobs currently visible in squeue.
```

## 2026-06-07 17:41 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `2079` across waves 1-10.
- Scheduled by wave: `{'wave1': 300, 'wave10': 42, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527, 'wave7': 120, 'wave8': 75, 'wave9': 42}`.
- Scheduled by model: `{'deepseek-v4-flash': 1298, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 677, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 1039, 'original': 1040}`.
- Completed result records so far: `2037`; reward-pass: `90`; saved trajectories: `359`; Pyxis start failures: `1678`.
- Results by model: total `{'deepseek-v4-flash': 1256, 'deepseek-v4-pro': 781}`, reward `{'deepseek-v4-flash': 76, 'deepseek-v4-pro': 14}`.
- Results by difficulty: total `{'easy': 635, 'hard': 158, 'medium': 1244}`, reward `{'easy': 67, 'hard': 4, 'medium': 19}`.
- Results by style: `{'deepswe': 1018, 'original': 1019}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC', 'window2': '2026-06-07 14:08:23 UTC'}`.

Queue snapshot:

```
229212_0  m7i-cpu2              swere-dsr10-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-87
         229211_13  m7i-cpu2              swere-dsr10-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-74
         229211_14  m7i-cpu2              swere-dsr10-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-74
         229211_15  m7i-cpu2              swere-dsr10-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-74
         229211_16  m7i-cpu2              swere-dsr10-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-74
         229211_17  m7i-cpu2              swere-dsr10-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-87
          229212_1  m7i-cpu2              swere-dsr10-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-87
          229212_2  m7i-cpu2              swere-dsr10-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-87
          229212_3  m7i-cpu2              swere-dsr10-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-88
          229212_4  m7i-cpu2              swere-dsr10-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-88
          229212_5  m7i-cpu2              swere-dsr10-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-88
          229212_6  m7i-cpu2              swere-dsr10-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-88
          229212_7  m7i-cpu2              swere-dsr10-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229212_8  m7i-cpu2              swere-dsr10-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229212_9  m7i-cpu2              swere-dsr10-och CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229211_0  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-84
          229210_1  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-140
          229210_2  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-75
          229210_3  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-75
          229210_4  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-75
          229210_5  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-112
          229210_6  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-112
          229210_7  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-111
          229210_8  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-111
          229210_9  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-111
         229210_10  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-82
         229210_11  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-81
         229210_12  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-81
         229210_13  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-84
          229211_1  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-83
          229211_2  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-83
          229211_3  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-83
          229211_4  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-91
          229211_5  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-89
          229211_6  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-89
          229211_7  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-89
          229211_8  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-89
          229211_9  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-90
         229211_10  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-90
         229211_11  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-90
         229211_12  m7i-cpu2              swere-dsr10-oew    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-90
          229210_0  m7i-cpu2              swere-dsr10-ewe    RUNNING       0:30 m7i-cpu2-dy-m7i-cpu-cr-0-86
```

## 2026-06-07 18:09 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `2079` across waves 1-10.
- Scheduled by wave: `{'wave1': 300, 'wave10': 42, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527, 'wave7': 120, 'wave8': 75, 'wave9': 42}`.
- Scheduled by model: `{'deepseek-v4-flash': 1298, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 677, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 1039, 'original': 1040}`.
- Completed result records so far: `2079`; reward-pass: `97`; saved trajectories: `388`; Pyxis start failures: `1689`.
- Results by model: total `{'deepseek-v4-flash': 1298, 'deepseek-v4-pro': 781}`, reward `{'deepseek-v4-flash': 83, 'deepseek-v4-pro': 14}`.
- Results by difficulty: total `{'easy': 677, 'hard': 158, 'medium': 1244}`, reward `{'easy': 74, 'hard': 4, 'medium': 19}`.
- Results by style: `{'deepswe': 1039, 'original': 1040}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC', 'window2': '2026-06-07 14:08:23 UTC'}`.

Queue snapshot:

```
No scheduled DeepSeek datagen jobs currently visible in squeue.
```

## 2026-06-07 18:13 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `2127` across waves 1-11.
- Scheduled by wave: `{'wave1': 300, 'wave10': 42, 'wave11': 48, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527, 'wave7': 120, 'wave8': 75, 'wave9': 42}`.
- Scheduled by model: `{'deepseek-v4-flash': 1346, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 725, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 1063, 'original': 1064}`.
- Completed result records so far: `2079`; reward-pass: `97`; saved trajectories: `388`; Pyxis start failures: `1689`.
- Results by model: total `{'deepseek-v4-flash': 1298, 'deepseek-v4-pro': 781}`, reward `{'deepseek-v4-flash': 83, 'deepseek-v4-pro': 14}`.
- Results by difficulty: total `{'easy': 677, 'hard': 158, 'medium': 1244}`, reward `{'easy': 74, 'hard': 4, 'medium': 19}`.
- Results by style: `{'deepswe': 1039, 'original': 1040}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC', 'window2': '2026-06-07 14:08:23 UTC'}`.

Queue snapshot:

```
229254_0  m7i-cpu2              swere-dsr11-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-2
          229253_3  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-74
          229253_4  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-74
          229253_5  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-74
          229253_6  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-74
          229253_7  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229253_8  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-115
          229253_9  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-115
         229253_10  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-115
         229253_11  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-116
         229253_12  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-116
         229253_13  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-116
         229253_14  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-116
         229253_15  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-2
         229253_16  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-2
         229253_17  m7i-cpu2              swere-dsr11-ewe CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-2
          229254_1  m7i-cpu2              swere-dsr11-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-3
          229254_2  m7i-cpu2              swere-dsr11-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-3
          229254_3  m7i-cpu2              swere-dsr11-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-3
          229254_4  m7i-cpu2              swere-dsr11-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-3
          229254_5  m7i-cpu2              swere-dsr11-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-4
          229254_6  m7i-cpu2              swere-dsr11-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-4
          229254_7  m7i-cpu2              swere-dsr11-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-4
          229254_8  m7i-cpu2              swere-dsr11-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-4
          229254_9  m7i-cpu2              swere-dsr11-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-87
         229254_10  m7i-cpu2              swere-dsr11-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-87
         229254_11  m7i-cpu2              swere-dsr11-oew CONFIGURIN       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-87
          229253_0  m7i-cpu2              swere-dsr11-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-83
          229252_1  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-114
          229252_2  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-140
          229252_3  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-91
          229252_4  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-91
          229252_5  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-75
          229252_6  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-75
          229252_7  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-75
          229252_8  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-112
          229252_9  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-112
         229252_10  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-111
         229252_11  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-111
         229252_12  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-111
         229252_13  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-82
         229252_14  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-81
         229252_15  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-81
         229252_16  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-84
         229252_17  m7i-cpu2              swere-dsr11-och    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-84
          229253_1  m7i-cpu2              swere-dsr11-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-83
          229253_2  m7i-cpu2              swere-dsr11-ewe    RUNNING       0:29 m7i-cpu2-dy-m7i-cpu-cr-0-83
          229252_0  m7i-cpu2              swere-dsr11-och    RUNNING       0:30 m7i-cpu2-dy-m7i-cpu-cr-0-86
```

## 2026-06-07 19:14 UTC

DeepSeek Docker-reset monitor update:

- Scheduled DeepSeek-only unique-container trials: `2127` across waves 1-11.
- Scheduled by wave: `{'wave1': 300, 'wave10': 42, 'wave11': 48, 'wave2': 300, 'wave3': 300, 'wave4': 300, 'wave5': 73, 'wave6': 527, 'wave7': 120, 'wave8': 75, 'wave9': 42}`.
- Scheduled by model: `{'deepseek-v4-flash': 1346, 'deepseek-v4-pro': 781}`.
- Scheduled by difficulty: `{'easy': 725, 'hard': 158, 'medium': 1244}`.
- Scheduled prompt styles: `{'deepswe': 1063, 'original': 1064}`.
- Completed result records so far: `2127`; reward-pass: `107`; saved trajectories: `427`; Pyxis start failures: `1698`.
- Results by model: total `{'deepseek-v4-flash': 1346, 'deepseek-v4-pro': 781}`, reward `{'deepseek-v4-flash': 93, 'deepseek-v4-pro': 14}`.
- Results by difficulty: total `{'easy': 725, 'hard': 158, 'medium': 1244}`, reward `{'easy': 84, 'hard': 4, 'medium': 19}`.
- Results by style: `{'deepswe': 1063, 'original': 1064}`.
- Probe success state: `{'window1:ewe': True, 'window1:och': True, 'window1:oew': True, 'window1b:ewe': True, 'window1b:och': True, 'window1b:oew': False, 'window2:ewe': True, 'window2:och': True, 'window2:oew': True}`.
- Release state: `{'window1': '2026-06-07 06:05:13 UTC', 'window1b': 'scheduled-start 2026-06-07 11:35:19 UTC', 'window2': '2026-06-07 14:08:23 UTC'}`.

Queue snapshot:

```
No scheduled DeepSeek datagen jobs currently visible in squeue.
```
