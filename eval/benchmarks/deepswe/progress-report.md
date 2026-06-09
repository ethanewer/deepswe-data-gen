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

## 2026-06-08 07:09 UTC

DeepSeek high-reasoning coverage recovery:

- Active objective: 50% high-quality task coverage for every difficulty using DeepSeek only, with `deepseek-v4-flash` for easy and `deepseek-v4-pro` for medium/hard. All new jobs use thinking enabled, `reasoning_effort=high`, `max_tokens=16384`, and CPU-only `m7i-cpu2`.
- Fixed the dependency-overlay startup issue by using the immutable Pyxis dependency overlay. Current r13/r14 startup samples have absolute `/wbl-fast/...` workspaces, Docker auth and pull success, and `dockerd://` Pyxis imports in progress.
- Submitted hard top-up `r12`: job `290710`, 120 unique hard tasks, DeepSeek v4 Pro. Current active hard coverage is 114 queued/running unique tasks.
- Submitted medium top-ups with corrected absolute workspaces: jobs `291950` and `291951` (`r13`, 900 rows each) plus job `292435` (`r14`, 900 rows). Current active medium coverage is 5,361 queued/running unique tasks.
- Submitted easy top-up with corrected absolute workspaces: job `292204` (`r13`, 1,000 rows). Current active easy coverage is 2,377 queued/running unique tasks.
- I canceled the bad `r12` easy/medium arrays (`290792`, `290966`, `291088`) because their generated workspace paths were relative and Pyxis rejected those mounts. Failure result records/traces from those starts were preserved; replacement `r13` workspaces are fresh.
- Current active queue by difficulty/model: easy `2,377` `deepseek-v4-flash`; medium `5,361` `deepseek-v4-pro`; hard `114` `deepseek-v4-pro`. Expanded `squeue` shows `0` GPU/H200 generation jobs.
- Using the last strict saved-trajectory snapshot as the completed baseline, active-plus-complete coverage projects above the 50% target for all difficulties: easy `469 + 2,377 >= 2,469`, medium `291 + 5,361 >= 4,588`, hard `123 + 114 >= 148`.
- The append-only result index currently contains recent Pyxis/startup failures from older and canceled waves; r13/r14 agent trajectories are not expected there until the current container imports and teacher runs finish.

## 2026-06-08 07:25 UTC

DeepSeek high-reasoning repair update:

- Root cause found for many r13/r14 non-trajectory starts: the shared Pyxis dependency overlay had no working `jinja2`, then a mismatched/missing `pydantic_core`. I repaired the overlay in place with the standalone `/wbl-fast` runtime Python and validated mini-swe-agent imports with `Jinja2`, `pydantic==2.13.4`, and `pydantic-core==2.46.4`.
- Confirmed saved host trajectories are being written despite the result-index undercount: sample host files include `agent/mini-swe-agent.trajectory.json` sizes in the hundreds of KB for r13 medium/easy and r12 hard tasks.
- Fixed the mini-swe-agent driver result-index locator for successful jobs. Successful jobs run with `--workspace /workspace`, so the old index locator could not find `pyxis-traces`; the driver now derives the host workspace from `$HOME` and records host-visible result, patch, and trajectory paths.
- Reduced hot-array throttles after seeing `/run/pyxis` no-space and pre-repair import failures: r14 medium to `20`, r13 easy to `60`, and r12 hard to `15`. This should reduce wasted startup failures while preserving queued coverage.
- Current monitoring remains CPU-only on `m7i-cpu2`; no H200/GPU generation jobs are visible.

## 2026-06-08 06:23 UTC

DeepSeek reasoning coverage ramp:

- Active target remains 50% high-quality task coverage for each difficulty, using `deepseek-v4-flash` for easy and `deepseek-v4-pro` for medium/hard.
- All current `swere-rsn` jobs are on `m7i-cpu2`; no DeepSeek datagen jobs are on H200/GPU partitions.
- Reasoning settings remain enabled for future DeepSeek starts: `extra_body_json={"thinking":{"type":"enabled"}}`, `reasoning_effort=high`, `max_tokens=16384`, and no temperature in thinking mode.
- Fixed startup blocker: repaired `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-overlay-immutable-20260608T0525Z` so `pydantic`, `pydantic_core`, `typing_extensions`, `litellm`, and `mini-swe-agent` import correctly.
- Throughput ramp: raised array throttles after a clean bounded startup sample. Current expanded Slurm state is `RUNNING=675`, `CONFIGURING=56`, `PENDING=5063`.
- Queue by difficulty/model: easy Flash `RUNNING=384`, `CONFIGURING=36`, `PENDING=1979`; medium Pro `RUNNING=287`, `CONFIGURING=20`, `PENDING=3084`; hard Pro `RUNNING=4`.
- Prompt style mix in the active/queued set remains balanced: deepswe `RUNNING=320`, `CONFIGURING=28`, `PENDING=2537`; original `RUNNING=355`, `CONFIGURING=28`, `PENDING=2526`.
- Bottleneck audit: Docker auth is not the current blocker. A temporary attempt to request Slurm `--tmp=20000` caused `BadConstraints` on `m7i-cpu2`, so it was reverted in place. Generated array scripts now set `ENROOT_TEMP_PATH=$WORKSPACE/enroot-tmp` under `/wbl-fast` for pending starts.
- Latest bounded active-log sample after throttle increase: `startup_ok=163`, still-importing `16`, one old Docker rate-limit marker, and no `No space left on device` or missing-module failures.
- Older r06/r09 generated scripts were still pointing at the mutable `runtime/pydeps-overlay`; those scripts now point at the immutable overlay, and the mutable overlay was repaired in place to rescue rows that had already started.
- Added a central append-only result index for future completions at `runs/swerebench-v2/datagen-20260608-pyxis-deepseek-reasoning1/manifest/result_index.jsonl` so hourly reports no longer require expensive recursive scans of trace directories. The append now uses file locking; `result.json` remains the source of truth and all failed traces/results are still saved.

## 2026-06-08 04:19 UTC

DeepSeek high-reasoning coverage push:

- All current DeepSeek generation uses thinking enabled, `reasoning_effort=high`, and `max_tokens=16384`; easy uses `deepseek-v4-flash`, medium/hard use `deepseek-v4-pro`.
- CPU-only status: all visible datagen jobs are on `m7i-cpu2`; no H200/GPU partition is being used.
- Runtime fix: rebuilt a clean mini-swe-agent/LiteLLM dependency overlay under `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-overlay` and atomically swapped it in after import validation. This fixes the missing `pydantic`, `typing_extensions`, `attrs`, and `anyio` failures seen in earlier r07/r06 starts.
- Throughput ramp: raised current array throttles to easy r06 `75` per shard, medium r06 `45` per shard, hard r06 `60`, and r07 retry arrays `50/50/50` where applicable.
- Submitted deduplicated r08 retry wave for rows that had `result.json` but no saved trajectory and were not already complete or active: easy `818` rows (`job 276392`, concurrency `80`), medium `585` rows (`job 276474`, concurrency `40`), hard `124` rows (`job 276475`, concurrency `30`).
- Current strict coverage snapshot counts only tasks with saved mini-swe-agent trajectories as complete:

| difficulty | high-quality total | 50% target | passed unique | complete unique | queued/running unique | complete + queued | remaining now | remaining if queued complete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 4938 | 2469 | 135 | 411 | 3453 | 3849 | 2058 | 0 |
| medium | 9175 | 4588 | 50 | 231 | 5180 | 5381 | 4357 | 0 |
| hard | 295 | 148 | 8 | 119 | 136 | 243 | 29 | 0 |

- Trial-level saved trajectories and pass counts: easy `272/826` passed, medium `100/466` passed, hard `11/190` passed.
- Current queue state at snapshot time: about `801` running array elements, with pending work held by array throttles rather than GPU/CPU partition mismatch.
- Next action: continue monitoring post-overlay logs; retry any remaining no-trajectory dependency/startup failures while maintaining high CPU throughput.

## 2026-06-08 04:39 UTC

DeepSeek high-reasoning throughput addendum:

- A few fresh r08 starts still saw the old overlay inode through shared-storage caching, so the validated clean dependency tree was also synced into `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-overlay.backup.20260608T040818Z`. Both the active overlay path and the backup inode now validate through `pydantic`, LiteLLM, and mini-swe-agent imports.
- Post-repair r08 sample is clean: latest r08 logs had no missing-module errors, no Docker `429`, and r08 has started writing trajectories.
- Third ramp applied: easy arrays to `125`, medium arrays to `60`, hard unchanged. Latest queue sample shows `1075` active/configuring/completing elements across `280` CPU nodes, all on `m7i-cpu2`.
- Recent trajectory throughput sample: `117` trajectories written in the last 15 minutes across r06/r08, with both `deepseek-v4-flash` and `deepseek-v4-pro` active.
- Next action: keep monitoring at the new throttle, then recompute strict task coverage and submit another deduplicated failed-only retry wave if no-trajectory startup failures remain.

## 2026-06-08 04:59 UTC

DeepSeek high-reasoning r09 recovery wave:

- Strict coverage recompute showed current queued work was no longer enough for 50% if only saved trajectories count: projected shortfalls were easy `1223`, medium `963`, hard `26`.
- Prepared deduplicated r09 failed-only retry manifests, excluding tasks already complete or active: easy `2756`, medium `1874`, hard `121`.
- Slurm array-size limit required splitting r09: easy shards `900/900/900/56`, medium shards `800/800/274`, hard `121`.
- Submitted r09 jobs on `m7i-cpu2`: easy `281902,281996,281997,281998`; medium `281999,282000,282001`; hard `282002`.
- Current r09 queue sample: hard is now running (`50` running, `71` pending); easy has `474` running and `2367` pending; medium has `460` running and `4210` pending.
- r09 startup health sample: no missing-module errors and no Docker `429`; r09 trajectories have not landed yet because the wave is still in startup/agent phase.
- Next action: continue timed monitoring, recompute strict coverage after r09 starts producing trajectories, and retry only no-trajectory failures that remain outside active/complete sets.

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


## 2026-06-07 19:43 UTC

DeepSeek coverage expansion after ethanewer Docker plan upgrade:

- New run root: `/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260607-pyxis-deepseek-coverage1`.
- Selection target: every high-quality task without a saved trajectory at selection time, prioritizing medium first because it had the lowest coverage.
- Coverage before this wave: `{'easy': {'pct': 10.51, 'remaining_without_trajectory': 4938, 'saved': 580, 'total': 5518}, 'hard': {'pct': 11.94, 'remaining_without_trajectory': 295, 'saved': 40, 'total': 335}, 'medium': {'pct': 2.84, 'remaining_without_trajectory': 9175, 'saved': 268, 'total': 9443}}`.
- Selected unique tasks: `14408` total; by difficulty `{'easy': 4938, 'hard': 295, 'medium': 9175}`; by model `{'deepseek-v4-flash': 4938, 'deepseek-v4-pro': 9470}`; prompt styles `{'deepswe': 7205, 'original': 7203}`.
- Submitted so far: `11118` rows across `15` Slurm arrays; by difficulty `{'medium': 9175, 'easy': 1648, 'hard': 295}`; by model `{'deepseek-v4-pro': 9470, 'deepseek-v4-flash': 1648}`; requested concurrency by model `{'deepseek-v4-pro': 475, 'deepseek-v4-flash': 455}`.
- Current result records: `2145`; saved trajectories `{'medium': 554, 'hard': 21, 'easy': 269}`; reward passes `{'medium': 108, 'hard': 2, 'easy': 93}`; exception types `{'': 844, 'PyxisContainerStartError': 1203, 'ValueError': 98}`.
- Current model results: total `{'deepseek-v4-pro': 1528, 'deepseek-v4-flash': 617}`; saved trajectories `{'deepseek-v4-pro': 575, 'deepseek-v4-flash': 269}`; reward passes `{'deepseek-v4-pro': 110, 'deepseek-v4-flash': 93}`.
- Docker/Pyxis behavior: future generated scripts now use an empty `/wbl-fast` Enroot config for the anonymous attempt, then retry with `ethanewer` credentials only if no `result.json` was produced. The first submitted medium arrays predated the empty-config fix but still use the authenticated retry path.
- Slurm note: broad array submission is being throttled by temporary `sbatch` controller backoff; a foreground retry loop is submitting the remaining easy Flash shards gradually.

## 2026-06-07 20:14 UTC

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

## 2026-06-07 20:47 UTC

DeepSeek high-quality coverage expansion status:

- Run root: `/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260607-pyxis-deepseek-coverage1`.
- First-pass unique coverage is fully submitted: `14408` high-quality tasks, split `{'medium': 9175, 'easy': 4938, 'hard': 295}`.
- Total scheduled rows are now `14879` across `26` arrays because I added `471` `r01` retries for no-trajectory dependency failures after repairing the shared LiteLLM runtime overlay.
- Scheduled rows by difficulty: `{'medium': 9375, 'easy': 5197, 'hard': 307}`.
- Scheduled rows by model: `{'deepseek-v4-pro': 9682, 'deepseek-v4-flash': 5197}`.
- Scheduled rows by difficulty/model: `{('medium', 'deepseek-v4-pro'): 9375, ('easy', 'deepseek-v4-flash'): 5197, ('hard', 'deepseek-v4-pro'): 307}`.
- Current result records: `{'medium': 3088, 'hard': 102, 'easy': 2658}`.
- Host-side mini-swe-agent trajectories saved: `{'medium': 1100, 'hard': 37, 'easy': 731}`.
- Reward passes among host-side trajectories: `{'medium': 190, 'hard': 3, 'easy': 242}`.
- Model trajectory pass rates: DeepSeek v4 Pro `193/1137 = 17.0%`; DeepSeek v4 Flash `242/731 = 33.1%`.
- Approximate high-quality trajectory coverage including previously saved trajectories and this run's host trajectories: easy `1311/5518 = 23.8%`, medium `1368/9443 = 14.5%`, hard `77/335 = 23.0%`. Medium remains the lowest-coverage difficulty as the queued first-pass jobs finish.
- Current exception mix: `{('medium', 'PyxisContainerStartError'): 1741, ('easy', 'PyxisContainerStartError'): 1587, ('medium', ''): 1100, ('easy', ''): 731, ('easy', 'ValueError'): 340, ('medium', 'ValueError'): 246, ('hard', 'PyxisContainerStartError'): 46, ('hard', ''): 37, ('hard', 'ValueError'): 19, ('medium', 'FileNotFoundError'): 1}`.
- Language trajectory counts: `{'js': 665, 'go': 496, 'python': 232, 'php': 162, 'ts': 96, 'java': 89, 'cpp': 48, 'c': 41, 'rust': 39}`.
- Runtime fix: `pydantic` and `anyio` were repaired in `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-overlay`; `litellm` import now succeeds for future DeepSeek starts.
- Queue state: `m7i-cpu2` only, with `1104` running array tasks and `23` queued array records at the time of this report. No coverage datagen was submitted to GPU partitions.

## 2026-06-07 20:56 UTC

Pyxis extraction-failure retry pilot:

- Submitter update: future generated Slurm scripts now export `ENROOT_REMAP_ROOT=yes` before `srun`, in addition to Pyxis `--container-remap-root`.
- Pilot target: no-trajectory `PyxisContainerStartError` rows, to test whether Enroot root remapping fixes the Go-module-cache extraction permission failures before launching a broad retry wave.
- Submitted pilot arrays on `m7i-cpu2`: `238012` medium/pro `64` rows, `238015` easy/flash `32` rows, `238017` hard/pro `24` rows.
- The pilot scripts were verified to include `ENROOT_REMAP_ROOT=yes`; no GPU partitions were used.

## 2026-06-07 21:16 UTC

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

## 2026-06-07 22:16 UTC

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

## 2026-06-07 23:16 UTC

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

## 2026-06-08 00:17 UTC

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

## 2026-06-08 01:17 UTC

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

## 2026-06-08 02:17 UTC

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

## 2026-06-08 03:14 UTC

DeepSeek reasoning-mode throughput update:

- DeepSeek docs check: thinking mode is enabled with `extra_body={"thinking":{"type":"enabled"}}`; default thinking effort is `high`; `temperature`, `top_p`, `presence_penalty`, and `frequency_penalty` are unsupported/no-op in thinking mode, so the driver now omits `temperature` for thinking requests.
- Datagen config changed for all future DeepSeek runs: `reasoning_effort=high`, `max_tokens=16384`, `model_timeout=600`, `extra_body_json={"thinking":{"type":"enabled"}}`.
- Bottleneck resolved: CPU-node Docker daemon is available on `m7i-cpu2`; a `dockerd://` Pyxis probe on previously failing `swerebenchv2/argoproj-argo:4413-1be03db` completed successfully on CPU-only Slurm in `00:02:46`.
- New run root: `/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/runs/swerebench-v2/datagen-20260608-pyxis-deepseek-reasoning1`.
- Remaining no-trajectory DeepSeek task pool prepared for `r03`: `easy=3922`, `medium=7150`, `hard=243`.
- Prepared pool by language: easy `{'c': 14, 'cpp': 14, 'go': 803, 'java': 162, 'js': 557, 'php': 190, 'python': 1212, 'rust': 356, 'ts': 614}`; medium `{'c': 17, 'cpp': 14, 'go': 1461, 'java': 350, 'js': 494, 'php': 307, 'python': 2170, 'rust': 1333, 'ts': 1004}`; hard `{'c': 1, 'cpp': 1, 'go': 54, 'java': 18, 'js': 12, 'python': 62, 'rust': 63, 'ts': 32}`.
- Submitted unique reasoning rows so far: `easy=3922` with `deepseek-v4-flash`; `medium=5364` with `deepseek-v4-pro`; `hard=243` with `deepseek-v4-pro`. Medium shards `s06/s07` remain prepared but not submitted yet (`1786` rows) due Slurm controller backpressure while current arrays configure.
- Submitted job IDs: pilot `244573`; hard `244583`; easy shards `244655,244656,244748,244749`; medium shards `244750,244751,244752,244753,244754,244755`.
- Pilot status: all 10 pilot trajectories were saved. Current reliable pilot result snapshot: Flash/easy has 4 completed results with 2 reward passes; several pilot result files were overwritten by cancelled duplicate full-array elements, so pilot passrate will be recomputed from stable artifacts after current cancellation noise settles.
- Queue/resource state at update time: `m7i-cpu2` is saturated as intended, with roughly `3806/4800` CPUs allocated and reasoning jobs in `RUNNING`, `CONFIGURING`, and `COMPLETING` states. No H200/GPU partitions are being used for this wave.
- Next action: monitor container/API startup and passrates; retry medium shards `s06/s07` when Slurm accepts more arrays; keep all new submissions on CPU-only nodes and under `/wbl-fast`.

## 2026-06-08 03:36 UTC

DeepSeek reasoning-mode bottleneck follow-up:

- Commit state pushed: `3537eec` enabled DeepSeek thinking/high/16k and `dockerd://`; `2428a62` switched Docker auth to the Docker Hub default endpoint; `c251f32` stopped calling the Docker login endpoint per task and writes per-workspace Docker auth config from the existing enroot credential; `dc38bd8` added retry/backoff for transient Docker pull failures.
- Bottleneck sequence observed:
  - `r03` proved `dockerd://` fixes rootless Enroot extraction failures, but default Docker login was initially wrong and many pulls hit the unauthenticated rate limit.
  - `r04/r05` fixed the auth target, then exposed Docker Hub login/API throttling under hundreds of simultaneous login calls.
  - Current launcher avoids Docker login calls and retries pulls with backoff on `429`, TLS timeout, reset, EOF, and timeout errors.
- Validation probe: CPU-only auth-config pull probe `263640` pulled `swerebenchv2/cloudflare-cloudflare-go:959-1190d57` successfully without calling `docker login`.
- Current stable wave: `r06` retry set excludes active jobs and any task already having `r03/r04/r05` trajectories. Prepared rows: `easy=3856`, `medium=7131`, `hard=189`.
- Accepted `r06` jobs so far: hard `273072`; easy `273073,273074,273125,273126,273127`; medium `273128,273184,273195,273196,273197,273198`. Medium `s06/s07` are still prepared but not submitted due Slurm `sbatch` backpressure.
- Current `r06` pull sample: `159` jobs reached Docker auth config, `142` reached `docker_pull=ok`, `146` entered `dockerd` Pyxis, and `20` used the new retry sleep path before continuing. This is the first stable Docker-throughput configuration in this wave.
- Current concurrency is intentionally lower while Docker Hub throttling recovers: hard `20`, easy `75`, medium `60` accepted from `s00`-`s05`. Increase only after pull success remains stable.

## 2026-06-08 03:17 UTC

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

## 2026-06-08 04:17 UTC

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

## 2026-06-08 05:17 UTC

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

## 2026-06-08 06:18 UTC

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

## 2026-06-08 08:33 UTC

DeepSeek high-reasoning recovery and current blocker:

- Repaired the shared mini-swe-agent runtime overlay in `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-overlay-immutable-20260608T0525Z`; validation now constructs a LiteLLM DeepSeek model with `thinking.enabled`, `reasoning_effort=high`, and `max_tokens=16384`.
- Confirmed active generation jobs are CPU-only on `m7i-cpu2`; no H200/GPU partition is involved.
- Canceled pending medium `deepseek-v4-pro` jobs after fresh runs reached the API and returned `Insufficient Balance`; already-running Pro jobs were left to drain so they can write result artifacts where possible.
- Built deduplicated r16 Flash replacement manifests from the coverage1 high-quality pool, skipping coverage1 tasks with saved trajectories and currently active task IDs. r16 selected `9,717` unique tasks: hard `243`, medium `7,143`, easy `2,331`; styles are balanced (`4,870` DeepSWE, `4,847` original).
- Submitted accepted r16 Flash arrays: hard job `300737` and medium jobs `300738`, `300879`, `300880`, `300980`, `301122`, `301204`, `301292`, `301293`. The easy r16 submission was attempted but Slurm returned temporary controller backpressure before accepting it.
- Early r16 logs show `deepseek-v4-flash` is also now returning `Insufficient Balance`. This makes the current bottleneck the DeepSeek account balance, not Docker, Pyxis, CPU capacity, or prompt/runtime configuration.
- To avoid wasting Docker pulls and CPU, all pending DeepSeek datagen array elements were canceled. Running elements were left to finish and write failure records/traces where the driver is already active; configuring elements were canceled before driver startup.
- Queue after pending/configuring cancellation: easy running `114`, hard r16 Flash running `60`, medium r16 Flash running `492` plus `2` completing. There are no pending DeepSeek datagen elements left at this checkpoint.

Current implication: the 50% coverage target cannot progress further until the DeepSeek key has usable balance or a replacement DeepSeek key is provided. When balance is restored, the r16 manifests are already prepared under `runs/swerebench-v2/datagen-20260608-pyxis-deepseek-reasoning1/manifest/` and can be resubmitted quickly.

## 2026-06-08 08:48 UTC

DeepSeek balance blocker follow-up:

- Rechecked DeepSeek directly with a tiny `deepseek-v4-flash` thinking request; the API still returns `Insufficient Balance`.
- No new jobs were submitted. Prepared r16 manifests remain available for immediate resubmission once the DeepSeek key is funded again.
- Pending/configuring DeepSeek jobs remain canceled. The visible queue drained from `272` to `47` CPU-only jobs during this monitoring pass; all remaining jobs are on `m7i-cpu2`.
- Current visible queue: easy Flash `5`, hard r16 Flash `5`, medium r16 Flash `35` running plus `2` completing. No pending DeepSeek datagen elements are visible.
- r16 hard/medium Flash artifacts are being preserved despite the failed API state. Current sampled r16 artifact counts: hard `55` result files, `45` trajectories, `41` balance-failure results; medium `459` result files, `410` trajectories, `402` balance-failure results, and `3` reward-pass results from jobs that completed before the balance failure dominated.

Current action: continue avoiding new DeepSeek submissions until the API probe succeeds. Resuming coverage generation is now gated on DeepSeek balance, not scheduler capacity, Docker pulls, Pyxis, or the harness.

## 2026-06-08 09:00 UTC

DeepSeek high-reasoning coverage status:

- Rechecked both `deepseek-v4-flash` and `deepseek-v4-pro` directly with tiny high-reasoning requests; both return `Insufficient Balance`.
- No new DeepSeek work was submitted. Submitting more while the API is in this state would only consume Docker/CPU and produce balance-failure traces.
- Remaining live datagen queue is CPU-only on `m7i-cpu2`: easy Flash `2` running, hard Flash `1` running, medium Flash `2` completing. The running jobs have already pulled their Docker images and are retrying on the DeepSeek balance error; they were left to drain so the mini-swe-agent driver can write failure trajectories/results.
- The current coverage snapshot, counting only tasks with saved mini-swe-agent trajectories as complete, is:
  - easy: passed unique `164`; complete saved unique `469`; queued/running unique `1849`; complete+queued `2318` of target `2469`.
  - medium: passed unique `70`; complete saved unique `291`; queued/running unique `3678`; complete+queued `3969` of target `4588`.
  - hard: passed unique `10`; complete saved unique `123`; queued/running unique `20`; complete+queued `143` of target `148`.
- Prepared r16 manifests remain ready for resubmission once DeepSeek balance is restored: `9,717` unique high-quality tasks selected, with hard `243`, medium `7,143`, easy `2,331`, balanced between original and DeepSWE prompt styles.

Current blocker: DeepSeek account balance. Docker auth/pulls, CPU partition selection, Pyxis startup, and the high-reasoning mini-swe-agent configuration have all been verified enough to resume quickly after the key is funded.

## 2026-06-08 14:59 UTC

DeepSeek high-reasoning post-drain coverage table:

- No `swere-rsn` datagen jobs are visible in `squeue`; the prior draining jobs have finished or disappeared from the queue.
- Current strict accounting was recomputed from manifest workspaces under `runs/swerebench-v2/datagen-20260608-pyxis-deepseek-reasoning1`, counting a task as complete only when `agent/mini-swe-agent.trajectory.json` exists and is non-empty.
- Unique saved-trajectory coverage by difficulty:

| difficulty | passed unique | complete saved unique | 50% target | complete coverage |
|---|---:|---:|---:|---:|
| easy | 553 | 1,827 | 2,469 | 74.0% |
| medium | 302 | 2,007 | 4,588 | 43.7% |
| hard | 16 | 203 | 148 | 137.2% |

- Trial-level saved trajectories: easy `2,172`, medium `2,379`, hard `283`.
- Trial-level passes: easy `568`, medium `308`, hard `17`.
- Remaining blocker for new coverage remains DeepSeek account balance; no additional DeepSeek jobs were submitted.

## 2026-06-08 17:55 UTC

MiMo/OpenRouter reasoning coverage expansion:

- Switched the active coverage push to the provided OpenRouter key only. The manifest maps easy/medium tasks to `xiaomi/mimo-v2.5` and hard tasks to `xiaomi/mimo-v2.5-pro`.
- Verified both MiMo model IDs return non-empty OpenRouter reasoning fields when called with `reasoning: {"effort": "high", "exclude": false}`. New Slurm manifests carry `extra_body_json={"reasoning":{"effort":"high","exclude":false}}`, `max_tokens=16384`, and `reasoning_effort=high`.
- Used the existing unique reasoning dataset at `/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/deepswe-highquality-unique-reasoning/data/train.jsonl` to select missing high-quality tasks. Missing count before this wave: `11,795` tasks (`easy=3,857`, `medium=7,809`, `hard=129`).
- Smoke jobs ran on CPU-only `m7i-cpu2` nodes. The saved smoke trajectories had reasoning in every checked assistant message.
- Initial scale submission exposed two infrastructure issues:
  - Docker pull wrapper retried anonymous first, then logged in, but did not retry after Docker Hub returned the exact `unauthenticated pull rate limit` message. This caused many preserved zero-call failure results.
  - The mounted mini-swe-agent overlay was missing runtime deps (`rich`, `typing_extensions`, `pydantic`, `pydantic_core`), causing auth-first jobs to fail after image import.
- Fixed the submit wrapper to retry immediately after Docker login and to treat `rate limit` as transient. Also added `PYTHONDONTWRITEBYTECODE=1` for newly generated scripts.
- Repaired the `/wbl-fast` runtime overlay in place from existing local dependency directories; validation now imports `rich`, `pydantic`, `litellm`, and `minisweagent` successfully.
- Canceled `7,432` pending old-script array elements before they started. Running elements were left to finish and preserve their results/traces.
- Built a replacement manifest with `9,729` unique rows: the canceled pending rows plus the `2,297` never-submitted deepswe rows. All `17` auth-first replacement arrays were accepted on `m7i-cpu2`:
  - `af00`-`af09`: jobs `320127`, `320153`-`320158`, `320260`-`320262`
  - `af10`-`af16`: jobs `321004`-`321006`, `321041`-`321044`
- Current MiMo queue after accepting all replacements: `9,360` visible array elements, `311` running and `9,049` pending, all on `m7i-cpu2`.
- Current MiMo result files: `2,436` unique tasks. Status breakdown: `Submitted=176`, `PyxisContainerStartError=2,245`, `ValueError=11`, `LimitsExceeded=3`, `FileNotFoundError=1`. Most `PyxisContainerStartError` entries are preserved infrastructure failures from before the Docker/runtime fixes and will need retry manifests rather than being counted as covered reasoning trajectories.
- Current result files by difficulty: `easy=912`, `medium=1,488`, `hard=36`. Reward passes so far: `easy=28`, `medium=28`, `hard=0`.
- Current result files by model: `xiaomi/mimo-v2.5=2,400`, `xiaomi/mimo-v2.5-pro=36`. Recorded cost so far: about `$37.48` for `xiaomi/mimo-v2.5` and `$2.68` for `xiaomi/mimo-v2.5-pro`.
- Saved trajectory files counted in the trace tree: `296`.
- Reasoning quality sample after fixing host/container path resolution: `80` submitted trajectories checked, `5,254` assistant messages checked, `0` assistant messages missing reasoning.

Next action: monitor post-fix auth-first jobs for real trajectory growth, then build retry manifests for zero-call Docker/runtime failures using a new rollout id so the failure artifacts remain preserved.

## 2026-06-08 19:36 UTC

MiMo/OpenRouter native reasoning retry:

- Active generation remains CPU-only on `m7i-cpu2`; no GPU/H200 jobs are involved.
- The unstable dependency path was isolated: copied packages in `.venv` and the main overlay were being removed/reverted. New Slurm scripts now prepend the stable Pydantic stack at `/wbl-fast/usrs/ee/code-swe-data/runtime/manual-pydeps/pydantic-stack-clean` before the mini-swe-agent overlay.
- The fixed-PythonPath retry initially reached task containers but exposed a separate LiteLLM import failure through `aiohttp`, reported by mini-swe-agent as `Unknown model class: litellm`. Pending LiteLLM-configured `fp*` elements were canceled; running elements were left to finish or pick up the patched driver if they had not entered the container yet.
- The driver now routes `openrouter/*` rows through mini-swe-agent's native `openrouter` model class, strips the `openrouter/` prefix for the API model name, passes `reasoning: {"effort": "high", "exclude": false}` as a top-level OpenRouter payload field, and uses a 600s OpenRouter request timeout for high-reasoning calls.
- Built OpenRouter-native retry manifest from the unique-reasoning missing set, skipping already saved all-reasoning trajectories and active running tasks. Retry rows: `10,460` total, difficulty split `easy=3,469`, `medium=6,868`, `hard=123`; model split `xiaomi/mimo-v2.5=10,337`, `xiaomi/mimo-v2.5-pro=123`.
- Retry language mix: python `2,689`, ts `1,835`, js `1,669`, rust `1,600`, go `1,573`, java `553`, php `476`, cpp `35`, c `30`.
- Accepted OpenRouter-native shards so far: `or00`-`or17` plus direct retry `or18`, covering `9,500` submitted rows. `or19` and `or20` (`960` rows total) are still hitting temporary Slurm `Resource temporarily unavailable` and will be retried after scheduler cooldown.
- Current visible MiMo queue after native submission: about `9.1k` array elements, all on `m7i-cpu2`, with roughly `414` running and the rest pending depending on the polling instant.
- Current MiMo trace tree snapshot from the latest scan: status counts include `Submitted=1,504`, `ValueError=1,305`, `PyxisContainerStartError=5,433`, `LimitsExceeded=47`, `TimeExceeded=12`, `FileNotFoundError=2`. The large Pyxis/ValueError counts are preserved failed attempts from earlier infrastructure/LiteLLM waves and are not counted as successful coverage.
- Current reward passes in this MiMo run: easy `170`, medium `166`, hard `0`.
- Native OpenRouter jobs are producing real trajectories now. Recent bounded sample: `80` submitted trajectories, `4,371` assistant messages checked, `0` assistant messages missing reasoning.

Next action: keep monitoring native OpenRouter jobs for pass rate and API stability, retry `or19`/`or20`, and build another failed/no-reasoning retry manifest after the active queue drains enough to avoid duplicates.

## 2026-06-08 20:20 UTC

MiMo/OpenRouter coverage expansion:

- Checkpointed and pushed the OpenRouter native/reasoning fixes as commit `96b370b`.
- Used the unique all-reasoning dataset file to select still-missing high-quality tasks. Current accounting: `3,501` tasks already in the unique reasoning dataset, `11,795` high-quality tasks missing from that dataset, `15,296` high-quality tasks total.
- Native retry manifest covers `10,460` of the missing tasks. Accepted native shards now cover `10,000` tasks: `or00`-`or19`. The only unaccepted native shard is `or20` with `460` tasks.
- Built a no-duplicate gap manifest for the tasks absent from both the unique reasoning dataset and the native retry manifest: `1,335` rows (`easy=388`, `medium=941`, `hard=6`), model split `xiaomi/mimo-v2.5=1,329`, `xiaomi/mimo-v2.5-pro=6`, style split `original=806`, `deepswe=529`.
- Split the gap manifest into three CPU arrays. Accepted gap shards: `gr03-00` job `338318` (`500` tasks) and `gr03-02` job `339911` (`335` tasks). Remaining unaccepted gap shard: `gr03-01` (`500` tasks).
- Accepted-or-complete coverage target accounting is now `14,336 / 15,296` high-quality tasks. Remaining scheduler-blocked tasks are `960` total: `or20=460`, `gr03-01=500`.
- Current visible MiMo queue: `9,049` array elements, `414` running and `8,635` pending, all on `m7i-cpu2`.
- Current MiMo trace tree snapshot: `9,295` result files, `2,945` non-empty trajectory files. Status counts include `Submitted=2,462`, `PyxisContainerStartError=5,435`, `ValueError=1,305`, `LimitsExceeded=70`, `TimeExceeded=21`, `FileNotFoundError=2`.
- Current reward passes in this MiMo run: `xiaomi/mimo-v2.5` easy `244`, medium `203`; hard/pro passes are still `0` in the current snapshot.
- Reasoning quality sample remains clean: `120` recent submitted trajectories checked, `7,493` assistant messages checked, `0` assistant messages missing reasoning.

Next action: continue the retry loop for `or20` and `gr03-01` until Slurm accepts both, while monitoring that newly submitted jobs preserve reasoning in every assistant message.

## 2026-06-08 21:03 UTC

MiMo/OpenRouter dependency-failure retry:

- All remaining coverage shards have been accepted by Slurm. The original missing-task queue is now complete at the task level: `3,501` tasks already in the unique reasoning dataset plus `11,795` accepted missing-task rows, for `15,296 / 15,296` high-quality tasks represented.
- Late log inspection found an infrastructure dependency regression in started native jobs: `jinja2` resolved to a broken namespace package from the repo `.venv`, and a later wave saw missing `pydantic`. These are zero-call failures saved as `PyxisContainerStartError`, so they do not contain useful traces and need retry.
- Repaired the mounted `/wbl-fast` dependency paths used by queued jobs:
  - `/wbl-fast/usrs/ee/code-swe-data/runtime/manual-pydeps/pydantic-stack-clean`
  - `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-miniswe-complete-20260608T1830Z`
  - patched `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-miniswe-upstream-11ec55d/minisweagent/models/openrouter_model.py` to honor `request_timeout=600` without sending it in the OpenRouter payload.
- Exact import validation now passes for `jinja2.StrictUndefined`, `pydantic`, `requests/urllib3`, `DefaultAgent`, and native `OpenRouterModel` under both the explicit complete-overlay path and the default pinned-overlay path.
- Built retry manifest `mimo-depfail-retry-r04.tsv` from saved no-call dependency failures, deduped by task and preserving existing failure artifacts. Rows: `1,792` total (`easy=265`, `medium=1,506`, `hard=21`); model split `xiaomi/mimo-v2.5=1,771`, `xiaomi/mimo-v2.5-pro=21`; style split `deepswe=919`, `original=873`.
- Submitted all dependency retry shards on CPU-only `m7i-cpu2`: `r04dep-00` job `344270` (`500` rows), `r04dep-01` job `344285` (`500`), `r04dep-02` job `344303` (`500`), `r04dep-03` job `344318` (`292`).
- Current visible MiMo queue: `8,206` array elements, `414` running and `7,792` pending, all on `m7i-cpu2`. The r04 dependency retries are accepted but have not started yet, so there are no r04dep logs to sample.

Next action: continue monitoring for r04dep startup; once they start, verify the dependency failure is gone and sample submitted trajectories for reasoning in every assistant message.

## 2026-06-08 22:00 UTC

MiMo/OpenRouter queue and runtime repair:

- Enforced the queue constraint that no MiMo datagen jobs should remain pending for `JobArrayTaskLimit`. A later Slurm reshuffle exposed `24` more `JobArrayTaskLimit` elements; those were canceled as well. Current visible MiMo queue after the second cancellation/recheck: `661` array elements, `397` running and `264` pending, with `JobArrayTaskLimit=0`.
- All visible MiMo datagen jobs are on CPU-only `m7i-cpu2` nodes. No new large retry wave was submitted while the runtime path was unstable.
- Canceled the r04 dependency-retry wave before it could add more bad no-call failures, then canceled the old pending tail that was blocked by array task limits.
- Created a stable symlink dependency overlay at `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-mimo-symlink-stable-20260608T2145Z` and redirected the old mounted dependency paths to it:
  - `/wbl-fast/usrs/ee/code-swe-data/runtime/manual-pydeps/pydantic-stack-clean`
  - `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-miniswe-complete-20260608T1830Z`
- Newly started jobs then exposed one more container-only dependency miss: `pydantic_core` imports `typing_extensions`, but `typing_extensions` was not in the overlay. Added `typing_extensions.py` and `typing_extensions-4.15.0.dist-info` symlinks from `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-datagen`.
- Revalidated the mounted runtime with `python3 -S` so it does not depend on local site-packages. Imports now pass for `typing_extensions`, `jinja2.StrictUndefined`, `pydantic`, `requests`, `urllib3`, `DefaultAgent`, and native `OpenRouterModel`.

Next action: keep monitoring newly starting MiMo jobs after the `typing_extensions` repair before submitting any additional retries, so the queue does not accumulate preserved infrastructure failures.

## 2026-06-08 22:08 UTC

MiMo/OpenRouter stale-wave cancellation and submitter fix:

- Follow-up log inspection showed the 22:00 runtime repair was not enough for already-generated Slurm scripts: `PYTHONPATH` placed the pinned mini-swe-agent overlay before the repaired dependency stack, so broken `jinja2`/missing `pydantic` entries could still win.
- Repointed the mounted dependency aliases to the validated full overlay `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-mimo-stable-20260608T2136Z`:
  - `/wbl-fast/usrs/ee/code-swe-data/runtime/manual-pydeps/pydantic-stack-clean`
  - `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-miniswe-complete-20260608T1830Z`
- Validated those exact alias paths with `python3 -S`; imports pass for `typing_extensions`, `jinja2.StrictUndefined`, `pydantic`, `pydantic_core`, `requests`, `urllib3`, `DefaultAgent`, and native `OpenRouterModel`.
- Patched the Pyxis submitter so future Slurm scripts put the repaired dependency stack before the pinned mini-swe-agent overlay in `PYTHONPATH`.
- Canceled the remaining old-script MiMo elements because they were actively producing preserved zero-call dependency failures. Final queue after cleanup: `0` visible MiMo elements and `JobArrayTaskLimit=0`.

Next action: submit only a small corrected probe wave first, verify no dependency failures and all assistant messages preserve reasoning, then resume controlled coverage retries without array throttles that create `JobArrayTaskLimit`.

## 2026-06-08 22:25 UTC

MiMo/OpenRouter corrected restart:

- The first corrected probe (`r05probe`) still failed because the full dependency overlay had an incomplete `jinja2` package: bytecode files were present, but source files such as `jinja2/__init__.py` were missing. The failure was preserved as a no-call trajectory/result artifact where applicable.
- Repaired `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-mimo-stable-20260608T2136Z` with `rsync` from the intact pinned mini-swe-agent overlay so `jinja2` and its dist-info are self-contained and container-visible.
- Submitted a fresh four-task probe (`r06probe`) on CPU-only `m7i-cpu2`, with no array backlog (`0-3%4`). All four tasks have now created `mini-swe-agent.trajectory.json` files.
- Reasoning quality check on `r06probe`: `4` trajectories, `356` assistant messages, `0` assistant messages missing reasoning.
- Submitted a conservative fixed-path retry wave `swere-mimo-r07fix-00` with `100` tasks and array concurrency equal to row count (`0-99%100`), so it does not create `JobArrayTaskLimit` pending elements. Difficulty mix: `20` easy, `70` medium, `10` hard; model mix: `90` `xiaomi/mimo-v2.5`, `10` `xiaomi/mimo-v2.5-pro`.
- Early r07 startup check found no dependency tracebacks. Current r07 trajectory count: `8 / 100`.
- Reasoning quality check on current r07 trajectories: `8` trajectories, `200` assistant messages, `0` assistant messages missing reasoning.
- Live MiMo queue at this check: `104` visible elements (`34` running, `70` configuring), `JobArrayTaskLimit=0`.

Next action: continue monitoring r07 for startup failures and trajectory growth; only submit the next batch after this wave remains clean, using row-count-equal array concurrency to avoid `JobArrayTaskLimit`.

## 2026-06-08 23:52 UTC

MiMo/OpenRouter packed CPU coverage restart:

- Found that the previous fixed overlay had regressed again: container-side imports saw missing source files such as `pydantic/__init__.py` and a missing `certifi/cacert.pem`. Rebuilt a complete immutable overlay at `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-mimo-complete-fixed-20260608T232518Z` from the intact pinned mini-swe-agent overlay, and repointed the mounted runtime aliases to it.
- Patched the Pyxis submitter to avoid adding duplicate dependency paths when the manual dependency alias resolves to the same overlay, and to export `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, and `CURL_CA_BUNDLE` from the validated overlay.
- Added a packed CPU-only submitter, `datagen/swerebench_v2/submit_pyxis_datagen_packed.py`, so each Slurm array element runs multiple task containers on one `m7i-cpu2` node. This reduces queue entries while keeping each CPU node busy.
- Fixed the packed submitter default memory to fit `m7i-cpu2` nodes (`56G`, not `64G+`) and made `sbatch` failures print Slurm's actual rejection message.
- Current usable reasoning coverage before this packed restart: `3,528 / 15,296` high-quality tasks (`23.07%`). By difficulty: easy `1,666 / 5,518` (`30.19%`), medium `1,653 / 9,443` (`17.51%`), hard `209 / 335` (`62.39%`).
- Submitted packed probe `swere-mimo-pack-r08probe` job `350436`: `8` unique uncovered tasks (`easy=2`, `medium=4`, `hard=2`), two array elements, four task containers per CPU node. Docker auth/pulls succeeded through `dockerd://`, and all `8 / 8` trajectories are saved. Reasoning check: `8` trajectories, `0` assistant messages missing reasoning.
- Submitted packed wave `swere-mimo-pack-r09` job `350678`: `1,024` unique uncovered tasks (`easy=300`, `medium=600`, `hard=124`), `86` array elements with `12` rows per element and `4` concurrent task containers per element. This includes all currently uncovered hard tasks in the missing manifest.
- Current visible packed MiMo queue: `88` elements total (`2` running probe elements, `86` configuring r09 elements), all on CPU-only `m7i-cpu2`, with `JobArrayTaskLimit=0`.

Next action: monitor r09 startup and first trajectories for Docker/Pyxis/import/API failures; if clean, prepare the next non-overlapping packed wave from the remaining uncovered easy/medium tasks while r09 runs.

## 2026-06-09 00:01 UTC

MiMo/OpenRouter packed throughput follow-up:

- r09 startup exposed a new Docker bottleneck rather than a model/API or Python dependency issue: several early rows failed before trajectory creation because `docker pull` hit CloudFront/layer-download `EOF` errors. There are still no observed `pydantic`, `jinja2`, `certifi`, auth, or `JobArrayTaskLimit` failures in the packed scripts.
- Patched and pushed commit `442dc33` so future submissions retry plain `EOF` / `failed to do request` Docker pull failures and serialize Docker pulls within each packed array element using `flock`. This keeps four task containers per CPU node for generation while reducing per-node simultaneous image pulls.
- Submitted r10 after that patch: `swere-mimo-pack-r10` job `350923`, `1,024` unique easy/medium tasks (`easy=224`, `medium=800`), `86` packed array elements, `12` rows per element, `4` concurrent task containers per element.
- Prepared but did not submit r11: `mimo-packed-wave-r11-1024.tsv`, `1,024` unique easy/medium tasks (`easy=174`, `medium=850`). Holding this manifest avoids unnecessary queue pressure while r10 is still starting.
- Current packed queue snapshot: `174` elements total across r08/r09/r10, `91` running, `7` configuring, `76` pending for `BeginTime`, all on CPU-only `m7i-cpu2`, `JobArrayTaskLimit=0`.
- Current saved/in-flight trace snapshot:
  - r08 probe: `8 / 8` trajectories, `8` all-reasoning, `6` result files, `1` pass so far.
  - r09: `252 / 1,024` trajectories, `251` all-reasoning, `1` trajectory with at least one assistant turn missing reasoning, `43` result files, `5` passes so far, `26` early no-trace Docker/Pyxis start failures.
  - r10: accepted/starting, no trajectories yet.
- Quality action: do not count the r09 trajectory for `aio-libs__aiohttp-9239` as usable coverage unless a later final trajectory has non-empty reasoning on every assistant turn. Add any such rows to the retry/filter list for full reasoning coverage.

Next action: keep r11 held until r10 is mostly running or r09 drains, then submit the next packed wave. Continue checking trajectory reasoning completeness and Docker pull failure rates before expanding further.

## 2026-06-09 00:13 UTC

MiMo/OpenRouter packed concurrency adjustment:

- r10 exposed a second Pyxis bottleneck: Docker pulls succeeded, but concurrent Pyxis image imports on the same node could fail while creating `/run/pyxis/...squashfs` with `No space left on device`.
- Canceled r10 job `350923` before the failure mode spread across the full wave. Existing r10 artifacts were left in place. One r10 row already had a completed all-reasoning trajectory, so it was excluded from the retry.
- Patched and pushed commit `9223090` to make the packed submitter default to the safer full-node shape: `parallel_rows=2`, `cpus_per_row=8`, `rows_per_job=12`.
- Submitted replacement r10b job `351254`: `1,023` retry rows (`easy=224`, `medium=799`), `86` packed array elements, `2` concurrent task containers per node, `8` CPUs per task container, serialized Docker pulls.
- r10b startup health check: `19` trajectories saved so far, all `19` with reasoning on every assistant turn; no r10b result/failure files yet and no observed Docker EOF, `/run/pyxis` space, dependency import, auth, or API errors.
- r09 continues to run: `560` trajectories saved, `559` all-reasoning, `343` result files, `75` passes so far. Known unusable reasoning row remains `aio-libs__aiohttp-9239`.
- Current packed queue snapshot: `172` visible elements across r08/r09/r10b, `100` running, `72` pending for normal `Resources`, all CPU-only `m7i-cpu2`, `JobArrayTaskLimit=0`.

Next action: keep r11 held until r10b has taken more of the freed CPU capacity and remains clean; then submit r11 with the safer 2-way packed defaults.

## 2026-06-09 00:49 UTC

MiMo/OpenRouter continued packed coverage:

- r10b remained clean under the safer two-way packing, so r11 was regenerated with the same defaults (`rows_per_job=12`, `parallel_rows=2`, `cpus_per_row=8`) and submitted as job `352383`.
- r11 manifest: `1,024` unique uncovered tasks (`easy=174`, `medium=850`), no hard tasks because r09 already selected all remaining hard IDs.
- Current submitted packed waves r08/r09/r10b/r11 represent `3,079` unique missing-task rows beyond the prior unique dataset/current traces: `easy=700`, `medium=2,253`, `hard=126`.
- Queue snapshot after r11 submission: `207` visible packed elements, `100` running and `107` pending, all CPU-only `m7i-cpu2`, `JobArrayTaskLimit=0`. Pending is normal scheduler `Resources`/`Priority`, not array throttling.
- Live progress at this checkpoint: r09 `968` results / `983` trajectories, r10b `116` results / `207` trajectories, r11 accepted but not yet producing trajectories.

Next action: continue holding r12 until r11 starts and pending elements drain; then submit r12 with the same safe packing. Keep filtering/retrying completed rows with missing reasoning turns.

## 2026-06-09 01:42 UTC

MiMo/OpenRouter r12 submission checkpoint:

- Investigated a residual r10b no-trace start failure and found another Docker transient: `502 Bad Gateway` during layer copy. Patched and pushed commit `a141478` so future Docker pulls retry `502` / `Bad Gateway` as well as EOF/timeouts.
- Regenerated r12 with the current safe packed defaults and the expanded Docker retry pattern, then submitted r12 job `353447`.
- r12 manifest: `1,024` unique uncovered tasks (`easy=124`, `medium=900`), no hard tasks.
- Current submitted packed waves r08/r09/r10b/r11/r12 represent `4,103` unique missing-task rows beyond the prior unique dataset/current traces: `easy=824`, `medium=3,153`, `hard=126`.
- Queue snapshot after r12 submission: `237` visible packed elements, `151` running and `86` pending, all CPU-only `m7i-cpu2`, `JobArrayTaskLimit=0`.
- Live progress at this checkpoint: r10b `803` results / `896` trajectories / `203` passes, r11 `105` results / `207` trajectories / `22` passes. r12 accepted but not yet producing trajectories.

Next action: stage r13 from the remaining easy/medium uncovered tasks, hold submission until r12 starts or pending drains, and continue filtering/retrying rows with completed trajectories missing reasoning turns.

## 2026-06-09 02:10 UTC

MiMo/OpenRouter r13 submission checkpoint:

- r12 started cleanly under the safe two-way packed settings with no sampled Docker/Pyxis/dependency/API errors, so r13 was submitted as job `354113`.
- r13 manifest: `1,024` unique uncovered tasks (`easy=124`, `medium=900`), no hard tasks.
- Current submitted packed waves r08/r09/r10b/r11/r12/r13 represent `5,127` unique missing-task rows beyond the prior unique dataset/current traces: `easy=948`, `medium=4,053`, `hard=126`.
- Queue snapshot after r13 submission: `283` visible packed elements, `197` running and `86` pending, all CPU-only `m7i-cpu2`, `JobArrayTaskLimit=0`.
- Live progress at this checkpoint: r10b `966` results / `1,003` trajectories / `240` passes, r11 `387` results / `515` trajectories / `84` passes, r12 `39` results / `102` trajectories / `9` passes. r13 accepted but not yet producing trajectories.

Next action: stage r14 from remaining easy/medium tasks and hold it until r13 starts or the pending queue drains. Continue requiring non-empty reasoning in every assistant turn for usable coverage.

## 2026-06-09 02:42 UTC

MiMo/OpenRouter dependency repair during r11-r13:

- Detected a new live dependency corruption after r11/r12/r13 started: `requests` resolved to an empty namespace package under the mounted overlay, causing `AttributeError: module 'requests' has no attribute 'post'` / `requests.exceptions`.
- This produced many preserved but unusable `AttributeError` rows in r11/r12/r13. Existing traces/results are left on disk, but those rows must be retried for usable coverage.
- Rebuilt a repaired overlay at `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-mimo-requests-fixed-20260609T023148Z` by replacing `requests` and `requests-2.34.2.dist-info` from the intact `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-datagen` package.
- Replaced the hard-coded path used by already-submitted Slurm scripts, `/wbl-fast/usrs/ee/code-swe-data/runtime/pydeps-mimo-complete-fixed-20260608T232518Z`, with a symlink to the repaired overlay. Also repointed the manual dependency aliases to the repaired overlay.
- Validation with `python3 -S` now passes at the hard-coded path: `requests.post=True`, `requests.exceptions=True`, `requests.__version__=2.34.2`, plus `certifi`, `jinja2.StrictUndefined`, `pydantic.BaseModel`, and `OpenRouterModel`.
- Post-repair monitor at 02:40 UTC: r13 has begun producing `Submitted` rows (`1` so far), while many rows that started before the repair are still finishing as `AttributeError`. Queue remains active with `203` visible elements, `200` running, `3` configuring, `0` pending, `JobArrayTaskLimit=0`.

Next action: hold r14 until post-repair r13 rows show sustained `Submitted` growth and no new dependency error mode. Then submit r14 with the repaired overlay path and safe two-way packing.

## 2026-06-09 02:58 UTC

MiMo/OpenRouter r15 submission checkpoint:

- Post-repair r12/r13 rows showed sustained `Submitted` growth and no new dependency error mode, so r14 was submitted and then r15 was submitted after r14 startup checked clean.
- r14 manifest: `1,024` unique uncovered tasks (`easy=124`, `medium=900`), no hard tasks.
- r15 manifest: `1,024` unique uncovered tasks (`easy=700`, `medium=324`), no hard tasks. This shifts selection toward easy because easy is now the lower selected-coverage difficulty.
- r16 has been staged and validated but not submitted: `1,024` unique tasks (`easy=700`, `medium=324`).
- Current submitted packed waves r08/r09/r10b/r11/r12/r13/r14/r15 represent `7,175` unique missing-task rows beyond the prior unique dataset/current traces: `easy=1,772`, `medium=5,277`, `hard=126`.
- Queue snapshot: `256` visible packed elements, `178` running and `78` pending for normal `Resources`, all CPU-only `m7i-cpu2`, `JobArrayTaskLimit=0`.

Next action: hold r16 until r15 starts or pending drains. Continue retry planning for rows lost to the pre-repair `requests` `AttributeError` failure and rows with completed trajectories missing reasoning turns.

## 2026-06-09 03:12 UTC

MiMo/OpenRouter r16 submission checkpoint:

- r15 startup checked clean (`38` sampled trajectories all with reasoning and no sampled Docker/Pyxis/dependency/API errors), so r16 was submitted as job `355555`.
- r16 manifest: `1,024` unique uncovered tasks (`easy=700`, `medium=324`), no hard tasks.
- r17 has been staged and validated but not submitted: `1,024` unique tasks (`easy=600`, `medium=424`).
- Current submitted packed waves r08/r09/r10b/r11/r12/r13/r14/r15/r16 represent `8,199` unique missing-task rows beyond the prior unique dataset/current traces: `easy=2,472`, `medium=5,601`, `hard=126`.
- Queue snapshot after r16 submission: `258` visible packed elements across the active tail, `160` running and `98` pending, all CPU-only `m7i-cpu2`, `JobArrayTaskLimit=0`.

Next action: hold r17 until r16 starts or pending drains. Continue monitoring post-repair rows for sustained `Submitted` results and no recurrence of the `requests` overlay failure.

## 2026-06-09 03:33 UTC

MiMo/OpenRouter packed coverage expansion:

- r17 was submitted as job `355703` after r16 startup checked clean. r17 manifest: `1,024` unique tasks (`easy=600`, `medium=424`), no hard tasks.
- The remaining uncovered-pool calculation initially risked counting the canceled r10 manifest as selected, so I corrected the reservation logic to exclude canceled r10 except for its usable all-reasoning traces. r10b already covers the intended retry set.
- r18 was staged, validated, and submitted as job `355720`. r18 manifest: `1,024` unique tasks (`easy=214`, `medium=807`, `hard=3`), including all remaining never-reserved hard tasks.
- r19 and r20 are staged and dry-run validated, but held to avoid queue overload:
  - r19: `1,024` unique tasks (`easy=378`, `medium=646`)
  - r20: `524` unique tasks (`easy=193`, `medium=331`)
- With r18 submitted and r19/r20 staged, every row in the current high-quality missing manifests is reserved exactly once across r08-r20: `easy=3,857`, `medium=7,809`, `hard=129`.
- Current MiMo wave generation counters from manifest/result scan:
  - `xiaomi/mimo-v2.5`: `5,345` completed result files, `5,784` saved trajectories, `4,422` all-reasoning trajectories, `926` passes.
  - `xiaomi/mimo-v2.5-pro`: `126` completed result files, `119` saved trajectories, `118` all-reasoning trajectories, `12` passes.
- Current all-reasoning saved trajectories from MiMo waves by difficulty: `easy=1,159`, `medium=3,263`, `hard=118`. These counts exclude pre-repair/malformed reasoning traces even when trace files are saved.
- Current queued/running snapshot after r18 submission: `514` visible packed elements, `291` running, `1` configuring, `222` pending; all are CPU-only `m7i-cpu2`, and `JobArrayTaskLimit=0`.

Next action: hold r19 until r18 starts or pending drops materially, then submit r19 and later r20 with the same safe packing. After first-pass reserved coverage completes, build targeted retry manifests for rows lost to pre-repair `requests` failures, transient container failures, and any completed trajectories missing assistant-turn reasoning.

## 2026-06-09 04:41 UTC

MiMo/OpenRouter r19 submission checkpoint:

- r16 continued cleanly after the previous checkpoint. The latest full r16 quality spot check saw `248` trajectories, `149` result files, `57` passes, and `0` missing-reasoning/JSON issues. A controller-log scan found no sampled Docker, auth, quota, Pyxis, or `requests` overlay errors.
- r17 began running once r16 was mostly resident on nodes. Before submitting r19, the compact queue snapshot was `328` visible packed elements, `189` running, `139` pending, and `JobArrayTaskLimit=0`; r17 had `33` running elements and `53` pending.
- r19 was submitted as job `355949` with the same CPU-only packed settings (`rows_per_job=12`, `parallel_rows=2`, `cpus_per_row=8`, `m7i-cpu2`, reasoning high, `max_tokens=16384`).
- r19 manifest: `1,024` unique remaining tasks (`easy=378`, `medium=646`), no hard tasks.
- r20 remains staged and validated but held as the final tail (`524` rows: `easy=193`, `medium=331`) to avoid unnecessary queue depth.
- Queue snapshot after r19 submission: `411` visible packed elements, `189` running, `222` pending, all CPU-only `m7i-cpu2`, `JobArrayTaskLimit=0`. Pending is scheduler `Resources`/`Priority`, not Docker or API failure.

Next action: keep r20 held until r18 or r19 starts draining; then submit the `524`-row tail. Continue using compact scheduler checks and targeted spot checks rather than broad scans over active trace directories, which can stall on shared storage.

## 2026-06-09 04:56 UTC

MiMo/OpenRouter r20 tail submission checkpoint:

- r17 became mostly resident on CPU nodes (`85` running, `1` pending) with no scheduler throttle pressure, so the final first-pass tail was submitted.
- r20 was submitted as job `356041` with the same packed CPU-only settings.
- r20 manifest: `524` unique tasks (`easy=193`, `medium=331`), no hard tasks.
- All rows in the current high-quality missing manifests are now submitted or already running across r08-r20. First-pass selected coverage beyond the prior unique dataset is `easy=3,857`, `medium=7,809`, `hard=129`.
- Queue snapshot after r20 submission: `400` visible packed elements, `189` running, `211` pending, all CPU-only `m7i-cpu2`, `JobArrayTaskLimit=0`.

Next action: monitor r17-r20 startup and completion using compact scheduler checks plus targeted quality spot checks. Once first-pass waves finish, generate retry manifests for tasks without an all-reasoning trajectory, especially pre-repair `requests` failures and any malformed/missing-reasoning traces.
