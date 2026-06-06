# Benchmark results

Date: 2026-06-06 UTC

These runs used the local 8-GPU setup. Runtimes are benchmark wall-clock times after a serving endpoint was available, unless a note says otherwise. For LiveCodeBench, the primary score is pass@1 and pass@3 is included as additional context. For SWE-bench Multilingual, the score is resolved instances over the 30-task subset used in these runs.

## LiveCodeBench

| Model | Setting | Score | Runtime | Run artifact |
| --- | --- | --- | --- | --- |
| `allenai/Olmo-3-7B-Think` | v6, 50 tasks, n=3 | pass@1 18.67%; pass@3 24.00% | 74.40s (0:01:14) | `runs/olmo3-7b-think-livecodebench-v6-8gpu-20260605T044048Z/timing.json` |
| `Qwen/Qwen3.5-9B` | v6, 50 tasks, n=3 | pass@1 26.67%; pass@3 32.00% | 55.26s (0:00:55) | `runs/qwen3_5_9b-livecodebench-v6-8gpu-20260605T045225Z/timing.json` |
| `Qwen/Qwen3.5-4B` | v6, 50 tasks, n=3 | pass@1 35.33%; pass@3 50.00% | 424.58s (0:07:05) | `runs/qwen3_5_4b-livecodebench-v6-8gpu-mtp-nothink-81920-20260605T193444Z/` |
| `Qwen/Qwen3-4B-Thinking-2507` | v6, 175 tasks, n=3 | pass@1 53.90%; pass@3 61.71% | 1148.71s (0:19:09) | `runs/qwen3_4b_thinking_2507-livecodebench-v6-175-8gpu-65k-20260605T200735Z/` |
| `Qwen/Qwen3-VL-8B-Thinking` | v6, 175 tasks, n=3 | pass@1 47.24%; pass@3 58.29% | 3882.62s (1:04:43) | `runs/qwen3_vl_8b_thinking-livecodebench-v6-175-8gpu-65k-20260605T231440Z/` |

## SWE-bench Multilingual

| Model | Setting | Score | Completed / empty / errors | Runtime | Report artifact |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-9B` | 30-task subset | 6/30 resolved (20.00%) | 16 / 13 / 1 | ~4217s (1:10:17) | `openai__Qwen__Qwen3.5-9B.qwen3_5_9b-swebench-multilingual-30-8gpu-131k-thinking-parser-stripreason-20260605T063818Z.json` |
| `Qwen/Qwen3.5-4B` | 30-task subset | 3/30 resolved (10.00%) | 13 / 17 / 0 | 4678.10s (1:17:58) | `openai__Qwen__Qwen3.5-4B.qwen3_5_4b-swebench-multilingual-30-8gpu-262k-mtp-tools-default-20260606T043842Z.json` |
| `Qwen/Qwen3-4B-Thinking-2507` | 30-task subset | 1/30 resolved (3.33%) | 7 / 17 / 6 | ~5248s (1:27:28) | `openai__Qwen__Qwen3-4B-Thinking-2507.qwen3_4b_thinking_2507-swebench-multilingual-30-8gpu-65k-default-eval-cpu-cli-20260605T224652Z.json` |
| `Qwen/Qwen3-VL-8B-Thinking` | 30-task subset | 2/30 resolved (6.67%) | 15 / 15 / 0 | 13673.07s (3:47:53) | `openai__Qwen__Qwen3-VL-8B-Thinking.qwen3_vl_8b_thinking-swebench-multilingual-30-8gpu-65k-default-20260606T002046Z.json` |

Notes:

- The `Qwen/Qwen3.5-9B` SWE runtime is reconstructed from generation and evaluation logs because that run did not leave a single `wall_time.txt`.
- The `Qwen/Qwen3-4B-Thinking-2507` SWE runtime is the useful end-to-end generation plus CPU evaluation rerun. The original wrapper wall-time file recorded only a failed intermediate segment.
- `allenai/Olmo-3-7B-Think` is not listed for SWE-bench Multilingual because the SWE attempts were diagnostic and abandoned after degenerate traces; no successful scored subset run was produced.
