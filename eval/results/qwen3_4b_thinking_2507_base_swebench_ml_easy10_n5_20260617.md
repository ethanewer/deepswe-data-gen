<!-- Full artifacts are the gitignored official report JSONs named:
     openai__Qwen__Qwen3-4B-Thinking-2507.qwen3-4b-thinking-base-base-65k-h200-2gpu-swebench-multilingual-easy-easy-n5-t*-20260616-manual.json  -->

# Qwen/Qwen3-4B-Thinking-2507 base — SWE-bench Multilingual easy-10 (n=5)

- **Model:** `Qwen/Qwen3-4B-Thinking-2507`
- **Benchmark:** SWE-bench Multilingual — easy-10 debugging subset
- **Harness:** mini-swe-agent -> official swebench.harness.run_evaluation (Docker)
- **Config:** H200 local vLLM serving, 2 GPUs per run, 65k context, temperature 0.6, top_p 0.95, top_k 20, max_tokens 8192, step_limit 250
- **Date:** 2026-06-17  ·  **Trials:** 5  ·  **Instances:** 10

> ⚠️ Easy-10 is a deliberately easier debugging slice, NOT the reportable swebench-multilingual score.

## Per-trial resolved

| Trial | Resolved | Empty patch | Eval error |
|-------|----------|-------------|------------|
| 1 | 0/10 (0%) | 4 | 4 |
| 2 | 1/10 (10%) | 4 | 2 |
| 3 | 1/10 (10%) | 3 | 2 |
| 4 | 0/10 (0%) | 5 | 1 |
| 5 | 1/10 (10%) | 3 | 2 |

## Aggregate

- **Mean resolved:** 0.6/10 = **6.0%**  (min 0, max 1, stdev 0.49)
- **Total resolved:** **3/50 = 6.0%**
- **pass@5** (solved in any trial): **1/10 = 10%**
- **pass^5** (solved in all 5 trials): **0/10 = 0%**

## Per-instance (out of 5 trials)

| Solves | Instance | t1 | t2 | t3 | t4 | t5 |
|--------|----------|----|----|----|----|----|
| 3/5 | `tokio-rs__tokio-4898` | ⬜ | ✅ | ✅ | ❌ | ✅ |
| 0/5 | `apache__lucene-12196` | ⚠️ | ⚠️ | ❌ | ⬜ | ⬜ |
| 0/5 | `briannesbitt__carbon-2752` | ⬜ | ⬜ | ⬜ | ⬜ | ⚠️ |
| 0/5 | `gin-gonic__gin-3741` | ⬜ | ❌ | ⬜ | ⬜ | ❌ |
| 0/5 | `google__gson-2024` | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| 0/5 | `nlohmann__json-4237` | ⚠️ | ⚠️ | ⚠️ | ⬜ | ⚠️ |
| 0/5 | `php-cs-fixer__php-cs-fixer-8256` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| 0/5 | `phpoffice__phpspreadsheet-4186` | ❌ | ⬜ | ❌ | ❌ | ❌ |
| 0/5 | `redis__redis-10764` | ⚠️ | ⬜ | ⚠️ | ❌ | ⬜ |
| 0/5 | `vuejs__core-11589` | ⚠️ | ❌ | ❌ | ❌ | ❌ |

Legend: ✅ resolved · ❌ unresolved (patch failed tests) · ⬜ empty patch · ⚠️ eval error

## Notes

- Only `tokio-rs__tokio-4898` solved in any trial, resolving in 3/5 trials.
- This easy-10 n=5 baseline was not present in `origin/main` before this commit; `origin/main` already had the older 30-task `Qwen/Qwen3-4B-Thinking-2507` baseline in `eval/benchmark_results.md`.

## Report files

```
openai__Qwen__Qwen3-4B-Thinking-2507.qwen3-4b-thinking-base-base-65k-h200-2gpu-swebench-multilingual-easy-easy-n5-t1-20260616-manual.json
openai__Qwen__Qwen3-4B-Thinking-2507.qwen3-4b-thinking-base-base-65k-h200-2gpu-swebench-multilingual-easy-easy-n5-t2-20260616-manual.json
openai__Qwen__Qwen3-4B-Thinking-2507.qwen3-4b-thinking-base-base-65k-h200-2gpu-swebench-multilingual-easy-easy-n5-t3-20260616-manual.json
openai__Qwen__Qwen3-4B-Thinking-2507.qwen3-4b-thinking-base-base-65k-h200-2gpu-swebench-multilingual-easy-easy-n5-t4-20260616-manual.json
openai__Qwen__Qwen3-4B-Thinking-2507.qwen3-4b-thinking-base-base-65k-h200-2gpu-swebench-multilingual-easy-easy-n5-t5-20260616-manual.json
```
