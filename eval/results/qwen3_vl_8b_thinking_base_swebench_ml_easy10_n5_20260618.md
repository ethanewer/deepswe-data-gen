<!-- Full artifacts (official reports and filled stopped-run predictions) live in the gitignored bundle:
     runs/swebench_ml/qwen3vl8b_base_stopped_h200_grading/  -->

# Qwen/Qwen3-VL-8B-Thinking base — SWE-bench Multilingual easy-10 (n=5)

- **Model:** `Qwen/Qwen3-VL-8B-Thinking`
- **Benchmark:** SWE-bench Multilingual — easy-10 debugging subset
- **Harness:** mini-swe-agent -> official swebench.harness.run_evaluation (Docker)
- **Config:** local H200 vLLM, 65k context, mini-swe-agent step_limit 250; t1 completed as a manual 1-GPU report, t2-t5 were single-node H200 jobs stopped before generation finished
- **Date:** 2026-06-18  ·  **Trials:** 5  ·  **Instances:** 10

> ⚠️ Easy-10 is a deliberately easier debugging slice, NOT the reportable swebench-multilingual score.

> ⚠️ This is a stopped-partial score, not a clean fully completed n=5 generation run. Trials 2-5 were stopped on request; generated predictions were graded with the official harness, and missing stopped instances were filled with empty patches counted unresolved.

## Per-trial resolved

| Trial | Run status | Generated before stop | Resolved | Empty patch | Eval error |
|-------|------------|-----------------------|----------|-------------|------------|
| 1 | complete official report | 10/10 | 3/10 (30%) | 4 | 0 |
| 2 | stopped, missing filled empty, official graded | 6/10 | 0/10 (0%) | 4 | 1 |
| 3 | stopped, missing filled empty, official graded | 6/10 | 0/10 (0%) | 6 | 0 |
| 4 | stopped, missing filled empty, official graded | 7/10 | 2/10 (20%) | 5 | 0 |
| 5 | stopped, missing filled empty, official graded | 7/10 | 1/10 (10%) | 5 | 1 |

## Aggregate

- **Mean resolved:** 1.2/10 = **12.0%**  (min 0, max 3, stdev 1.17)
- **Stopped-partial total:** **6/50 = 12.0%**
- **pass@5** (solved in any trial): **4/10 = 40%**
- **pass^5** (solved in all 5 trials): **0/10 = 0%**

## Per-instance (out of 5 trials)

| Solves | Instance | t1 | t2 | t3 | t4 | t5 |
|--------|----------|----|----|----|----|----|
| 2/5 | `gin-gonic__gin-3741` | ✅ | ❌ | ❌ | ✅ | ❌ |
| 2/5 | `nlohmann__json-4237` | ✅ | ⬜ | ⬜ | ✅ | ⬜ |
| 1/5 | `apache__lucene-12196` | ✅ | ⬜ | ⬜ | ⬜ | ⬜ |
| 1/5 | `phpoffice__phpspreadsheet-4186` | ⬜ | ❌ | ❌ | ❌ | ✅ |
| 0/5 | `briannesbitt__carbon-2752` | ⬜ | ⬜ | ⬜ | ❌ | ⬜ |
| 0/5 | `google__gson-2024` | ❌ | ⚠️ | ⬜ | ⬜ | ⚠️ |
| 0/5 | `php-cs-fixer__php-cs-fixer-8256` | ⬜ | ❌ | ⬜ | ⬜ | ⬜ |
| 0/5 | `redis__redis-10764` | ❌ | ❌ | ❌ | ❌ | ❌ |
| 0/5 | `tokio-rs__tokio-4898` | ❌ | ❌ | ❌ | ⬜ | ❌ |
| 0/5 | `vuejs__core-11589` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

Legend: ✅ resolved · ❌ unresolved (patch failed tests) · ⬜ empty patch · ⚠️ eval error

## Notes

- The earlier conservative score of 3/50 counted every stopped trial as a full failure. Official grading of partial stopped predictions added two resolves from t4 and one resolve from t5, yielding the saved 6/50 score.
- New-checkpoint eval jobs were cancelled on request and are excluded from this base-checkpoint result.
- The briefly tested LR 1e-5 checkpoint failed startup before cancellation with a vLLM weight-load mismatch: embedding/lm-head tensors were `[75968, 4096]` while the config expected vocab size `151936`.

## Layout

```
runs/swebench_ml/qwen3vl8b_base_swebench_multilingual_easy_n5_score_summary.json
runs/swebench_ml/qwen3vl8b_base_stopped_h200_grading/
  openai__Qwen__Qwen3-VL-8B-Thinking.*.json
  t2_preds_missing_empty.json
  t3_preds_missing_empty.json
  t4_preds_missing_empty.json
  t5_preds_missing_empty.json
```
