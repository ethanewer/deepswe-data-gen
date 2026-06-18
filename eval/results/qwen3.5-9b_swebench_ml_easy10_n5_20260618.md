<!-- Full artifacts (traces, predictions, reports, logs) live in the gitignored bundle:
     runs/qwen3.5-9b_swebench-ml-easy10_n5_20260618/  -->

# qwen/qwen3.5-9b — SWE-bench Multilingual easy-10 (n=5)

- **Model:** `qwen/qwen3.5-9b` → resolves to `qwen/qwen3.5-9b-20260310 (OpenRouter, Venice provider)`
- **Benchmark:** SWE-bench Multilingual — easy-10 debugging subset
- **Harness:** mini-swe-agent -> official swebench.harness.run_evaluation (Docker)
- **Config:** temperature 0, max_tokens 4096, step_limit 250, provider order venice/together/siliconflow fp8
- **Date:** 2026-06-18  ·  **Trials:** 5  ·  **Instances:** 10

> ⚠️ Easy-10 is a deliberately easier debugging slice, NOT the reportable swebench-multilingual score.

## Per-trial resolved

| Trial | Resolved | Empty patch | Eval error |
|-------|----------|-------------|------------|
| 1 | 5/10 (50%) | 3 | 1 |
| 2 | 5/10 (50%) | 3 | 0 |
| 3 | 6/10 (60%) | 1 | 1 |
| 4 | 5/10 (50%) | 2 | 0 |
| 5 | 5/10 (50%) | 3 | 2 |

## Aggregate

- **Mean resolved:** 5.2/10 = **52.0%**  (min 5, max 6, stdev 0.4)
- **pass@5** (solved in any trial): **8/10 = 80%**
- **pass^5** (solved in all 5 trials): **0/10 = 0%**

## Per-instance (out of 5 trials)

| Solves | Instance | t1 | t2 | t3 | t4 | t5 |
|--------|----------|----|----|----|----|----|
| 4/5 | `nlohmann__json-4237` | ✅ | ✅ | ⬜ | ✅ | ✅ |
| 4/5 | `briannesbitt__carbon-2752` | ✅ | ✅ | ✅ | ✅ | ⬜ |
| 4/5 | `redis__redis-10764` | ✅ | ⬜ | ✅ | ✅ | ✅ |
| 4/5 | `google__gson-2024` | ✅ | ✅ | ✅ | ⬜ | ✅ |
| 4/5 | `php-cs-fixer__php-cs-fixer-8256` | ✅ | ✅ | ✅ | ✅ | ⬜ |
| 3/5 | `phpoffice__phpspreadsheet-4186` | ⬜ | ⬜ | ✅ | ✅ | ✅ |
| 2/5 | `gin-gonic__gin-3741` | ⚠️ | ✅ | ❌ | ❌ | ✅ |
| 1/5 | `vuejs__core-11589` | ⬜ | ⬜ | ✅ | ⬜ | ⬜ |
| 0/5 | `tokio-rs__tokio-4898` | ❌ | ❌ | ⚠️ | ❌ | ⚠️ |
| 0/5 | `apache__lucene-12196` | ⬜ | ❌ | ❌ | ❌ | ⚠️ |

Legend: ✅ resolved · ❌ unresolved (patch failed tests) · ⬜ empty patch · ⚠️ eval error

## Notes

- tokio-rs__tokio-4898 (Rust) and apache__lucene-12196 (Java) never solved in any trial.
- pass^5=0: no instance solved in all 5 trials; the 4/5 instances each had one empty-patch/failing trial (max_tokens=4096 variance for a reasoning model).
- trial-4 vuejs__core-11589 was manually stopped after ~54min (zero source edits in the live container -> guaranteed empty patch) and counted as a failure, per user authorization.

## Layout

```
aggregate.json          machine-readable results
instance_ids.txt        the 10-task subset
reports/trial-N.report.json    official swebench evaluation reports
predictions/trial-N.preds.json model patches per instance
traces/trial-N/<instance>/<instance>.traj.json   full agent trajectories
logs/                   runner/agent/eval logs + launch scripts output
run_scripts/            launcher scripts used
```
