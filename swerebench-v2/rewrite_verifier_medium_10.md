# Rewritten Medium Prompt Verifier Check

Generated 10 medium task rewrites with a mixed language sample:

- Go: 4 tasks
- Python: 3 tasks
- TypeScript: 3 tasks

Rewrite command:

```bash
python3 scripts/rewrite_prompts.py \
  --model gpt-5.4-mini \
  --output-dir swerebench-v2/rewritten-prompts-medium-10 \
  --limit 10 \
  --instance-id 0xpolygonhermez__zkevm-node-1044 \
  --instance-id 0xpolygonhermez__zkevm-node-1321 \
  --instance-id 33cn__plugin-813 \
  --instance-id 99designs__aws-vault-1196 \
  --instance-id 3yourmind__django-migration-linter-186 \
  --instance-id arm-doe__act-651 \
  --instance-id aspp__pelita-412 \
  --instance-id 0xs34n__starknet.js-490 \
  --instance-id 0xs34n__starknet.js-508 \
  --instance-id 0xs34n__starknet.js-520
```

Generated task set:

```bash
python3 scripts/generate_harbor_tasks.py \
  --clean \
  --instruction-style rewritten \
  --rewrites-file swerebench-v2/rewritten-prompts-medium-10/rewrites.jsonl \
  --limit 10
```

Verifier run:

```bash
pier run -p swerebench-v2/harbor-tasks \
  --agent oracle \
  --jobs-dir runs/pier-oracle-rewritten-medium-conc5 \
  --n-tasks 10 \
  --n-concurrent 5 \
  --yes
```

Result directory:

```text
runs/pier-oracle-rewritten-medium-conc5/2026-06-03__14-05-56
```

## Result

| Task | Language | Reward | Notes |
|---|---|---:|---|
| `0xpolygonhermez__zkevm-node-1044` | Go | 1.0 | Reference solution passed verifier. |
| `0xpolygonhermez__zkevm-node-1321` | Go | 1.0 | Reference solution passed verifier. |
| `33cn__plugin-813` | Go | 0.0 | Verifier tried to fetch Go modules from `proxy.golang.org`, but Pier ran the task with network disabled. |
| `99designs__aws-vault-1196` | Go | 1.0 | Reference solution passed verifier. |
| `3yourmind__django-migration-linter-186` | Python | 1.0 | Reference solution passed verifier. |
| `arm-doe__act-651` | Python | 0.0 | Existing plotting verifier failed at `test_plot_datarose` with a dimension mismatch. |
| `aspp__pelita-412` | Python | n/a | Verifier hung after `test/test_simplesetup.py .`; run was cancelled after no progress. |
| `0xs34n__starknet.js-490` | TypeScript | 0.0 | Starknet.js verifier had multiple failing suites; failures were concentrated in provider/endpoint-dependent tests. |
| `0xs34n__starknet.js-508` | TypeScript | 0.0 | Starknet.js verifier had multiple failing suites; failures were concentrated in provider/endpoint-dependent tests. |
| `0xs34n__starknet.js-520` | TypeScript | 0.0 | Starknet.js verifier had multiple failing suites; failures were concentrated in provider/endpoint-dependent tests. |

Summary:

- 10 rewritten medium tasks were generated and manually inspected.
- Pier ran the oracle/reference solution with `--n-concurrent 5`.
- 4/10 produced reward `1.0`.
- 5/10 produced reward `0.0`.
- 1/10 was cancelled after the verifier stopped making progress.

## Interpretation

The medium sample is mixed: Go and Python tasks with self-contained verifier
dependencies mostly worked, while tasks depending on disabled network access,
local endpoints, or unstable plotting/runtime behavior did not. These failures
do not indicate that the rewritten prompts are unsolvable by themselves; they
show that this sampled Harbor packaging plus Pier verifier environment is not
uniformly clean for medium tasks.

As with the easy check, this validates the rewritten-prompt packaging and
reference-solution verifier path, not agent solve rate from the prompt.
