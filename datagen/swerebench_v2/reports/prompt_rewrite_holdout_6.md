# Prompt Rewrite Holdout Check: 6 Tasks

Date: 2026-06-03

## Purpose

After the 15-task validation and pipeline revision, a small unseen holdout sample was rewritten to check for obvious regressions without doing a full rollout.

## Sample

| Difficulty | Task | Language |
| --- | --- | --- |
| easy | `99designs__gqlgen-3276` | Go |
| easy | `3yourmind__django-migration-linter-258` | Python |
| medium | `0xs34n__starknet.js-520` | TypeScript |
| medium | `0xpolygonhermez__zkevm-node-1321` | Go |
| hard | `gridtools__gt4py-1449` | Python |
| hard | `rezact__rezact-43` | TypeScript |

Command:

```bash
python3 -m datagen.swerebench_v2.rewrite_prompts \
  --model gpt-5.4-mini \
  --output-dir datagen/swerebench_v2/examples/rewritten-prompts-validation-holdout-6 \
  --limit 6 \
  --instance-id 99designs__gqlgen-3276 \
  --instance-id 3yourmind__django-migration-linter-258 \
  --instance-id 0xs34n__starknet.js-520 \
  --instance-id 0xpolygonhermez__zkevm-node-1321 \
  --instance-id gridtools__gt4py-1449 \
  --instance-id rezact__rezact-43
```

## Result

All six holdout rewrites had zero quality warnings after the current checker was applied.

Manual spot-check:

- `0xs34n__starknet.js-520` preserved the `Account.getStarkName()` public method and the `""` versus `"stark"` edge case.
- `0xpolygonhermez__zkevm-node-1321` preserved `etherman.GetPublicAddress` and `Client.IsReadOnly`.
- `3yourmind__django-migration-linter-258` preserved the partial-index `IS NOT NULL` versus true `NOT NULL` constraint distinction.
- `rezact__rezact-43` kept the fragment/signal behavior example because the JSX example is central to the user-visible bug.

The only initial warning was `missing_edge_literal:prediction`, caused by a quoted model name inside a fenced reproduction code block. The checker was refined to ignore quoted literals inside fenced code examples, and warnings were recomputed.

## Verdict

The holdout check did not reveal any new prompt-rewrite pipeline issues.
