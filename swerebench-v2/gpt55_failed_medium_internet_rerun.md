# gpt-5.5 Rerun: Failed Medium Tasks With Internet

Date: 2026-06-03

## Setup

The rerun targeted the six medium rewritten-prompt tasks that failed or did not complete in the prior oracle/reference verifier run. Those prompts had been rewritten with `gpt-5.4-mini`.

- `0xs34n__starknet.js-490`
- `0xs34n__starknet.js-508`
- `0xs34n__starknet.js-520`
- `33cn__plugin-813`
- `arm-doe__act-651`
- `aspp__pelita-412`

Generated Harbor tasks used rewritten instructions and enabled task internet access:

```bash
python3 scripts/generate_harbor_tasks.py \
  --clean \
  --output-dir swerebench-v2/harbor-tasks \
  --instruction-style rewritten \
  --rewrites-file swerebench-v2/rewritten-prompts-medium-10/rewrites.jsonl \
  --allow-internet \
  --instance-id 0xs34n__starknet.js-490 \
  --instance-id 0xs34n__starknet.js-508 \
  --instance-id 0xs34n__starknet.js-520 \
  --instance-id 33cn__plugin-813 \
  --instance-id arm-doe__act-651 \
  --instance-id aspp__pelita-412
```

The rerun command was:

```bash
python3 scripts/run_deepswe.py \
  --tasks-dir swerebench-v2/harbor-tasks \
  --model openai/gpt-5.5 \
  --limit 6 \
  --n-concurrent 5 \
  --jobs-dir runs/pier-gpt55-failed-medium-internet-pip-py311
```

Run directory:

```text
runs/pier-gpt55-failed-medium-internet-pip-py311/2026-06-03__14-48-40
```

Note: Pier's packaged mini-swe-agent installer was locally patched during this run because the Docker build path failed on TLS certificate verification while downloading `uv`. The local workaround installed `mini-swe-agent` with container Python/pip and trusted PyPI hosts. This was a local harness-install workaround and is not committed to this repository.

## Aggregate Result

Pier reported:

```json
{
  "n_total_trials": 6,
  "n_completed_trials": 6,
  "n_errored_trials": 1,
  "n_running_trials": 0,
  "n_pending_trials": 0,
  "n_cancelled_trials": 1,
  "n_input_tokens": 3826555,
  "n_cache_tokens": 3442688,
  "n_output_tokens": 57287,
  "cost_usd": 5.359289
}
```

No task passed. Five tasks reached the verifier and received reward `0.0`. `33cn__plugin-813` did not reach a verifier result because the run was cancelled after the agent became stuck repeatedly launching long-running Go tests under the per-command timeout.

## Task Results

| Task | Language | Result | Failure classification |
| --- | --- | --- | --- |
| `0xs34n__starknet.js-490` | TypeScript | Reward `0.0` | Verifier environment issue dominates: Starknet tests require local devnet at `127.0.0.1:5050`, which refused connections. |
| `0xs34n__starknet.js-508` | TypeScript | Reward `0.0` | Model likely solved the task-specific short-string issue; focused tests passed. Full verifier still failed on missing local Starknet devnet. |
| `0xs34n__starknet.js-520` | TypeScript | Reward `0.0` | Mixed: missing local Starknet devnet plus an actual model miss in `useDecoded([])`, which returned `"stark"` instead of `""`. |
| `33cn__plugin-813` | Go | Cancelled | Harness/runtime issue. Internet allowed setup progress, but the agent became stuck around Go test commands and no verifier result was produced. |
| `arm-doe__act-651` | Python | Reward `0.0` | Model did not complete the task-specific histogram behavior; verifier also had unrelated plotting fragility. |
| `aspp__pelita-412` | Python | Reward `0.0` | Model understood the broad task but missed package export wiring, causing `ImportError` for `SteppingPlayer`. |

## Failure Analysis

### `0xs34n__starknet.js-490`

The verifier failed primarily because the Starknet test suite tried to connect to `http://127.0.0.1:5050/feeder_gateway/...` and got `ECONNREFUSED`. This is not a web-access failure. It requires a local Starknet devnet/service inside the verifier environment.

Classification: environment/verifier setup, not prompt misunderstanding.

### `0xs34n__starknet.js-508`

The agent changed `src/utils/shortString.ts` and its focused short-string test passed locally inside the agent run, including decimal and unprefixed hex decode coverage. The full verifier still failed because suites such as `account.test.ts` and `contract.test.ts` required the same missing local Starknet devnet endpoint.

Classification: likely task-specific success masked by verifier environment failure.

### `0xs34n__starknet.js-520`

The agent changed Starknet ID handling but missed a concrete expected behavior: verifier output showed `useDecoded([])` expected `""` and received `"stark"`. This task also had the same missing local devnet failures as the other Starknet tasks.

Classification: model/incomplete-solution failure plus environment noise.

### `33cn__plugin-813`

The run did not produce a verifier result. The agent repeatedly ran Go tests that exceeded mini-swe-agent's command timeout and left long-running test processes. The trial was manually interrupted and recorded `CancelledError`.

Classification: harness/runtime failure; not enough evidence to decide whether `gpt-5.5` solved the task.

### `arm-doe__act-651`

The verifier reported both a task-specific `test_histogram_kwargs` assertion failure and an unrelated `test_plot_datarose` `ValueError` about dimension mismatch. The task-specific assertion means the generated patch was not sufficient.

Classification: model/incomplete-solution failure, with additional unrelated verifier fragility.

### `aspp__pelita-412`

The agent implemented/renamed `SteppingPlayer` in `pelita/player/base.py`, but the verifier failed during collection because `SteppingPlayer` could not be imported from `pelita.player`. The likely missing piece was export wiring in `pelita/player/__init__.py`.

Classification: model/incomplete-solution failure, not prompt misunderstanding.

## Conclusion

Rerunning the failed medium tasks with `gpt-5.5` and internet access was not enough to make them pass.

Internet access helped avoid the no-network class of setup failures, but it did not fix local-service dependencies. The three Starknet tasks need verifier support for the local devnet at `127.0.0.1:5050` or a narrower verifier command that excludes endpoint-dependent suites. Among the remaining tasks, `arm-doe__act-651` and `aspp__pelita-412` were genuine incomplete model solutions, while `33cn__plugin-813` needs harness/runtime handling before the model result can be judged.
