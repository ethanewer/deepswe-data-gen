# Prompt Rewrite Validation: 15-Task Sample

Date: 2026-06-03

## Sample

The validation sample used 5 easy, 5 medium, and 5 hard tasks, with each difficulty split across Python, TypeScript, and Go where possible:

| Difficulty | Python | TypeScript | Go |
| --- | --- | --- | --- |
| easy | `3yourmind__django-migration-linter-113`, `3yourmind__django-migration-linter-222` | `8398a7__action-slack-120`, `8398a7__action-slack-184` | `99designs__aws-vault-1178` |
| medium | `3yourmind__django-migration-linter-186`, `arm-doe__act-651` | `0xs34n__starknet.js-490`, `0xs34n__starknet.js-508` | `0xpolygonhermez__zkevm-node-1044` |
| hard | `axelrod-python__axelrod-975`, `dai-lab__copulas-100` | `designliquido__delegua-742`, `nomicfoundation__hardhat-ignition-505` | `tbd54566975__ssi-service-552` |

Initial rewrite command:

```bash
python3 -m datagen.swerebench_v2.rewrite_prompts \
  --model gpt-5.4-mini \
  --output-dir datagen/swerebench_v2/examples/rewritten-prompts-validation-15 \
  --limit 15 \
  --instance-id ...
```

The final current-pipeline sample is saved in:

```text
datagen/swerebench_v2/examples/rewritten-prompts-validation-15-final
```

## Subagent Review

Three subagents reviewed the initial 15-task sample by difficulty:

- Easy: 4 good, 1 issue.
- Medium: 4 good, 1 issue.
- Hard: 3 good, 2 issues.

The four issues were:

- `3yourmind__django-migration-linter-113`: lost too much of `MigrationLinter.read_migrations_list` public return/error contract.
- `0xpolygonhermez__zkevm-node-1044`: lost exported config/function names such as `EncodeUnsignedTransaction`, `NetworkConfig.L1ChainID`, `NetworkConfig.L2ChainID`, and `Config.ChainID`.
- `tbd54566975__ssi-service-552`: lost concrete public behavior for `Container.IsValid`, `NewCredentialContainerFromJWT`, and `NewCredentialContainerFromMap`.
- `dai-lab__copulas-100`: over-compressed constructor/default/from_dict/_fit_params compatibility for public univariate distribution APIs.

The reviewers also identified many quality-warning false positives caused by illustrative literals, schema dumps, redacted placeholders, screenshot captions, and example function names.

## Pipeline Changes

The pipeline was revised to:

- Treat generated interface notes as public compatibility notes when they describe callable signatures, constructor/default behavior, return shapes, exported fields, serialized/config keys, or import/export compatibility.
- Explicitly list likely required public symbols in the LLM input and instruct the model to keep their exact names unless clearly private/internal.
- Accept split qualified references like "`simulateTransaction` on `Account`" when checking for `Account.simulateTransaction`.
- Reduce noisy literal warnings by ignoring contractions, numeric IDs, redacted placeholders, screenshot captions, long prose fragments, ellipsis examples, version strings, file paths, and quoted literals without behavior-signaling context.
- Instruct the model to keep the same natural language as the original task and avoid introducing unrelated writing systems.

## Final Validation

After the pipeline changes:

- The four previously problematic tasks were rerun and produced rewrites preserving the missing public API/compatibility contracts.
- A final 15-task sample was rerun with the current pipeline.
- An additional warning-filter refinement ignored quoted literals inside fenced reproduction code blocks.
- After recomputing quality warnings with the current checker, all 15 final sample rewrites had zero quality warnings.

Local verification:

```bash
python3 -m pytest -q
python3 -m py_compile datagen/swerebench_v2/*.py
```

Result: 8 tests passed; compile checks passed.

## Verdict

The improved pipeline is working correctly on the requested 15-task validation sample. The main substantive gap found by manual review was under-preservation of public interface contracts; the current pipeline now addresses that by making public symbols and compatibility details explicit inputs to the rewrite model and by checking the output afterward.

No full corpus rollout was performed.
