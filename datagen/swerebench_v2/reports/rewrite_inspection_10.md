# Rewrite Inspection: First 10 Prompts

Model: `gpt-5.4-mini`

Command:

```bash
python3 -m datagen.swerebench_v2.rewrite_prompts --model gpt-5.4-mini --limit 10
```

Output directory:

```text
datagen/swerebench_v2/examples/rewritten-prompts/
```

## Verdict

The first 10 rewrites are usable as DeepSWE-style prompts. They preserve the
core behavioral requirements while removing issue-template boilerplate, external
links, stack traces, PR/file references, and most implementation guidance.

Two automated flags were reviewed manually and accepted:

- `aftership__clickhouse-sql-parser-123` contains `test-event` as literal SQL example data, not a benchmark-test reference.
- `burntsushi__toml-418` contains `Metadata.Keys()` because it is the public API under test and is necessary for the task.

## Rewritten Tasks

### 99designs__aws-vault-1178

Good. The rewrite keeps the role + web identity credential-source behavior and
removes issue checkboxes and URLs.

### 99designs__gqlgen-3276

Good. The rewrite preserves the `Accept: text/event-stream` transport-selection
bug and distinguishes regular queries/mutations from subscriptions.

### aftership__clickhouse-sql-parser-123

Good. The rewrite keeps the `NOT isZeroOrNull(...)` semantic requirement and
includes a concrete SQL example. It is longer than the others, but the example
is useful behavior context rather than issue noise.

### aftership__clickhouse-sql-parser-130

Good. The rewrite is concise and preserves the panic-to-normal-error behavior
for `SELECT * FROM`.

### aftership__clickhouse-sql-parser-132

Good. The rewrite preserves materialized view parsing with `ORDER BY` and the
`AS SELECT` ambiguity.

### aftership__clickhouse-sql-parser-147

Good. The rewrite preserves support for `$` in column identifiers and the
expected parser behavior.

### azure__azure-workload-identity-1108

Good. The rewrite preserves sidecar insertion order and the startup/restart
motivation without naming implementation files.

### burntsushi__toml-339

Good. The rewrite preserves empty TOML array decoding into non-nil empty slices,
including nested/interface-typed values.

### burntsushi__toml-418

Good. The rewrite preserves the public `Metadata.Keys()` and `Undecoded`
behavior. The API name should remain in the prompt.

### clusterlabs__ha_cluster_exporter-216

Good. The rewrite preserves parsing tab-vs-space whitespace around `=` in
`corosync-cfgtool -s` output.
