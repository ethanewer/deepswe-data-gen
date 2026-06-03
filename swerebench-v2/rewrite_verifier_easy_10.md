# Rewritten Easy Prompt Verifier Check

Generated task set:

```bash
python3 scripts/generate_harbor_tasks.py --clean --limit 10 --difficulty easy --instruction-style rewritten
```

Verifier run:

```bash
pier run -p swerebench-v2/harbor-tasks \
  --agent oracle \
  --jobs-dir runs/pier-oracle-rewritten-easy-serial-10 \
  --n-tasks 10 \
  --n-concurrent 1 \
  --yes
```

Result directory:

```text
runs/pier-oracle-rewritten-easy-serial-10/2026-06-03__13-22-19
```

## Result

| Task | Reward | Notes |
|---|---:|---|
| `99designs__aws-vault-1178` | 1.0 | Reference solution passed verifier. |
| `99designs__gqlgen-3276` | 1.0 | Reference solution passed verifier. |
| `aftership__clickhouse-sql-parser-123` | 1.0 | Reference solution passed verifier. |
| `aftership__clickhouse-sql-parser-130` | 1.0 | Reference solution passed verifier. |
| `aftership__clickhouse-sql-parser-132` | 1.0 | Reference solution passed verifier. |
| `aftership__clickhouse-sql-parser-147` | 1.0 | Reference solution passed verifier. |
| `azure__azure-workload-identity-1108` | 0.0 | Verifier tried to fetch Go modules from `proxy.golang.org`, but Pier ran the task with network disabled. |
| `burntsushi__toml-339` | 1.0 | Reference solution passed verifier. |
| `burntsushi__toml-418` | 1.0 | Reference solution passed verifier. |
| `clusterlabs__ha_cluster_exporter-216` | 1.0 | Reference solution passed verifier. |

Summary:

- 10 rewritten easy tasks were run through Pier with the oracle/reference solution.
- 9/10 produced reward `1.0`.
- 1/10 produced reward `0.0` due to missing network access for dependency downloads, not prompt content.
- Pier reported zero exceptions in the final run.

## Interpretation

This validates that the rewritten prompts can be packaged as Pier/Harbor tasks
without breaking the verifier path, and that the known reference patches still
solve 9 of the 10 sampled tasks under the verifier.

This does not prove that `mini-swe-agent` can solve the tasks from the rewritten
prompts. The earlier mini-swe-agent smoke run was blocked during Docker agent
installation by a local certificate-chain error while fetching `uv`.
