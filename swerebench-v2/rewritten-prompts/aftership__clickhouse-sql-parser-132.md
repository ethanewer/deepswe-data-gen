# aftership__clickhouse-sql-parser-132

- repo: AfterShip/clickhouse-sql-parser
- language: go
- difficulty: easy

## Rewritten Prompt

Materialized view definitions should parse correctly when an `ORDER BY` clause is present. In particular, a statement like `CREATE MATERIALIZED VIEW IF NOT EXISTS ... ENGINE = ReplacingMergeTree() PRIMARY KEY (id) ORDER BY (id) AS SELECT * FROM test_table;` should be accepted.

The parser should treat the `AS SELECT ...` part as the materialized view query, not as an alias attached to the `ORDER BY` expression.

## Preserved Requirements

- Materialized view creation must work when `ORDER BY` is specified.
- `AS SELECT * FROM test_table` must be parsed as the materialized view query.
- The parser must not misinterpret `AS SELECT` after `ORDER BY (id)` as an alias for the `id` expression.

## Removed Noise

- Issue-template style exposition and reproduction formatting.
- The exact parser error message and caret location.
- External GitHub URL and code reference.
- Mention of a specific source line and internal implementation hint.
- PR/test references and metadata.

## Risk Notes

- The intended behavior is inferred from one failing SQL example; other valid MV syntaxes may also need to continue parsing.
- The prompt does not specify whether the fix should affect only materialized views or other statements that can combine `ORDER BY` with `AS SELECT`.

## Original Prompt

Create materialized view fails when specifying an ORDER BY clause
See this MV setup:

```
CREATE MATERIALIZED VIEW IF NOT EXISTS test_mv
ENGINE = ReplacingMergeTree()
PRIMARY KEY (id)
ORDER BY (id)
AS
SELECT * FROM test_table;
```

Which fails to create:

```
Received unexpected error:
line 5:7 <EOF> or ';' was expected, but got: "*"
SELECT * FROM test_table;
       ^
```

The issue seems to be that the parser interprets `AS SELECT` in `ORDER BY (id) AS SELECT * FROM test_table` as alias for the id column instead of the query for the materialized view in [here](https://github.com/AfterShip/clickhouse-sql-parser/blob/d03ad5b77f8c0621486596d2bc7e2bb8af2ffe7a/parser/parser_table.go#L672).

## Original Interface

No new interfaces are introduced.
