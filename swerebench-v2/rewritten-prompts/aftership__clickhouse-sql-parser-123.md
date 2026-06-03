# aftership__clickhouse-sql-parser-123

- repo: AfterShip/clickhouse-sql-parser
- language: go
- difficulty: easy

## Rewritten Prompt

The SQL parser should preserve logical negation when parsing `NOT` applied to a function call. In particular, a query containing `NOT isZeroOrNull(x)` must be represented as a `NOT` expression, not rewritten into arithmetic negation or any other invalid form.

Given a statement like the one below, the parser should keep the `WHERE` condition semantically equivalent and produce valid SQL output:

```sql
CREATE MATERIALIZED VIEW infra_bm.view_name
    ON CLUSTER 'default_cluster' TO infra_bm.table_name
(
  `f1` DateTime64(3),
  `f2` String,
  `f3` String,
  `f4` String,
  `f5` String,
  `f6` Int64
) AS
SELECT f1,
       f2,
       visitParamExtractString(properties, 'f3') AS f3,
       visitParamExtractString(properties, 'f4') AS f4,
       visitParamExtractString(properties, 'f5') AS f5,
       visitParamExtractInt(properties, 'f6') AS f6
FROM infra_bm.table_name1
WHERE infra_bm.table_name1.event = 'test-event' AND
      NOT isZeroOrNull(f2)
```

The translated result must not turn `NOT isZeroOrNull(f2)` into `-isZeroOrNull(f2)` or another invalid expression.

## Preserved Requirements

- Parsing should preserve `NOT` applied to a function call.
- `NOT isZeroOrNull(x)` must remain semantically equivalent in the parsed/translated SQL.
- The parser must not rewrite the negation into `-isZeroOrNull(x)` or an invalid expression.
- The example materialized view query should remain valid after parsing/translation.

## Removed Noise

- Issue template / bug report framing.
- Reference to current behavior and incorrect translated output as a verbose before/after comparison.
- Mention of ClickHouse error code `43`.
- Comment about generated interface notes being none.
- Repository metadata and language metadata.
- External URLs, PR/test references, and solution hints (none present explicitly).

## Risk Notes

- The exact AST shape for negation is not specified; only the observable SQL behavior is clear.
- The parser may have multiple serialization paths, so the fix should cover any path that can misrender `NOT` on function calls.
- The example includes `CREATE MATERIALIZED VIEW` context, but the core requirement is broader: preserve `NOT` semantics in parsed SQL output.

## Original Prompt

Bug in Query Parser - Incorrect Translation of NOT isZeroOrNull(x)
### Description:

We have identified a bug in the query parser responsible for parsing  statements. The parser incorrectly translates the NOT isZeroOrNull(x) function, transforming it into an invalid statement. This issue needs to be addressed to ensure the parser correctly handles negations of functions.

**Current Behavior:**
The statement:

```
CREATE MATERIALIZED VIEW infra_bm.view_name 
    ON CLUSTER 'default_cluster' TO infra_bm.table_name
(
  `f1` DateTime64(3), 
  `f2` String, 
  `f3` String, 
  `f4` String, 
  `f5` String, 
  `f6` Int64
) AS
SELECT f1,
       f2,
       visitParamExtractString(properties, 'f3') AS f3,
       visitParamExtractString(properties, 'f4') AS f4,
       visitParamExtractString(properties, 'f5') AS f5,
       visitParamExtractInt(properties, 'f6') AS f6
FROM infra_bm.table_name1
WHERE infra_bm.table_name1.event = 'test-event' AND
      NOT isZeroOrNull(f2)
```

is incorrectly translated by the parser into:

```
CREATE MATERIALIZED VIEW infra_bm.view_name 
    ON CLUSTER 'default_cluster' TO infra_bm.table_name
(
  `f1` DateTime64(3), 
  `f2` String, 
  `f3` String, 
  `f4` String, 
  `f5` String, 
  `f6` Int64
) AS
SELECT f1,
       f2,
       visitParamExtractString(properties, 'f3') AS f3,
       visitParamExtractString(properties, 'f4') AS f4,
       visitParamExtractString(properties, 'f5') AS f5,
       visitParamExtractInt(properties, 'f6') AS f6
FROM infra_bm.table_name1
WHERE infra_bm.table_name1.event = 'test-event' AND
      -isZeroOrNull(f2)
COMMENT 'Comment for table';
```

The incorrect translation is invalid due to the use of -isZeroOrNull(f2) instead of NOT isZeroOrNull(f2). In some case clickhouse throws error code `43`.

## Original Interface

No new interfaces are introduced.
