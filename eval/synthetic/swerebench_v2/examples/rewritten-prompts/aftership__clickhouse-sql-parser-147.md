# aftership__clickhouse-sql-parser-147

- repo: AfterShip/clickhouse-sql-parser
- language: go
- difficulty: easy

## Rewritten Prompt

The SQL parser should accept column identifiers containing `$` without requiring backticks. For example, a query like `SELECT service$$name FROM test_table` should parse successfully, matching ClickHouse behavior where such names may be handled as quoted identifiers.

The parser should not fail on this input with an unexpected end-of-input or stray `"$"` error.

## Preserved Requirements

- The parser must accept column names containing `$` in unquoted form.
- The query `SELECT service$$name FROM test_table` should parse successfully.
- The behavior should match ClickHouse, which treats such column names as quoted identifiers.
- The parser should not produce an error for this input.

## Removed Noise

- Issue-style wording such as "Failed to parse query with $ in column name"
- Exact error message text from the original failure
- The code block formatting of example queries
- The note about ClickHouse automatically adding backticks
- Any solution hint about how the parser should recognize the identifier

## Risk Notes

- The desired behavior may overlap with broader identifier-tokenization rules, so changes could affect other special characters in identifiers.
- The prompt does not specify whether this should apply to all identifier positions or only column names in SELECT lists.

## Original Prompt

Failed to parse query with $ in column name
The following query throws error **_line 0:30 <EOF> or ';' was expected, but got: "$"_**

```sql
SELECT service$$name FROM test_table
```

On the other hand, this works
```sql
SELECT `service$$name` FROM test_table
```

When we run `SELECT service$$name FROM test_table`, CH automatically add "``" to such columns. The parser should recognise that and not throw an error.

## Original Interface

No new interfaces are introduced.
