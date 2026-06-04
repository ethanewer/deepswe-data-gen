# aftership__clickhouse-sql-parser-130

- repo: AfterShip/clickhouse-sql-parser
- language: go
- difficulty: easy

## Rewritten Prompt

Parsing an incomplete SELECT query with a trailing FROM must not panic. For input like `SELECT * FROM`, the parser should return a normal error instead of dereferencing nil or crashing.

Make sure malformed queries of this shape are handled safely and reported as an error that indicates a table name or subquery was expected.

## Preserved Requirements

- Parsing `SELECT * FROM` must not panic.
- The parser should return an error instead of crashing.
- The error should indicate that a table name or subquery was expected.
- Malformed queries of this shape must be handled safely.

## Removed Noise

- Issue template / bug report framing
- Stack trace and panic details
- Redacted stack frames
- Repository and language metadata
- Mention of a specific code line and suspected nil pointer cause
- References to tests, patches, PRs, or hidden evaluation
- External URLs

## Risk Notes

- The exact error text may already vary across parser paths; preserve the expected meaning without over-constraining wording.
- The query is intentionally incomplete, so the fix should focus on safe error handling rather than accepting the query as valid.

## Original Prompt

nil pointer dereference on a simple query with no table name after FROM
query: `SELECT * FROM` 

```
panic: runtime error: invalid memory address or nil pointer dereference [recovered]
	panic: runtime error: invalid memory address or nil pointer dereference
[signal SIGSEGV: segmentation violation code=0x2 addr=0x10 pc=0x102ee9d80]

[redacted]
github.com/AfterShip/clickhouse-sql-parser/parser.(*Parser).parseJoinTableExpr(0x1400009ef30, 0x102ee2e74?)
	/pkg/mod/github.com/AfterShip/clickhouse-sql-parser@v0.0.0-20250301225821-9825d50f553f/parser/parser_query.go:217 +0x2d0
[redacted]
```

the problematic code line:
```
return nil, fmt.Errorf("expected table name or subquery, got %s", p.last().Kind)
```
i would guess `p.last()` is nil

## Original Interface

No new interfaces are introduced.
