# clusterlabs__ha_cluster_exporter-216

- repo: ClusterLabs/ha_cluster_exporter
- language: go
- difficulty: easy

## Rewritten Prompt

The collector should correctly parse `corosync-cfgtool -s` output for ring/link entries. It currently fails when matching the `id` and `status` fields because the command emits tabs after those labels rather than spaces.

Update the parsing logic so entries are recognized whether the whitespace before `=` is spaces or tabs, and the extracted prefix, number, address, and status values remain correct.

## Preserved Requirements

- Parse `corosync-cfgtool -s` ring/link entries correctly.
- Handle tab characters after `id` and `status` labels.
- Accept both spaces and tabs in the whitespace around the `=` separators.
- Keep extracting the prefix, number, address, and status fields correctly.

## Removed Noise

- Issue template/bug-report phrasing.
- Direct file reference and line number.
- Exact regular expression snippet from the report.
- Explanation of the current regex bug in implementation terms.
- Code-style fix hint showing the exact replacement.
- Any mention of tests or PR/test references.

## Risk Notes

- The input format may vary slightly across corosync versions, so the parser should remain tolerant of whitespace differences without broadening matches too much.
- Preserving existing field extraction behavior is important; changing the pattern should not alter how values are captured.

## Original Prompt

regexp parsing corosync-cfgtool output will not work
In collector/corosync/parser.go line 154
There is a bug in the regexp when parsing corosync-cfgtool -s output:
```re := regexp.MustCompile(`(?m)(?P<prefix>RING|Link) ID (?P<number>\d+)\s+(?P<id>id|addr) \s*= (?P<address>.+)\s+status \s*= (?P<status>.+)`)```
corosync-cfgtool outputs a tab character after "id" and "status", so the regexp with an additional space after id/status will not match. Without the spaces in front of "\s" the regexp will work. 
```...(?P<id>id|addr)\s*= (?P<address>.+)\s+status\s*=...```

## Original Interface

No new interfaces are introduced.
