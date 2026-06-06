# Prompt Style Analysis

Basis: DeepSWE emphasizes short, natural, behavior-focused prompts, broad
repository exploration, and behavioral verifiers. SWE-rebench prompts are
derived from public issues/PRs, so many include issue-template boilerplate,
external links, implementation hints, or generated interface blocks.

## Conclusion

Yes. 14,584 of 15,296 high-quality confidence-filtered SWE-rebench prompts contain at least one signal that should be changed or reviewed before treating them as DeepSWE-style prompts.

The generated Harbor tasks therefore default to `instruction-style=deepswe`,
which removes the generated interface section and keeps only a cleaned,
natural task request. That improves prompt shape, but it does not make the
tasks contamination-free or replace DeepSWE's hand-authored behavioral
verifiers.

## Prompt Length

- p50: 950 characters
- p90: 3,007 characters
- p95: 4,369 characters

## Change Signals

- code_fences: 9,601
- interface_dump: 9,198
- signature_blocks: 9,198
- external_urls: 7,732
- file_path_hints: 4,083
- long_prompt: 2,122
- issue_template_boilerplate: 1,306

## Needs Change By Difficulty

- easy: 5,177
- medium: 9,082
- hard: 325

## Needs Change By Language

- c: 80
- cpp: 87
- go: 2,820
- java: 756
- js: 2,086
- php: 752
- python: 4,008
- ruby: 0
- rust: 1,820
- ts: 2,175
