# Prompt Style Analysis

Basis: DeepSWE emphasizes short, natural, behavior-focused prompts, broad
repository exploration, and behavioral verifiers. SWE-rebench prompts are
derived from public issues/PRs, so many include issue-template boilerplate,
external links, implementation hints, or generated interface blocks.

## Conclusion

Yes. 9,003 of 9,409 high-quality confidence-filtered SWE-rebench prompts contain at least one signal that should be changed or reviewed before treating them as DeepSWE-style prompts.

The generated Harbor tasks therefore default to `instruction-style=deepswe`,
which removes the generated interface section and keeps only a cleaned,
natural task request. That improves prompt shape, but it does not make the
tasks contamination-free or replace DeepSWE's hand-authored behavioral
verifiers.

## Prompt Length

- p50: 973 characters
- p90: 3,179 characters
- p95: 4,659 characters

## Change Signals

- interface_dump: 6,116
- signature_blocks: 6,116
- code_fences: 5,712
- external_urls: 4,729
- file_path_hints: 3,139
- long_prompt: 1,423
- issue_template_boilerplate: 974

## Needs Change By Difficulty

- easy: 3,289
- medium: 5,534
- hard: 180

## Needs Change By Language

- python: 4,008
- ts: 2,175
- go: 2,820
