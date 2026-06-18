# terminal_bench (vendored Terminus-2 agent library)

This directory is a **vendored agent library**, not a standalone benchmark. It
provides the Terminus-2 agent (terminal session, LLM/chat plumbing, prompt
templates) used by the `terminus-2` generation harness in
`eval/benchmarks/swebench_multilingual/run.py` (one of the supported harnesses
alongside the default `mini-swe-agent`).

It has no `run.py` of its own and is invoked only through the SWE-bench
Multilingual runner with `--harness terminus-2`.
