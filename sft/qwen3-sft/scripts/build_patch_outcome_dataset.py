#!/usr/bin/env python3
"""Build mini-swe-agent patch-outcome SFT rows from successful repair traces.

This is intentionally different from trajectory cloning: each row teaches a
complete source-code repair outcome in the exact mini-swe-agent interface by
applying a verified teacher patch, inspecting the final patch, and submitting.
Rows remain raw chat JSONL; tokenization and packing happen online.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


RAW_ROOT = Path("/wbl-fast/usrs/ee/code-swe-data/data/code-swe-terminal-agentic-sft")

BASH_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the shell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
]

MINI_SWE_SYSTEM = "You are a helpful assistant that can interact with a computer shell to solve programming tasks."
MINI_SWE_SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
MINI_SWE_STRICT_USER_TEMPLATE = """<pr_description>
Consider the following PR description:
{task}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and issue AT LEAST ONE command, see the result, then think and issue your next command(s).</important>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide one or more bash tool calls to execute

## Important Boundaries

- MODIFY: Regular source code files in /testbed (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call. You can make MULTIPLE tool calls in a single response when the commands are independent (e.g., searching multiple files, reading different parts of the codebase).
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

Example of a CORRECT response:
<example_response>
I need to understand the Builder-related code. Let me find relevant files and check the project structure.

[Makes multiple bash tool calls: {{"command": "ls -la"}}, {{"command": "find src -name '*.java' | grep -i builder"}}, {{"command": "cat README.md | head -50"}}]
</example_response>

## Environment Details

- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- You can use bash commands or invoke any tool that is available in the environment
- If a tool isn't available, you can also install it

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
Do NOT commit your changes.

<IMPORTANT>
The patch must only contain changes to the specific source files you modified to fix the issue.
Do not submit file creations or changes to any of the following files:

- test and reproduction files
- helper scripts, tests, or tools that you created
- installation, build, packaging, configuration, or setup scripts unless they are directly part of the issue you were fixing (you can assume that the environment is already set up for your client)
- binary or compiled files
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit:

```bash
{submit_command}
```

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- Creating/viewing the patch and submitting it MUST be separate commands (not combined with &&).
- If you modify patch.txt after verifying, you SHOULD verify again before submitting.
- You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
</CRITICAL>
</instructions>"""


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def bash_call(command: str) -> dict[str, Any]:
    return {"function": {"name": "bash", "arguments": {"command": command}}}


def assistant(command: str, content: str = "") -> dict[str, Any]:
    return {"role": "assistant", "content": content, "tool_calls": [bash_call(command)]}


def tool_output(output: str, returncode: int = 0) -> dict[str, str]:
    return {
        "role": "tool",
        "content": f"<returncode>{returncode}</returncode>\n<output>\n{output.strip()}\n</output>",
    }


def extract_task_text(content: str) -> str:
    for start_tag, end_tag in (("<issue_description>", "</issue_description>"), ("<pr_description>", "</pr_description>")):
        if start_tag in content and end_tag in content:
            start = content.find(start_tag) + len(start_tag)
            end = content.find(end_tag, start)
            text = content[start:end].strip()
            if text:
                return text
    if "ISSUE:" in content:
        return content.split("ISSUE:", 1)[1].strip()
    if "Task Description:" in content:
        return content.split("Task Description:", 1)[1].strip()
    return content.strip()


def first_user_text(trajectory: Iterable[Any]) -> str:
    for message in trajectory:
        role = str(message.get("role", "")).lower()
        if role in {"user", "human"}:
            content = message.get("content")
            if content is None:
                content = message.get("text")
            content = str(content or "").strip()
            if content:
                return extract_task_text(content)
    return ""


def is_bad_patch_path(path: str) -> bool:
    p = path.strip().lower()
    name = Path(p).name
    parts = set(Path(p).parts)
    if not p:
        return True
    if name in {"patch.txt", "install.sh", "run_tests.sh"}:
        return True
    if name.startswith(("reproduce", "debug_", "check_", "run_", "try_", "tmp_", "comprehensive_test")):
        return True
    if name.endswith((".sh", ".patch", ".diff")):
        return True
    if name.endswith((".md", ".rst")):
        return True
    if any(part in {"test", "tests", "testing", "__tests__", "spec", "specs"} for part in parts):
        return True
    if name.startswith("test_") or name.endswith(("_test.py", ".test.js", ".spec.js", ".spec.ts")):
        return True
    if name in {"pyproject.toml", "setup.cfg", "setup.py", "package.json", "package-lock.json", "poetry.lock"}:
        return True
    return False


def diff_block_path(block: list[str]) -> str:
    for line in block[:8]:
        if line.startswith("+++ b/"):
            return line[len("+++ b/") :].strip()
    first = block[0] if block else ""
    match = re.match(r"diff --git a/(.*?) b/(.*)", first)
    return match.group(2).strip() if match else ""


def filter_patch(patch: str) -> str:
    patch = patch.strip()
    if not patch.startswith("diff --git "):
        pos = patch.find("\ndiff --git ")
        if pos == -1:
            return ""
        patch = patch[pos + 1 :]

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)

    kept: list[str] = []
    for block in blocks:
        path = diff_block_path(block)
        if is_bad_patch_path(path):
            continue
        kept.extend(block)
    return "\n".join(kept).strip() + ("\n" if kept else "")


def shell_quote_heredoc(text: str) -> str:
    # Single-quoted heredocs are literal except for the delimiter line itself.
    return text.replace("\nPATCH\n", "\nPATCH_CONTENT\n")


def patch_apply_command(patch: str) -> str:
    patch = shell_quote_heredoc(patch)
    return "\n".join(
        [
            "cat > patch.txt <<'PATCH'",
            patch.rstrip(),
            "PATCH",
            "git apply patch.txt",
        ]
    )


def build_row(task: str, patch: str, *, source: str, row_id: str) -> dict[str, Any]:
    user = MINI_SWE_STRICT_USER_TEMPLATE.format(task=task.strip(), submit_command=MINI_SWE_SUBMIT_COMMAND)
    apply_cmd = patch_apply_command(patch)
    inspect_cmd = "git diff > patch.txt && sed -n '1,240p' patch.txt"
    messages = [
        {"role": "system", "content": MINI_SWE_SYSTEM},
        {"role": "user", "content": user},
        assistant(
            apply_cmd,
            "<thought>\nI have enough information to make the source-code change. I will apply the verified source patch and then inspect the resulting diff before submitting.\n</thought>",
        ),
        tool_output("Applied patch cleanly."),
        assistant(
            inspect_cmd,
            "<thought>\nThe patch is applied. I will inspect the final diff in patch.txt before submitting.\n</thought>",
        ),
        tool_output(patch[:9500]),
        assistant(
            MINI_SWE_SUBMIT_COMMAND,
            "<thought>\nThe source diff is ready and patch.txt contains the final answer. I will submit it now.\n</thought>",
        ),
    ]
    return {
        "messages": messages,
        "tools": BASH_TOOL,
        "metadata": {
            "source": source,
            "row_id": row_id,
            "patch_chars": len(patch),
        },
    }


def iter_swe_hero_rows(root: Path, max_rows: int) -> Iterable[dict[str, Any]]:
    emitted = 0
    for path in sorted((root / "nvidia__SWE-Hero-openhands-trajectories" / "data").glob("*.parquet")):
        df = pd.read_parquet(path, engine="pyarrow")
        for idx, row in df.iterrows():
            task = first_user_text(row["trajectory"])
            patch = filter_patch(str(row.get("model_patch") or ""))
            if task and 200 <= len(patch) <= 20_000:
                yield build_row(task, patch, source="nvidia_swe_hero_patch_outcome", row_id=f"{path.name}:{idx}")
                emitted += 1
                if emitted >= max_rows:
                    return


def iter_nebius_rows(root: Path, max_rows: int) -> Iterable[dict[str, Any]]:
    emitted = 0
    for path in sorted((root / "nebius__SWE-agent-trajectories" / "data").glob("*.parquet")):
        df = pd.read_parquet(path, engine="pyarrow")
        df = df[df["target"] == True]  # noqa: E712
        for idx, row in df.iterrows():
            logs = str(row.get("eval_logs") or "")
            if "All tests passed" not in logs:
                continue
            task = first_user_text(row["trajectory"])
            patch = filter_patch(str(row.get("generated_patch") or ""))
            if task and 200 <= len(patch) <= 12_000:
                yield build_row(task, patch, source="nebius_swe_agent_patch_outcome", row_id=f"{path.name}:{idx}")
                emitted += 1
                if emitted >= max_rows:
                    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-swe-hero-rows", type=int, default=3500)
    parser.add_argument("--max-nebius-rows", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=33333)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    out_dir = args.output_root / "patch_outcome_v22"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        *iter_swe_hero_rows(args.raw_root, args.max_swe_hero_rows),
        *iter_nebius_rows(args.raw_root, args.max_nebius_rows),
    ]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    out_path = out_dir / "data.jsonl"
    patch_chars = 0
    sources: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as out:
        for row in rows:
            source = row["metadata"]["source"]
            sources[source] = sources.get(source, 0) + 1
            patch_chars += int(row["metadata"]["patch_chars"])
            out.write(json_dumps(row) + "\n")

    manifest = {
        "output_root": str(args.output_root),
        "rows": len(rows),
        "sources": sources,
        "patch_chars": patch_chars,
        "avg_patch_chars": patch_chars / max(len(rows), 1),
        "selection": "source-only verified teacher patches rendered as exact mini-swe apply/inspect/submit trajectories",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
