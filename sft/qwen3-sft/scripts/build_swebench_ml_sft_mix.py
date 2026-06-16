#!/usr/bin/env python3
"""Build small raw SFT mixtures for SWE-bench Multilingual experiments.

This writes normalized chat JSONL, not token IDs. Training still applies the
Qwen chat template, tokenization, and THD packing online.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import random
import re
import shutil
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qwen_agentic_sft.data import discover_raw_files, iter_normalized_examples_from_files


RAW_ROOT = Path("/wbl-fast/usrs/ee/code-swe-data/data/code-swe-terminal-agentic-sft")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_dir: str
    include: tuple[str, ...]
    token_budget: int
    avg_tokens_per_row: int
    note: str
    transform: str = "none"


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
MINI_SWE_SUBMIT_MARKER = "complete_task_and_submit_final_output"
MINI_SWE_USER_TEMPLATE = """<pr_description>
{task}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes in the current directory in a way that is general and consistent with the codebase.

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish.
2. Provide one or more bash tool calls to execute.

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files.
2. Create a script or command to reproduce the issue when applicable.
3. Edit the source code to resolve the issue.
4. Verify your fix works by running the relevant command or test again.
5. Inspect the final patch before submitting.

## Submit

When the task is complete, submit with this exact bash command:

```bash
{submit_command}
```
</instructions>"""
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
- You can also create new tools or scripts to help you with the task
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
MINI_SWE_FORMAT_ERROR = """Tool call error:

<error>
No tool calls found in the response. Every response MUST include at least one tool call.
</error>

Here is general guidance on how to submit correct toolcalls:

Every response needs to use the 'bash' tool at least once to execute commands.

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {"command": "your_command_here"}

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that).
"""


PRESET_V0_180M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "nvidia_nemotron_sft_swe_v2_openhands",
        "nvidia__Nemotron-SFT-SWE-v2",
        ("data/swe.jsonl",),
        25_000_000,
        32_164,
        "OpenHands SWE, Qwen3-Coder-480B/DeepSeek-R1 teacher",
    ),
    SourceSpec(
        "nvidia_nemotron_swe_v1_r2e",
        "nvidia__Nemotron-SWE-v1",
        ("data/r2e_gym.jsonl",),
        15_000_000,
        53_395,
        "OpenHands/R2E, Qwen3-Coder-480B teacher",
    ),
    SourceSpec(
        "nvidia_swe_hero_openhands",
        "nvidia__SWE-Hero-openhands-trajectories",
        ("data/*.parquet",),
        20_000_000,
        54_498,
        "OpenHands SWE-Hero, Qwen3-Coder-480B teacher",
    ),
    SourceSpec(
        "nvidia_swe_zero_openhands",
        "nvidia__SWE-Zero-openhands-trajectories",
        ("data/*.parquet",),
        12_000_000,
        33_204,
        "OpenHands SWE-Zero, Qwen3-Coder-480B teacher; not AlienKevin SWE-ZERO",
    ),
    SourceSpec(
        "nebius_swe_agent",
        "nebius__SWE-agent-trajectories",
        ("**/*.jsonl", "**/*.parquet"),
        12_000_000,
        17_816,
        "SWE-agent harness diversity",
    ),
    SourceSpec(
        "sera_46_best_subset",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        25_000_000,
        51_778,
        "SVG issue-to-patch traces, GLM-4.6 teacher, verified subset",
    ),
    SourceSpec(
        "sera_45_lite_t2",
        "allenai__Sera-4.5A-Lite-T2",
        ("*.jsonl",),
        10_000_000,
        71_821,
        "SVG issue-to-patch traces, GLM-4.5-Air teacher, verified",
    ),
    SourceSpec(
        "terminal_corpus_swe",
        "nvidia__Nemotron-Terminal-Corpus",
        ("dataset_adapters/swe.parquet",),
        18_000_000,
        21_595,
        "Terminus-2 SWE terminal traces, DeepSeek-V3.2 teacher",
    ),
    SourceSpec(
        "opencode_cli",
        "nvidia__Nemotron-SFT-OpenCode-v1",
        ("**/*.jsonl", "**/*.parquet"),
        10_000_000,
        10_500,
        "OpenCode CLI traces, Qwen3-Coder-480B teacher",
    ),
    SourceSpec(
        "agenttrove_verified",
        "open-thoughts__AgentTrove",
        ("data/*.parquet",),
        10_000_000,
        10_015,
        "Diverse verified Terminus-2 traces with strong mixed teachers",
    ),
    SourceSpec(
        "cascade_swe_agentic",
        "nvidia__Nemotron-Cascade-2-SFT-Data",
        ("swe/swe_agentic.jsonl",),
        10_000_000,
        34_376,
        "Cascade SWE agentic traces with mixed strong teachers",
    ),
    SourceSpec(
        "cascade_terminal_agent",
        "nvidia__Nemotron-Cascade-2-SFT-Data",
        ("terminal_agent/terminal_agent.jsonl",),
        10_000_000,
        29_033,
        "Cascade Terminus-2 terminal agent traces",
    ),
)


PRESET_V1_BASH_TOOL_180M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "sera46_best_bash_tool",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        35_000_000,
        51_778,
        "Verified Sera GLM-4.6 SWE traces, remapped to mini-swe-agent bash tool calls",
        "bash_tool",
    ),
    SourceSpec(
        "sera46_lite_t2_bash_tool",
        "allenai__Sera-4.6-Lite-T2",
        ("*.jsonl",),
        25_000_000,
        60_140,
        "Verified Sera GLM-4.6 T2 traces, remapped to mini-swe-agent bash tool calls",
        "bash_tool",
    ),
    SourceSpec(
        "kimi_swesmith_bash_tool",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        25_000_000,
        24_000,
        "Verified Kimi-2.5 SWE-Smith terminal traces, command JSON converted to bash tool calls",
        "bash_tool",
    ),
    SourceSpec(
        "kimi_r2egym_bash_tool",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        15_000_000,
        22_000,
        "Verified Kimi-2.5 R2E-Gym terminal traces, command JSON converted to bash tool calls",
        "bash_tool",
    ),
    SourceSpec(
        "glm46_swesmith_bash_tool",
        "penfever__glm46-swesmith-maxeps-131k",
        ("data/*.parquet",),
        20_000_000,
        24_000,
        "Verified GLM-4.6 SWE-Smith terminal traces, command JSON converted to bash tool calls",
        "bash_tool",
    ),
    SourceSpec(
        "glm46_r2egym_bash_tool",
        "penfever__glm-4.6-r2egym-32ep-32k",
        ("data/*.parquet",),
        10_000_000,
        20_000,
        "GLM-4.6 R2E-Gym terminal traces, command JSON converted to bash tool calls",
        "bash_tool",
    ),
    SourceSpec(
        "minimax_nemotron_oracle_bash_tool",
        "penfever__nemotron-code-oracle-filtered-minimax-m27-131k-traces",
        ("data/*.parquet",),
        15_000_000,
        18_000,
        "Verified MiniMax-M2.7 Nemotron-code oracle terminal traces, command JSON converted to bash tool calls",
        "bash_tool",
    ),
    SourceSpec(
        "minimax_inferredbugs_bash_tool",
        "penfever__inferredbugs-sandboxes-verifier-minimax-m27-131k-traces",
        ("data/*.parquet",),
        15_000_000,
        18_000,
        "Verified MiniMax-M2.7 inferred-bugs terminal traces, command JSON converted to bash tool calls",
        "bash_tool",
    ),
    SourceSpec(
        "glm46_nl2bash_verified_bash_tool",
        "penfever__GLM-4.6-nl2bash-verified-32ep-32k-reasoning",
        ("data/*.parquet",),
        10_000_000,
        12_000,
        "Verified GLM-4.6 nl2bash traces for exact shell-command behavior",
        "bash_tool",
    ),
    SourceSpec(
        "nvidia_swe_v2_bash_tool",
        "nvidia__Nemotron-SFT-SWE-v2",
        ("data/swe.jsonl",),
        10_000_000,
        32_164,
        "OpenHands SWE, Qwen3-Coder-480B/DeepSeek-R1 teacher, remapped to bash-only tools",
        "bash_tool",
    ),
)


PRESET_V2_REASONING_TOOL_180M: tuple[SourceSpec, ...] = tuple(
    SourceSpec(
        spec.name.replace("_bash_tool", "_reasoning_tool"),
        spec.dataset_dir,
        spec.include,
        spec.token_budget,
        spec.avg_tokens_per_row,
        spec.note.replace("bash tool calls", "bash tool calls with Qwen reasoning/tool boundaries"),
        "bash_tool_reasoning",
    )
    for spec in PRESET_V1_BASH_TOOL_180M
)


PRESET_V3_MINISWE_SUBMIT_200M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "sera46_best_miniswe_submit",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        35_000_000,
        51_778,
        "Verified Sera GLM-4.6 SWE traces, strict mini-swe prompt, visible reasoning, bash tool calls, submit-filtered",
        "bash_tool_submit_strict",
    ),
    SourceSpec(
        "sera46_lite_t2_miniswe_submit",
        "allenai__Sera-4.6-Lite-T2",
        ("*.jsonl",),
        25_000_000,
        60_140,
        "Verified Sera GLM-4.6 T2 traces, strict mini-swe prompt, visible reasoning, bash tool calls, submit-filtered",
        "bash_tool_submit_strict",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_submit",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        30_000_000,
        24_000,
        "Verified Kimi-2.5 SWE-Smith traces, strict mini-swe prompt, visible reasoning, bash tool calls, submit-filtered",
        "bash_tool_submit_strict",
    ),
    SourceSpec(
        "kimi_r2egym_miniswe_submit",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        15_000_000,
        22_000,
        "Verified Kimi-2.5 R2E-Gym traces, strict mini-swe prompt, visible reasoning, bash tool calls, submit-filtered",
        "bash_tool_submit_strict",
    ),
    SourceSpec(
        "glm46_swesmith_miniswe_submit",
        "penfever__glm46-swesmith-maxeps-131k",
        ("data/*.parquet",),
        25_000_000,
        24_000,
        "Verified GLM-4.6 SWE-Smith traces, strict mini-swe prompt, visible reasoning, bash tool calls, submit-filtered",
        "bash_tool_submit_strict",
    ),
    SourceSpec(
        "glm46_r2egym_miniswe_submit",
        "penfever__glm-4.6-r2egym-32ep-32k",
        ("data/*.parquet",),
        15_000_000,
        20_000,
        "GLM-4.6 R2E-Gym traces, strict mini-swe prompt, visible reasoning, bash tool calls, submit-filtered",
        "bash_tool_submit_strict",
    ),
    SourceSpec(
        "minimax_nemotron_oracle_miniswe_submit",
        "penfever__nemotron-code-oracle-filtered-minimax-m27-131k-traces",
        ("data/*.parquet",),
        20_000_000,
        18_000,
        "Verified MiniMax-M2.7 oracle traces, strict mini-swe prompt, visible reasoning, bash tool calls, submit-filtered",
        "bash_tool_submit_strict",
    ),
    SourceSpec(
        "minimax_inferredbugs_miniswe_submit",
        "penfever__inferredbugs-sandboxes-verifier-minimax-m27-131k-traces",
        ("data/*.parquet",),
        20_000_000,
        18_000,
        "Verified MiniMax-M2.7 inferred-bugs traces, strict mini-swe prompt, visible reasoning, bash tool calls, submit-filtered",
        "bash_tool_submit_strict",
    ),
    SourceSpec(
        "nvidia_swe_v2_miniswe_submit",
        "nvidia__Nemotron-SFT-SWE-v2",
        ("data/swe.jsonl",),
        10_000_000,
        32_164,
        "OpenHands SWE traces, Qwen3-Coder/DeepSeek teacher, strict mini-swe prompt, submit-filtered",
        "bash_tool_submit_strict",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_recovery",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        5_000_000,
        8_000,
        "Short recovery rows: exact mini-swe no-tool error followed by a valid bash tool call",
        "bash_tool_recovery_strict",
    ),
    SourceSpec(
        "sera46_best_miniswe_recovery",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        5_000_000,
        8_000,
        "Short recovery rows: exact mini-swe no-tool error followed by a valid bash tool call",
        "bash_tool_recovery_strict",
    ),
)


PRESET_V4_MINISWE_TOOLOBS_128M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "sera46_best_miniswe_toolobs",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        20_000_000,
        51_778,
        "Verified Sera GLM-4.6 SWE traces, strict mini-swe prompt, bash tool calls, submit-filtered, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs",
    ),
    SourceSpec(
        "sera46_lite_t2_miniswe_toolobs",
        "allenai__Sera-4.6-Lite-T2",
        ("*.jsonl",),
        15_000_000,
        60_140,
        "Verified Sera GLM-4.6 T2 traces, strict mini-swe prompt, bash tool calls, submit-filtered, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_toolobs",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        25_000_000,
        24_000,
        "Verified Kimi-2.5 SWE-Smith traces, strict mini-swe prompt, bash tool calls, submit-filtered, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs",
    ),
    SourceSpec(
        "kimi_r2egym_miniswe_toolobs",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        12_000_000,
        22_000,
        "Verified Kimi-2.5 R2E-Gym traces, strict mini-swe prompt, bash tool calls, submit-filtered, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs",
    ),
    SourceSpec(
        "glm46_swesmith_miniswe_toolobs",
        "penfever__glm46-swesmith-maxeps-131k",
        ("data/*.parquet",),
        20_000_000,
        24_000,
        "Verified GLM-4.6 SWE-Smith traces, strict mini-swe prompt, bash tool calls, submit-filtered, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs",
    ),
    SourceSpec(
        "glm46_r2egym_miniswe_toolobs",
        "penfever__glm-4.6-r2egym-32ep-32k",
        ("data/*.parquet",),
        12_000_000,
        20_000,
        "GLM-4.6 R2E-Gym traces, strict mini-swe prompt, bash tool calls, submit-filtered, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs",
    ),
    SourceSpec(
        "minimax_nemotron_oracle_miniswe_toolobs",
        "penfever__nemotron-code-oracle-filtered-minimax-m27-131k-traces",
        ("data/*.parquet",),
        15_000_000,
        18_000,
        "Verified MiniMax-M2.7 oracle traces, strict mini-swe prompt, bash tool calls, submit-filtered, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs",
    ),
    SourceSpec(
        "minimax_inferredbugs_miniswe_toolobs",
        "penfever__inferredbugs-sandboxes-verifier-minimax-m27-131k-traces",
        ("data/*.parquet",),
        15_000_000,
        18_000,
        "Verified MiniMax-M2.7 inferred-bugs traces, strict mini-swe prompt, bash tool calls, submit-filtered, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs",
    ),
    SourceSpec(
        "nvidia_swe_v2_miniswe_toolobs",
        "nvidia__Nemotron-SFT-SWE-v2",
        ("data/swe.jsonl",),
        3_000_000,
        32_164,
        "OpenHands SWE traces, Qwen3-Coder/DeepSeek teacher, strict mini-swe prompt, submit-filtered, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_recovery_toolobs",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        4_000_000,
        8_000,
        "Short recovery rows: exact mini-swe no-tool error followed by a valid bash tool call",
        "bash_tool_recovery_strict_toolobs",
    ),
    SourceSpec(
        "sera46_best_miniswe_recovery_toolobs",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        4_000_000,
        8_000,
        "Short recovery rows: exact mini-swe no-tool error followed by a valid bash tool call",
        "bash_tool_recovery_strict_toolobs",
    ),
)

PRESET_V5_MINISWE_REASONING_TOOLOBS_128M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "sera46_best_miniswe_reasoning_toolobs",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        20_000_000,
        51_800,
        "Verified Sera GLM-4.6 best traces, strict mini-swe prompt, bash tool calls, submit-filtered, reasoning_content, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_reasoning",
    ),
    SourceSpec(
        "sera46_lite_t2_miniswe_reasoning_toolobs",
        "allenai__Sera-4.6-Lite-T2",
        ("*.jsonl",),
        15_000_000,
        60_140,
        "Verified Sera GLM-4.6 T2 traces, strict mini-swe prompt, bash tool calls, submit-filtered, reasoning_content, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_reasoning",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_reasoning_toolobs",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        25_000_000,
        24_000,
        "Verified Kimi-2.5 SWE-Smith traces, strict mini-swe prompt, bash tool calls, submit-filtered, reasoning_content, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_reasoning",
    ),
    SourceSpec(
        "kimi_r2egym_miniswe_reasoning_toolobs",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        12_000_000,
        22_000,
        "Verified Kimi-2.5 R2E-Gym traces, strict mini-swe prompt, bash tool calls, submit-filtered, reasoning_content, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_reasoning",
    ),
    SourceSpec(
        "glm46_swesmith_miniswe_reasoning_toolobs",
        "penfever__glm46-swesmith-maxeps-131k",
        ("data/*.parquet",),
        20_000_000,
        24_000,
        "Verified GLM-4.6 SWE-Smith traces, strict mini-swe prompt, bash tool calls, submit-filtered, reasoning_content, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_reasoning",
    ),
    SourceSpec(
        "glm46_r2egym_miniswe_reasoning_toolobs",
        "penfever__glm-4.6-r2egym-32ep-32k",
        ("data/*.parquet",),
        12_000_000,
        20_000,
        "GLM-4.6 R2E-Gym traces, strict mini-swe prompt, bash tool calls, submit-filtered, reasoning_content, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_reasoning",
    ),
    SourceSpec(
        "minimax_nemotron_oracle_miniswe_reasoning_toolobs",
        "penfever__nemotron-code-oracle-filtered-minimax-m27-131k-traces",
        ("data/*.parquet",),
        15_000_000,
        18_000,
        "Verified MiniMax-M2.7 oracle traces, strict mini-swe prompt, bash tool calls, submit-filtered, reasoning_content, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_reasoning",
    ),
    SourceSpec(
        "minimax_inferredbugs_miniswe_reasoning_toolobs",
        "penfever__inferredbugs-sandboxes-verifier-minimax-m27-131k-traces",
        ("data/*.parquet",),
        15_000_000,
        18_000,
        "Verified MiniMax-M2.7 inferred-bugs traces, strict mini-swe prompt, bash tool calls, submit-filtered, reasoning_content, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_reasoning",
    ),
    SourceSpec(
        "nvidia_swe_v2_miniswe_reasoning_toolobs",
        "nvidia__Nemotron-SFT-SWE-v2",
        ("data/swe.jsonl",),
        3_000_000,
        32_164,
        "OpenHands SWE traces, Qwen3-Coder/DeepSeek teacher, strict mini-swe prompt, submit-filtered, reasoning_content, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_reasoning",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_recovery_reasoning_toolobs",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        4_000_000,
        8_000,
        "Short recovery rows: exact mini-swe no-tool error followed by a valid bash tool call, reasoning_content, terminal observations converted to tool role",
        "bash_tool_recovery_strict_toolobs_reasoning",
    ),
    SourceSpec(
        "sera46_best_miniswe_recovery_reasoning_toolobs",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        4_000_000,
        8_000,
        "Short recovery rows: exact mini-swe no-tool error followed by a valid bash tool call, reasoning_content, terminal observations converted to tool role",
        "bash_tool_recovery_strict_toolobs_reasoning",
    ),
)

PRESET_V6_MINISWE_TOOLONLY_TOOLOBS_128M: tuple[SourceSpec, ...] = tuple(
    SourceSpec(
        spec.name.replace("_reasoning_toolobs", "_toolonly_toolobs"),
        spec.dataset_dir,
        spec.include,
        spec.token_budget,
        spec.avg_tokens_per_row,
        spec.note.replace("reasoning_content, ", "tool-call-only, "),
        spec.transform.replace("toolobs_reasoning", "toolobs_toolonly"),
    )
    for spec in PRESET_V5_MINISWE_REASONING_TOOLOBS_128M
)

PRESET_V7_MINISWE_FIRSTTURN_RECOVERY_TOOLONLY_180M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "kimi_swesmith_miniswe_toolonly_full_v7",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        20_000_000,
        24_000,
        "Full strict mini-swe trajectories, exact mini-swe prompt, tool-call-only bash targets, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_best_miniswe_toolonly_full_v7",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        15_000_000,
        51_800,
        "Full strict mini-swe trajectories, exact mini-swe prompt, tool-call-only bash targets, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "glm46_swesmith_miniswe_toolonly_full_v7",
        "penfever__glm46-swesmith-maxeps-131k",
        ("data/*.parquet",),
        15_000_000,
        24_000,
        "Full strict mini-swe trajectories, exact mini-swe prompt, tool-call-only bash targets, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "minimax_nemotron_miniswe_toolonly_full_v7",
        "penfever__nemotron-code-oracle-filtered-minimax-m27-131k-traces",
        ("data/*.parquet",),
        10_000_000,
        18_000,
        "Full strict mini-swe trajectories, exact mini-swe prompt, tool-call-only bash targets, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_t2_miniswe_toolonly_full_v7",
        "allenai__Sera-4.6-Lite-T2",
        ("*.jsonl",),
        10_000_000,
        60_140,
        "Full strict mini-swe trajectories, exact mini-swe prompt, tool-call-only bash targets, terminal observations converted to tool role",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_firstturn_v7",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        25_000_000,
        4_500,
        "First assistant turn only, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_first_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_best_miniswe_firstturn_v7",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        15_000_000,
        4_500,
        "First assistant turn only, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_first_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "glm46_swesmith_miniswe_firstturn_v7",
        "penfever__glm46-swesmith-maxeps-131k",
        ("data/*.parquet",),
        15_000_000,
        4_500,
        "First assistant turn only, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_first_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "minimax_nemotron_miniswe_firstturn_v7",
        "penfever__nemotron-code-oracle-filtered-minimax-m27-131k-traces",
        ("data/*.parquet",),
        10_000_000,
        4_500,
        "First assistant turn only, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_first_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "kimi_r2egym_miniswe_firstturn_v7",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        10_000_000,
        4_500,
        "First assistant turn only, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_first_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_recovery_toolonly_v7",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        15_000_000,
        4_800,
        "No-tool format-error recovery row, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_recovery_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_best_miniswe_recovery_toolonly_v7",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        10_000_000,
        4_800,
        "No-tool format-error recovery row, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_recovery_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "glm46_swesmith_miniswe_recovery_toolonly_v7",
        "penfever__glm46-swesmith-maxeps-131k",
        ("data/*.parquet",),
        10_000_000,
        4_800,
        "No-tool format-error recovery row, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_recovery_strict_toolobs_toolonly",
    ),
)

PRESET_V8_MINISWE_CURATED_FIRSTTURN_RECOVERY_TOOLONLY_180M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "kimi_swesmith_miniswe_toolonly_full_v8",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        25_000_000,
        24_000,
        "Full exact mini-swe trajectories with /testbed-compatible first commands, tool-call-only bash targets",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "kimi_r2egym_miniswe_toolonly_full_v8",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        15_000_000,
        22_000,
        "Full exact mini-swe trajectories with /testbed-compatible first commands, tool-call-only bash targets",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_best_miniswe_toolonly_full_v8",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        20_000_000,
        51_800,
        "Full exact mini-swe trajectories with /testbed-compatible first commands, tool-call-only bash targets",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_t2_miniswe_toolonly_full_v8",
        "allenai__Sera-4.6-Lite-T2",
        ("*.jsonl",),
        15_000_000,
        60_140,
        "Full exact mini-swe trajectories with /testbed-compatible first commands, tool-call-only bash targets",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_firstturn_v8",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        30_000_000,
        4_500,
        "First assistant turn only, exact mini-swe prompt, /testbed-compatible tool-call-only bash target",
        "bash_tool_first_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "kimi_r2egym_miniswe_firstturn_v8",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        15_000_000,
        4_500,
        "First assistant turn only, exact mini-swe prompt, /testbed-compatible tool-call-only bash target",
        "bash_tool_first_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_best_miniswe_firstturn_v8",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        20_000_000,
        4_500,
        "First assistant turn only, exact mini-swe prompt, /testbed-compatible tool-call-only bash target",
        "bash_tool_first_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_t2_miniswe_firstturn_v8",
        "allenai__Sera-4.6-Lite-T2",
        ("*.jsonl",),
        10_000_000,
        4_500,
        "First assistant turn only, exact mini-swe prompt, /testbed-compatible tool-call-only bash target",
        "bash_tool_first_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_recovery_toolonly_v8",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        15_000_000,
        4_800,
        "No-tool format-error recovery row, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_recovery_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "kimi_r2egym_miniswe_recovery_toolonly_v8",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        5_000_000,
        4_800,
        "No-tool format-error recovery row, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_recovery_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_best_miniswe_recovery_toolonly_v8",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        10_000_000,
        4_800,
        "No-tool format-error recovery row, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_recovery_strict_toolobs_toolonly",
    ),
)

PRESET_V9_MINISWE_CURATED_FULL_TRAJ_TOOLONLY_180M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "kimi_swesmith_miniswe_toolonly_full_v9",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        45_000_000,
        24_000,
        "Full exact mini-swe trajectories, curated /testbed-compatible sources, tool-call-only bash targets",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "kimi_r2egym_miniswe_toolonly_full_v9",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        30_000_000,
        22_000,
        "Full exact mini-swe trajectories, curated /testbed-compatible sources, tool-call-only bash targets",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_best_miniswe_toolonly_full_v9",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        45_000_000,
        51_800,
        "Full exact mini-swe trajectories, curated /testbed-compatible sources, tool-call-only bash targets",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_t2_miniswe_toolonly_full_v9",
        "allenai__Sera-4.6-Lite-T2",
        ("*.jsonl",),
        30_000_000,
        60_140,
        "Full exact mini-swe trajectories, curated /testbed-compatible sources, tool-call-only bash targets",
        "bash_tool_submit_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_firstturn_v9",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        10_000_000,
        4_500,
        "Small first-turn anchor, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_first_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_best_miniswe_firstturn_v9",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        10_000_000,
        4_500,
        "Small first-turn anchor, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_first_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_recovery_toolonly_v9",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        5_000_000,
        4_800,
        "Small no-tool format-error recovery row, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_recovery_strict_toolobs_toolonly",
    ),
    SourceSpec(
        "sera46_best_miniswe_recovery_toolonly_v9",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        5_000_000,
        4_800,
        "Small no-tool format-error recovery row, exact mini-swe prompt, tool-call-only bash target",
        "bash_tool_recovery_strict_toolobs_toolonly",
    ),
)

PRESET_V10_MINISWE_SINGLECALL_FULL_TRAJ_TOOLONLY_180M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "kimi_swesmith_miniswe_singlecall_full_v10",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        50_000_000,
        24_000,
        "Full exact mini-swe trajectories with one bash call per assistant turn and cleaned per-command observations",
        "bash_tool_submit_strict_toolobs_toolonly_single",
    ),
    SourceSpec(
        "kimi_r2egym_miniswe_singlecall_full_v10",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        30_000_000,
        22_000,
        "Full exact mini-swe trajectories with one bash call per assistant turn and cleaned per-command observations",
        "bash_tool_submit_strict_toolobs_toolonly_single",
    ),
    SourceSpec(
        "sera46_best_miniswe_singlecall_full_v10",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        45_000_000,
        51_800,
        "Full exact mini-swe trajectories with one bash call per assistant turn and cleaned per-command observations",
        "bash_tool_submit_strict_toolobs_toolonly_single",
    ),
    SourceSpec(
        "sera46_t2_miniswe_singlecall_full_v10",
        "allenai__Sera-4.6-Lite-T2",
        ("*.jsonl",),
        30_000_000,
        60_140,
        "Full exact mini-swe trajectories with one bash call per assistant turn and cleaned per-command observations",
        "bash_tool_submit_strict_toolobs_toolonly_single",
    ),
    SourceSpec(
        "kimi_swesmith_miniswe_recovery_toolonly_v10",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        10_000_000,
        4_800,
        "No-tool format-error recovery row, exact mini-swe prompt, single-call tool-only bash target",
        "bash_tool_recovery_strict_toolobs_toolonly_single",
    ),
    SourceSpec(
        "sera46_best_miniswe_recovery_toolonly_v10",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        10_000_000,
        4_800,
        "No-tool format-error recovery row, exact mini-swe prompt, single-call tool-only bash target",
        "bash_tool_recovery_strict_toolobs_toolonly_single",
    ),
)

PRESET_V11_MINISWE_EMPTYTHINK_SINGLECALL_FULL_TRAJ_TOOLONLY_166M: tuple[SourceSpec, ...] = tuple(
    SourceSpec(
        spec.name.replace("_v10", "_v11").replace("singlecall", "emptythink_singlecall"),
        spec.dataset_dir,
        spec.include,
        spec.token_budget,
        spec.avg_tokens_per_row,
        spec.note + "; assistant labels include the empty Qwen thinking block used by eval",
        spec.transform + "_emptythink",
    )
    for spec in PRESET_V10_MINISWE_SINGLECALL_FULL_TRAJ_TOOLONLY_180M
)

PRESET_V12_MINISWE_EMPTYTHINK_TRUNCATED_SINGLECALL_FULL_TRAJ_TOOLONLY_166M: tuple[SourceSpec, ...] = tuple(
    SourceSpec(
        spec.name.replace("_v11", "_v12").replace("emptythink_singlecall", "emptythink_truncated_singlecall"),
        spec.dataset_dir,
        spec.include,
        spec.token_budget,
        spec.avg_tokens_per_row,
        spec.note + "; tool observations use the mini-swe-agent long-output truncation format",
        spec.transform,
    )
    for spec in PRESET_V11_MINISWE_EMPTYTHINK_SINGLECALL_FULL_TRAJ_TOOLONLY_166M
)

PRESET_V14_MINISWE_LANGBALANCED_EMPTYTHINK_SINGLECALL_220M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "kimi_swesmith_miniswe_singlecall_full_v14",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        30_000_000,
        24_000,
        "Existing SWE-Smith anchor, exact mini-swe prompt, single-call tool-only empty-think targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "kimi_r2egym_miniswe_singlecall_full_v14",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        15_000_000,
        22_000,
        "Existing R2E-Gym anchor, exact mini-swe prompt, single-call tool-only empty-think targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "sera46_best_miniswe_singlecall_full_v14",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        25_000_000,
        51_800,
        "Existing SERA GLM-4.6 anchor, exact mini-swe prompt, single-call tool-only empty-think targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "sera46_t2_miniswe_singlecall_full_v14",
        "allenai__Sera-4.6-Lite-T2",
        ("*.jsonl",),
        15_000_000,
        60_140,
        "Existing SERA T2 anchor, exact mini-swe prompt, single-call tool-only empty-think targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "minimax_inferredbugs_singlecall_full_v14",
        "penfever__inferredbugs-sandboxes-verifier-minimax-m27-131k-traces",
        ("data/*.parquet",),
        35_000_000,
        18_000,
        "Verifier-backed MiniMax inferred-bugs traces with Java, C#, and other non-Python code-repair tasks",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "minimax_nemotron_junit_singlecall_full_v14",
        "penfever__exp_rpt_nemotron-junit-minimax-m27-131k-traces",
        ("data/*.parquet",),
        30_000_000,
        18_000,
        "MiniMax JUnit/Nemotron repair traces to teach Java repo inspection and test execution",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "minimax_stack_junit_singlecall_full_v14",
        "penfever__exp_rpt_stack-junit-v6-minimax-m27-131k-traces",
        ("data/*.parquet",),
        15_000_000,
        18_000,
        "MiniMax Stack/JUnit traces with Java test-oriented exploration",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "glm46_defects4j_navigation_v14",
        "penfever__glm46-defects4j-32ep-131k",
        ("data/*.parquet",),
        25_000_000,
        24_000,
        "Defects4J Java navigation/debugging traces; no-submit rows retained as early-turn exploration targets",
        "bash_tool_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "minimax_inferredbugs_recovery_v14",
        "penfever__inferredbugs-sandboxes-verifier-minimax-m27-131k-traces",
        ("data/*.parquet",),
        5_000_000,
        4_800,
        "No-tool format-error recovery rows from verifier-backed MiniMax inferred-bugs traces",
        "bash_tool_recovery_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "kimi_swesmith_recovery_v14",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        5_000_000,
        4_800,
        "No-tool format-error recovery rows from the existing SWE-Smith anchor",
        "bash_tool_recovery_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "sera46_best_recovery_v14",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        5_000_000,
        4_800,
        "No-tool format-error recovery rows from the existing SERA anchor",
        "bash_tool_recovery_strict_toolobs_toolonly_single_emptythink",
    ),
)

PRESET_V15_MINISWE_LANGBALANCED_EMPTYTHINK_SINGLECALL_215M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "kimi_swesmith_miniswe_singlecall_full_v15",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        30_000_000,
        24_000,
        "Existing SWE-Smith anchor, exact mini-swe prompt, single-call tool-only empty-think targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "kimi_r2egym_miniswe_singlecall_full_v15",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        15_000_000,
        22_000,
        "Existing R2E-Gym anchor, exact mini-swe prompt, single-call tool-only empty-think targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "sera46_best_miniswe_singlecall_full_v15",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        25_000_000,
        51_800,
        "Existing SERA GLM-4.6 anchor, exact mini-swe prompt, single-call tool-only empty-think targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "sera46_t2_miniswe_singlecall_full_v15",
        "allenai__Sera-4.6-Lite-T2",
        ("*.jsonl",),
        15_000_000,
        60_140,
        "Existing SERA T2 anchor, exact mini-swe prompt, single-call tool-only empty-think targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "minimax_inferredbugs_singlecall_full_v15",
        "penfever__inferredbugs-sandboxes-verifier-minimax-m27-131k-traces",
        ("data/*.parquet",),
        35_000_000,
        18_000,
        "Verifier-backed MiniMax inferred-bugs traces with Java, C#, and other non-Python code-repair tasks",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "kimi_inferredbugs_singlecall_full_v15",
        "penfever__Kimi-2.5-inferredbugs-sandboxes-maxeps-32k",
        ("data/*.parquet",),
        30_000_000,
        18_000,
        "Kimi inferred-bugs traces for additional non-Python repair diversity with submit-complete episodes",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "minimax_nemotron_junit_singlecall_full_v15",
        "penfever__exp_rpt_nemotron-junit-minimax-m27-131k-traces",
        ("data/*.parquet",),
        30_000_000,
        18_000,
        "MiniMax JUnit/Nemotron repair traces to teach Java repo inspection and test execution",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "glm46_inferredbugs_reasoning_singlecall_full_v15",
        "penfever__GLM-4.6-inferredbugs-32ep-65k-reasoning",
        ("data/*.parquet",),
        15_000_000,
        18_000,
        "GLM-4.6 inferred-bugs traces with Java and C# exploration, submit-filtered",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "minimax_stack_junit_singlecall_full_v15",
        "penfever__exp_rpt_stack-junit-v6-minimax-m27-131k-traces",
        ("data/*.parquet",),
        5_000_000,
        18_000,
        "MiniMax Stack/JUnit traces with Java test-oriented exploration",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "minimax_inferredbugs_recovery_v15",
        "penfever__inferredbugs-sandboxes-verifier-minimax-m27-131k-traces",
        ("data/*.parquet",),
        5_000_000,
        4_800,
        "No-tool format-error recovery rows from verifier-backed MiniMax inferred-bugs traces",
        "bash_tool_recovery_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "kimi_swesmith_recovery_v15",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        5_000_000,
        4_800,
        "No-tool format-error recovery rows from the existing SWE-Smith anchor",
        "bash_tool_recovery_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "sera46_best_recovery_v15",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        5_000_000,
        4_800,
        "No-tool format-error recovery rows from the existing SERA anchor",
        "bash_tool_recovery_strict_toolobs_toolonly_single_emptythink",
    ),
)


def v16_visible_transform(transform: str) -> str:
    if transform == "bash_tool_submit_strict_toolobs_toolonly_single_emptythink":
        return "bash_tool_submit_strict_toolobs_single_visible"
    if transform == "bash_tool_recovery_strict_toolobs_toolonly_single_emptythink":
        return "bash_tool_recovery_strict_toolobs_single_visible"
    raise ValueError(f"Unexpected v15 transform for v16 conversion: {transform}")


PRESET_V16_MINISWE_LANGBALANCED_VISIBLE_SINGLECALL_215M: tuple[SourceSpec, ...] = tuple(
    SourceSpec(
        spec.name.replace("_v15", "_v16").replace("emptythink", "visible"),
        spec.dataset_dir,
        spec.include,
        spec.token_budget,
        spec.avg_tokens_per_row,
        spec.note.replace("tool-only empty-think targets", "visible reasoning text plus single bash tool targets")
        .replace("single-call tool-only empty-think targets", "single-call visible reasoning plus bash tool targets")
        .replace("single-call tool-only", "single-call visible reasoning plus bash tool")
        .replace("No-tool format-error recovery rows", "No-tool format-error recovery rows with visible reasoning targets"),
        v16_visible_transform(spec.transform),
    )
    for spec in PRESET_V15_MINISWE_LANGBALANCED_EMPTYTHINK_SINGLECALL_215M
)


PRESET_V29_MINISWE_VISIBLE_STRONG_AGENTIC_240M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "agenttrove_visible_submit_v29",
        "open-thoughts__AgentTrove",
        ("data/*.parquet",),
        30_000_000,
        16_000,
        "Verified AgentTrove Terminus-2 traces with mixed strong teachers, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "nebius_swe_agent_visible_submit_v29",
        "nebius__SWE-agent-trajectories",
        ("data/*.parquet",),
        20_000_000,
        17_816,
        "Verified SWE-agent traces for harness diversity, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "nvidia_swe_v2_visible_submit_v29",
        "nvidia__Nemotron-SFT-SWE-v2",
        ("data/swe.jsonl",),
        15_000_000,
        32_164,
        "OpenHands SWE traces from Qwen3-Coder/DeepSeek teachers, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "sera46_best_visible_submit_v29",
        "allenai__SERA-4.6-Lite-Best-Subset",
        ("*.jsonl",),
        25_000_000,
        51_800,
        "Verified SERA GLM-4.6 repair traces, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "sera46_t2_visible_submit_v29",
        "allenai__Sera-4.6-Lite-T2",
        ("*.jsonl",),
        15_000_000,
        60_140,
        "Verified SERA GLM-4.6 T2 repair traces, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "kimi_swesmith_visible_submit_v29",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        25_000_000,
        24_000,
        "Verified Kimi-2.5 SWE-Smith repair traces, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "kimi_inferredbugs_visible_submit_v29",
        "penfever__Kimi-2.5-inferredbugs-sandboxes-maxeps-32k",
        ("data/*.parquet",),
        20_000_000,
        18_000,
        "Kimi-2.5 inferred-bugs repair traces for non-Python diversity, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "kimi_r2egym_visible_submit_v29",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        10_000_000,
        22_000,
        "Verified Kimi-2.5 R2E-Gym traces, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "minimax_nemotron_visible_submit_v29",
        "penfever__nemotron-code-oracle-filtered-minimax-m27-131k-traces",
        ("data/*.parquet",),
        20_000_000,
        18_000,
        "Oracle-filtered MiniMax-M2.7 code repair traces, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "minimax_inferredbugs_visible_submit_v29",
        "penfever__inferredbugs-sandboxes-verifier-minimax-m27-131k-traces",
        ("data/*.parquet",),
        20_000_000,
        18_000,
        "Verifier-backed MiniMax-M2.7 inferred-bugs traces, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "minimax_freelancer_visible_submit_v29",
        "penfever__llm-verifier-freelancer-minimax-m27-131k-traces",
        ("data/*.parquet",),
        15_000_000,
        18_000,
        "Verifier-backed MiniMax-M2.7 freelancer repair traces, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "glm46_swesmith_visible_submit_v29",
        "penfever__glm46-swesmith-maxeps-131k",
        ("data/*.parquet",),
        15_000_000,
        24_000,
        "Verified GLM-4.6 SWE-Smith traces, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "glm46_code_feedback_visible_submit_v29",
        "penfever__glm46-code-feedback-maxeps-131k",
        ("data/*.parquet",),
        15_000_000,
        18_000,
        "Verified GLM-4.6 code feedback traces, visible reasoning plus single bash tool targets",
        "bash_tool_submit_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "glm46_defects4j_visible_navigation_v29",
        "penfever__glm46-defects4j-32ep-131k",
        ("data/*.parquet",),
        5_000_000,
        24_000,
        "Small verified GLM-4.6 Defects4J navigation slice for Java repo inspection without submit filtering",
        "bash_tool_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "kimi_swesmith_visible_recovery_v29",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        5_000_000,
        4_800,
        "No-tool format-error recovery rows from verified Kimi SWE-Smith traces with visible reasoning targets",
        "bash_tool_recovery_strict_toolobs_single_visible",
    ),
    SourceSpec(
        "minimax_inferredbugs_visible_recovery_v29",
        "penfever__inferredbugs-sandboxes-verifier-minimax-m27-131k-traces",
        ("data/*.parquet",),
        5_000_000,
        4_800,
        "No-tool format-error recovery rows from verifier-backed MiniMax traces with visible reasoning targets",
        "bash_tool_recovery_strict_toolobs_single_visible",
    ),
)


PRESET_V30_MINISWE_VERIFIED_TOOLONLY_180M: tuple[SourceSpec, ...] = (
    SourceSpec(
        "kimi_swesmith_verified_toolonly_v30",
        "penfever__Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k",
        ("data/*.parquet",),
        50_000_000,
        24_000,
        "Oracle-verified Kimi-2.5 SWE-Smith repair traces; eval-aligned empty-think single bash tool targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "minimax_nemotron_oracle_toolonly_v30",
        "penfever__nemotron-code-oracle-filtered-minimax-m27-131k-traces",
        ("data/*.parquet",),
        45_000_000,
        18_000,
        "Oracle-filtered MiniMax-M2.7 code repair traces; eval-aligned empty-think single bash tool targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "minimax_inferredbugs_verified_toolonly_v30",
        "penfever__inferredbugs-sandboxes-verifier-minimax-m27-131k-traces",
        ("data/*.parquet",),
        35_000_000,
        18_000,
        "Verifier-backed MiniMax-M2.7 inferred-bugs repair traces; eval-aligned empty-think single bash tool targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "minimax_freelancer_verified_toolonly_v30",
        "penfever__llm-verifier-freelancer-minimax-m27-131k-traces",
        ("data/*.parquet",),
        25_000_000,
        18_000,
        "Verifier-backed MiniMax-M2.7 freelancer repair traces; eval-aligned empty-think single bash tool targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "kimi_inferredbugs_toolonly_v30",
        "penfever__Kimi-2.5-inferredbugs-sandboxes-maxeps-32k",
        ("data/*.parquet",),
        15_000_000,
        18_000,
        "Kimi-2.5 inferred-bugs repair traces with verifier metadata; eval-aligned empty-think single bash tool targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
    SourceSpec(
        "kimi_r2egym_toolonly_v30",
        "penfever__Kimi-2.5-r2egym_sandboxes-maxeps-32k",
        ("data/*.parquet",),
        10_000_000,
        22_000,
        "Kimi-2.5 R2E-Gym repair traces with verifier metadata; eval-aligned empty-think single bash tool targets",
        "bash_tool_submit_strict_toolobs_toolonly_single_emptythink",
    ),
)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def included_files(dataset_root: Path, include: tuple[str, ...], seed: int) -> list[Path]:
    files = discover_raw_files(dataset_root)
    selected: list[Path] = []
    for path in files:
        rel = path.relative_to(dataset_root).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in include):
            selected.append(path)
    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected


def bash_call(command: str) -> dict[str, Any]:
    return {"function": {"name": "bash", "arguments": {"command": command}}}


def json_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def python_replace_command(path: str, old: str, new: str) -> str:
    return "\n".join(
        [
            "python3 - <<'PY'",
            "from pathlib import Path",
            f"p = Path({json_literal(path)})",
            f"old = {json_literal(old)}",
            f"new = {json_literal(new)}",
            "text = p.read_text()",
            "if old not in text:",
            "    raise SystemExit('old string not found')",
            "p.write_text(text.replace(old, new, 1))",
            "PY",
        ]
    )


def python_insert_command(path: str, line: int, text: str) -> str:
    return "\n".join(
        [
            "python3 - <<'PY'",
            "from pathlib import Path",
            f"p = Path({json_literal(path)})",
            f"line = {int(line)}",
            f"insert = {json_literal(text)}",
            "lines = p.read_text().splitlines(True)",
            "idx = max(0, min(len(lines), line))",
            "if insert and not insert.endswith('\\n'):",
            "    insert += '\\n'",
            "lines.insert(idx, insert)",
            "p.write_text(''.join(lines))",
            "PY",
        ]
    )


def editor_view_command(args: dict[str, Any]) -> str | None:
    path = args.get("path")
    if not path:
        return None
    quoted = shlex.quote(str(path))
    view_range = args.get("view_range")
    if isinstance(view_range, list) and view_range:
        start = max(1, int(view_range[0]))
        end = int(view_range[1]) if len(view_range) > 1 and view_range[1] not in (None, -1) else start + 200
        return f"sed -n '{start},{end}p' {quoted} | cat -n"
    return (
        f"if [ -d {quoted} ]; then "
        f"find {quoted} -maxdepth 2 -mindepth 1 | sort | head -200; "
        f"else sed -n '1,240p' {quoted} | cat -n; fi"
    )


def native_tool_to_bash(call: dict[str, Any]) -> dict[str, Any] | None:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = str(function.get("name", "")).lower()
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"command": args}
    if not isinstance(args, dict):
        args = {}

    if name in {"bash", "execute_bash"}:
        command = args.get("command") or args.get("cmd")
        if command:
            return bash_call(str(command).rstrip())
    if name in {"submit", "finish"}:
        return bash_call(MINI_SWE_SUBMIT_COMMAND)
    if name == "str_replace_editor":
        command = str(args.get("command", "")).lower()
        path = str(args.get("path", ""))
        if command == "view":
            view_command = editor_view_command(args)
            return bash_call(view_command) if view_command else None
        if command in {"str_replace", "replace"} and path and args.get("old_str") is not None and args.get("new_str") is not None:
            return bash_call(python_replace_command(path, str(args["old_str"]), str(args["new_str"])))
        if command == "insert" and path and args.get("insert_line") is not None and args.get("new_str") is not None:
            return bash_call(python_insert_command(path, int(args["insert_line"]), str(args["new_str"])))
        if command == "create" and path and args.get("file_text") is not None:
            quoted = shlex.quote(path)
            return bash_call(f"cat > {quoted} <<'EOF'\n{args['file_text']}\nEOF")
    return None


def extract_think_text(content: str) -> str:
    start = content.find("<think>")
    end = content.find("</think>")
    if start == -1 or end == -1 or end < start:
        return ""
    return content[start + len("<think>") : end].strip()


def split_thinking_and_visible(content: str) -> tuple[str, str]:
    start = content.find("<think>")
    end = content.find("</think>")
    if start == -1 or end == -1 or end < start:
        return "", content.strip()
    thinking = content[start + len("<think>") : end].strip()
    visible = content[end + len("</think>") :].strip()
    return thinking, visible


def tool_reasoning_from_content(content: str) -> str:
    thinking, visible = split_thinking_and_visible(content)
    return "\n\n".join(part for part in (thinking, visible) if part).strip()


def strip_think_tags(content: str) -> str:
    return content.replace("<think>", "").replace("</think>", "").strip()


def iter_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    idx = 0
    while True:
        start = text.find("{", idx)
        if start == -1:
            return objects
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        idx = start + max(end, 1)


def command_json_to_bash(content: str) -> tuple[str, list[dict[str, Any]]]:
    payload = None
    for candidate in iter_json_objects(content):
        if isinstance(candidate.get("commands"), list):
            payload = candidate
    if payload is None:
        return content, []

    calls: list[dict[str, Any]] = []
    for item in payload.get("commands") or []:
        if not isinstance(item, dict):
            continue
        command = str(item.get("keystrokes") or "").rstrip()
        if command:
            calls.append(bash_call(command))
    if not calls and payload.get("task_complete") is True:
        calls.append(bash_call(MINI_SWE_SUBMIT_COMMAND))
    if not calls:
        return content, []

    thinking = extract_think_text(content)
    analysis = str(payload.get("analysis") or "").strip()
    plan = str(payload.get("plan") or "").strip()
    response_parts = [part for part in (analysis, plan) if part]
    response = "\n\n".join(response_parts)
    if thinking:
        response = f"<think>\n{thinking}\n</think>\n{response}".rstrip()
    elif response:
        response = f"<think>\n\n</think>\n{response}".rstrip()
    return response or content, calls


def fenced_command_to_bash(content: str) -> tuple[str, list[dict[str, Any]]]:
    if "```" not in content:
        return content, []
    pieces = content.split("```")
    if len(pieces) < 3:
        return content, []
    block = pieces[1]
    if "\n" in block:
        lang, command = block.split("\n", 1)
        if lang.strip().lower() not in {"", "bash", "sh", "shell", "console"}:
            return content, []
    else:
        command = block
    command = command.strip()
    if not command or "\n" in command.strip("\n"):
        return content, []
    return content, [bash_call(command)]


def force_mini_swe_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if messages and messages[0].get("role") == "system":
        return [{"role": "system", "content": MINI_SWE_SYSTEM}, *messages[1:]]
    return [{"role": "system", "content": MINI_SWE_SYSTEM}, *messages]


def rewrite_json_command_prompt(content: str) -> tuple[str, bool]:
    if "Format your response as JSON" not in content or "Task Description:" not in content:
        return content, False
    task = content.split("Task Description:", 1)[1].strip()
    if not task:
        return content, False
    return MINI_SWE_USER_TEMPLATE.format(task=task, submit_command=MINI_SWE_SUBMIT_COMMAND), True


def extract_task_text(content: str) -> str:
    for start_tag, end_tag in (
        ("<pr_description>", "</pr_description>"),
        ("<issue_description>", "</issue_description>"),
        ("<issue>", "</issue>"),
    ):
        if start_tag in content and end_tag in content:
            start = content.find(start_tag) + len(start_tag)
            end = content.find(end_tag, start)
            inner = content[start:end].strip()
            if inner.startswith("Consider the following PR description:"):
                inner = inner.split("Consider the following PR description:", 1)[1].strip()
            if inner:
                return inner
    if "Task Description:" in content:
        task = content.split("Task Description:", 1)[1].strip()
        if task:
            return task
    if content.strip().startswith("Please solve this issue:"):
        task = content.split("Please solve this issue:", 1)[1].strip()
        if task:
            return task
    return content.strip()


def strict_mini_swe_prompt(content: str) -> str:
    return MINI_SWE_STRICT_USER_TEMPLATE.format(
        task=extract_task_text(content),
        submit_command=MINI_SWE_SUBMIT_COMMAND,
    )


def message_has_submit_command(message: dict[str, Any]) -> bool:
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"command": args}
        if isinstance(args, dict) and MINI_SWE_SUBMIT_MARKER in str(args.get("command", "")).lower():
            return True
    return MINI_SWE_SUBMIT_MARKER in str(message.get("content", "")).lower()


def has_submit_command(messages: list[dict[str, Any]]) -> bool:
    return any(message_has_submit_command(message) for message in messages if message.get("role") == "assistant")


TOOL_OBSERVATION_PREFIXES = (
    "New Terminal Output:",
    "Current terminal state:",
    "Current Terminal Screen:",
)
SHELL_PROMPT_RE = re.compile(r"^(?:\([^)]*\)\s*)?[^@\n#\s]+@[^#\n]*#.*$")
MINI_SWE_OUTPUT_LIMIT_CHARS = 10_000
MINI_SWE_OUTPUT_HEAD_CHARS = 5_000
MINI_SWE_OUTPUT_TAIL_CHARS = 5_000
MINI_SWE_LONG_OUTPUT_WARNING = """The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file."""


def is_terminal_observation(content: str) -> bool:
    stripped = content.lstrip()
    return stripped.startswith(TOOL_OBSERVATION_PREFIXES) or stripped.startswith("<returncode>")


def parse_wrapped_tool_observation(content: str) -> tuple[str, str] | None:
    rc_match = re.search(r"<returncode>\s*(.*?)\s*</returncode>", content, re.DOTALL)
    out_match = re.search(r"<output>\n?(.*?)\n?</output>", content, re.DOTALL)
    if out_match:
        output = out_match.group(1)
    else:
        head_match = re.search(r"<output_head>\n?(.*?)\n?</output_head>", content, re.DOTALL)
        tail_match = re.search(r"<output_tail>\n?(.*?)\n?</output_tail>", content, re.DOTALL)
        if not head_match and not tail_match:
            return None
        output = "\n".join(
            part
            for part in (
                head_match.group(1) if head_match else "",
                tail_match.group(1) if tail_match else "",
            )
            if part
        )
    returncode = (rc_match.group(1).strip() if rc_match else "0") or "0"
    return returncode, output


def normalize_tool_observation(content: str) -> str:
    stripped = content.strip()
    json_observation = format_json_tool_observation(stripped)
    if json_observation is not None:
        return json_observation
    for prefix in TOOL_OBSERVATION_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
            break
    if stripped.startswith("<returncode>") or stripped.startswith("<output>"):
        parsed = parse_wrapped_tool_observation(stripped)
        if parsed is None:
            return stripped
        return format_tool_observation(*parsed)
    return f"<returncode>0</returncode>\n<output>\n{stripped}\n</output>"


def parse_tool_observation(content: str) -> tuple[str, str] | None:
    normalized = normalize_tool_observation(content)
    return parse_wrapped_tool_observation(normalized)


def format_tool_observation(returncode: str, output: str) -> str:
    output = output.strip("\n")
    parts = [f"<returncode>{returncode}</returncode>"]
    if len(output) < MINI_SWE_OUTPUT_LIMIT_CHARS:
        parts.append(f"<output>\n{output}\n</output>")
    else:
        elided_chars = len(output) - MINI_SWE_OUTPUT_LIMIT_CHARS
        parts.extend(
            [
                f"<warning>\n{MINI_SWE_LONG_OUTPUT_WARNING}\n</warning>",
                f"<output_head>\n{output[:MINI_SWE_OUTPUT_HEAD_CHARS]}\n</output_head>",
                f"<elided_chars>\n{elided_chars} characters elided\n</elided_chars>",
                f"<output_tail>\n{output[-MINI_SWE_OUTPUT_TAIL_CHARS:]}\n</output_tail>",
            ]
        )
    return "\n".join(parts)


def format_json_tool_observation(content: str) -> str | None:
    if not content.startswith("{"):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if not any(key in payload for key in ("returncode", "output", "exception_info")):
        return None
    returncode = str(payload.get("returncode", 0))
    output = payload.get("output", "")
    if not isinstance(output, str):
        output = json_dumps(output)
    observation = format_tool_observation(returncode, output)
    exception = payload.get("exception_info")
    if exception:
        return f"<exception>{exception}</exception>\n{observation}"
    return observation


def strip_shell_echo_lines(output: str) -> str:
    lines = output.splitlines()
    while lines and (SHELL_PROMPT_RE.match(lines[0]) or lines[0].startswith("> ")):
        lines.pop(0)
    while lines and (SHELL_PROMPT_RE.match(lines[-1]) or lines[-1].startswith("> ")):
        lines.pop()
    return "\n".join(lines).strip("\n")


def command_from_bash_call(call: dict[str, Any]) -> str:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"command": args}
    if not isinstance(args, dict):
        return ""
    return str(args.get("command") or "")


def command_line_bounds(output: str, command: str, start: int) -> tuple[int, int, int] | None:
    if not command or "\n" in command:
        return None
    pos = output.find(command, start)
    if pos == -1:
        return None
    line_start = output.rfind("\n", 0, pos) + 1
    line_end = output.find("\n", pos)
    if line_end == -1:
        line_end = len(output)
    return line_start, line_end, pos


def split_tool_observation_by_commands(content: str, commands: list[str]) -> list[str] | None:
    parsed = parse_tool_observation(content)
    if parsed is None or not commands:
        return None
    returncode, output = parsed
    bounds: list[tuple[int, int, int]] = []
    cursor = 0
    for command in commands:
        found = command_line_bounds(output, command, cursor)
        if found is None:
            return None
        bounds.append(found)
        cursor = found[1]

    chunks: list[str] = []
    for idx, (_, line_end, _) in enumerate(bounds):
        next_line_start = bounds[idx + 1][0] if idx + 1 < len(bounds) else len(output)
        chunk = strip_shell_echo_lines(output[line_end + 1 : next_line_start])
        chunks.append(format_tool_observation(returncode, chunk))
    return chunks


def split_tool_observation_by_prompt_lines(content: str, expected_count: int) -> list[str] | None:
    parsed = parse_tool_observation(content)
    if parsed is None or expected_count <= 1:
        return None
    returncode, output = parsed
    matches = list(re.finditer(r"(?m)^(?:\([^)]*\)\s*)?[^@\n#\s]+@[^#\n]*#.*$", output))
    if len(matches) < expected_count:
        return None
    chunks: list[str] = []
    for idx in range(expected_count):
        start = matches[idx].end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(output)
        chunks.append(format_tool_observation(returncode, strip_shell_echo_lines(output[start:end])))
    return chunks


def clean_tool_observation_for_command(content: str, command: str) -> str:
    parsed = parse_tool_observation(content)
    if parsed is None:
        return normalize_tool_observation(content)
    returncode, output = parsed
    found = command_line_bounds(output, command, 0)
    if found is None:
        matches = list(re.finditer(r"(?m)^(?:\([^)]*\)\s*)?[^@\n#\s]+@[^#\n]*#.*$", output))
        if not matches:
            return normalize_tool_observation(content)
        return format_tool_observation(returncode, strip_shell_echo_lines(output[matches[0].end() :]))
    _, line_end, _ = found
    return format_tool_observation(returncode, strip_shell_echo_lines(output[line_end + 1 :]))


def expand_single_tool_calls(messages: list[dict[str, Any]], stats: dict[str, int]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    idx = 0
    while idx < len(messages):
        message = messages[idx]
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            expanded.append(message)
            idx += 1
            continue

        calls = list(message.get("tool_calls") or [])
        next_message = messages[idx + 1] if idx + 1 < len(messages) else None
        if not next_message or next_message.get("role") != "tool":
            expanded.append(message)
            idx += 1
            continue

        commands = [command_from_bash_call(call) for call in calls]
        if len(calls) == 1:
            assistant_msg = {"role": "assistant", "content": message.get("content", ""), "tool_calls": calls}
            if "reasoning_content" in message:
                assistant_msg["reasoning_content"] = message["reasoning_content"]
            expanded.append(assistant_msg)
            expanded.append(
                {
                    "role": "tool",
                    "content": clean_tool_observation_for_command(str(next_message.get("content", "")), commands[0]),
                }
            )
            stats["tool_observations_cleaned"] = stats.get("tool_observations_cleaned", 0) + 1
            idx += 2
            continue

        chunks = split_tool_observation_by_commands(str(next_message.get("content", "")), commands)
        if chunks is None:
            chunks = split_tool_observation_by_prompt_lines(str(next_message.get("content", "")), len(calls))
        if chunks is None or len(chunks) != len(calls):
            stats["single_call_failed_splits"] = stats.get("single_call_failed_splits", 0) + 1
            expanded.append(message)
            expanded.append(next_message)
            idx += 2
            continue

        for call, chunk in zip(calls, chunks, strict=True):
            assistant_msg = {"role": "assistant", "content": message.get("content", ""), "tool_calls": [call]}
            if "reasoning_content" in message:
                assistant_msg["reasoning_content"] = message["reasoning_content"]
            expanded.append(assistant_msg)
            expanded.append({"role": "tool", "content": chunk})
        stats["single_call_expanded_turns"] = stats.get("single_call_expanded_turns", 0) + 1
        stats["single_call_split_observations"] = stats.get("single_call_split_observations", 0) + len(calls)
        idx += 2
    return expanded


def first_user_and_assistant(messages: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    first_user = None
    first_assistant = None
    for message in messages:
        if first_user is None and message.get("role") == "user":
            first_user = message
        if first_assistant is None and message.get("role") == "assistant":
            first_assistant = message
        if first_user is not None and first_assistant is not None:
            break
    return first_user, first_assistant


def adapt_to_bash_tool(
    example: dict[str, Any],
    *,
    reasoning_tool_boundary: bool = False,
    tool_call_only: bool = False,
    strict_prompt: bool = False,
    require_submit: bool = False,
    first_turn_only: bool = False,
    recovery_only: bool = False,
    tool_observation_roles: bool = False,
    single_tool_calls: bool = False,
    empty_think_tool_calls: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    stats = {
        "assistant_turns": 0,
        "assistant_turns_with_bash": 0,
        "native_tool_turns": 0,
        "command_json_turns": 0,
        "fenced_command_turns": 0,
        "dropped_missing_bash": 0,
        "json_prompts_rewritten": 0,
        "strict_prompts_rewritten": 0,
        "dropped_missing_submit": 0,
        "dropped_missing_first_turn": 0,
        "dropped_missing_recovery_turn": 0,
        "terminal_observations_to_tool": 0,
    }
    messages: list[dict[str, Any]] = []
    seen_user = False
    for msg in force_mini_swe_system(list(example["messages"])):
        role = msg.get("role")
        if role != "assistant":
            content = str(msg.get("content", ""))
            if (
                tool_observation_roles
                and role == "user"
                and messages
                and messages[-1].get("role") == "assistant"
                and messages[-1].get("tool_calls")
                and is_terminal_observation(content)
            ):
                messages.append({"role": "tool", "content": normalize_tool_observation(content)})
                stats["terminal_observations_to_tool"] += 1
                continue
            if tool_observation_roles and role == "tool":
                content = normalize_tool_observation(content)
            if role == "user" and not seen_user:
                if strict_prompt:
                    content = strict_mini_swe_prompt(content)
                    stats["strict_prompts_rewritten"] += 1
                else:
                    content, rewritten = rewrite_json_command_prompt(content)
                    if rewritten:
                        stats["json_prompts_rewritten"] += 1
                seen_user = True
            messages.append({"role": role, "content": content})
            continue

        stats["assistant_turns"] += 1
        content = str(msg.get("content") or "")
        calls = [
            converted
            for converted in (native_tool_to_bash(call) for call in (msg.get("tool_calls") or []))
            if converted is not None
        ]
        if calls:
            stats["native_tool_turns"] += 1
        if not calls:
            content, calls = command_json_to_bash(content)
            if calls:
                stats["command_json_turns"] += 1
        if not calls:
            content, calls = fenced_command_to_bash(content)
            if calls:
                stats["fenced_command_turns"] += 1
        if not calls:
            stats["dropped_missing_bash"] += 1
            return None, stats

        stats["assistant_turns_with_bash"] += 1
        if tool_call_only:
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": calls}
            if empty_think_tool_calls:
                assistant_msg["reasoning_content"] = "\n"
            messages.append(assistant_msg)
        elif reasoning_tool_boundary:
            reasoning = tool_reasoning_from_content(content)
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": calls,
            }
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            messages.append(assistant_msg)
        else:
            if strict_prompt:
                content = strip_think_tags(tool_reasoning_from_content(content) or content)
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})

    if require_submit and not has_submit_command(messages):
        stats["dropped_missing_submit"] += 1
        return None, stats

    if single_tool_calls:
        messages = expand_single_tool_calls(messages, stats)
        if stats.get("single_call_failed_splits", 0):
            return None, stats
        seen_user_after_rewrite = False
        for message in messages:
            if message.get("role") != "user":
                continue
            if seen_user_after_rewrite:
                stats["dropped_extra_user_turn"] = stats.get("dropped_extra_user_turn", 0) + 1
                return None, stats
            seen_user_after_rewrite = True
        if any(
            message.get("role") == "assistant" and len(message.get("tool_calls") or []) > 1
            for message in messages
        ):
            stats["dropped_multi_tool_after_expand"] = stats.get("dropped_multi_tool_after_expand", 0) + 1
            return None, stats

    if first_turn_only:
        system = messages[0] if messages and messages[0].get("role") == "system" else {"role": "system", "content": MINI_SWE_SYSTEM}
        first_user, first_assistant = first_user_and_assistant(messages)
        if first_user is None or first_assistant is None:
            stats["dropped_missing_first_turn"] += 1
            return None, stats
        messages = [
            system,
            first_user,
            first_assistant,
        ]

    if recovery_only:
        system = messages[0] if messages and messages[0].get("role") == "system" else {"role": "system", "content": MINI_SWE_SYSTEM}
        first_user, first_assistant = first_user_and_assistant(messages)
        if first_user is None or first_assistant is None:
            stats["dropped_missing_recovery_turn"] += 1
            return None, stats
        messages = [
            system,
            first_user,
            {"role": "user", "content": MINI_SWE_FORMAT_ERROR},
            first_assistant,
        ]

    return {"messages": messages, "tools": BASH_TOOL}, stats


def build_source(
    spec: SourceSpec,
    *,
    raw_root: Path,
    output_root: Path,
    seed: int,
    parquet_batch_size: int,
) -> dict[str, Any]:
    dataset_root = raw_root / spec.dataset_dir
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    files = included_files(dataset_root, spec.include, seed)
    if not files:
        raise RuntimeError(f"{spec.name}: no files matched {spec.include}")

    out_dir = output_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.jsonl"
    rows = 0
    rows_seen = 0
    rows_skipped = 0
    estimated_tokens = 0
    transform_stats: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as out:
        for example in iter_normalized_examples_from_files(
            files,
            parquet_batch_size=parquet_batch_size,
        ):
            rows_seen += 1
            if spec.transform == "bash_tool":
                example, stats = adapt_to_bash_tool(example)
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_reasoning":
                example, stats = adapt_to_bash_tool(example, reasoning_tool_boundary=True)
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_submit_strict":
                example, stats = adapt_to_bash_tool(
                    example,
                    strict_prompt=True,
                    require_submit=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_submit_strict_toolobs":
                example, stats = adapt_to_bash_tool(
                    example,
                    strict_prompt=True,
                    require_submit=True,
                    tool_observation_roles=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_submit_strict_toolobs_reasoning":
                example, stats = adapt_to_bash_tool(
                    example,
                    reasoning_tool_boundary=True,
                    strict_prompt=True,
                    require_submit=True,
                    tool_observation_roles=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_submit_strict_toolobs_toolonly":
                example, stats = adapt_to_bash_tool(
                    example,
                    tool_call_only=True,
                    strict_prompt=True,
                    require_submit=True,
                    tool_observation_roles=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_submit_strict_toolobs_toolonly_single":
                example, stats = adapt_to_bash_tool(
                    example,
                    tool_call_only=True,
                    strict_prompt=True,
                    require_submit=True,
                    tool_observation_roles=True,
                    single_tool_calls=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_submit_strict_toolobs_toolonly_single_emptythink":
                example, stats = adapt_to_bash_tool(
                    example,
                    tool_call_only=True,
                    strict_prompt=True,
                    require_submit=True,
                    tool_observation_roles=True,
                    single_tool_calls=True,
                    empty_think_tool_calls=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_submit_strict_toolobs_single_visible":
                example, stats = adapt_to_bash_tool(
                    example,
                    strict_prompt=True,
                    require_submit=True,
                    tool_observation_roles=True,
                    single_tool_calls=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_strict_toolobs_single_visible":
                example, stats = adapt_to_bash_tool(
                    example,
                    strict_prompt=True,
                    require_submit=False,
                    tool_observation_roles=True,
                    single_tool_calls=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_strict_toolobs_toolonly_single_emptythink":
                example, stats = adapt_to_bash_tool(
                    example,
                    tool_call_only=True,
                    strict_prompt=True,
                    require_submit=False,
                    tool_observation_roles=True,
                    single_tool_calls=True,
                    empty_think_tool_calls=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_first_strict_toolobs_toolonly":
                example, stats = adapt_to_bash_tool(
                    example,
                    tool_call_only=True,
                    strict_prompt=True,
                    require_submit=True,
                    first_turn_only=True,
                    tool_observation_roles=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_recovery_strict_toolobs":
                example, stats = adapt_to_bash_tool(
                    example,
                    strict_prompt=True,
                    require_submit=True,
                    recovery_only=True,
                    tool_observation_roles=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_recovery_strict_toolobs_toolonly":
                example, stats = adapt_to_bash_tool(
                    example,
                    tool_call_only=True,
                    strict_prompt=True,
                    require_submit=True,
                    recovery_only=True,
                    tool_observation_roles=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_recovery_strict_toolobs_toolonly_single":
                example, stats = adapt_to_bash_tool(
                    example,
                    tool_call_only=True,
                    strict_prompt=True,
                    require_submit=True,
                    recovery_only=True,
                    tool_observation_roles=True,
                    single_tool_calls=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_recovery_strict_toolobs_toolonly_single_emptythink":
                example, stats = adapt_to_bash_tool(
                    example,
                    tool_call_only=True,
                    strict_prompt=True,
                    require_submit=True,
                    recovery_only=True,
                    tool_observation_roles=True,
                    single_tool_calls=True,
                    empty_think_tool_calls=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_recovery_strict_toolobs_single_visible":
                example, stats = adapt_to_bash_tool(
                    example,
                    strict_prompt=True,
                    require_submit=True,
                    recovery_only=True,
                    tool_observation_roles=True,
                    single_tool_calls=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_recovery_strict_toolobs_reasoning":
                example, stats = adapt_to_bash_tool(
                    example,
                    reasoning_tool_boundary=True,
                    strict_prompt=True,
                    require_submit=True,
                    recovery_only=True,
                    tool_observation_roles=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform == "bash_tool_recovery_strict":
                example, stats = adapt_to_bash_tool(
                    example,
                    strict_prompt=True,
                    require_submit=True,
                    recovery_only=True,
                )
                for key, value in stats.items():
                    transform_stats[key] = transform_stats.get(key, 0) + value
                if example is None:
                    rows_skipped += 1
                    continue
            elif spec.transform != "none":
                raise ValueError(f"unknown transform {spec.transform!r}")
            row = {
                "messages": example["messages"],
                "source": spec.name,
                "source_note": spec.note,
            }
            if "tools" in example:
                row["tools"] = example["tools"]
            out.write(json_dumps(row) + "\n")
            rows += 1
            estimated_tokens += spec.avg_tokens_per_row
            if estimated_tokens >= spec.token_budget:
                break

    return {
        "name": spec.name,
        "dataset_dir": spec.dataset_dir,
        "include": list(spec.include),
        "note": spec.note,
        "rows": rows,
        "rows_seen": rows_seen,
        "rows_skipped": rows_skipped,
        "avg_tokens_per_row": spec.avg_tokens_per_row,
        "target_tokens": spec.token_budget,
        "estimated_tokens": estimated_tokens,
        "output": str(out_path),
        "source_files_considered": len(files),
        "transform": spec.transform,
        "transform_stats": transform_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--preset",
        choices=[
            "v0_180m",
            "v1_bash_tool_180m",
            "v2_reasoning_tool_180m",
            "v3_miniswe_submit_200m",
            "v4_miniswe_toolobs_128m",
            "v5_miniswe_reasoning_toolobs_128m",
            "v6_miniswe_toolonly_toolobs_128m",
            "v7_miniswe_firstturn_recovery_toolonly_180m",
            "v8_miniswe_curated_firstturn_recovery_toolonly_180m",
            "v9_miniswe_curated_full_traj_toolonly_180m",
            "v10_miniswe_singlecall_full_traj_toolonly_180m",
            "v11_miniswe_emptythink_singlecall_full_traj_toolonly_166m",
            "v12_miniswe_emptythink_truncated_singlecall_full_traj_toolonly_166m",
            "v14_miniswe_langbalanced_emptythink_singlecall_220m",
            "v15_miniswe_langbalanced_emptythink_singlecall_215m",
            "v16_miniswe_langbalanced_visible_singlecall_215m",
            "v29_miniswe_visible_strong_agentic_240m",
            "v30_miniswe_verified_toolonly_180m",
        ],
        default="v0_180m",
    )
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--parquet-batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.preset == "v0_180m":
        specs = PRESET_V0_180M
    elif args.preset == "v1_bash_tool_180m":
        specs = PRESET_V1_BASH_TOOL_180M
    elif args.preset == "v2_reasoning_tool_180m":
        specs = PRESET_V2_REASONING_TOOL_180M
    elif args.preset == "v3_miniswe_submit_200m":
        specs = PRESET_V3_MINISWE_SUBMIT_200M
    elif args.preset == "v4_miniswe_toolobs_128m":
        specs = PRESET_V4_MINISWE_TOOLOBS_128M
    elif args.preset == "v5_miniswe_reasoning_toolobs_128m":
        specs = PRESET_V5_MINISWE_REASONING_TOOLOBS_128M
    elif args.preset == "v6_miniswe_toolonly_toolobs_128m":
        specs = PRESET_V6_MINISWE_TOOLONLY_TOOLOBS_128M
    elif args.preset == "v7_miniswe_firstturn_recovery_toolonly_180m":
        specs = PRESET_V7_MINISWE_FIRSTTURN_RECOVERY_TOOLONLY_180M
    elif args.preset == "v8_miniswe_curated_firstturn_recovery_toolonly_180m":
        specs = PRESET_V8_MINISWE_CURATED_FIRSTTURN_RECOVERY_TOOLONLY_180M
    elif args.preset == "v9_miniswe_curated_full_traj_toolonly_180m":
        specs = PRESET_V9_MINISWE_CURATED_FULL_TRAJ_TOOLONLY_180M
    elif args.preset == "v10_miniswe_singlecall_full_traj_toolonly_180m":
        specs = PRESET_V10_MINISWE_SINGLECALL_FULL_TRAJ_TOOLONLY_180M
    elif args.preset == "v11_miniswe_emptythink_singlecall_full_traj_toolonly_166m":
        specs = PRESET_V11_MINISWE_EMPTYTHINK_SINGLECALL_FULL_TRAJ_TOOLONLY_166M
    elif args.preset == "v12_miniswe_emptythink_truncated_singlecall_full_traj_toolonly_166m":
        specs = PRESET_V12_MINISWE_EMPTYTHINK_TRUNCATED_SINGLECALL_FULL_TRAJ_TOOLONLY_166M
    elif args.preset == "v14_miniswe_langbalanced_emptythink_singlecall_220m":
        specs = PRESET_V14_MINISWE_LANGBALANCED_EMPTYTHINK_SINGLECALL_220M
    elif args.preset == "v15_miniswe_langbalanced_emptythink_singlecall_215m":
        specs = PRESET_V15_MINISWE_LANGBALANCED_EMPTYTHINK_SINGLECALL_215M
    elif args.preset == "v16_miniswe_langbalanced_visible_singlecall_215m":
        specs = PRESET_V16_MINISWE_LANGBALANCED_VISIBLE_SINGLECALL_215M
    elif args.preset == "v29_miniswe_visible_strong_agentic_240m":
        specs = PRESET_V29_MINISWE_VISIBLE_STRONG_AGENTIC_240M
    elif args.preset == "v30_miniswe_verified_toolonly_180m":
        specs = PRESET_V30_MINISWE_VERIFIED_TOOLONLY_180M
    else:
        raise AssertionError(args.preset)
    summaries = [
        build_source(
            spec,
            raw_root=args.raw_root,
            output_root=args.output_root,
            seed=args.seed + idx,
            parquet_batch_size=args.parquet_batch_size,
        )
        for idx, spec in enumerate(specs)
    ]
    manifest = {
        "preset": args.preset,
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "estimated_tokens": sum(item["estimated_tokens"] for item in summaries),
        "target_tokens": sum(spec.token_budget for spec in specs),
        "sources": summaries,
        "excluded": [
            {
                "dataset": "AlienKevin__SWE-ZERO-12M-trajectories",
                "reason": "weak mini-coder-1.7B teacher, explicitly excluded",
            }
        ],
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
