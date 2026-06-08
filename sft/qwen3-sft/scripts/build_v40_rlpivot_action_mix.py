#!/usr/bin/env python3
"""Build v40 prefix-target rows from RL pivot SWE action supervision.

The current Qwen3-VL SFT checkpoints can make parsed mini-swe bash calls but
stay in read-only loops. This builder emphasizes the missing next-action
transitions by converting single-step RL pivot expected actions into exact
mini-swe bash-tool prefix rows. Prior actions are retained as masked context;
only the final expected action is trainable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


RL_PIVOT = Path("/wbl-fast/usrs/ee/code-swe-data/data/extra/nvidia__Nemotron-RL-Agentic-SWE-Pivot-v1/train.jsonl")

BASH_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the shell.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
]

MINI_SWE_SYSTEM = "You are a helpful assistant that can interact with a computer shell to solve programming tasks."
MINI_SWE_SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
MINI_SWE_USER_TEMPLATE = """<pr_description>
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

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
Do NOT commit your changes.

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit:

```bash
{submit_command}
```
</instructions>"""

MINI_SWE_OUTPUT_LIMIT_CHARS = 10_000
MINI_SWE_OUTPUT_HEAD_CHARS = 5_000
MINI_SWE_OUTPUT_TAIL_CHARS = 5_000
MINI_SWE_LONG_OUTPUT_WARNING = """The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file."""

EDIT_ACTIONS = {"apply_patch", "edit", "str_replace_editor", "write"}
SHELL_ACTIONS = {"execute_bash", "shell_command", "bash"}
READ_ACTIONS = {"read_file", "read", "grep", "grep_files", "glob", "list_dir"}
SKIP_ACTIONS = {"todo_write", "update_plan", "task_tracker", "think"}


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def stable_fraction(*parts: object) -> float:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def extract_between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        return ""
    left = text.find(start) + len(start)
    right = text.find(end, left)
    return text[left:right].strip() if right >= 0 else ""


def extract_task_text(content: str) -> str:
    content = content.strip()
    for start, end in (
        ("[ISSUE]", "[/ISSUE]"),
        ("<issue_description>", "</issue_description>"),
        ("<github_issue_description>", "</github_issue_description>"),
        ("<pr_description>", "</pr_description>"),
    ):
        text = extract_between(content, start, end)
        if text:
            return text
    match = re.search(r"Consider the following issue description:\s*(.*)", content, flags=re.S)
    if match:
        return match.group(1).strip()
    return content


def first_user_task(items: list[dict[str, Any]]) -> str:
    for item in items:
        if item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, list):
                content = "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
            return extract_task_text(str(content or ""))
    return ""


def normalize_path(path: str) -> str:
    path = str(path or "").strip()
    path = re.sub(r"^/workspace/[^/\s'\"`]+/", "", path)
    path = re.sub(r"^/testbed/", "", path)
    path = path.lstrip("./")
    return path


def normalize_command(command: str) -> str:
    command = str(command or "").strip()
    command = re.sub(r"cd\s+/workspace/[^&;\n]+&&\s*", "", command)
    command = re.sub(r"cd\s+/workspace/[^&;\n]+;\s*", "", command)
    command = re.sub(r"cd\s+/testbed\s*&&\s*", "", command)
    command = re.sub(r"/workspace/[^/\s'\"`]+/", "", command)
    command = command.replace("/testbed/", "")
    return command.strip()


def bash_call(command: str) -> dict[str, Any]:
    return {"function": {"name": "bash", "arguments": {"command": command}}}


def assistant(command: str, thought: str, *, trainable: bool, emptythink_toolonly: bool = False) -> dict[str, Any]:
    if emptythink_toolonly:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "\n",
            "tool_calls": [bash_call(command)],
        }
    else:
        message = {
            "role": "assistant",
            "content": f"<thought>\n{thought}\n</thought>",
            "tool_calls": [bash_call(command)],
        }
    if not trainable:
        message["trainable"] = False
    return message


def normalize_tool_output(output: str) -> str:
    text = str(output or "").strip("\n")
    returncode = "0"
    match = re.search(r"(?:\[Command finished with exit code|\[The command completed with exit code)\s+(-?\d+)\]$", text)
    if match:
        returncode = match.group(1)
    parts = [f"<returncode>{returncode}</returncode>"]
    if len(text) <= MINI_SWE_OUTPUT_LIMIT_CHARS:
        parts.append(f"<output>\n{text}\n</output>")
    else:
        elided = len(text) - MINI_SWE_OUTPUT_LIMIT_CHARS
        parts.extend(
            [
                f"<warning>\n{MINI_SWE_LONG_OUTPUT_WARNING}\n</warning>",
                f"<output_head>\n{text[:MINI_SWE_OUTPUT_HEAD_CHARS]}\n</output_head>",
                f"<elided_chars>\n{elided} characters elided\n</elided_chars>",
                f"<output_tail>\n{text[-MINI_SWE_OUTPUT_TAIL_CHARS:]}\n</output_tail>",
            ]
        )
    return "\n".join(parts)


def tool_message(output: str) -> dict[str, str]:
    return {"role": "tool", "content": normalize_tool_output(output)}


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def python_replace_command(path: str, old: str, new: str) -> str:
    payload = {
        "path": normalize_path(path),
        "old": old,
        "new": new,
    }
    return (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        f"data = {json.dumps(payload, ensure_ascii=False)!r}\n"
        "import json\n"
        "data = json.loads(data)\n"
        "path = Path(data['path'])\n"
        "text = path.read_text()\n"
        "old = data['old']\n"
        "if old not in text:\n"
        "    raise SystemExit('old string not found')\n"
        "path.write_text(text.replace(old, data['new'], 1))\n"
        "PY"
    )


def cat_write_command(path: str, content: str) -> str:
    return f"cat > {shell_quote(normalize_path(path))} <<'EOF'\n{content.rstrip()}\nEOF"


def command_category(command: str, action_name: str = "") -> str:
    name = action_name.lower()
    text = command.strip().lower()
    if name in {"apply_patch", "edit"}:
        return "edit"
    if name == "str_replace_editor" and text:
        return "edit" if "python - <<'py'" in text or "apply_patch" in text else "inspect"
    if name == "write":
        return "edit"
    if name == "finish" or "complete_task_and_submit_final_output" in text:
        return "submit"
    if "git diff" in text:
        return "diff"
    if any(marker in text for marker in ("apply_patch", "git apply", "sed -i", "perl -", "cat >", "tee ", "python - <<", "python3 - <<")):
        return "edit"
    if any(marker in text for marker in ("pytest", "mvn", "gradle", "npm test", "go test", "cargo test", "phpunit", "unittest")):
        return "test"
    if any(marker in text for marker in ("grep", "rg ", "find ", "cat ", "sed -n", "head ", "tail ", "ls ")):
        return "inspect"
    return "other"


def thought_for(category: str) -> str:
    return {
        "edit": "I have enough context to edit the source file now.",
        "diff": "The source change is applied. I will inspect the final diff.",
        "test": "I will run a focused verification command for the change.",
        "submit": "The source diff is ready in patch.txt. I will submit it now.",
        "inspect": "I will inspect the repository with a focused shell command.",
    }.get(category, "I will take the next focused shell action.")


def is_bad_edit_path(path: str) -> bool:
    p = normalize_path(path).lower()
    if not p:
        return True
    name = Path(p).name
    parts = set(Path(p).parts)
    if any(part in {"test", "tests", "testing", "__tests__", "spec", "specs"} for part in parts):
        return True
    if name.startswith(("test_", "reproduce", "debug_", "check_", "scratch_", "tmp_")):
        return True
    if name.endswith(("_test.py", ".test.js", ".spec.js", ".spec.ts", ".sh", ".md", ".rst", ".patch", ".diff")):
        return True
    if name in {"patch.txt", "pyproject.toml", "setup.cfg", "setup.py", "package.json", "poetry.lock"}:
        return True
    return False


def patch_has_bad_paths(patch: str) -> bool:
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            paths.append(line[len("+++ b/") :].strip())
        elif line.startswith("*** Update File: "):
            paths.append(line[len("*** Update File: ") :].strip())
        elif line.startswith("*** Add File: ") or line.startswith("*** Delete File: "):
            return True
    return any(is_bad_edit_path(path) for path in paths) if paths else False


def convert_action(action: dict[str, Any] | None) -> tuple[str, str, str] | None:
    if not isinstance(action, dict):
        return None
    name = str(action.get("name") or "")
    args = parse_json(action.get("arguments") or {})
    if not isinstance(args, dict):
        return None

    command = ""
    if name in SHELL_ACTIONS:
        command = normalize_command(str(args.get("command") or ""))
    elif name == "apply_patch":
        patch = str(args.get("input") or args.get("patch") or "")
        if not patch.strip() or patch_has_bad_paths(patch):
            return None
        command = f"apply_patch <<'PATCH'\n{patch.rstrip()}\nPATCH"
    elif name in {"edit", "str_replace_editor"}:
        editor_cmd = str(args.get("command") or "")
        if editor_cmd and editor_cmd not in {"str_replace", "edit"}:
            if editor_cmd == "view":
                path = normalize_path(str(args.get("path") or args.get("file_path") or ""))
                view_range = args.get("view_range")
                if isinstance(view_range, list) and len(view_range) == 2:
                    command = f"sed -n '{int(view_range[0])},{int(view_range[1])}p' {shell_quote(path)}"
                else:
                    command = f"sed -n '1,200p' {shell_quote(path)}"
            else:
                return None
        else:
            path = str(args.get("file_path") or args.get("path") or "")
            old = str(args.get("old_string") or args.get("old_str") or "")
            new = str(args.get("new_string") or args.get("new_str") or "")
            if is_bad_edit_path(path) or not old:
                return None
            command = python_replace_command(path, old, new)
    elif name == "write":
        path = str(args.get("file_path") or args.get("path") or "")
        content = str(args.get("content") or "")
        if is_bad_edit_path(path) or not content:
            return None
        command = cat_write_command(path, content)
    elif name == "read_file":
        path = normalize_path(str(args.get("path") or args.get("file_path") or ""))
        command = f"sed -n '1,200p' {shell_quote(path)}" if path else ""
    elif name == "read":
        path = normalize_path(str(args.get("file_path") or args.get("path") or ""))
        command = f"sed -n '1,200p' {shell_quote(path)}" if path else ""
    elif name in {"grep", "grep_files"}:
        pattern = str(args.get("pattern") or args.get("query") or "")
        path = normalize_path(str(args.get("path") or args.get("directory") or "."))
        include = str(args.get("include") or "")
        include_part = f" --include {shell_quote(include)}" if include else ""
        command = f"grep -RIn{include_part} {shell_quote(pattern)} {shell_quote(path or '.')}"
    elif name == "glob":
        pattern = str(args.get("pattern") or "*")
        path = normalize_path(str(args.get("path") or "."))
        command = f"find {shell_quote(path or '.')} -path {shell_quote(pattern)}"
    elif name == "list_dir":
        path = normalize_path(str(args.get("path") or "."))
        command = f"ls -la {shell_quote(path or '.')}"
    elif name == "finish":
        command = MINI_SWE_SUBMIT_COMMAND
    elif name in SKIP_ACTIONS:
        return None
    else:
        return None

    command = normalize_command(command)
    if not command or command == "bash":
        return None
    category = command_category(command, name)
    return command, category, name


def convert_history(items: list[dict[str, Any]], max_pairs: int, *, emptythink_toolonly: bool = False) -> list[dict[str, Any]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    pending: dict[str, Any] | None = None
    for item in items:
        item_type = item.get("type")
        if item_type == "function_call":
            converted = convert_action(item)
            if converted is None:
                pending = None
                continue
            command, category, _ = converted
            pending = assistant(command, thought_for(category), trainable=False, emptythink_toolonly=emptythink_toolonly)
            pairs.append((pending, None))
            continue
        if item_type == "function_call_output" and pairs and pairs[-1][1] is None:
            pairs[-1] = (pairs[-1][0], tool_message(str(item.get("output") or "")))
            pending = None
            continue
        if item.get("role") == "assistant":
            continue
        pending = pending

    messages: list[dict[str, Any]] = []
    for assistant_msg, tool_msg in pairs[-max_pairs:]:
        messages.append(assistant_msg)
        if tool_msg is not None:
            messages.append(tool_msg)
    return messages


def build_rl_row(
    raw: dict[str, Any],
    line_number: int,
    max_history_pairs: int,
    *,
    emptythink_toolonly: bool = False,
) -> dict[str, Any] | None:
    params = raw.get("responses_create_params") or {}
    items = params.get("input") or []
    if not isinstance(items, list):
        return None
    task = first_user_task(items)
    if not task:
        return None
    converted = convert_action(raw.get("expected_action"))
    if converted is None:
        return None
    command, category, action_name = converted
    if category not in {"edit", "diff", "test", "submit"}:
        return None
    history = convert_history(items[2:], max_history_pairs, emptythink_toolonly=emptythink_toolonly)
    messages = [
        {"role": "system", "content": MINI_SWE_SYSTEM},
        {
            "role": "user",
            "content": MINI_SWE_USER_TEMPLATE.format(task=task.strip(), submit_command=MINI_SWE_SUBMIT_COMMAND),
        },
        *history,
        assistant(command, thought_for(category), trainable=True, emptythink_toolonly=emptythink_toolonly),
    ]
    return {
        "messages": messages,
        "tools": BASH_TOOL,
        "metadata": {
            "v40_source": "nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1",
            "source_line": line_number,
            "target_category": category,
            "target_action_name": action_name,
            "instance_id": (raw.get("metadata") or {}).get("instance_id"),
            "history_pairs": sum(1 for message in history if message.get("role") == "assistant"),
        },
    }


def iter_jsonl_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield line_number, value


class ShardedWriter:
    def __init__(self, output_dir: Path, shards: int) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.paths = [output_dir / f"shard-{idx:03d}.jsonl" for idx in range(shards)]
        self.handles = [path.open("w", encoding="utf-8") for path in self.paths]
        self.rows = 0

    def write(self, row: dict[str, Any]) -> None:
        self.handles[self.rows % len(self.handles)].write(json_dumps(row) + "\n")
        self.rows += 1

    def close(self) -> int:
        for handle in self.handles:
            handle.close()
        nonempty = 0
        for path in self.paths:
            if path.stat().st_size:
                nonempty += 1
            else:
                path.unlink()
        return nonempty


def cap_probability(category: str, count: int, caps: dict[str, int]) -> float:
    cap = caps.get(category, 0)
    if cap <= 0 or count <= 0:
        return 0.0
    return min(1.0, cap / count)


def parse_caps(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        result[key.strip()] = int(value)
    return result


def build_rl_pivot(
    *,
    input_path: Path,
    output_root: Path,
    shards: int,
    caps: dict[str, int],
    seed: int,
    max_history_pairs: int,
    tag: str,
    emptythink_toolonly: bool,
) -> dict[str, Any]:
    rows: list[tuple[int, dict[str, Any]]] = []
    available: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    for line_number, raw in iter_jsonl_rows(input_path):
        row = build_rl_row(raw, line_number, max_history_pairs, emptythink_toolonly=emptythink_toolonly)
        if row is None:
            dropped["unusable_or_non_target"] += 1
            continue
        category = row["metadata"]["target_category"]
        available[category] += 1
        rows.append((line_number, row))

    writer = ShardedWriter(output_root / f"rlpivot_action_targets_{tag}", shards)
    selected: Counter[str] = Counter()
    action_names: Counter[str] = Counter()
    try:
        for line_number, row in rows:
            category = row["metadata"]["target_category"]
            prob = cap_probability(category, available[category], caps)
            if stable_fraction(seed, "rlpivot", line_number, category) >= prob:
                continue
            row["metadata"]["selection_probability"] = prob
            writer.write(row)
            selected[category] += 1
            action_names[row["metadata"]["target_action_name"]] += 1
    finally:
        nonempty = writer.close()
    return {
        "source": str(input_path),
        "rows_available": sum(available.values()),
        "available_by_category": dict(available),
        "rows_out": sum(selected.values()),
        "rows_out_by_category": dict(selected),
        "action_names": dict(action_names),
        "dropped": dict(dropped),
        "output_files": nonempty,
        "caps": caps,
        "emptythink_toolonly": emptythink_toolonly,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-pivot", type=Path, default=RL_PIVOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=40040)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--tag", default="v40")
    parser.add_argument("--max-history-pairs", type=int, default=8)
    parser.add_argument("--caps", default="edit=18000,diff=5000,test=8000,submit=2000")
    parser.add_argument("--emptythink-toolonly", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rl_summary = build_rl_pivot(
        input_path=args.rl_pivot,
        output_root=args.output_root,
        shards=args.shards,
        caps=parse_caps(args.caps),
        seed=args.seed,
        max_history_pairs=args.max_history_pairs,
        tag=args.tag,
        emptythink_toolonly=args.emptythink_toolonly,
    )
    target_style = "empty Qwen thinking + tool-call-only" if args.emptythink_toolonly else "visible <thought> content + tool call"
    manifest = {
        "output_root": str(args.output_root),
        "seed": args.seed,
        "tag": args.tag,
        "max_history_pairs": args.max_history_pairs,
        "emptythink_toolonly": args.emptythink_toolonly,
        "selection": (
            f"{args.tag} RL pivot prefix-target action rows; assistant targets use {target_style}; "
            "trainable targets are edit, diff, test, and submit bash-tool transitions after realistic SWE histories"
        ),
        "rows_out": rl_summary["rows_out"],
        "sources": [rl_summary],
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
