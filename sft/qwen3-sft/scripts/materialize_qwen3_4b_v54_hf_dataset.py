#!/usr/bin/env python3
"""Materialize the exact processed v54 Qwen3-4B SFT dataset for HF upload."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qwen_agentic_sft.data import (  # noqa: E402
    assistant_has_reasoning,
    assistant_has_valid_tool_calls,
    iter_jsonl_rows,
    json_dumps,
    normalize_row,
    text_from_content,
)
from qwen_agentic_sft.online_packed_dataset import (  # noqa: E402
    assistant_turn_action,
)


DEFAULT_INPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260616/"
    "swerebench-raw2030-targeted-limitations-strict-passed-miniswe-aligned/data"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "qwen3-4b-thinking-sft-v54-raw2030-strictpassed-processed"
)
DEFAULT_SOURCE_DATASET_ID = (
    "eewer/swerebench-traces-raw-source-targeted-limitations-compaction-full-20260616-2030"
)
DEFAULT_REPO_ID = "eewer/qwen3-4b-thinking-sft-v54-raw2030-strictpassed-processed"
DEFAULT_CHAT_TEMPLATE = (
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/"
    "qwen3_thinking_acc.jinja2"
)

MINI_SWE_SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
PATCH_TXT_PATH_PATTERN = r"(?<![\w.-])(?:\./|/testbed/)?patch\.txt(?![\w.-])"
PATCH_LIKE_PATH_PATTERN = (
    r"(?<![\w./-])(?:/tmp/|/testbed/|\./)?[\w./-]+\.(?:txt|patch|diff)(?![\w./-])"
)
PATCH_TXT_WRITE_PATTERN = (
    rf"(>\s*{PATCH_TXT_PATH_PATTERN}"
    rf"|\|\s*tee\s+(-a\s+)?{PATCH_TXT_PATH_PATTERN}"
    rf"|\btee\s+(-a\s+)?{PATCH_TXT_PATH_PATTERN})"
)
PATCH_TXT_SHELL_CREATE_PATTERN = (
    rf"(^|[;&|\n]\s*)(touch|truncate)\b[^;&|\n]*{PATCH_TXT_PATH_PATTERN}"
    rf"|(^|[;&|\n]\s*)(:|true)\s*>\s*{PATCH_TXT_PATH_PATTERN}"
    rf"|(^|[;&|\n]\s*)cp\s+/dev/null\s+{PATCH_TXT_PATH_PATTERN}"
)

SOURCE_MAPPING_KEYS = (
    "uuid",
    "task_id",
    "source_shard",
    "source_row_index",
    "row_source",
    "source_group",
    "language",
    "teacher",
    "difficulty",
    "agent_exit_status",
    "model_patch_bytes",
    "percent_messages_with_reasoning",
    "api_calls",
    "assistant_message_count",
    "trajectory_bytes",
    "trajectory_chars",
)


class ZstdJsonlWriter:
    def __init__(self, path: Path, *, level: int):
        self.path = path
        self.level = int(level)
        self.process: subprocess.Popen[bytes] | None = None
        self.stdin: Any = None

    def __enter__(self) -> "ZstdJsonlWriter":
        self.process = subprocess.Popen(
            ["zstd", "-q", f"-{self.level}", "-T0", "-f", "-o", str(self.path), "-"],
            stdin=subprocess.PIPE,
        )
        assert self.process.stdin is not None
        self.stdin = self.process.stdin
        return self

    def write_row(self, row: dict[str, Any]) -> None:
        self.stdin.write((json_dumps(row) + "\n").encode("utf-8"))

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self.process is not None
        self.stdin.close()
        return_code = self.process.wait()
        if return_code != 0 and exc_type is None:
            raise RuntimeError(f"zstd failed while writing {self.path} with exit code {return_code}")


def command_references_git_diff(command: str) -> bool:
    text = command.lower()
    return bool(
        re.search(r"\bgit\s+diff\b", text)
        or re.search(r"['\"]git['\"]\s*,\s*['\"]diff['\"]", text)
    )


def command_mentions_patch_file(command: str) -> bool:
    return bool(re.search(PATCH_TXT_PATH_PATTERN, command.lower()))


def command_has_script_patch_write(command: str) -> bool:
    text = command.lower()
    if not command_mentions_patch_file(command):
        return False
    return bool(
        re.search(r"\bopen\s*\([^)\n]*patch\.txt[^)\n]*['\"]w", text)
        or re.search(r"patch\.txt[^;\n]*\.open\s*\([^)\n]*['\"]w", text)
        or ("patch.txt" in text and "write_text" in text)
        or re.search(r"\bwritefilesync\s*\([^)\n]*patch\.txt", text)
    )


def _normalized_shell_path(path: str) -> str:
    path = path.strip().strip("'\"")
    if path.startswith("./"):
        path = path[2:]
    if path.startswith("/testbed/"):
        path = path[len("/testbed/") :]
    return path


def _same_patch_txt_path(path: str) -> bool:
    return _normalized_shell_path(path) == "patch.txt"


def _path_has_prior_git_diff_write(command: str, path: str, end: int) -> bool:
    target = re.escape(path)
    prior = command[:end]
    for segment in re.split(r"(?:&&|\n)", prior):
        if not command_references_git_diff(segment):
            continue
        if re.search(rf"(?:>\s*|\|\s*tee\s+(-a\s+)?){target}(?:\s|$|[;&|])", segment):
            return True
    return False


def command_has_untrusted_patch_assembly(command: str) -> bool:
    text = command.lower()
    copy_like = re.finditer(
        rf"(^|[;&|\n]\s*)(cat|cp|mv)\b(?P<body>[^;&|\n]*)\s+(?:>\s*)?{PATCH_TXT_PATH_PATTERN}",
        text,
    )
    for match in copy_like:
        body = match.group("body")
        source_paths = [
            path
            for path in re.findall(PATCH_LIKE_PATH_PATTERN, body)
            if not _same_patch_txt_path(path)
        ]
        if not source_paths and re.search(r"\b/dev/null\b", body):
            return True
        if not source_paths:
            continue
        for path in source_paths:
            if not _path_has_prior_git_diff_write(text, path, match.start()):
                return True
    return False


def command_has_untrusted_patch_append(command: str) -> bool:
    text = command.lower()
    append_segments = re.finditer(
        rf"(^|[;&\n]\s*)(?P<body>[^;&\n]*?)"
        rf"(?:>>\s*{PATCH_TXT_PATH_PATTERN}|\|\s*tee\s+-a\s+{PATCH_TXT_PATH_PATTERN})",
        text,
    )
    for match in append_segments:
        if not command_references_git_diff(match.group("body")):
            return True
    return False


def command_has_untrusted_patch_redirect(command: str) -> bool:
    text = command.lower()
    redirect_segments = re.finditer(
        rf"(^|[;&|\n]\s*)(cat|tee|echo|printf)\b(?P<body>[^;&|\n]*?)"
        rf"(?:>\s*{PATCH_TXT_PATH_PATTERN}|\|\s*tee\s+(-a\s+)?{PATCH_TXT_PATH_PATTERN}"
        rf"|\btee\s+(-a\s+)?{PATCH_TXT_PATH_PATTERN})",
        text,
    )
    for match in redirect_segments:
        source_paths = [
            path
            for path in re.findall(PATCH_LIKE_PATH_PATTERN, match.group("body"))
            if not _same_patch_txt_path(path)
        ]
        if source_paths and all(
            _path_has_prior_git_diff_write(text, path, match.start())
            for path in source_paths
        ):
            continue
        if not command_references_git_diff(match.group("body")):
            return True
    return False


def assistant_tool_command(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls or not isinstance(calls[0], dict):
        return ""
    function = calls[0].get("function", {})
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    if not isinstance(args, dict):
        return ""
    return str(args.get("command") or args.get("cmd") or "")


def assistant_reasoning_from_content(message: dict[str, Any]) -> str:
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    content = text_from_content(message.get("content"))
    for open_tag, close_tag in (("<think>", "</think>"), ("<thought>", "</thought>")):
        start = content.find(open_tag)
        end = content.find(close_tag, start + len(open_tag))
        if start != -1 and end != -1:
            return content[start + len(open_tag) : end].strip()
    return ""


def drop_assistant_content_preserving_reasoning(message: dict[str, Any]) -> None:
    reasoning = assistant_reasoning_from_content(message)
    if reasoning:
        message["reasoning_content"] = reasoning
    message["content"] = ""


def assistant_has_manual_patch_target(message: dict[str, Any]) -> bool:
    command = assistant_tool_command(message)
    text = command.lower()
    if not command_mentions_patch_file(command):
        return False
    if text.strip() == MINI_SWE_SUBMIT_COMMAND.lower():
        return False
    if re.search(PATCH_TXT_SHELL_CREATE_PATTERN, text):
        return True
    if command_has_untrusted_patch_append(command):
        return True
    if "diff -u /dev/null" in text:
        return True
    if command_has_untrusted_patch_assembly(command):
        return True
    if command_has_untrusted_patch_redirect(command):
        return True
    if command_has_script_patch_write(command) and not command_references_git_diff(command):
        return True
    return False


def is_submit_command(command: str) -> bool:
    return "complete_task_and_submit_final_output" in command.lower()


def command_prepares_patch_for_submit(command: str) -> bool:
    text = command.lower()
    if not command_mentions_patch_file(command) or is_submit_command(text):
        return False
    if command_references_git_diff(command) and re.search(rf"\|\s*tee\s+(-a\s+)?{PATCH_TXT_PATH_PATTERN}", text):
        return True
    if re.search(
        rf"(^|[;&|\n]\s*)(cat|grep|sed|head|tail)\b[^;&]*{PATCH_TXT_PATH_PATTERN}",
        text,
        flags=re.DOTALL,
    ):
        return True
    return False


def command_writes_patch_file(command: str) -> bool:
    text = command.lower()
    if not command_mentions_patch_file(command) or is_submit_command(text):
        return False
    return bool(re.search(PATCH_TXT_WRITE_PATTERN, text) or command_has_script_patch_write(command))


def text_has_unified_diff_header(text: str) -> bool:
    return bool(
        "diff --git" in text
        and re.search(r"(?m)^--- (?:a/|/dev/null)", text)
        and re.search(r"(?m)^\+\+\+ (?:b/|/dev/null)", text)
    )


def observation_has_visible_patch_output(observation: str) -> bool:
    matches = re.findall(r"<output>\n?(.*?)</output>", observation, flags=re.DOTALL)
    if matches:
        return any(text_has_unified_diff_header(match) for match in matches)
    return text_has_unified_diff_header(observation)


def source_value(example: dict[str, Any], key: str) -> Any:
    if key in example:
        return example[key]
    outcome = example.get("source_outcome")
    if isinstance(outcome, dict) and key in outcome:
        return outcome[key]
    metadata = example.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    return None


def example_passed(example: dict[str, Any]) -> bool:
    for key in ("passed", "pass", "resolved"):
        value = source_value(example, key)
        if value is not None:
            return bool(value)
    return False


def apply_v54_assistant_loss_policy(example: dict[str, Any]) -> dict[str, Any]:
    """Apply the v54-era policy from the training log to materialized messages."""
    previous_assistant_command = ""
    previous_assistant_observations: list[str] = []
    visible_patch_since_write = False
    patch_file_tainted = False
    seen_submit_command = False
    seen_manual_patch_target = False
    nonpassing_row = not example_passed(example)

    for message in example.get("messages", []):
        if message.get("role") != "assistant":
            if previous_assistant_command:
                previous_assistant_observations.append(text_from_content(message.get("content")))
            continue

        command = assistant_tool_command(message)
        has_tool_calls = assistant_has_valid_tool_calls(message)
        if previous_assistant_command and observation_has_visible_patch_output("\n".join(previous_assistant_observations)):
            visible_patch_since_write = True

        if has_tool_calls:
            drop_assistant_content_preserving_reasoning(message)

        has_manual_patch_target = assistant_has_manual_patch_target(message)
        is_submit = is_submit_command(command)
        if seen_submit_command:
            message["loss"] = False
        if seen_manual_patch_target:
            message["loss"] = False
        if has_manual_patch_target:
            message["loss"] = False
        if (
            is_submit
            and (
                not command_prepares_patch_for_submit(previous_assistant_command)
                or not visible_patch_since_write
                or patch_file_tainted
            )
        ):
            message["loss"] = False
        if nonpassing_row and is_submit:
            message["loss"] = False
        if not assistant_has_reasoning(message):
            message["loss"] = False
        if not has_tool_calls:
            message["loss"] = False

        if is_submit:
            seen_submit_command = True
        if has_manual_patch_target:
            seen_manual_patch_target = True
        if command:
            if command_writes_patch_file(command):
                visible_patch_since_write = False
                patch_file_tainted = bool(has_manual_patch_target)
            previous_assistant_command = command
            previous_assistant_observations = []
    return example


def build_output_row(
    example: dict[str, Any],
    *,
    source_dataset_id: str,
    source_training_view: str,
    training_index: int,
    training_shard: str,
    training_row_index: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "messages": example["messages"],
        "source_dataset_id": source_dataset_id,
        "source_training_view": source_training_view,
        "training_index": training_index,
        "training_shard": training_shard,
        "training_row_index": training_row_index,
        "passed": bool(source_value(example, "passed")),
    }
    tools = example.get("tools")
    if tools not in (None, "", [], {}):
        out["tools"] = tools
    for key in ("reward", "source", "source_note"):
        value = source_value(example, key)
        if value not in (None, ""):
            out[key] = value
    for key in SOURCE_MAPPING_KEYS:
        value = source_value(example, key)
        if value not in (None, "", [], {}):
            out["source_uuid" if key == "uuid" else key] = value
    return out


def iter_input_files(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    files = sorted(input_root.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no .jsonl shards found under {input_root}")
    return files


def update_stats(stats: dict[str, Any], example: dict[str, Any]) -> None:
    messages = example.get("messages") or []
    stats["messages"] += len(messages)
    for message in messages:
        role = message.get("role")
        stats["roles"][str(role)] += 1
        if role != "assistant":
            continue
        stats["assistant_turns"] += 1
        action = assistant_turn_action(message)
        stats["assistant_actions"][action] += 1
        if assistant_has_reasoning(message):
            stats["assistant_turns_with_reasoning"] += 1
        if assistant_has_valid_tool_calls(message):
            stats["assistant_turns_with_valid_tool_calls"] += 1
        if message.get("loss") is False:
            stats["assistant_turns_masked"] += 1
        else:
            stats["assistant_turns_trainable"] += 1


def write_dataset_info(output_root: Path) -> None:
    info = {
        "configs": [
            {
                "config_name": "default",
                "data_files": [{"split": "train", "path": "data/*.jsonl.zst"}],
            }
        ]
    }
    (output_root / "dataset_info.json").write_text(json_dumps(info) + "\n", encoding="utf-8")
    (output_root / ".gitattributes").write_text("*.zst filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")


def write_readme(output_root: Path, manifest: dict[str, Any], repo_id: str) -> None:
    readme = f"""---
license: other
pretty_name: Qwen3 4B Thinking SFT v54 Processed Training View
task_categories:
- text-generation
language:
- en
tags:
- code
- software-engineering
- agent-traces
- swe-bench
- qwen3
size_categories:
- 1K<n<10K
---

# Qwen3 4B Thinking SFT v54 Processed Training View

This dataset is the processed and filtered training view used by the v54
Qwen3-4B-Thinking SFT recipe. It starts from
`{manifest["source_dataset_id"]}` and uses the strict-passed raw2030 mini-swe
aligned view.

Rows are compressed JSONL.zst files under `data/`. Each row contains a
top-level `messages` column, optional `tools`, and scalar source mapping fields
such as `source_uuid`, `task_id`, `source_shard`, and `source_row_index` when
available. Assistant turns that v54 did not train on have `loss: false`.

Important counts:

- Rows: {manifest["rows_written"]:,}
- Input training shards: {manifest["input_files"]:,}
- Assistant turns: {manifest["assistant_turns"]:,}
- Trainable assistant turns: {manifest["assistant_turns_trainable"]:,}
- Masked assistant turns: {manifest["assistant_turns_masked"]:,}

Default repo id: `{repo_id}`.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def upload_folder(output_root: Path, repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required for upload") from exc
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_root),
        commit_message="Upload processed qwen3-4b v54 SFT training dataset",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    data_dir = args.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "source_dataset_id": args.source_dataset_id,
        "source_training_view": str(args.input_root),
        "chat_template_path": args.chat_template_path,
        "rows_seen": 0,
        "rows_written": 0,
        "rows_dropped_normalization": 0,
        "input_files": 0,
        "output_files": 0,
        "messages": 0,
        "assistant_turns": 0,
        "assistant_turns_with_reasoning": 0,
        "assistant_turns_with_valid_tool_calls": 0,
        "assistant_turns_trainable": 0,
        "assistant_turns_masked": 0,
        "roles": Counter(),
        "assistant_actions": Counter(),
        "policy": {
            "require_assistant_reasoning_for_loss": True,
            "require_assistant_tool_calls_for_loss": True,
            "drop_assistant_content_for_tool_calls": True,
            "assistant_loss_target": "assistant",
            "reject_manual_patch_targets": True,
            "reject_unverified_submit_targets": True,
            "reject_nonpassing_submit_targets": True,
        },
    }

    training_index = 0
    for input_file in iter_input_files(args.input_root):
        stats["input_files"] += 1
        output_file = data_dir / f"{input_file.stem}.jsonl.zst"
        stats["output_files"] += 1
        with ZstdJsonlWriter(output_file, level=args.zstd_level) as writer:
            for training_row_index, raw_row in enumerate(iter_jsonl_rows(input_file)):
                stats["rows_seen"] += 1
                example = normalize_row(raw_row)
                if example is None:
                    stats["rows_dropped_normalization"] += 1
                    continue
                example = apply_v54_assistant_loss_policy(example)
                row = build_output_row(
                    example,
                    source_dataset_id=args.source_dataset_id,
                    source_training_view=str(args.input_root),
                    training_index=training_index,
                    training_shard=input_file.name,
                    training_row_index=training_row_index,
                )
                update_stats(stats, row)
                writer.write_row(row)
                stats["rows_written"] += 1
                training_index += 1

    stats["roles"] = dict(stats["roles"])
    stats["assistant_actions"] = dict(stats["assistant_actions"])
    (args.output_root / "manifest.json").write_text(json_dumps(stats) + "\n", encoding="utf-8")
    write_dataset_info(args.output_root)
    write_readme(args.output_root, stats, args.repo_id)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-dataset-id", default=DEFAULT_SOURCE_DATASET_ID)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--chat-template-path", default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--zstd-level", type=int, default=19)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("zstd") is None:
        raise SystemExit("zstd CLI is required to write compressed JSONL.zst shards")
    stats = build(args)
    print(
        "materialized processed dataset: "
        f"rows={stats['rows_written']} files={stats['output_files']} "
        f"masked_assistant_turns={stats['assistant_turns_masked']} "
        f"output={args.output_root}"
    )
    if args.upload:
        upload_folder(args.output_root, args.repo_id, private=args.private)
        print(f"uploaded dataset to hf://datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
