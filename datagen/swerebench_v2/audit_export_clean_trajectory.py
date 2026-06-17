#!/usr/bin/env python3
"""Audit datagen trajectories and export benchmark-shaped clean traces."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml
from jinja2 import StrictUndefined, Template

from eval.paths import REPO_ROOT


DEFAULT_TARGET_CONFIG = REPO_ROOT / "datagen" / "swerebench_v2" / "minisweagent_swebench_multilingual.yaml"
REMINDER_RE = re.compile(
    r"\n?\s*<(?:datagen_)?reminder>\s*.*?\s*</(?:datagen_)?reminder>\s*|"
    r"\n?\s*<blocks_reminder>\s*.*?\s*</blocks_reminder>\s*",
    re.DOTALL,
)
DIFF_FILE_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)
FORBIDDEN_PATH_RE = re.compile(
    r"(^|/)(tests?|__tests__|fixtures?|snapshots?|goldens?)(/|$)|"
    r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Cargo\.lock|poetry\.lock|"
    r"Pipfile\.lock|composer\.lock|Gemfile\.lock|go\.sum)$|"
    r"(^|/)(pyproject\.toml|setup\.cfg|setup\.py|tox\.ini|noxfile\.py)$|"
    r"(^|/)(patch|fix|model)\.(txt|patch|diff)$|"
    r"(^|/).*\.(snap|snapshot|golden|min\.js)$"
)
MANUAL_PATCH_RE = re.compile(
    r"(^|[;&|]\s*)(echo|printf)\b[\s\S]*?>+\s*(/tmp/)?[^;&|\n]*patch[^;&|\n]*|"
    r"(^|[;&|]\s*)cat\s+<<['\"]?\w+['\"]?\s*>+\s*(/tmp/)?[^;&|\n]*patch[^;&|\n]*|"
    r"(^|[;&|]\s*)cat\s+>+\s*(/tmp/)?[^;&|\n]*patch[^;&|\n]*|"
    r"(^|[;&|]\s*)tee\s+[^;&|\n]*patch[^;&|\n]*",
    re.IGNORECASE,
)
WEAK_PATCH_CHECK_RE = re.compile(r"(^|[;&|]\s*)(head|tail|grep|wc)\b[^;&|\n]*patch\.txt\b")
FULL_PATCH_READ_RE = re.compile(r"(^|[;&|]\s*)cat\s+[^;&|\n]*patch\.txt\b")
FINAL_PATCH_REDIRECT_RE = re.compile(r"(?:^|[;&|]\s*)[^;&|\n]*(?:>>?|1>)\s*(?:/testbed/)?(?:\./)?patch\.txt\b")
FINAL_PATCH_COPY_RE = re.compile(r"(^|[;&|]\s*)(cp|mv)\b[^;&|\n]*(?:/testbed/)?(?:\./)?patch\.txt\b")
EMPTY_PATCH_CREATE_RE = re.compile(
    r"(^|[;&|]\s*)(touch|truncate)\b[^;&|\n]*(?:/testbed/)?(?:\./)?patch\.txt\b|"
    r"(^|[;&|]\s*):\s*>\s*(?:/testbed/)?(?:\./)?patch\.txt\b|"
    r"(^|[;&|]\s*)cp\s+/dev/null\s+(?:/testbed/)?(?:\./)?patch\.txt\b"
)
NONEMPTY_PATCH_CHECK_RE = re.compile(
    r"test\s+-s\s+(?:/testbed/)?(?:\./)?patch\.txt\b|"
    r"\[\s+-s\s+(?:/testbed/)?(?:\./)?patch\.txt\s+\]"
)
SED_PATCH_PREVIEW_RE = re.compile(
    r"(^|[;&|]\s*)sed\s+-n\s+['\"]?1(?:,\d+)?p['\"]?\s+(?:/testbed/)?(?:\./)?patch\.txt\b"
)
PATCH_NAMED_SOURCE_SCRIPT_RE = re.compile(
    r">\s*\S*patch\.(?:js|ts|jsx|tsx|py|sh|go|rs|java|php|c|cc|cpp|h|hpp)\b"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--trajectory-json", type=Path)
    parser.add_argument("--patch-file", type=Path)
    parser.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_CONFIG)
    parser.add_argument("--output-trajectory", type=Path, required=True)
    parser.add_argument(
        "--rejected-trajectory",
        type=Path,
        help=(
            "Where to save the raw trajectory when audit rejects it. Defaults to "
            "<output-trajectory parent>/rejected/<output-trajectory name>."
        ),
    )
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument(
        "--apply-check-workdir",
        help="Optional clean git workdir where the final patch should pass git apply --check.",
    )
    parser.add_argument("--fail-on-reject", action="store_true")
    return parser.parse_args(argv)


def default_rejected_trajectory_path(output_trajectory: Path) -> Path:
    return output_trajectory.parent / "rejected" / output_trajectory.name


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def missing_trajectory_placeholder(result: dict[str, Any], trajectory_path: Path) -> dict[str, Any]:
    return {
        "trajectory_format": "missing",
        "messages": [],
        "info": {
            "missing_trajectory": True,
            "expected_trajectory_path": str(trajectory_path),
            "instance_id": result.get("instance_id"),
            "rollout_id": result.get("rollout_id"),
            "agent_exit_status": result.get("agent_exit_status"),
            "agent_exception": result.get("agent_exception"),
        },
    }


def missing_result_placeholder(result_path: Path, row: dict[str, Any] | None = None) -> dict[str, Any]:
    row = row or {}
    return {
        "missing_result": True,
        "result_path": str(result_path),
        "instance_id": row.get("instance_id"),
        "rollout_id": row.get("rollout_id"),
        "model": row.get("model"),
        "agent_exception": {
            "type": "MissingResultJson",
            "message": f"missing result.json at {result_path}",
        },
        "reward": 0,
        "api_calls": 0,
    }


def load_trajectory_or_placeholder(path: Path, result: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if path.exists():
        return load_json(path), False
    return missing_trajectory_placeholder(result, path), True


def workspace_from_result(result_json: Path) -> Path:
    return result_json.resolve().parent


def host_path(path_value: str | None, workspace: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute() and str(path).startswith("/workspace/"):
        return workspace / path.relative_to("/workspace")
    return path


def infer_paths(args: argparse.Namespace, result: dict[str, Any]) -> tuple[Path, Path]:
    workspace = workspace_from_result(args.result_json)
    trajectory = args.trajectory_json or host_path(result.get("trajectory_path"), workspace)
    patch = args.patch_file or host_path(result.get("patch_path"), workspace)
    if trajectory is None:
        trajectory = workspace / "agent" / "mini-swe-agent.trajectory.json"
    if patch is None:
        patch = workspace / "model.patch"
    return trajectory, patch


def changed_files_from_patch(patch_text: str) -> list[str]:
    files: list[str] = []
    for old, new in DIFF_FILE_RE.findall(patch_text):
        for value in (old, new):
            if value != "/dev/null" and value not in files:
                files.append(value)
    return files


def output_block_text(content: str) -> str:
    match = re.search(r"<output>\n?(.*?)\n?</output>", content, re.DOTALL)
    if match:
        return match.group(1)
    return content


def submitted_patch_text(trajectory: dict[str, Any]) -> str:
    messages = trajectory.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "exit":
            content = message.get("content")
            if isinstance(content, str) and "diff --git " in content:
                return content.strip() + "\n"
            submission = (message.get("extra") or {}).get("submission")
            if isinstance(submission, str) and "diff --git " in submission:
                return submission.strip() + "\n"

    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for command in iter_tool_commands({"messages": [message]}):
            if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
                for followup in messages[index + 1 :]:
                    if followup.get("role") == "assistant":
                        break
                    if followup.get("role") != "tool":
                        continue
                    content = followup.get("content")
                    if isinstance(content, str):
                        output = output_block_text(content)
                        if "diff --git " in output:
                            return output.strip() + "\n"
    return ""


def audit_patch_text(trajectory: dict[str, Any], workspace_patch_text: str) -> tuple[str, str]:
    submitted = submitted_patch_text(trajectory)
    if patch_has_unified_headers(submitted):
        return submitted, "submission"
    return workspace_patch_text, "workspace_diff"


def patch_has_unified_headers(patch_text: str) -> bool:
    if not patch_text.strip():
        return False
    return (
        "diff --git " in patch_text
        and re.search(r"^--- (a/|/dev/null)", patch_text, re.MULTILINE) is not None
        and re.search(r"^\+\+\+ (b/|/dev/null)", patch_text, re.MULTILINE) is not None
    )


def iter_tool_commands(trajectory: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for message in trajectory.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for action in (message.get("extra") or {}).get("actions") or []:
            command = action.get("command")
            if isinstance(command, str):
                commands.append(command)
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if function.get("name") != "bash":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            command = arguments.get("command")
            if isinstance(command, str):
                commands.append(command)
    return commands


def command_manually_writes_patch(command: str) -> bool:
    if "patch" not in command:
        return False
    if PATCH_NAMED_SOURCE_SCRIPT_RE.search(command) and not re.search(r"diff --git|--- a/|\+\+\+ b/", command):
        return False
    if "git diff" in command and not re.search(r"\b(echo|printf|cat\s+<<|cat\s+>|tee)\b", command):
        return False
    return MANUAL_PATCH_RE.search(command) is not None


def command_writes_final_patch(command: str) -> bool:
    if "patch.txt" not in command:
        return False
    return (
        FINAL_PATCH_REDIRECT_RE.search(command) is not None
        or FINAL_PATCH_COPY_RE.search(command) is not None
        or EMPTY_PATCH_CREATE_RE.search(command) is not None
    )


def command_writes_final_patch_from_git_diff(command: str) -> bool:
    if "git diff" not in command or "patch.txt" not in command:
        return False
    return FINAL_PATCH_REDIRECT_RE.search(command) is not None


def command_assembles_patch_artifact(command: str) -> bool:
    """Detect final patch writes that are not direct git diff output."""

    if not command_writes_final_patch(command):
        return False
    if EMPTY_PATCH_CREATE_RE.search(command):
        return True
    if command_writes_final_patch_from_git_diff(command) and not command_manually_writes_patch(command):
        return False
    return True


def final_submit_index(commands: list[str]) -> int | None:
    for index in range(len(commands) - 1, -1, -1):
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in commands[index]:
            return index
    return None


def has_full_patch_read_before_submit(commands: list[str], submit_index: int) -> bool:
    return any(FULL_PATCH_READ_RE.search(command) for command in commands[max(0, submit_index - 5) : submit_index])


def weak_patch_check_before_submit(commands: list[str], submit_index: int) -> bool:
    window = commands[max(0, submit_index - 3) : submit_index]
    return any(WEAK_PATCH_CHECK_RE.search(command) for command in window)


def has_nonempty_patch_check_before_submit(commands: list[str], submit_index: int) -> bool:
    return any(NONEMPTY_PATCH_CHECK_RE.search(command) for command in commands[max(0, submit_index - 5) : submit_index])


def has_sed_patch_preview_before_submit(commands: list[str], submit_index: int) -> bool:
    return any(SED_PATCH_PREVIEW_RE.search(command) for command in commands[max(0, submit_index - 5) : submit_index])


def has_git_diff_patch_creation_before_submit(commands: list[str], submit_index: int) -> bool:
    return any(command_writes_final_patch_from_git_diff(command) for command in commands[:submit_index])


def run_apply_check(patch_path: Path, workdir: str | None) -> dict[str, Any] | None:
    if not workdir:
        return None
    repo = Path(workdir)
    if not repo.exists() or not (repo / ".git").exists():
        return None
    command = ["git", "-C", str(repo), "apply", "--check", str(patch_path)]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    return {
        "command": " ".join(shlex.quote(part) for part in command),
        "returncode": result.returncode,
        "output_tail": result.stdout[-4000:],
    }


def result_mentions(result: dict[str, Any], pattern: str) -> bool:
    return re.search(pattern, json.dumps(result, sort_keys=True, default=str), re.IGNORECASE) is not None


def classify_outcome(
    result: dict[str, Any],
    reasons: list[str],
    *,
    patch_bytes: int,
    changed_files: list[str],
    trajectory_missing: bool,
) -> str:
    if not reasons:
        return "resolved"
    if trajectory_missing or result.get("missing_result"):
        return "harness_error"

    agent_exception = result.get("agent_exception") or {}
    api_calls = int(result.get("api_calls") or 0)
    if agent_exception and api_calls == 0:
        return "harness_error"
    if result_mentions(result, r"patch apply failed|git apply.*fail|apply.*patch.*fail"):
        return "patch_apply_failed"
    if "missing_submit_command" in reasons:
        return "missing_submit"
    if patch_bytes == 0 or not changed_files:
        return "empty_patch"
    if "patch_touches_forbidden_paths" in reasons:
        return "forbidden_path_patch"
    if (
        "manual_patch_construction" in reasons
        or "patch_artifact_assembly" in reasons
        or "patch_missing_real_unified_diff_headers" in reasons
        or "git_apply_check_failed" in reasons
    ):
        return "malformed_or_manual_patch"
    if "verifier_reward_not_1" in reasons:
        return "unresolved"
    return reasons[0]


def audit(
    result: dict[str, Any],
    trajectory: dict[str, Any],
    patch_text: str,
    patch_path: Path,
    *,
    apply_check_workdir: str | None = None,
    trajectory_missing: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    if result.get("missing_result"):
        reasons.append("missing_result")
    if trajectory_missing:
        reasons.append("missing_trajectory")

    reward = int(result.get("reward") or 0)
    if reward != 1:
        reasons.append("verifier_reward_not_1")

    if not patch_has_unified_headers(patch_text):
        reasons.append("patch_missing_real_unified_diff_headers")

    changed_files = changed_files_from_patch(patch_text)
    if not changed_files:
        reasons.append("patch_changed_files_empty")

    forbidden_files = [path for path in changed_files if FORBIDDEN_PATH_RE.search(path)]
    if forbidden_files:
        reasons.append("patch_touches_forbidden_paths")

    commands = iter_tool_commands(trajectory)
    manual_patch_commands = [command for command in commands if command_manually_writes_patch(command)]
    if manual_patch_commands:
        reasons.append("manual_patch_construction")
    patch_artifact_commands = [command for command in commands if command_assembles_patch_artifact(command)]
    if patch_artifact_commands:
        reasons.append("patch_artifact_assembly")

    submit_index = final_submit_index(commands)
    full_patch_read_before_submit = False
    weak_patch_check = False
    nonempty_patch_check_before_submit = False
    sed_patch_preview_before_submit = False
    git_diff_patch_creation_before_submit = False
    if submit_index is None:
        reasons.append("missing_submit_command")
    else:
        full_patch_read_before_submit = has_full_patch_read_before_submit(commands, submit_index)
        weak_patch_check = weak_patch_check_before_submit(commands, submit_index)
        nonempty_patch_check_before_submit = has_nonempty_patch_check_before_submit(commands, submit_index)
        sed_patch_preview_before_submit = has_sed_patch_preview_before_submit(commands, submit_index)
        git_diff_patch_creation_before_submit = has_git_diff_patch_creation_before_submit(commands, submit_index)
        if weak_patch_check and not full_patch_read_before_submit:
            reasons.append("weak_patch_check_before_submit")

    apply_check = run_apply_check(patch_path, apply_check_workdir)
    if apply_check is not None and apply_check["returncode"] != 0:
        reasons.append("git_apply_check_failed")

    assistant_messages = sum(1 for message in trajectory.get("messages") or [] if message.get("role") == "assistant")
    tool_messages = sum(1 for message in trajectory.get("messages") or [] if message.get("role") == "tool")
    patch_bytes = len(patch_text.encode("utf-8"))
    outcome_category = classify_outcome(
        result,
        reasons,
        patch_bytes=patch_bytes,
        changed_files=changed_files,
        trajectory_missing=trajectory_missing,
    )
    return {
        "accepted": not reasons,
        "reject_reasons": reasons,
        "outcome_category": outcome_category,
        "reward": reward,
        "patch_bytes": patch_bytes,
        "changed_file_count": len(changed_files),
        "changed_files": changed_files,
        "forbidden_files": forbidden_files,
        "manual_patch_command_count": len(manual_patch_commands),
        "manual_patch_command_samples": manual_patch_commands[:3],
        "patch_artifact_command_count": len(patch_artifact_commands),
        "patch_artifact_command_samples": patch_artifact_commands[:3],
        "full_patch_read_before_submit": full_patch_read_before_submit,
        "weak_patch_check_before_submit": weak_patch_check,
        "nonempty_patch_check_before_submit": nonempty_patch_check_before_submit,
        "sed_patch_preview_before_submit": sed_patch_preview_before_submit,
        "git_diff_patch_creation_before_submit": git_diff_patch_creation_before_submit,
        "assistant_message_count": assistant_messages,
        "tool_message_count": tool_messages,
        "apply_check": apply_check,
    }


def write_patch_artifact(path: Path, patch_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(patch_text, encoding="utf-8")


def render_target_messages(config_path: Path, task: str) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    agent = config["agent"]
    template_vars = {"task": task}
    system = Template(agent["system_template"], undefined=StrictUndefined).render(**template_vars)
    user = Template(agent["instance_template"], undefined=StrictUndefined).render(**template_vars)
    return {"role": "system", "content": system}, {"role": "user", "content": user}, config


def strip_reminder(content: Any) -> Any:
    if isinstance(content, str):
        return REMINDER_RE.sub("", content)
    return content


def task_text(result: dict[str, Any], trajectory: dict[str, Any]) -> str:
    task_dir = result.get("task_dir")
    if task_dir:
        instruction = host_path(str(Path(task_dir) / "instruction.md"), workspace_from_result(Path(result["result_path"])) if result.get("result_path") else Path("."))
        if instruction and instruction.exists():
            return instruction.read_text(encoding="utf-8", errors="replace")
    messages = trajectory.get("messages") or []
    if len(messages) > 1:
        content = messages[1].get("content") or ""
        match = re.search(r"<pr_description>\s*Consider the following PR description:\s*(.*?)\s*</pr_description>", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"Please solve this issue:\s*(.*?)(?:\n\n|$)", content, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def export_trajectory(
    trajectory: dict[str, Any],
    result: dict[str, Any],
    target_config: Path,
) -> dict[str, Any]:
    exported = copy.deepcopy(trajectory)
    messages = exported.get("messages") or []
    if len(messages) < 2:
        raise ValueError("trajectory has fewer than two messages")
    system_message, user_message, config = render_target_messages(target_config, task_text(result, trajectory))
    messages[0] = system_message
    messages[1] = user_message
    for message in messages[2:]:
        message["content"] = strip_reminder(message.get("content"))
    info = exported.setdefault("info", {})
    info["config"] = copy.deepcopy(config)
    return exported


def main() -> None:
    args = parse_args()
    result = load_json(args.result_json)
    result.setdefault("result_path", str(args.result_json))
    trajectory_path, patch_path = infer_paths(args, result)
    trajectory, trajectory_missing = load_trajectory_or_placeholder(trajectory_path, result)
    workspace_patch_text = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.exists() else ""
    patch_text, patch_source = audit_patch_text(trajectory, workspace_patch_text)

    audit_record = audit(
        result,
        trajectory,
        patch_text,
        patch_path,
        apply_check_workdir=args.apply_check_workdir,
        trajectory_missing=trajectory_missing,
    )
    audit_record.update(
        {
            "result_json": str(args.result_json),
            "trajectory_json": str(trajectory_path),
            "patch_file": str(patch_path),
            "workspace_patch_bytes": len(workspace_patch_text.encode("utf-8")),
            "audit_patch_source": patch_source,
            "output_trajectory": str(args.output_trajectory),
            "rejected_trajectory": str(
                args.rejected_trajectory or default_rejected_trajectory_path(args.output_trajectory)
            ),
            "target_config": str(args.target_config),
        }
    )

    if audit_record["accepted"]:
        exported = export_trajectory(trajectory, result, args.target_config)
        args.output_trajectory.parent.mkdir(parents=True, exist_ok=True)
        args.output_trajectory.write_text(json.dumps(exported, indent=2) + "\n", encoding="utf-8")
        saved_patch_path = args.output_trajectory.with_suffix(".patch")
        write_patch_artifact(saved_patch_path, patch_text)
        audit_record["saved_trace_path"] = str(args.output_trajectory)
        audit_record["saved_patch_path"] = str(saved_patch_path)
        audit_record["saved_trace_kind"] = "benchmark_shaped_accepted"
    else:
        rejected_path = args.rejected_trajectory or default_rejected_trajectory_path(args.output_trajectory)
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_path.write_text(json.dumps(trajectory, indent=2) + "\n", encoding="utf-8")
        saved_patch_path = rejected_path.with_suffix(".patch")
        write_patch_artifact(saved_patch_path, patch_text)
        audit_record["saved_trace_path"] = str(rejected_path)
        audit_record["saved_patch_path"] = str(saved_patch_path)
        audit_record["saved_trace_kind"] = "raw_rejected"

    audit_path = args.audit_json or args.output_trajectory.with_suffix(".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit_record, indent=2, sort_keys=True))
    if args.fail_on_reject and not audit_record["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
