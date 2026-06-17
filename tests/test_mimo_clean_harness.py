import csv
import json
from pathlib import Path

from datagen.swerebench_v2 import audit_export_clean_run
from datagen.swerebench_v2 import audit_export_clean_trajectory as exporter
from datagen.swerebench_v2 import build_mimo_easy_assignments
from datagen.swerebench_v2.pyxis_miniswe_agent_driver import reminder_for_output


def test_reminder_flags_empty_plain_diff_and_empty_patch_read():
    output = {"returncode": 0, "output": ""}

    assert "no patch/source diff" in reminder_for_output("git diff -- src/app.py", output)
    assert "patch.txt is empty" in reminder_for_output("cat patch.txt", output)
    assert reminder_for_output("git diff -- src/app.py > patch.txt", output) == ""


def test_reminder_flags_weak_patch_check():
    output = {"returncode": 0, "output": "--- a/src/app.py\n+++ b/src/app.py\n"}

    assert "only checked part" in reminder_for_output("head -5 patch.txt", output)
    assert reminder_for_output("cat patch.txt", output) == ""


def test_build_mimo_easy_assignments_prefers_small_easy_changes(tmp_path: Path):
    tasks_csv = tmp_path / "tasks.csv"
    output_csv = tmp_path / "assignments.csv"
    fields = [
        "instance_id",
        "repo",
        "language",
        "difficulty",
        "confidence",
        "num_modified_files",
        "num_modified_lines",
        "fail_to_pass_count",
        "pass_to_pass_count",
    ]
    rows = [
        {
            "instance_id": "repo__large-1",
            "repo": "repo/large",
            "language": "python",
            "difficulty": "easy",
            "confidence": "0.99",
            "num_modified_files": "3",
            "num_modified_lines": "40",
            "fail_to_pass_count": "2",
            "pass_to_pass_count": "5",
        },
        {
            "instance_id": "repo__small-1",
            "repo": "repo/small",
            "language": "go",
            "difficulty": "easy",
            "confidence": "0.95",
            "num_modified_files": "1",
            "num_modified_lines": "4",
            "fail_to_pass_count": "1",
            "pass_to_pass_count": "2",
        },
        {
            "instance_id": "repo__medium-1",
            "repo": "repo/medium",
            "language": "rust",
            "difficulty": "medium",
            "confidence": "1.0",
            "num_modified_files": "1",
            "num_modified_lines": "1",
            "fail_to_pass_count": "1",
            "pass_to_pass_count": "1",
        },
    ]
    with tasks_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    args = [
        "--tasks-csv",
        str(tasks_csv),
        "--output-csv",
        str(output_csv),
        "--limit",
        "1",
    ]
    old_parse_args = build_mimo_easy_assignments.parse_args
    build_mimo_easy_assignments.parse_args = lambda: old_parse_args(args)
    try:
        build_mimo_easy_assignments.main()
    finally:
        build_mimo_easy_assignments.parse_args = old_parse_args

    with output_csv.open(newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    assert selected[0]["instance_id"] == "repo__small-1"
    assert selected[0]["assigned_model"] == "xiaomi/mimo-v2.5"
    assert selected[0]["difficulty"] == "easy"


def assistant(command: str) -> dict:
    return {
        "role": "assistant",
        "content": "THOUGHT: next",
        "extra": {"actions": [{"command": command}]},
    }


def tool(content: str) -> dict:
    return {"role": "tool", "content": content}


def valid_patch() -> str:
    return (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )


def test_audit_rejects_manual_patch_construction(tmp_path: Path):
    patch = valid_patch()
    trajectory = {
        "messages": [
            {"role": "system", "content": "raw"},
            {"role": "user", "content": "Please solve this issue: fix it"},
            assistant("cat > /tmp/app_patch.txt <<'EOF'\ndiff --git a/src/app.py b/src/app.py\nEOF"),
            tool(""),
            assistant("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
        ]
    }

    audit = exporter.audit({"reward": 1}, trajectory, patch, tmp_path / "model.patch")

    assert audit["accepted"] is False
    assert "manual_patch_construction" in audit["reject_reasons"]


def test_audit_rejects_patch_fragment_assembly(tmp_path: Path):
    patch = valid_patch()
    trajectory = {
        "messages": [
            {"role": "system", "content": "raw"},
            {"role": "user", "content": "Please solve this issue: fix it"},
            assistant("cat /tmp/accept_new.patch /tmp/modified.patch > patch.txt"),
            tool(""),
            assistant("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
        ]
    }

    audit = exporter.audit({"reward": 1}, trajectory, patch, tmp_path / "model.patch")

    assert audit["accepted"] is False
    assert "patch_artifact_assembly" in audit["reject_reasons"]


def test_audit_allows_source_edit_script_named_patch(tmp_path: Path):
    patch = valid_patch()
    trajectory = {
        "messages": [
            {"role": "system", "content": "raw"},
            {"role": "user", "content": "Please solve this issue: fix it"},
            assistant("cat > /tmp/checkbox_interface_patch.js <<'EOF'\nconsole.log('edit source')\nEOF\nnode /tmp/checkbox_interface_patch.js"),
            tool(""),
            assistant("git diff -- src/app.py > patch.txt"),
            tool(valid_patch()),
            assistant("cat patch.txt"),
            tool(valid_patch()),
            assistant("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
        ]
    }

    audit = exporter.audit({"reward": 1}, trajectory, patch, tmp_path / "model.patch")

    assert audit["accepted"] is True
    assert audit["git_diff_patch_creation_before_submit"] is True
    assert audit["full_patch_read_before_submit"] is True


def test_audit_rejects_empty_patch_creation(tmp_path: Path):
    patch = valid_patch()
    trajectory = {
        "messages": [
            {"role": "system", "content": "raw"},
            {"role": "user", "content": "Please solve this issue: fix it"},
            assistant(": > patch.txt"),
            tool(""),
            assistant("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
        ]
    }

    audit = exporter.audit({"reward": 1}, trajectory, patch, tmp_path / "model.patch")

    assert audit["accepted"] is False
    assert "patch_artifact_assembly" in audit["reject_reasons"]


def test_export_strips_reminders_and_replaces_prompt(tmp_path: Path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Fix the bug.", encoding="utf-8")
    result = {
        "reward": 1,
        "task_dir": str(task_dir),
        "result_path": str(tmp_path / "result.json"),
    }
    trajectory = {
        "info": {"config": {"agent": {"system_template": "raw"}}},
        "messages": [
            {"role": "system", "content": "raw system"},
            {"role": "user", "content": "raw user"},
            assistant("cat patch.txt"),
            tool("<returncode>0</returncode><output>x</output><blocks_reminder>remove me</blocks_reminder>"),
            assistant("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
        ],
        "trajectory_format": "mini-swe-agent-1.1",
    }

    exported = exporter.export_trajectory(
        trajectory,
        result,
        Path("datagen/swerebench_v2/minisweagent_swebench_multilingual.yaml"),
    )

    assert exported["messages"][0]["content"].startswith("You are a helpful assistant")
    assert "Fix the bug." in exported["messages"][1]["content"]
    assert "<blocks_reminder>" not in json.dumps(exported["messages"])
    assert "remove me" not in json.dumps(exported["messages"])


def test_batch_auditor_saves_rejected_raw_trace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Fix the bug.", encoding="utf-8")
    trajectory_path = workspace / "agent" / "mini-swe-agent.trajectory.json"
    trajectory_path.parent.mkdir(parents=True)
    trajectory = {
        "messages": [
            {"role": "system", "content": "raw"},
            {"role": "user", "content": "Please solve this issue: Fix the bug."},
            assistant("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
        ]
    }
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    patch_path = workspace / "model.patch"
    patch_path.write_text("", encoding="utf-8")
    result_path = workspace / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "reward": 0,
                "instance_id": "repo__issue-1",
                "rollout_id": "r00",
                "model": "xiaomi/mimo-v2.5",
                "task_dir": str(task_dir),
                "trajectory_path": str(trajectory_path),
                "patch_path": str(patch_path),
            }
        ),
        encoding="utf-8",
    )
    result_index = tmp_path / "result_index.jsonl"
    result_index.write_text(
        json.dumps(
            {
                "instance_id": "repo__issue-1",
                "rollout_id": "r00",
                "model": "xiaomi/mimo-v2.5",
                "result_path": str(result_path),
                "trajectory_path": str(trajectory_path),
                "patch_path": str(patch_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "exported"

    old_parse_args = audit_export_clean_run.parse_args
    audit_export_clean_run.parse_args = lambda: old_parse_args(
        [
            "--result-index",
            str(result_index),
            "--output-dir",
            str(output_dir),
        ]
    )
    try:
        audit_export_clean_run.main()
    finally:
        audit_export_clean_run.parse_args = old_parse_args

    rejected = list((output_dir / "rejected").glob("*.trajectory.json"))
    audits = list((output_dir / "audits").glob("*.audit.json"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(rejected) == 1
    assert len(audits) == 1
    assert json.loads(rejected[0].read_text(encoding="utf-8")) == trajectory
    assert summary["accepted"] == 0
    assert summary["rejected"] == 1


def test_batch_auditor_attaches_manifest_metadata(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Fix the bug.", encoding="utf-8")
    trajectory_path = workspace / "agent" / "mini-swe-agent.trajectory.json"
    trajectory_path.parent.mkdir(parents=True)
    trajectory = {
        "messages": [
            {"role": "system", "content": "raw"},
            {"role": "user", "content": "Please solve this issue: Fix the bug."},
            assistant("git diff -- src/app.py > patch.txt"),
            tool(valid_patch()),
            assistant("cat patch.txt"),
            tool(valid_patch()),
            assistant("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
        ]
    }
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    patch_path = workspace / "model.patch"
    patch_path.write_text(valid_patch(), encoding="utf-8")
    result_path = workspace / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "reward": 1,
                "instance_id": "repo__issue-2",
                "rollout_id": "r00",
                "model": "xiaomi/mimo-v2.5",
                "task_dir": str(task_dir),
                "trajectory_path": str(trajectory_path),
                "patch_path": str(patch_path),
            }
        ),
        encoding="utf-8",
    )
    result_index = tmp_path / "result_index.jsonl"
    result_index.write_text(
        json.dumps(
            {
                "instance_id": "repo__issue-2",
                "rollout_id": "r00",
                "model": "xiaomi/mimo-v2.5",
                "result_path": str(result_path),
                "trajectory_path": str(trajectory_path),
                "patch_path": str(patch_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_tsv = tmp_path / "manifest.tsv"
    manifest_tsv.write_text(
        "\t".join(
            [
                "7",
                "r00",
                "repo__issue-2",
                str(task_dir),
                str(workspace),
                "docker.io/example/image:tag",
                "xiaomi/mimo-v2.5",
                "openrouter/xiaomi/mimo-v2.5",
                "OPENROUTER_API_KEY",
                "-",
                "{}",
                "easy",
                "rust",
                "original",
                "org/repo",
                "false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "exported"

    old_parse_args = audit_export_clean_run.parse_args
    audit_export_clean_run.parse_args = lambda: old_parse_args(
        [
            "--result-index",
            str(result_index),
            "--manifest-tsv",
            str(manifest_tsv),
            "--output-dir",
            str(output_dir),
        ]
    )
    try:
        audit_export_clean_run.main()
    finally:
        audit_export_clean_run.parse_args = old_parse_args

    audit_path = next((output_dir / "audits").glob("*.audit.json"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    accepted_raw = list((output_dir / "accepted_raw").glob("*.trajectory.json"))
    assert audit["accepted"] is True
    assert audit["language"] == "rust"
    assert audit["difficulty"] == "easy"
    assert audit["teacher"] == "xiaomi/mimo-v2.5"
    assert audit["outcome_category"] == "resolved"
    assert len(accepted_raw) == 1
    assert json.loads(accepted_raw[0].read_text(encoding="utf-8")) == trajectory
    assert audit["accepted_raw_trace_path"] == str(accepted_raw[0])
    assert summary["accepted_by_language"] == {"rust": 1}
    assert summary["by_outcome_category"] == {"resolved": 1}


def test_batch_auditor_records_missing_result_as_rejected(tmp_path: Path):
    result_index = tmp_path / "result_index.jsonl"
    result_index.write_text(
        json.dumps(
            {
                "instance_id": "repo__missing-1",
                "rollout_id": "r00",
                "model": "xiaomi/mimo-v2.5",
                "result_path": str(tmp_path / "missing" / "result.json"),
                "trajectory_path": "/workspace/agent/mini-swe-agent.trajectory.json",
                "patch_path": "/workspace/logs/artifacts/model.patch",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "exported"

    old_parse_args = audit_export_clean_run.parse_args
    audit_export_clean_run.parse_args = lambda: old_parse_args(
        [
            "--result-index",
            str(result_index),
            "--output-dir",
            str(output_dir),
        ]
    )
    try:
        audit_export_clean_run.main()
    finally:
        audit_export_clean_run.parse_args = old_parse_args

    audit = json.loads(next((output_dir / "audits").glob("*.audit.json")).read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert audit["accepted"] is False
    assert audit["outcome_category"] == "harness_error"
    assert "missing_result" in audit["reject_reasons"]
    assert summary["errors"] == 0
    assert summary["rejected"] == 1
