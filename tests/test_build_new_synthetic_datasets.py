from scripts import build_new_synthetic_datasets
from scripts import build_260609_reasoning_datasets


def test_remove_rollout_hints_for_sft_replaces_first_user_prompt(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    rollout = "Fix the bug.\n\nAdditional guidance:\n- Inspect cache invalidation.\n"
    sft = "Fix the bug.\n"
    (task_dir / "instruction.md").write_text(rollout, encoding="utf-8")
    (task_dir / "instruction.sft.md").write_text(sft, encoding="utf-8")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": f"Please solve this issue: {rollout}"},
        {"role": "assistant", "content": "I will inspect the cache."},
    ]

    updated, metadata = build_new_synthetic_datasets.remove_rollout_hints_for_sft(
        messages,
        tmp_path,
        {},
        {"task_dir": str(task_dir)},
        "planned",
    )

    assert metadata["sft_prompt_variant"] == "unhinted"
    assert metadata["sft_prompt_substitution"] == "replaced_first_user_prompt"
    assert "Additional guidance" not in updated[1]["content"]
    assert "Fix the bug." in updated[1]["content"]
    assert updated[2] == messages[2]


def test_remove_rollout_hints_for_sft_leaves_non_planned_prompts_unchanged(tmp_path):
    messages = [{"role": "user", "content": "Original benchmark prompt."}]

    updated, metadata = build_new_synthetic_datasets.remove_rollout_hints_for_sft(
        messages,
        tmp_path,
        {},
        {},
        "deepswe",
    )

    assert updated is messages
    assert metadata == {"sft_prompt_variant": "rollout"}


def test_260609_index_exposes_sft_prompt_substitution_metadata():
    row = {
        "uuid": "u1",
        "task_id": "repo__task-1",
        "teacher": "teacher",
        "reward": 1,
        "passed": True,
        "percent_messages_with_reasoning": 1.0,
        "metadata": {
            "assistant_message_count": 2,
            "assistant_messages_with_reasoning": 2,
            "assistant_messages_without_reasoning": 0,
            "instruction_style": "planned",
            "prompt": "Please solve this issue: Fix the bug.\n",
            "sft_prompt_variant": "unhinted",
            "sft_prompt_substitution": "replaced_first_user_prompt",
            "rollout_instruction_sha256": "rollout-sha",
            "sft_instruction_sha256": "sft-sha",
        },
    }

    index = build_260609_reasoning_datasets.record_to_index(row, line_number=3)

    assert index["sft_prompt_variant"] == "unhinted"
    assert index["sft_prompt_substitution"] == "replaced_first_user_prompt"
    assert index["rollout_instruction_sha256"] == "rollout-sha"
    assert index["sft_instruction_sha256"] == "sft-sha"
