from __future__ import annotations

import json
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "sft" / "qwen3-sft" / "src"
sys.path.insert(0, str(SRC_DIR))

from qwen_agentic_sft.data import (  # noqa: E402
    assistant_has_reasoning,
    assistant_has_valid_tool_calls,
    normalize_row,
)
from qwen_agentic_sft.online_packed_dataset import (  # noqa: E402
    IGNORE_INDEX,
    apply_assistant_loss_policy,
    assistant_has_manual_patch_target,
    tokenize_chat_example,
)


class CharTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": [ord(char) for char in text]}

    def apply_chat_template(
        self,
        *,
        conversation,
        tokenize: bool,
        add_generation_prompt: bool,
        tools=None,
        chat_template=None,
    ) -> str:
        assert not tokenize
        rendered = ""
        for message in conversation:
            role = message["role"]
            content = str(message.get("content") or "")
            if role == "assistant":
                reasoning = ""
                if "</think>" in content:
                    reasoning = content.split("</think>")[0].split("<think>", 1)[-1].strip()
                    content = content.split("</think>", 1)[1].lstrip("\n")
                rendered += f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content.lstrip()}"
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function", tool_call)
                    if content:
                        rendered += "\n"
                    rendered += (
                        '<tool_call>\n{"name": "'
                        + function["name"]
                        + '", "arguments": '
                        + json.dumps(function["arguments"])
                        + "}\n</tool_call>"
                    )
                rendered += "<|im_end|>\n"
            else:
                rendered += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


def test_reasoning_only_tool_turn_is_trainable_after_normalization() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "You can use bash."},
            {"role": "user", "content": "Inspect the repo."},
            {
                "role": "assistant",
                "content": None,
                "reasoning": "I need to inspect the repository first.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "ls -la"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "<returncode>0</returncode>"},
            {"role": "exit", "content": "Submitted"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
    }

    normalized = normalize_row(row)

    assert normalized is not None
    assert [message["role"] for message in normalized["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assistant = normalized["messages"][2]
    assert assistant_has_reasoning(assistant)
    assert assistant_has_valid_tool_calls(assistant)
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"command": "ls -la"}


def test_empty_think_block_is_not_reasoning() -> None:
    assert not assistant_has_reasoning({"role": "assistant", "content": "<think>\n\n</think>"})


def test_structured_reasoning_and_tool_call_content_is_trainable() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Run the tests."},
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "text": "I should inspect the failure first."},
                    {
                        "type": "tool_call",
                        "name": "bash",
                        "arguments": {"command": "pytest -q"},
                    },
                ],
            },
        ]
    }

    normalized = normalize_row(row)
    assert normalized is not None

    assistant = normalized["messages"][1]
    assert assistant_has_reasoning(assistant)
    assert assistant_has_valid_tool_calls(assistant)

    filtered = apply_assistant_loss_policy(
        normalized,
        require_assistant_reasoning_for_loss=True,
        require_assistant_tool_calls_for_loss=True,
    )
    assert filtered["messages"][1].get("loss") is not False


def test_loss_policy_masks_assistant_turn_without_tool_call() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Summarize the repo."},
            {"role": "assistant", "reasoning": "No tool use needed.", "content": "It is a Python project."},
        ]
    }

    normalized = normalize_row(row)
    assert normalized is not None

    filtered = apply_assistant_loss_policy(
        normalized,
        require_assistant_reasoning_for_loss=True,
        require_assistant_tool_calls_for_loss=True,
    )
    assert filtered["messages"][1]["loss"] is False


def test_loss_policy_drops_no_reasoning_tool_call_context() -> None:
    example = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "sed -i 's/bad/good/' foo.py"}}}],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output></output>"},
            {
                "role": "assistant",
                "reasoning": "I should inspect the patch before submitting.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "git diff -- foo.py"}}}],
            },
        ]
    }

    filtered = apply_assistant_loss_policy(
        example,
        require_assistant_reasoning_for_loss=True,
        require_assistant_tool_calls_for_loss=True,
    )

    assert filtered["messages"][1]["loss"] is False
    assert filtered.get("drop") is True


def test_loss_policy_can_drop_visible_content_on_tool_call_turn() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Inspect the repo."},
            {
                "role": "assistant",
                "reasoning": "I should list files.",
                "content": "Let me inspect the repository first.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls"}}}],
            },
        ]
    }

    normalized = normalize_row(row)
    assert normalized is not None

    filtered = apply_assistant_loss_policy(
        normalized,
        require_assistant_reasoning_for_loss=True,
        require_assistant_tool_calls_for_loss=True,
        drop_assistant_content_for_tool_calls=True,
    )

    assistant = filtered["messages"][1]
    assert assistant.get("loss") is not False
    assert "I should list files." in assistant["reasoning_content"]
    assert assistant["content"] == ""
    assert "Let me inspect" not in assistant["content"]


def test_tool_call_loss_target_masks_reasoning_and_visible_content() -> None:
    example = {
        "messages": [
            {"role": "user", "content": "Inspect the repo."},
            {
                "role": "assistant",
                "content": "<think>\nI should list files.\n</think>\nLet me inspect first.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls"}}}],
            },
        ]
    }
    filtered = apply_assistant_loss_policy(
        example,
        require_assistant_reasoning_for_loss=True,
        require_assistant_tool_calls_for_loss=True,
        drop_assistant_content_for_tool_calls=True,
    )

    input_ids, labels = next(
        tokenize_chat_example(
            filtered,
            CharTokenizer(),
            chat_template="unused",
            sequence_length=4096,
            assistant_loss_target="tool_calls",
        )
    )
    labelled = "".join(chr(token) for token, label in zip(input_ids, labels) if label != IGNORE_INDEX)

    assert labelled.startswith("<tool_call>")
    assert "I should list files" not in labelled
    assert "Let me inspect" not in labelled
    assert '"command": "ls"' in labelled
    assert labelled.endswith("<|im_end|>\n")


def test_loss_policy_taints_absolute_patch_txt_manual_write() -> None:
    example = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "reasoning": "I should create the final patch.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {
                                "command": (
                                    "cat > /testbed/patch.txt <<'EOF'\n"
                                    "diff --git a/foo.py b/foo.py\n"
                                    "--- a/foo.py\n"
                                    "+++ b/foo.py\n"
                                    "@@ -1 +1 @@\n"
                                    "-bad\n"
                                    "+good\n"
                                    "EOF"
                                )
                            },
                        }
                    }
                ],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output></output>"},
            {
                "role": "assistant",
                "reasoning": "I should inspect the patch before submitting.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "cat /testbed/patch.txt"}}}],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output>diff --git a/foo.py b/foo.py</output>"},
            {
                "role": "assistant",
                "reasoning": "The patch is ready.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"},
                        }
                    }
                ],
            },
        ]
    }

    filtered = apply_assistant_loss_policy(
        example,
        require_assistant_reasoning_for_loss=True,
        require_assistant_tool_calls_for_loss=True,
        reject_manual_patch_targets=True,
        reject_unverified_submit_targets=True,
    )

    assert filtered["messages"][1]["loss"] is False
    assert filtered["messages"][3]["loss"] is False
    assert filtered["messages"][5]["loss"] is False


def test_loss_policy_recognizes_cmd_tool_arguments() -> None:
    example = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "reasoning": "I should create the final patch.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {
                                "cmd": (
                                    "cat > /testbed/patch.txt <<'EOF'\n"
                                    "diff --git a/foo.py b/foo.py\n"
                                    "--- a/foo.py\n"
                                    "+++ b/foo.py\n"
                                    "@@ -1 +1 @@\n"
                                    "-bad\n"
                                    "+good\n"
                                    "EOF"
                                )
                            },
                        }
                    }
                ],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output></output>"},
            {
                "role": "assistant",
                "reasoning": "I should submit the patch.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"cmd": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"}',
                        }
                    }
                ],
            },
        ]
    }

    filtered = apply_assistant_loss_policy(
        example,
        require_assistant_reasoning_for_loss=True,
        require_assistant_tool_calls_for_loss=True,
        reject_manual_patch_targets=True,
        reject_unverified_submit_targets=True,
    )

    assert assistant_has_manual_patch_target(filtered["messages"][1])
    assert filtered["messages"][1]["loss"] is False
    assert filtered["messages"][3]["loss"] is False


def test_loss_policy_masks_turns_after_submit() -> None:
    example = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "reasoning": "I should submit the patch.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"},
                        }
                    }
                ],
            },
            {"role": "tool", "content": "<returncode>1</returncode><output>cat: patch.txt: No such file</output>"},
            {
                "role": "assistant",
                "reasoning": "I need to create the patch file now.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "git diff -- foo.py > patch.txt"}}}],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output></output>"},
            {
                "role": "assistant",
                "reasoning": "I should inspect the patch.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "cat patch.txt"}}}],
            },
        ]
    }

    filtered = apply_assistant_loss_policy(
        example,
        require_assistant_reasoning_for_loss=True,
        require_assistant_tool_calls_for_loss=True,
        reject_unverified_submit_targets=True,
    )

    assert filtered["messages"][1]["loss"] is False
    assert filtered["messages"][3]["loss"] is False
    assert filtered["messages"][5]["loss"] is False
