from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "sft" / "qwen3-sft" / "src"
sys.path.insert(0, str(SRC_DIR))

from qwen_agentic_sft.data import (  # noqa: E402
    assistant_has_reasoning,
    assistant_has_valid_tool_calls,
    normalize_row,
)
from qwen_agentic_sft.online_packed_dataset import apply_assistant_loss_policy  # noqa: E402


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
