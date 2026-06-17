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
    assistant_has_risky_source_edit_target,
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


def test_empty_think_remainder_is_tool_call_reasoning() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Inspect the repo."},
            {
                "role": "assistant",
                "content": "<think>\n\n</think>\nI should inspect the repository first.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls"}}}],
            },
        ]
    }

    normalized = normalize_row(row)
    assert normalized is not None
    assert assistant_has_reasoning(normalized["messages"][1])

    filtered = apply_assistant_loss_policy(
        normalized,
        require_assistant_reasoning_for_loss=True,
        require_assistant_tool_calls_for_loss=True,
        drop_assistant_content_for_tool_calls=True,
    )

    assistant = filtered["messages"][1]
    assert filtered.get("drop") is not True
    assert assistant["reasoning_content"] == "I should inspect the repository first."
    assert assistant["content"] == ""


def test_empty_think_remainder_without_tool_call_is_not_reasoning() -> None:
    assert not assistant_has_reasoning(
        {"role": "assistant", "content": "<think>\n\n</think>\nThe final answer is ready."}
    )


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


def test_loss_policy_masks_no_reasoning_tool_call_turn_only() -> None:
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
    assert filtered["messages"][3].get("loss") is not False
    assert filtered.get("drop") is not True


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
    assert filtered.get("drop") is not True


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
    assert filtered.get("drop") is not True


def test_manual_patch_detector_allows_same_command_git_diff_assembly() -> None:
    message = {
        "role": "assistant",
        "reasoning": "I should assemble the final patch from real source diffs.",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": {
                        "command": (
                            "cd /testbed && "
                            "git diff -- src/foo.py > /tmp/patch_part1.txt && "
                            "git diff --no-index /dev/null src/new.py > /tmp/patch_part2.txt 2>/dev/null; "
                            "cat /tmp/patch_part1.txt /tmp/patch_part2.txt > patch.txt"
                        )
                    },
                }
            }
        ],
    }

    assert not assistant_has_manual_patch_target(message)


def test_manual_patch_detector_allows_sed_adjusted_git_diff_fragment() -> None:
    message = {
        "role": "assistant",
        "reasoning": "I should assemble the final patch from adjusted git diff output.",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": {
                        "command": (
                            "cd /testbed && "
                            "git diff src/foo.py > /tmp/patch1.txt && "
                            "git diff --no-index /dev/null src/new.py | "
                            "sed 's|/dev/null|a/src/new.py|' > /tmp/patch2.txt && "
                            "cat /tmp/patch1.txt /tmp/patch2.txt > patch.txt"
                        )
                    },
                }
            }
        ],
    }

    assert not assistant_has_manual_patch_target(message)


def test_manual_patch_detector_allows_git_diff_append_to_patch_txt() -> None:
    message = {
        "role": "assistant",
        "reasoning": "I should include the new file diff in the final patch.",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": {
                        "command": (
                            "cd /testbed && "
                            "git diff -- src/foo.py > patch.txt && "
                            "git diff --no-index /dev/null src/new.py >> patch.txt 2>/dev/null || true"
                        )
                    },
                }
            }
        ],
    }

    assert not assistant_has_manual_patch_target(message)


def test_manual_patch_detector_allows_git_diff_write_then_inspection() -> None:
    message = {
        "role": "assistant",
        "reasoning": "I should write the patch and verify it contains a diff.",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": {
                        "command": (
                            "cd /testbed && "
                            "git add src/new.py && "
                            "git diff --cached > patch.txt && "
                            "cat patch.txt | grep \"diff --git\""
                        )
                    },
                }
            }
        ],
    }

    assert not assistant_has_manual_patch_target(message)


def test_manual_patch_detector_rejects_imported_temp_patch_fragment() -> None:
    message = {
        "role": "assistant",
        "reasoning": "I should assemble the final patch.",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": {
                        "command": (
                            "cd /testbed && "
                            "git diff -- src/foo.py > /tmp/modified.patch && "
                            "cat /tmp/accept_new.patch /tmp/modified.patch > patch.txt"
                        )
                    },
                }
            }
        ],
    }

    assert assistant_has_manual_patch_target(message)


def test_manual_patch_detector_rejects_manual_redirect_even_with_later_git_diff() -> None:
    message = {
        "role": "assistant",
        "reasoning": "I should create a manual patch then append a source diff.",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": {
                        "command": (
                            "cd /testbed && "
                            "printf 'diff --git a/new.py b/new.py\\n' > patch.txt && "
                            "git diff -- src/foo.py >> patch.txt"
                        )
                    },
                }
            }
        ],
    }

    assert assistant_has_manual_patch_target(message)


def test_manual_patch_detector_rejects_manual_patch_append() -> None:
    message = {
        "role": "assistant",
        "reasoning": "I should append a handmade diff fragment.",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": {
                        "command": (
                            "cd /testbed && "
                            "git diff -- src/foo.py > patch.txt && "
                            "cat /tmp/accept_new.patch >> patch.txt"
                        )
                    },
                }
            }
        ],
    }

    assert assistant_has_manual_patch_target(message)


def test_manual_patch_detector_rejects_empty_patch_creation() -> None:
    for command in (
        "cd /testbed && touch patch.txt",
        "cd /testbed && truncate -s 0 patch.txt",
        "cd /testbed && : > patch.txt",
        "cd /testbed && cp /dev/null patch.txt",
    ):
        message = {
            "role": "assistant",
            "reasoning": "I should create an empty patch.",
            "tool_calls": [{"function": {"name": "bash", "arguments": {"command": command}}}],
        }
        assert assistant_has_manual_patch_target(message), command


def test_manual_patch_context_masks_later_targets_without_dropping_example() -> None:
    example = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "reasoning": "I should not hand-write the final patch.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {
                                "command": (
                                    "cat > patch.txt <<'PATCH'\n"
                                    "diff --git a/foo.py b/foo.py\n"
                                    "--- a/foo.py\n"
                                    "+++ b/foo.py\n"
                                    "@@ -1 +1 @@\n"
                                    "-bad\n"
                                    "+good\n"
                                    "PATCH"
                                )
                            },
                        }
                    }
                ],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output></output>"},
            {
                "role": "assistant",
                "reasoning": "I should edit the source file instead.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "sed -i 's/bad/good/' foo.py"}}}],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output></output>"},
            {
                "role": "assistant",
                "reasoning": "Now I can inspect the real source diff.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "git diff -- foo.py"}}}],
            },
        ]
    }

    filtered = apply_assistant_loss_policy(
        example,
        require_assistant_reasoning_for_loss=True,
        require_assistant_tool_calls_for_loss=True,
        reject_manual_patch_targets=True,
    )

    assert filtered["messages"][1]["loss"] is False
    assert filtered["messages"][3]["loss"] is False
    assert filtered["messages"][5]["loss"] is False
    assert filtered.get("drop") is not True


def test_unverified_submit_masks_only_submit_turn() -> None:
    example = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "reasoning": "I should inspect the repository.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls"}}}],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output>foo.py</output>"},
            {
                "role": "assistant",
                "reasoning": "I should submit after inspection.",
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
        reject_unverified_submit_targets=True,
    )

    assert filtered["messages"][1].get("loss") is not False
    assert filtered["messages"][3]["loss"] is False
    assert filtered.get("drop") is not True


def test_unverified_submit_requires_visible_unified_diff() -> None:
    example = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "reasoning": "I should write the current source diff to patch.txt.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "git diff -- foo.py > patch.txt"}}}],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output></output>"},
            {
                "role": "assistant",
                "reasoning": "I should inspect the patch before submitting.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "cat patch.txt"}}}],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output>Only in src: stale.py</output>"},
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
        reject_unverified_submit_targets=True,
    )

    assert filtered["messages"][1].get("loss") is not False
    assert filtered["messages"][3].get("loss") is not False
    assert filtered["messages"][5]["loss"] is False
    assert filtered.get("drop") is not True


def test_unverified_submit_allows_visible_unified_diff() -> None:
    patch = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-bad\n"
        "+good\n"
    )
    example = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "reasoning": "I should write the current source diff to patch.txt.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "git diff -- foo.py > patch.txt"}}}],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output></output>"},
            {
                "role": "assistant",
                "reasoning": "I should inspect the patch before submitting.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "cat patch.txt"}}}],
            },
            {"role": "tool", "content": f"<returncode>0</returncode><output>{patch}</output>"},
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
        reject_unverified_submit_targets=True,
    )

    assert filtered["messages"][1].get("loss") is not False
    assert filtered["messages"][3].get("loss") is not False
    assert filtered["messages"][5].get("loss") is not False
    assert filtered.get("drop") is not True


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


def test_risky_source_edit_detector_flags_linewise_source_construction() -> None:
    risky_append = {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": {"command": "echo '  BAD_ENUM,' >> gson/src/main/java/Foo.java"},
                }
            }
        ],
    }
    risky_overwrite = {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": {"command": "printf '%s\\n' 'export const x = 1' > src/index.ts"},
                }
            }
        ],
    }
    safe_sed = {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": {"command": "sed -i 's/old/new/' src/index.ts"},
                }
            }
        ],
    }
    safe_patch = {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": {"command": "git diff -- src/index.ts > patch.txt"},
                }
            }
        ],
    }

    assert assistant_has_risky_source_edit_target(risky_append)
    assert assistant_has_risky_source_edit_target(risky_overwrite)
    assert not assistant_has_risky_source_edit_target(safe_sed)
    assert not assistant_has_risky_source_edit_target(safe_patch)


def test_loss_policy_masks_risky_source_edit_target_only() -> None:
    example = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "reasoning": "I should inspect the file.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "sed -n '1,80p' src/index.ts"}}}],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output>old</output>"},
            {
                "role": "assistant",
                "reasoning": "I should append a replacement line.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "echo 'export const x = 1' >> src/index.ts"}}}],
            },
            {"role": "tool", "content": "<returncode>0</returncode><output></output>"},
            {
                "role": "assistant",
                "reasoning": "I should inspect the diff.",
                "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "git diff -- src/index.ts"}}}],
            },
        ]
    }

    filtered = apply_assistant_loss_policy(
        example,
        require_assistant_reasoning_for_loss=True,
        require_assistant_tool_calls_for_loss=True,
        mask_risky_source_edit_targets=True,
    )

    assert filtered["messages"][1].get("loss") is not False
    assert filtered["messages"][3]["loss"] is False
    assert filtered["messages"][5].get("loss") is not False
