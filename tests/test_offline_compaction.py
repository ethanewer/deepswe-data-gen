import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sft/qwen3-sft/src"))

from qwen_agentic_sft.offline_compaction import (  # noqa: E402
    ChatMeasurer,
    CompactionConfig,
    CompactionStats,
    compact_example,
)


class CharTokenizer:
    eos_token = "<eos>"
    pad_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, tools=None, chat_template=None):
        rendered = []
        if tools:
            rendered.append("<tools>" + str(tools) + "</tools>\n")
        for message in conversation:
            rendered.append(f"<{message['role']}>\n{message.get('content', '')}\n</{message['role']}>\n")
            for call in message.get("tool_calls") or []:
                rendered.append(f"<tool_call>{call}</tool_call>\n")
        if add_generation_prompt:
            rendered.append("<assistant>\n")
        text = "".join(rendered)
        if tokenize:
            return self(text, add_special_tokens=False)["input_ids"]
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(idx)) for idx in ids)


def make_example(turns: int = 12):
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the repository bug. " * 4},
    ]
    for idx in range(turns):
        messages.extend(
            [
                {"role": "assistant", "content": f"I will inspect file {idx}. " * 3},
                {"role": "tool", "content": f"file {idx} output " * 8},
                {"role": "assistant", "content": f"Next action {idx}. " * 3},
            ]
        )
    return {"messages": messages, "tools": [{"type": "function", "function": {"name": "bash"}}]}


def test_included_mode_adds_prompt_and_summary_and_respects_cap():
    measurer = ChatMeasurer(CharTokenizer(), chat_template="")
    config = CompactionConfig(
        max_sequence_length=1200,
        boundary_tokens=760,
        include_compaction=True,
        summary_token_budget=180,
    )
    stats = CompactionStats()

    rows = compact_example(make_example(), measurer, config, stats, source_index=7)

    assert len(rows) > 1
    assert any(row["metadata"]["compaction"] == "included" for row in rows)
    assert any("compact the conversation" in msg["content"] for row in rows for msg in row["messages"])
    assert all(measurer.count_chat(row["messages"], row.get("tools")) <= 1200 for row in rows)
    assert rows[1]["messages"][1]["role"] == "system"
    assert rows[1]["messages"][1]["content"].startswith("[synthetic-compaction-summary]")


def test_excluded_mode_starts_next_chunk_with_summary_without_training_prompt():
    measurer = ChatMeasurer(CharTokenizer(), chat_template="")
    config = CompactionConfig(
        max_sequence_length=1200,
        boundary_tokens=760,
        include_compaction=False,
        summary_token_budget=180,
    )

    rows = compact_example(make_example(), measurer, config, CompactionStats())

    assert len(rows) > 1
    assert all("compact the conversation" not in msg["content"] for row in rows for msg in row["messages"])
    assert all(row["metadata"]["compaction"] != "included" for row in rows)
    assert rows[1]["messages"][1]["role"] == "system"
    assert rows[1]["messages"][1]["content"].startswith("[synthetic-compaction-summary]")
    assert all(measurer.count_chat(row["messages"], row.get("tools")) <= 1200 for row in rows)


def test_oversized_single_message_is_truncated_at_token_boundary():
    measurer = ChatMeasurer(CharTokenizer(), chat_template="")
    example = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u" * 2000},
            {"role": "assistant", "content": "done"},
        ]
    }
    config = CompactionConfig(
        max_sequence_length=500,
        boundary_tokens=300,
        include_compaction=True,
        summary_token_budget=80,
    )
    stats = CompactionStats()

    rows = compact_example(example, measurer, config, stats)

    assert rows
    assert stats.messages_truncated >= 1
    assert all(measurer.count_chat(row["messages"], row.get("tools")) <= 500 for row in rows)
