#!/usr/bin/env python3
"""Online Qwen chat-template tokenization and THD sequence packing."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .data import (
    RAW_ROOT,
    assistant_has_reasoning,
    assistant_has_valid_tool_calls,
    discover_raw_files,
    iter_normalized_examples_from_files,
    text_from_content,
)


IGNORE_INDEX = -100
DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_CHAT_TEMPLATE = Path(
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/"
    "qwen3_thinking_acc.jinja2"
)
THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>"
ASSISTANT_LOSS_TARGETS = ("assistant", "tool_calls")


READ_COMMAND_RE = re.compile(
    r"\b(cat|cd|find|grep|head|ls|pwd|rg|sed\s+-n|stat|tail|tree|wc|git\s+(status|diff|show|log|grep|ls-files))\b"
)
TEST_COMMAND_RE = re.compile(
    r"\b(pytest|tox|nox|go\s+test|cargo\s+test|npm\s+test|pnpm\s+test|yarn\s+test|"
    r"mvn\s+test|gradle\s+test|make\s+(test|check)|ctest|phpunit)\b"
)
WRITE_COMMAND_RE = re.compile(
    r"(apply_patch|sed\s+-i|perl\s+-pi|python\s+- <<|python3\s+- <<|node\s+- <<|ruby\s+- <<|"
    r"\btee\b|>\s*/|>>\s*/|\bmv\b|\bcp\b|\brm\b|\btouch\b|\bchmod\b|\bgit\s+apply\b)"
)
SUBMIT_COMMAND_RE = re.compile(r"\bsubmit\b|/submit|submit\.py")
PATCH_TXT_WRITE_RE = re.compile(r"\bgit\s+diff\b.*(>\s*\S*patch\.txt|\|\s*tee\s+\S*patch\.txt)")
PATCH_TXT_READ_RE = re.compile(r"\b(cat|sed|head|tail|wc|grep|stat|ls|test)\b.*\bpatch\.txt\b")
MANUAL_PATCH_TXT_RE = re.compile(
    r"(cat|tee|printf|echo).*(>\s*\S*patch\.txt|>>\s*\S*patch\.txt|\btee\s+\S*patch\.txt)"
)
VERIFY_COMMAND_RE = re.compile(
    r"(\b(cat|sed|head|tail|wc|grep|stat|ls|test)\b.*\bpatch\.txt\b|"
    r"\bgit\s+(diff|status)\b)"
)


def _rank_world() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def load_chat_template(path: str | Path | None) -> str | None:
    if path is None or str(path).strip() == "":
        return None
    return Path(path).read_text(encoding="utf-8")


def tokenizer_name_or_path(tokenizer: Any) -> str:
    for attr in ("name_or_path", "_name_or_path"):
        value = getattr(tokenizer, attr, None)
        if value:
            return str(value)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        for key in ("name_or_path", "pretrained_model_name_or_path"):
            value = init_kwargs.get(key)
            if value:
                return str(value)
    return DEFAULT_MODEL


def render_chat(tokenizer: Any, messages: list[dict[str, Any]], tools: Any, chat_template: str | None) -> str:
    kwargs: dict[str, Any] = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if chat_template is None:
        return tokenizer.apply_chat_template(**kwargs)
    try:
        return tokenizer.apply_chat_template(chat_template=chat_template, **kwargs)
    except TypeError:
        old_template = getattr(tokenizer, "chat_template", None)
        tokenizer.chat_template = chat_template
        try:
            return tokenizer.apply_chat_template(**kwargs)
        finally:
            tokenizer.chat_template = old_template


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def assistant_reasoning_from_content(message: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning", "thinking", "thought"):
        value = message.get(key)
        if value not in (None, "", [], {}):
            return text_from_content(value).strip()
    content = text_from_content(message.get("content"))
    start = content.find("<think>")
    end = content.find("</think>", start + len("<think>"))
    if start == -1 or end == -1:
        return ""
    return content[start + len("<think>") : end].strip()


def drop_assistant_content_preserving_reasoning(message: dict[str, Any]) -> None:
    reasoning = assistant_reasoning_from_content(message)
    if reasoning:
        message["content"] = f"{THINK_OPEN}{reasoning}{THINK_CLOSE}"
    else:
        message["content"] = ""


def apply_assistant_loss_policy(
    example: dict[str, Any],
    *,
    require_assistant_reasoning_for_loss: bool = False,
    require_assistant_tool_calls_for_loss: bool = False,
    drop_assistant_content_for_tool_calls: bool = False,
    mask_tool_call_error_recovery: bool = False,
    mask_manual_patch_artifact_turns: bool = False,
    enable_turn_loss_weights: bool = False,
    read_loss_weight: float = 0.5,
    write_loss_weight: float = 1.0,
    test_loss_weight: float = 1.0,
    verify_loss_weight: float = 1.5,
    submit_loss_weight: float = 2.0,
    default_loss_weight: float = 1.0,
    nonpassing_loss_multiplier: float = 1.0,
    mask_nonpassing_submit_turns: bool = False,
) -> dict[str, Any]:
    if (
        not require_assistant_reasoning_for_loss
        and not require_assistant_tool_calls_for_loss
        and not drop_assistant_content_for_tool_calls
        and not mask_tool_call_error_recovery
        and not mask_manual_patch_artifact_turns
        and not enable_turn_loss_weights
        and not mask_nonpassing_submit_turns
    ):
        return example

    messages = example.get("messages", [])
    passed = example_passed(example)
    mask_next_assistant = False
    for message in messages:
        if message.get("role") == "tool":
            if mask_tool_call_error_recovery and "tool call error" in text_from_content(message.get("content")).lower():
                mask_next_assistant = True
            continue
        if message.get("role") != "assistant":
            continue
        has_tool_calls = assistant_has_valid_tool_calls(message)
        if drop_assistant_content_for_tool_calls and has_tool_calls:
            drop_assistant_content_preserving_reasoning(message)
        if require_assistant_reasoning_for_loss and not assistant_has_reasoning(message):
            message["loss"] = False
        if require_assistant_tool_calls_for_loss and not has_tool_calls:
            message["loss"] = False
        action = assistant_turn_action(message)
        if mask_next_assistant:
            message["loss"] = False
            mask_next_assistant = False
        if mask_manual_patch_artifact_turns and action == "manual_patch_artifact":
            message["loss"] = False
        if mask_nonpassing_submit_turns and not passed and action == "submit":
            message["loss"] = False
        if enable_turn_loss_weights:
            if action == "submit":
                weight = submit_loss_weight
            elif action == "verify":
                weight = verify_loss_weight
            elif action == "write":
                weight = write_loss_weight
            elif action == "test":
                weight = test_loss_weight
            elif action == "read":
                weight = read_loss_weight
            else:
                weight = default_loss_weight
            if not passed:
                weight *= nonpassing_loss_multiplier
            message["loss_weight"] = float(weight)
    return example


def example_passed(example: dict[str, Any]) -> bool:
    for key in ("passed", "pass", "resolved"):
        if key in example:
            return bool(example[key])
    outcome = example.get("source_outcome")
    if isinstance(outcome, dict):
        for key in ("passed", "pass", "resolved"):
            if key in outcome:
                return bool(outcome[key])
    metadata = example.get("metadata")
    if isinstance(metadata, dict):
        for key in ("passed", "pass", "resolved"):
            if key in metadata:
                return bool(metadata[key])
    return False


def assistant_tool_commands(message: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        if isinstance(arguments, dict):
            command = arguments.get("command")
            if command not in (None, ""):
                commands.append(str(command))
    return commands


def assistant_turn_action(message: dict[str, Any]) -> str:
    commands = assistant_tool_commands(message)
    if not commands:
        return "other"
    joined = "\n".join(commands)
    lowered = joined.lower()
    if "patch.txt" in lowered and MANUAL_PATCH_TXT_RE.search(lowered):
        return "manual_patch_artifact"
    if PATCH_TXT_WRITE_RE.search(lowered) or SUBMIT_COMMAND_RE.search(lowered):
        return "submit"
    if VERIFY_COMMAND_RE.search(lowered):
        return "verify"
    if TEST_COMMAND_RE.search(lowered):
        return "test"
    if WRITE_COMMAND_RE.search(lowered):
        return "write"
    if READ_COMMAND_RE.search(lowered):
        return "read"
    return "other"


def tool_call_label_span(rendered: str, assistant_start_char: int) -> tuple[int, int] | None:
    assistant_end = rendered.find("<|im_end|>", assistant_start_char)
    search_end = len(rendered) if assistant_end == -1 else assistant_end
    tool_start = rendered.find("<tool_call>", assistant_start_char, search_end)
    if tool_start == -1:
        return None
    if assistant_end != -1:
        tool_end = assistant_end + len("<|im_end|>")
        if rendered.startswith("\n", tool_end):
            tool_end += 1
        return tool_start, tool_end
    close = rendered.rfind("</tool_call>", tool_start)
    if close == -1:
        return None
    return tool_start, close + len("</tool_call>")


def assistant_label_span(rendered: str, assistant_start_char: int) -> tuple[int, int] | None:
    assistant_end = rendered.find("<|im_end|>", assistant_start_char)
    if assistant_end == -1:
        return None
    end = assistant_end + len("<|im_end|>")
    if rendered.startswith("\n", end):
        end += 1
    return assistant_start_char, end


def token_span_from_offsets(offsets: list[tuple[int, int]], start_char: int, end_char: int) -> tuple[int, int] | None:
    start_token = None
    end_token = len(offsets)
    for idx, (start, end) in enumerate(offsets):
        if start_token is None and end > start_char:
            start_token = idx
        if start >= end_char:
            end_token = idx
            break
    if start_token is None or end_token <= start_token:
        return None
    return start_token, end_token


def encode_rendered_with_offsets(tokenizer: Any, rendered: str) -> tuple[list[int], list[tuple[int, int]]] | None:
    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        return None
    offsets = encoded.get("offset_mapping")
    input_ids = encoded.get("input_ids")
    if offsets is None or input_ids is None:
        return None
    return list(input_ids), [(int(start), int(end)) for start, end in offsets]


def label_chat_example_slow(
    rendered: str,
    input_ids: list[int],
    messages: list[dict[str, Any]],
    tools: Any,
    tokenizer: Any,
    chat_template: str,
    assistant_loss_target: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(len(input_ids), IGNORE_INDEX, dtype=np.int64)
    loss_weights = np.zeros(len(input_ids), dtype=np.float32)
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if message.get("trainable") is False or message.get("loss") is False:
            continue
        prefix = "" if idx == 0 else render_chat(tokenizer, messages[:idx], tools, chat_template)
        current = render_chat(tokenizer, messages[: idx + 1], tools, chat_template)
        if assistant_loss_target == "tool_calls":
            span = tool_call_label_span(current, len(prefix))
            if span is None:
                continue
            start_char, end_char = span
            start = len(encode_text(tokenizer, current[:start_char]))
            end = len(encode_text(tokenizer, current[:end_char]))
        else:
            start = len(encode_text(tokenizer, prefix))
            end = len(encode_text(tokenizer, current))
        if end > start:
            labels[start:end] = np.asarray(input_ids[start:end], dtype=np.int64)
            loss_weights[start:end] = float(message.get("loss_weight", 1.0))
    return labels, loss_weights


def label_chat_example_fast(
    rendered: str,
    input_ids: list[int],
    offsets: list[tuple[int, int]],
    messages: list[dict[str, Any]],
    assistant_loss_target: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    labels = np.full(len(input_ids), IGNORE_INDEX, dtype=np.int64)
    loss_weights = np.zeros(len(input_ids), dtype=np.float32)
    search_pos = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        assistant_start = rendered.find("<|im_start|>assistant", search_pos)
        if assistant_start == -1:
            return None
        search_pos = assistant_start + len("<|im_start|>assistant")
        if message.get("trainable") is False or message.get("loss") is False:
            continue
        if assistant_loss_target == "tool_calls":
            char_span = tool_call_label_span(rendered, assistant_start)
        else:
            char_span = assistant_label_span(rendered, assistant_start)
        if char_span is None:
            continue
        token_span = token_span_from_offsets(offsets, *char_span)
        if token_span is None:
            continue
        start, end = token_span
        labels[start:end] = np.asarray(input_ids[start:end], dtype=np.int64)
        loss_weights[start:end] = float(message.get("loss_weight", 1.0))
    return labels, loss_weights


def tokenize_chat_example(
    example: dict[str, Any],
    tokenizer: Any,
    *,
    chat_template: str,
    sequence_length: int,
    overlength_strategy: str = "split",
    min_label_tokens: int = 1,
    assistant_loss_target: str = "assistant",
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if assistant_loss_target not in ASSISTANT_LOSS_TARGETS:
        raise ValueError(f"assistant_loss_target must be one of {ASSISTANT_LOSS_TARGETS}; got {assistant_loss_target}")

    messages = example["messages"]
    tools = example.get("tools")
    if not messages:
        return

    rendered = render_chat(tokenizer, messages, tools, chat_template)
    encoded_with_offsets = encode_rendered_with_offsets(tokenizer, rendered)
    if encoded_with_offsets is None:
        input_ids = encode_text(tokenizer, rendered)
        labels, loss_weights = label_chat_example_slow(
            rendered,
            input_ids,
            messages,
            tools,
            tokenizer,
            chat_template,
            assistant_loss_target,
        )
    else:
        input_ids, offsets = encoded_with_offsets
        fast_labels = label_chat_example_fast(
            rendered,
            input_ids,
            offsets,
            messages,
            assistant_loss_target,
        )
        if fast_labels is None:
            labels, loss_weights = label_chat_example_slow(
                rendered,
                input_ids,
                messages,
                tools,
                tokenizer,
                chat_template,
                assistant_loss_target,
            )
        else:
            labels, loss_weights = fast_labels
    if not input_ids:
        return

    ids = np.asarray(input_ids, dtype=np.int64)
    if int(np.count_nonzero(labels != IGNORE_INDEX)) < min_label_tokens:
        return
    if ids.size <= sequence_length:
        yield ids, labels, loss_weights
        return

    if overlength_strategy == "drop":
        return
    if overlength_strategy == "truncate":
        chunk_ids = ids[:sequence_length]
        chunk_labels = labels[:sequence_length]
        if int(np.count_nonzero(chunk_labels != IGNORE_INDEX)) >= min_label_tokens:
            yield chunk_ids, chunk_labels, loss_weights[:sequence_length]
        return
    if overlength_strategy != "split":
        raise ValueError(f"unknown overlength_strategy: {overlength_strategy}")

    for start in range(0, ids.size, sequence_length):
        end = min(start + sequence_length, ids.size)
        chunk_ids = ids[start:end]
        chunk_labels = labels[start:end]
        if int(np.count_nonzero(chunk_labels != IGNORE_INDEX)) >= min_label_tokens:
            yield chunk_ids, chunk_labels, loss_weights[start:end]


@dataclass
class PackedSample:
    input_ids: np.ndarray
    labels: np.ndarray
    loss_weights: np.ndarray | None
    position_ids: np.ndarray
    seq_lens: list[int]
    seq_lens_padded: list[int]
    attention_mask: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "labels": self.labels,
            "position_ids": self.position_ids,
            "seq_lens": self.seq_lens,
            "seq_lens_padded": self.seq_lens_padded,
            "attention_mask": self.attention_mask,
            **({} if self.loss_weights is None else {"loss_weights": self.loss_weights}),
        }


class StreamingTHDPacker:
    def __init__(self, *, sequence_length: int, pad_token_id: int):
        self.sequence_length = int(sequence_length)
        self.pad_token_id = int(pad_token_id)
        self._ids: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._loss_weights: list[np.ndarray] = []
        self._doc_lens: list[int] = []
        self._used = 0

    def add(self, input_ids: np.ndarray, labels: np.ndarray, loss_weights: np.ndarray | None = None) -> Iterator[PackedSample]:
        if input_ids.size > self.sequence_length:
            raise ValueError("documents must be split before packing")
        if self._used and self._used + input_ids.size > self.sequence_length:
            yield self.flush()
        self._ids.append(input_ids.astype(np.int64, copy=False))
        self._labels.append(labels.astype(np.int64, copy=False))
        if loss_weights is not None:
            self._loss_weights.append(loss_weights.astype(np.float32, copy=False))
        self._doc_lens.append(int(input_ids.size))
        self._used += int(input_ids.size)
        if self._used == self.sequence_length:
            yield self.flush()

    @property
    def has_partial(self) -> bool:
        return bool(self._ids)

    def flush(self) -> PackedSample:
        if not self._ids:
            raise RuntimeError("cannot flush an empty packer")
        input_ids = np.concatenate(self._ids)
        labels = np.concatenate(self._labels)
        loss_weights = np.concatenate(self._loss_weights) if self._loss_weights else None
        attention_mask = np.ones(input_ids.size, dtype=np.int64)
        seq_lens = list(self._doc_lens)
        pad = self.sequence_length - input_ids.size
        if pad:
            input_ids = np.pad(input_ids, (0, pad), constant_values=self.pad_token_id)
            labels = np.pad(labels, (0, pad), constant_values=IGNORE_INDEX)
            if loss_weights is not None:
                loss_weights = np.pad(loss_weights, (0, pad), constant_values=0.0)
            attention_mask = np.pad(attention_mask, (0, pad), constant_values=0)
            seq_lens.append(int(pad))

        position_chunks = [np.arange(length, dtype=np.int64) for length in seq_lens]
        position_ids = np.concatenate(position_chunks)
        sample = PackedSample(
            input_ids=input_ids,
            labels=labels,
            loss_weights=loss_weights,
            position_ids=position_ids,
            seq_lens=seq_lens,
            seq_lens_padded=list(seq_lens),
            attention_mask=attention_mask,
        )
        self._ids = []
        self._labels = []
        self._loss_weights = []
        self._doc_lens = []
        self._used = 0
        return sample

    def zero_loss_sample(self) -> PackedSample:
        input_ids = np.full(self.sequence_length, self.pad_token_id, dtype=np.int64)
        labels = np.full(self.sequence_length, IGNORE_INDEX, dtype=np.int64)
        loss_weights = np.zeros(self.sequence_length, dtype=np.float32)
        attention_mask = np.zeros(self.sequence_length, dtype=np.int64)
        position_ids = np.arange(self.sequence_length, dtype=np.int64)
        return PackedSample(
            input_ids=input_ids,
            labels=labels,
            loss_weights=loss_weights,
            position_ids=position_ids,
            seq_lens=[self.sequence_length],
            seq_lens_padded=[self.sequence_length],
            attention_mask=attention_mask,
        )


class OnlinePackedChatDataset(IterableDataset):
    """Stream raw local chat data, tokenize with Qwen's template, and pack online.

    Each rank and dataloader worker gets a disjoint file shard. Packing happens
    inside the worker, so there is no pre-tokenized dataset and no full packed
    cache materialization.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        raw_root: str | Path = RAW_ROOT,
        chat_template_path: str | Path | None = DEFAULT_CHAT_TEMPLATE,
        sequence_length: int = 65536,
        max_rows_per_file: int = 0,
        max_examples: int = 0,
        parquet_batch_size: int = 128,
        overlength_strategy: str = "split",
        min_label_tokens: int = 1,
        shuffle_files: bool = True,
        seed: int = 33333,
        repeat: bool = True,
        pad_to_pack_count: int = 0,
        skip_roots: list[str | Path] | None = None,
        log_every_packs: int = 100,
        shard_rank: int | None = None,
        shard_world_size: int | None = None,
        require_assistant_reasoning_for_loss: bool = False,
        require_assistant_tool_calls_for_loss: bool = False,
        drop_assistant_content_for_tool_calls: bool = False,
        assistant_loss_target: str = "assistant",
        mask_tool_call_error_recovery: bool = False,
        mask_manual_patch_artifact_turns: bool = False,
        enable_turn_loss_weights: bool = False,
        read_loss_weight: float = 0.5,
        write_loss_weight: float = 1.0,
        test_loss_weight: float = 1.0,
        verify_loss_weight: float = 1.5,
        submit_loss_weight: float = 2.0,
        default_loss_weight: float = 1.0,
        nonpassing_loss_multiplier: float = 1.0,
        mask_nonpassing_submit_turns: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_name_or_path = tokenizer_name_or_path(tokenizer)
        self.raw_root = Path(raw_root)
        self.chat_template_path = None if chat_template_path is None or str(chat_template_path).strip() == "" else Path(chat_template_path)
        self.sequence_length = int(sequence_length)
        self.max_rows_per_file = int(max_rows_per_file)
        self.max_examples = int(max_examples)
        self.parquet_batch_size = int(parquet_batch_size)
        self.overlength_strategy = overlength_strategy
        self.min_label_tokens = int(min_label_tokens)
        self.shuffle_files = bool(shuffle_files)
        self.seed = int(seed)
        self.repeat = bool(repeat)
        self.pad_to_pack_count = int(pad_to_pack_count)
        self.skip_roots = [Path(root) for root in (skip_roots or [])]
        self.log_every_packs = int(log_every_packs)
        self.shard_rank = None if shard_rank is None else int(shard_rank)
        self.shard_world_size = None if shard_world_size is None else int(shard_world_size)
        self.require_assistant_reasoning_for_loss = bool(require_assistant_reasoning_for_loss)
        self.require_assistant_tool_calls_for_loss = bool(require_assistant_tool_calls_for_loss)
        self.drop_assistant_content_for_tool_calls = bool(drop_assistant_content_for_tool_calls)
        self.mask_tool_call_error_recovery = bool(mask_tool_call_error_recovery)
        self.mask_manual_patch_artifact_turns = bool(mask_manual_patch_artifact_turns)
        self.enable_turn_loss_weights = bool(enable_turn_loss_weights)
        self.read_loss_weight = float(read_loss_weight)
        self.write_loss_weight = float(write_loss_weight)
        self.test_loss_weight = float(test_loss_weight)
        self.verify_loss_weight = float(verify_loss_weight)
        self.submit_loss_weight = float(submit_loss_weight)
        self.default_loss_weight = float(default_loss_weight)
        self.nonpassing_loss_multiplier = float(nonpassing_loss_multiplier)
        self.mask_nonpassing_submit_turns = bool(mask_nonpassing_submit_turns)
        if assistant_loss_target not in ASSISTANT_LOSS_TARGETS:
            raise ValueError(f"assistant_loss_target must be one of {ASSISTANT_LOSS_TARGETS}; got {assistant_loss_target}")
        self.assistant_loss_target = assistant_loss_target
        self.chat_template = load_chat_template(self.chat_template_path)
        self._configure_tokenizer()

    def _configure_tokenizer(self) -> None:
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = int(self.tokenizer.pad_token_id)

    def _ensure_tokenizer(self) -> Any:
        if self.tokenizer is None:
            try:
                from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer

                self.tokenizer = NeMoAutoTokenizer.from_pretrained(
                    self.tokenizer_name_or_path,
                    trust_remote_code=True,
                )
            except Exception:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name_or_path,
                    trust_remote_code=True,
                )
            self._configure_tokenizer()
        return self.tokenizer

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # NeMo's tokenizer wrapper is not reliably picklable under spawned
        # dataloader workers. Reload it lazily inside each worker instead.
        state["tokenizer"] = None
        return state

    def shard(self, num_shards: int, index: int) -> "OnlinePackedChatDataset":
        if index < 0 or index >= num_shards:
            raise ValueError(f"invalid shard index {index} for {num_shards} shards")
        dataset = copy.copy(self)
        dataset.shard_rank = int(index)
        dataset.shard_world_size = int(num_shards)
        return dataset

    def _sharded_files(self, epoch: int) -> tuple[list[Path], int, int, int, int, int | None, int | None]:
        if self.shard_rank is None or self.shard_world_size is None:
            rank, world_size = _rank_world()
        else:
            rank, world_size = self.shard_rank, self.shard_world_size
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        shard_id = rank * num_workers + worker_id
        num_shards = max(1, world_size * num_workers)

        files = discover_raw_files(self.raw_root, self.skip_roots)
        if self.shuffle_files:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(files)
        if len(files) >= num_shards:
            return files[shard_id::num_shards], rank, world_size, worker_id, num_workers, None, None
        return files, rank, world_size, worker_id, num_workers, shard_id, num_shards

    def __iter__(self) -> Iterator[dict[str, Any]]:
        tokenizer = self._ensure_tokenizer()
        epoch = 0
        emitted_total = 0
        while True:
            files, rank, _world_size, worker_id, _num_workers, row_shard_id, row_num_shards = self._sharded_files(epoch)
            packer = StreamingTHDPacker(
                sequence_length=self.sequence_length,
                pad_token_id=self.pad_token_id,
            )
            examples_seen = 0
            packs_emitted = 0
            start_time = time.time()
            for row_idx, example in enumerate(iter_normalized_examples_from_files(
                files,
                max_rows_per_file=self.max_rows_per_file,
                parquet_batch_size=self.parquet_batch_size,
                max_examples=self.max_examples,
            )):
                if (
                    row_shard_id is not None
                    and row_num_shards is not None
                    and row_idx % row_num_shards != row_shard_id
                ):
                    continue
                examples_seen += 1
                example = apply_assistant_loss_policy(
                    example,
                    require_assistant_reasoning_for_loss=self.require_assistant_reasoning_for_loss,
                    require_assistant_tool_calls_for_loss=self.require_assistant_tool_calls_for_loss,
                    drop_assistant_content_for_tool_calls=self.drop_assistant_content_for_tool_calls,
                    mask_tool_call_error_recovery=self.mask_tool_call_error_recovery,
                    mask_manual_patch_artifact_turns=self.mask_manual_patch_artifact_turns,
                    enable_turn_loss_weights=self.enable_turn_loss_weights,
                    read_loss_weight=self.read_loss_weight,
                    write_loss_weight=self.write_loss_weight,
                    test_loss_weight=self.test_loss_weight,
                    verify_loss_weight=self.verify_loss_weight,
                    submit_loss_weight=self.submit_loss_weight,
                    default_loss_weight=self.default_loss_weight,
                    nonpassing_loss_multiplier=self.nonpassing_loss_multiplier,
                    mask_nonpassing_submit_turns=self.mask_nonpassing_submit_turns,
                )
                for input_ids, labels, loss_weights in tokenize_chat_example(
                    example,
                    tokenizer,
                    chat_template=self.chat_template,
                    sequence_length=self.sequence_length,
                    overlength_strategy=self.overlength_strategy,
                    min_label_tokens=self.min_label_tokens,
                    assistant_loss_target=self.assistant_loss_target,
                ):
                    for packed in packer.add(input_ids, labels, loss_weights if self.enable_turn_loss_weights else None):
                        packs_emitted += 1
                        emitted_total += 1
                        if self.log_every_packs and packs_emitted % self.log_every_packs == 0:
                            elapsed = max(time.time() - start_time, 1e-9)
                            print(
                                f"[online-pack rank={rank} worker={worker_id}] "
                                f"epoch={epoch} packs={packs_emitted:,} "
                                f"examples={examples_seen:,} rate={packs_emitted / elapsed:,.2f} packs/s",
                                flush=True,
                            )
                        yield packed.as_dict()

            if packer.has_partial:
                packed = packer.flush()
                packs_emitted += 1
                emitted_total += 1
                yield packed.as_dict()

            if self.pad_to_pack_count > 0:
                while packs_emitted < self.pad_to_pack_count:
                    packs_emitted += 1
                    emitted_total += 1
                    yield packer.zero_loss_sample().as_dict()
                if packs_emitted > self.pad_to_pack_count:
                    raise RuntimeError(
                        f"worker emitted {packs_emitted:,} packs, exceeding pad_to_pack_count={self.pad_to_pack_count:,}"
                    )

            if not self.repeat:
                return
            epoch += 1
            if emitted_total == 0:
                raise RuntimeError(
                    f"no packed samples produced from {self.raw_root}; check normalization and data paths"
                )


def inspect_packer(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    dataset = OnlinePackedChatDataset(
        tokenizer,
        raw_root=args.raw_root,
        chat_template_path=args.chat_template,
        sequence_length=args.sequence_length,
        max_rows_per_file=args.max_rows_per_file,
        max_examples=args.max_examples,
        overlength_strategy=args.overlength_strategy,
        shuffle_files=False,
        repeat=False,
        log_every_packs=0,
        require_assistant_reasoning_for_loss=args.require_assistant_reasoning_for_loss,
        require_assistant_tool_calls_for_loss=args.require_assistant_tool_calls_for_loss,
        drop_assistant_content_for_tool_calls=args.drop_assistant_content_for_tool_calls,
        assistant_loss_target=args.assistant_loss_target,
        mask_tool_call_error_recovery=args.mask_tool_call_error_recovery,
        mask_manual_patch_artifact_turns=args.mask_manual_patch_artifact_turns,
        enable_turn_loss_weights=args.enable_turn_loss_weights,
        read_loss_weight=args.read_loss_weight,
        write_loss_weight=args.write_loss_weight,
        test_loss_weight=args.test_loss_weight,
        verify_loss_weight=args.verify_loss_weight,
        submit_loss_weight=args.submit_loss_weight,
        default_loss_weight=args.default_loss_weight,
        nonpassing_loss_multiplier=args.nonpassing_loss_multiplier,
        mask_nonpassing_submit_turns=args.mask_nonpassing_submit_turns,
    )
    stats = {
        "packs": 0,
        "tokens": 0,
        "label_tokens": 0,
        "effective_label_weight": 0.0,
        "max_docs_per_pack": 0,
        "sequence_length": args.sequence_length,
    }
    for sample in dataset:
        stats["packs"] += 1
        stats["tokens"] += int(len(sample["input_ids"]))
        stats["label_tokens"] += int(np.count_nonzero(np.asarray(sample["labels"]) != IGNORE_INDEX))
        if "loss_weights" in sample:
            stats["effective_label_weight"] += float(np.asarray(sample["loss_weights"], dtype=np.float32).sum())
        stats["max_docs_per_pack"] = max(stats["max_docs_per_pack"], len(sample["seq_lens"]))
        if args.max_packs and stats["packs"] >= args.max_packs:
            break
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0 if stats["packs"] else 1


def _dataset_for_count(args: argparse.Namespace, tokenizer: Any, *, shard_rank: int, shard_world_size: int) -> OnlinePackedChatDataset:
    return OnlinePackedChatDataset(
        tokenizer,
        raw_root=args.raw_root,
        chat_template_path=args.chat_template,
        sequence_length=args.sequence_length,
        max_rows_per_file=args.max_rows_per_file,
        max_examples=args.max_examples,
        overlength_strategy=args.overlength_strategy,
        shuffle_files=False,
        repeat=False,
        log_every_packs=0,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        require_assistant_reasoning_for_loss=args.require_assistant_reasoning_for_loss,
        require_assistant_tool_calls_for_loss=args.require_assistant_tool_calls_for_loss,
        drop_assistant_content_for_tool_calls=args.drop_assistant_content_for_tool_calls,
        assistant_loss_target=args.assistant_loss_target,
        mask_tool_call_error_recovery=args.mask_tool_call_error_recovery,
        mask_manual_patch_artifact_turns=args.mask_manual_patch_artifact_turns,
        enable_turn_loss_weights=args.enable_turn_loss_weights,
        read_loss_weight=args.read_loss_weight,
        write_loss_weight=args.write_loss_weight,
        test_loss_weight=args.test_loss_weight,
        verify_loss_weight=args.verify_loss_weight,
        submit_loss_weight=args.submit_loss_weight,
        default_loss_weight=args.default_loss_weight,
        nonpassing_loss_multiplier=args.nonpassing_loss_multiplier,
        mask_nonpassing_submit_turns=args.mask_nonpassing_submit_turns,
    )


def _count_one_shard(args_dict: dict[str, Any], shard_id: int, total_workers: int) -> dict[str, Any]:
    from transformers import AutoTokenizer

    args = argparse.Namespace(**args_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    rank = shard_id // args.num_workers
    worker = shard_id % args.num_workers
    dataset = _dataset_for_count(args, tokenizer, shard_rank=shard_id, shard_world_size=total_workers)
    packs = 0
    label_tokens = 0
    effective_label_weight = 0.0
    for sample in dataset:
        packs += 1
        label_tokens += int(np.count_nonzero(np.asarray(sample["labels"]) != IGNORE_INDEX))
        if "loss_weights" in sample:
            effective_label_weight += float(np.asarray(sample["loss_weights"], dtype=np.float32).sum())
    return {
        "rank": rank,
        "worker": worker,
        "shard_id": shard_id,
        "packs": packs,
        "label_tokens": label_tokens,
        "effective_label_weight": effective_label_weight,
    }


def count_shards(args: argparse.Namespace) -> int:
    total_workers = args.world_size * args.num_workers
    worker_stats: list[dict[str, Any]]
    if args.count_processes <= 1:
        worker_stats = []
        for shard_id in range(total_workers):
            worker_stats.append(_count_one_shard(vars(args), shard_id, total_workers))
    else:
        count_processes = min(int(args.count_processes), total_workers)
        worker_stats = []
        with ProcessPoolExecutor(max_workers=count_processes) as executor:
            futures = [
                executor.submit(_count_one_shard, vars(args), shard_id, total_workers)
                for shard_id in range(total_workers)
            ]
            for future in as_completed(futures):
                worker_stats.append(future.result())
        worker_stats.sort(key=lambda item: item["shard_id"])

    max_worker_packs = max((item["packs"] for item in worker_stats), default=0)
    packs_per_worker_multiple = max(1, args.packs_per_worker_multiple)
    pad_to_pack_count = ceil(max_worker_packs / packs_per_worker_multiple) * packs_per_worker_multiple
    local_samples_per_rank = pad_to_pack_count * args.num_workers
    local_samples_per_step = args.local_batch_size * args.grad_accum_steps
    if local_samples_per_rank % local_samples_per_step != 0:
        pad_to_pack_count = ceil(local_samples_per_rank / local_samples_per_step) * local_samples_per_step
        pad_to_pack_count = ceil(pad_to_pack_count / args.num_workers)
        pad_to_pack_count = ceil(pad_to_pack_count / packs_per_worker_multiple) * packs_per_worker_multiple
        local_samples_per_rank = pad_to_pack_count * args.num_workers
    max_steps = local_samples_per_rank // local_samples_per_step

    result = {
        "raw_root": str(args.raw_root),
        "chat_template": str(args.chat_template),
        "sequence_length": args.sequence_length,
        "world_size": args.world_size,
        "num_workers": args.num_workers,
        "local_batch_size": args.local_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "global_batch_size": args.world_size * args.local_batch_size * args.grad_accum_steps,
        "real_packs_total": sum(item["packs"] for item in worker_stats),
        "real_label_tokens_total": sum(item["label_tokens"] for item in worker_stats),
        "real_effective_label_weight_total": sum(item["effective_label_weight"] for item in worker_stats),
        "max_worker_packs": max_worker_packs,
        "pad_to_pack_count": pad_to_pack_count,
        "padded_packs_total": pad_to_pack_count * total_workers,
        "padding_packs_total": pad_to_pack_count * total_workers - sum(item["packs"] for item in worker_stats),
        "max_steps": max_steps,
        "final_checkpoint_step": max_steps - 1,
        "worker_stats": worker_stats,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if max_worker_packs else 1


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--sequence-length", type=int, default=65536)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--overlength-strategy", choices=["split", "truncate", "drop"], default="split")
    parser.add_argument("--require-assistant-reasoning-for-loss", action="store_true")
    parser.add_argument("--require-assistant-tool-calls-for-loss", action="store_true")
    parser.add_argument("--drop-assistant-content-for-tool-calls", action="store_true")
    parser.add_argument("--assistant-loss-target", choices=ASSISTANT_LOSS_TARGETS, default="assistant")
    parser.add_argument("--mask-tool-call-error-recovery", action="store_true")
    parser.add_argument("--mask-manual-patch-artifact-turns", action="store_true")
    parser.add_argument("--enable-turn-loss-weights", action="store_true")
    parser.add_argument("--read-loss-weight", type=float, default=0.5)
    parser.add_argument("--write-loss-weight", type=float, default=1.0)
    parser.add_argument("--test-loss-weight", type=float, default=1.0)
    parser.add_argument("--verify-loss-weight", type=float, default=1.5)
    parser.add_argument("--submit-loss-weight", type=float, default=2.0)
    parser.add_argument("--default-loss-weight", type=float, default=1.0)
    parser.add_argument("--nonpassing-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--mask-nonpassing-submit-turns", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    add_common_dataset_args(inspect)
    inspect.add_argument("--max-packs", type=int, default=2)
    count = subparsers.add_parser("count-shards")
    add_common_dataset_args(count)
    count.set_defaults(max_examples=0)
    count.add_argument("--world-size", type=int, default=8)
    count.add_argument("--num-workers", type=int, default=2)
    count.add_argument("--local-batch-size", type=int, default=2)
    count.add_argument("--grad-accum-steps", type=int, default=4)
    count.add_argument("--packs-per-worker-multiple", type=int, default=4)
    count.add_argument("--count-processes", type=int, default=1)
    count.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "inspect":
        return inspect_packer(args)
    if args.command == "count-shards":
        return count_shards(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
