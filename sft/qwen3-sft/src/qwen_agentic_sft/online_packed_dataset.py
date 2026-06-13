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
from dataclasses import dataclass
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
MINI_SWE_SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
PATCH_TXT_PATH_PATTERN = r"(?:\./|/testbed/)?patch\.txt"


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
        message["reasoning_content"] = reasoning
        message["content"] = ""
    else:
        message["content"] = ""


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


def assistant_has_manual_patch_target(message: dict[str, Any]) -> bool:
    """Detect targets that hand-write patch.txt instead of editing the tree."""
    command = assistant_tool_command(message)
    text = command.lower()
    if "patch.txt" not in text:
        return False
    if text.strip() == MINI_SWE_SUBMIT_COMMAND.lower():
        return False
    if re.search(rf">>\s*{PATCH_TXT_PATH_PATTERN}", text):
        return True
    if "diff -u /dev/null" in text:
        return True
    patch_writer = re.search(r"(^|[;&|\n]\s*)(cat|tee|echo|printf)\b", text, flags=re.DOTALL)
    patch_redirect = re.search(
        rf"(>\s*{PATCH_TXT_PATH_PATTERN}|\btee\s+(-a\s+)?{PATCH_TXT_PATH_PATTERN})",
        text,
        flags=re.DOTALL,
    )
    writes_patch = bool(patch_writer and patch_redirect)
    manual_diff_markers = ("diff --git", "--- /dev/null", "+++ /dev/null", "new file mode", "index 0000000")
    if writes_patch and any(marker in text for marker in manual_diff_markers):
        return True
    if writes_patch and "git diff" not in text:
        return True
    return False


def is_submit_command(command: str) -> bool:
    return "complete_task_and_submit_final_output" in command.lower()


def command_prepares_patch_for_submit(command: str) -> bool:
    text = command.lower()
    if "patch.txt" not in text or is_submit_command(text):
        return False
    if "git diff" in text and re.search(rf"\|\s*tee\s+(-a\s+)?{PATCH_TXT_PATH_PATTERN}", text):
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
    if "patch.txt" not in text or is_submit_command(text):
        return False
    return bool(re.search(rf"(>\s*{PATCH_TXT_PATH_PATTERN}|\|\s*tee\s+(-a\s+)?{PATCH_TXT_PATH_PATTERN})", text))


def observation_has_visible_patch_output(observation: str) -> bool:
    matches = re.findall(r"<output>\n?(.*?)</output>", observation, flags=re.DOTALL)
    if matches:
        return any(match.strip() for match in matches)
    return bool(observation.strip())


def apply_assistant_loss_policy(
    example: dict[str, Any],
    *,
    require_assistant_reasoning_for_loss: bool = False,
    require_assistant_tool_calls_for_loss: bool = False,
    drop_assistant_content_for_tool_calls: bool = False,
    reject_manual_patch_targets: bool = False,
    reject_unverified_submit_targets: bool = False,
) -> dict[str, Any]:
    if (
        not require_assistant_reasoning_for_loss
        and not require_assistant_tool_calls_for_loss
        and not drop_assistant_content_for_tool_calls
        and not reject_manual_patch_targets
        and not reject_unverified_submit_targets
    ):
        return example

    previous_assistant_command = ""
    previous_assistant_observations: list[str] = []
    visible_patch_since_write = False
    patch_file_tainted = False
    seen_submit_command = False
    seen_manual_patch_target = False
    for message in example.get("messages", []):
        if message.get("role") != "assistant":
            if previous_assistant_command:
                previous_assistant_observations.append(str(message.get("content") or ""))
            continue
        command = assistant_tool_command(message)
        has_tool_calls = assistant_has_valid_tool_calls(message)
        if previous_assistant_command and observation_has_visible_patch_output(
            "\n".join(previous_assistant_observations)
        ):
            visible_patch_since_write = True
        if drop_assistant_content_for_tool_calls and has_tool_calls:
            drop_assistant_content_preserving_reasoning(message)
        has_manual_patch_target = assistant_has_manual_patch_target(message)
        is_submit = is_submit_command(command)
        if seen_submit_command:
            message["loss"] = False
        if seen_manual_patch_target:
            message["loss"] = False
        if reject_manual_patch_targets and has_manual_patch_target:
            message["loss"] = False
        if (
            reject_unverified_submit_targets
            and is_submit
            and (
                not command_prepares_patch_for_submit(previous_assistant_command)
                or not visible_patch_since_write
                or patch_file_tainted
            )
        ):
            message["loss"] = False
        if require_assistant_reasoning_for_loss and not assistant_has_reasoning(message):
            message["loss"] = False
        if require_assistant_tool_calls_for_loss and not has_tool_calls:
            message["loss"] = False
        if is_submit:
            seen_submit_command = True
        if reject_manual_patch_targets and has_manual_patch_target:
            seen_manual_patch_target = True
        if command:
            if command_writes_patch_file(command):
                visible_patch_since_write = False
                patch_file_tainted = bool(has_manual_patch_target)
            previous_assistant_command = command
            previous_assistant_observations = []
    return example


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


def tokenize_chat_example(
    example: dict[str, Any],
    tokenizer: Any,
    *,
    chat_template: str,
    sequence_length: int,
    overlength_strategy: str = "split",
    min_label_tokens: int = 1,
    assistant_loss_target: str = "assistant",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if assistant_loss_target not in ASSISTANT_LOSS_TARGETS:
        raise ValueError(f"assistant_loss_target must be one of {ASSISTANT_LOSS_TARGETS}; got {assistant_loss_target}")

    messages = example["messages"]
    tools = example.get("tools")
    if not messages:
        return

    rendered = render_chat(tokenizer, messages, tools, chat_template)
    input_ids = encode_text(tokenizer, rendered)
    if not input_ids:
        return

    labels = np.full(len(input_ids), IGNORE_INDEX, dtype=np.int64)
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

    ids = np.asarray(input_ids, dtype=np.int64)
    if int(np.count_nonzero(labels != IGNORE_INDEX)) < min_label_tokens:
        return
    if ids.size <= sequence_length:
        yield ids, labels
        return

    if overlength_strategy == "drop":
        return
    if overlength_strategy == "truncate":
        chunk_ids = ids[:sequence_length]
        chunk_labels = labels[:sequence_length]
        if int(np.count_nonzero(chunk_labels != IGNORE_INDEX)) >= min_label_tokens:
            yield chunk_ids, chunk_labels
        return
    if overlength_strategy != "split":
        raise ValueError(f"unknown overlength_strategy: {overlength_strategy}")

    for start in range(0, ids.size, sequence_length):
        end = min(start + sequence_length, ids.size)
        chunk_ids = ids[start:end]
        chunk_labels = labels[start:end]
        if int(np.count_nonzero(chunk_labels != IGNORE_INDEX)) >= min_label_tokens:
            yield chunk_ids, chunk_labels


@dataclass
class PackedSample:
    input_ids: np.ndarray
    labels: np.ndarray
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
        }


class StreamingTHDPacker:
    def __init__(self, *, sequence_length: int, pad_token_id: int):
        self.sequence_length = int(sequence_length)
        self.pad_token_id = int(pad_token_id)
        self._ids: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._doc_lens: list[int] = []
        self._used = 0

    def add(self, input_ids: np.ndarray, labels: np.ndarray) -> Iterator[PackedSample]:
        if input_ids.size > self.sequence_length:
            raise ValueError("documents must be split before packing")
        if self._used and self._used + input_ids.size > self.sequence_length:
            yield self.flush()
        self._ids.append(input_ids.astype(np.int64, copy=False))
        self._labels.append(labels.astype(np.int64, copy=False))
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
        attention_mask = np.ones(input_ids.size, dtype=np.int64)
        seq_lens = list(self._doc_lens)
        pad = self.sequence_length - input_ids.size
        if pad:
            input_ids = np.pad(input_ids, (0, pad), constant_values=self.pad_token_id)
            labels = np.pad(labels, (0, pad), constant_values=IGNORE_INDEX)
            attention_mask = np.pad(attention_mask, (0, pad), constant_values=0)
            seq_lens.append(int(pad))

        position_chunks = [np.arange(length, dtype=np.int64) for length in seq_lens]
        position_ids = np.concatenate(position_chunks)
        sample = PackedSample(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            seq_lens=seq_lens,
            seq_lens_padded=list(seq_lens),
            attention_mask=attention_mask,
        )
        self._ids = []
        self._labels = []
        self._doc_lens = []
        self._used = 0
        return sample


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
        skip_roots: list[str | Path] | None = None,
        log_every_packs: int = 100,
        shard_rank: int | None = None,
        shard_world_size: int | None = None,
        require_assistant_reasoning_for_loss: bool = False,
        require_assistant_tool_calls_for_loss: bool = False,
        drop_assistant_content_for_tool_calls: bool = False,
        assistant_loss_target: str = "assistant",
        reject_manual_patch_targets: bool = False,
        reject_unverified_submit_targets: bool = False,
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
        self.skip_roots = [Path(root) for root in (skip_roots or [])]
        self.log_every_packs = int(log_every_packs)
        self.shard_rank = None if shard_rank is None else int(shard_rank)
        self.shard_world_size = None if shard_world_size is None else int(shard_world_size)
        self.require_assistant_reasoning_for_loss = bool(require_assistant_reasoning_for_loss)
        self.require_assistant_tool_calls_for_loss = bool(require_assistant_tool_calls_for_loss)
        self.drop_assistant_content_for_tool_calls = bool(drop_assistant_content_for_tool_calls)
        self.reject_manual_patch_targets = bool(reject_manual_patch_targets)
        self.reject_unverified_submit_targets = bool(reject_unverified_submit_targets)
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
                    reject_manual_patch_targets=self.reject_manual_patch_targets,
                    reject_unverified_submit_targets=self.reject_unverified_submit_targets,
                )
                for input_ids, labels in tokenize_chat_example(
                    example,
                    tokenizer,
                    chat_template=self.chat_template,
                    sequence_length=self.sequence_length,
                    overlength_strategy=self.overlength_strategy,
                    min_label_tokens=self.min_label_tokens,
                    assistant_loss_target=self.assistant_loss_target,
                ):
                    for packed in packer.add(input_ids, labels):
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
        reject_manual_patch_targets=args.reject_manual_patch_targets,
        reject_unverified_submit_targets=args.reject_unverified_submit_targets,
    )
    stats = {
        "packs": 0,
        "tokens": 0,
        "label_tokens": 0,
        "max_docs_per_pack": 0,
        "sequence_length": args.sequence_length,
    }
    for sample in dataset:
        stats["packs"] += 1
        stats["tokens"] += int(len(sample["input_ids"]))
        stats["label_tokens"] += int(np.count_nonzero(np.asarray(sample["labels"]) != IGNORE_INDEX))
        stats["max_docs_per_pack"] = max(stats["max_docs_per_pack"], len(sample["seq_lens"]))
        if stats["packs"] >= args.max_packs:
            break
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0 if stats["packs"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--model", default=DEFAULT_MODEL)
    inspect.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    inspect.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    inspect.add_argument("--sequence-length", type=int, default=65536)
    inspect.add_argument("--max-rows-per-file", type=int, default=0)
    inspect.add_argument("--max-examples", type=int, default=64)
    inspect.add_argument("--max-packs", type=int, default=2)
    inspect.add_argument("--overlength-strategy", choices=["split", "truncate", "drop"], default="split")
    inspect.add_argument("--require-assistant-reasoning-for-loss", action="store_true")
    inspect.add_argument("--require-assistant-tool-calls-for-loss", action="store_true")
    inspect.add_argument("--drop-assistant-content-for-tool-calls", action="store_true")
    inspect.add_argument("--assistant-loss-target", choices=ASSISTANT_LOSS_TARGETS, default="assistant")
    inspect.add_argument("--reject-manual-patch-targets", action="store_true")
    inspect.add_argument("--reject-unverified-submit-targets", action="store_true")
    inspect.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "inspect":
        return inspect_packer(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
