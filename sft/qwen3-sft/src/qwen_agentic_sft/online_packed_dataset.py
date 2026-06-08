#!/usr/bin/env python3
"""Online Qwen chat-template tokenization and THD sequence packing."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .data import RAW_ROOT, discover_raw_files, iter_normalized_examples_from_files


IGNORE_INDEX = -100
DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_CHAT_TEMPLATE = Path(
    "/wbl-fast/usrs/ee/code-swe-data/deepswe-data-gen/eval/chat_templates/"
    "qwen3_thinking_acc.jinja2"
)


def _rank_world() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def load_chat_template(path: str | Path) -> str:
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


def render_chat(tokenizer: Any, messages: list[dict[str, Any]], tools: Any, chat_template: str) -> str:
    kwargs: dict[str, Any] = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if tools is not None:
        kwargs["tools"] = tools
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


def tokenize_chat_example(
    example: dict[str, Any],
    tokenizer: Any,
    *,
    chat_template: str,
    sequence_length: int,
    overlength_strategy: str = "split",
    min_label_tokens: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
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
        start = 0 if idx == 0 else len(encode_text(tokenizer, render_chat(tokenizer, messages[:idx], tools, chat_template)))
        end = len(encode_text(tokenizer, render_chat(tokenizer, messages[: idx + 1], tools, chat_template)))
        if end <= start:
            continue
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
        chat_template_path: str | Path = DEFAULT_CHAT_TEMPLATE,
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
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_name_or_path = tokenizer_name_or_path(tokenizer)
        self.raw_root = Path(raw_root)
        self.chat_template_path = Path(chat_template_path)
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

    def _sharded_files(self, epoch: int) -> tuple[list[Path], int, int, int, int]:
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
        return files[shard_id::num_shards], rank, world_size, worker_id, num_workers

    def __iter__(self) -> Iterator[dict[str, Any]]:
        tokenizer = self._ensure_tokenizer()
        epoch = 0
        emitted_total = 0
        while True:
            files, rank, _world_size, worker_id, _num_workers = self._sharded_files(epoch)
            packer = StreamingTHDPacker(
                sequence_length=self.sequence_length,
                pad_token_id=self.pad_token_id,
            )
            examples_seen = 0
            packs_emitted = 0
            start_time = time.time()
            for example in iter_normalized_examples_from_files(
                files,
                max_rows_per_file=self.max_rows_per_file,
                parquet_batch_size=self.parquet_batch_size,
                max_examples=self.max_examples,
            ):
                examples_seen += 1
                for input_ids, labels in tokenize_chat_example(
                    example,
                    tokenizer,
                    chat_template=self.chat_template,
                    sequence_length=self.sequence_length,
                    overlength_strategy=self.overlength_strategy,
                    min_label_tokens=self.min_label_tokens,
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
    inspect.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "inspect":
        return inspect_packer(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
