#!/usr/bin/env python3
"""Rolling online tokenization cache and loader for local OLMo-3 SFT runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional for JSONL-only smoke tests.
    pa = None
    pq = None

try:
    from olmo_core.aliases import PathOrStr
    from olmo_core.data.collator import DataCollator
    from olmo_core.data.data_loader import TextDataLoaderBase
    from olmo_core.data.tokenizer import TokenizerConfig as OLMoTokenizerConfig
except ModuleNotFoundError:  # The producer runs in the Open Instruct venv.
    PathOrStr = str  # type: ignore
    DataCollator = None  # type: ignore
    TextDataLoaderBase = object  # type: ignore
    OLMoTokenizerConfig = Any  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
OPEN_INSTRUCT_DIR = Path(os.environ.get("OPEN_INSTRUCT_DIR", "/wbl-fast/usrs/mk/open-instruct"))
if str(OPEN_INSTRUCT_DIR) not in sys.path:
    sys.path.insert(0, str(OPEN_INSTRUCT_DIR))

import prepare_code_swe_sft_data as prep  # noqa: E402

RAW_ROOT = Path("/wbl-fast/usrs/ee/code-swe-data/data/code-swe-terminal-agentic-sft")
TOKENIZER = Path("/wbl-fast/usrs/mk/data/tokenizers/Olmo-3-7B-Think-SFT")
DEFAULT_CACHE_DIR = Path("/wbl-fast/usrs/ee/code-swe-data/sft/olmo3-sft/work/online-cache")
DEFAULT_OPEN_INSTRUCT_PYTHON = OPEN_INSTRUCT_DIR / ".venv/bin/python"


def get_dataset_transformation():
    from open_instruct import dataset_transformation

    return dataset_transformation


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def discover_raw_files(raw_root: Path, skip_roots: Iterable[Path] = ()) -> list[Path]:
    skip_resolved = [root.resolve() for root in skip_roots]
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(raw_root):
        current = Path(dirpath).resolve()
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname != ".cache"
            and not any(prep.is_under((current / dirname).resolve(), root) for root in skip_resolved)
        ]
        if any(prep.is_under(current, root) for root in skip_resolved):
            continue
        for filename in filenames:
            if filename.endswith((".jsonl", ".parquet")):
                files.append(current / filename)
    files.sort(key=lambda p: str(p.relative_to(raw_root)))
    return files


def _iter_jsonl_rows(path: Path, max_rows: int = 0) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_rows and idx >= max_rows:
                return
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _iter_parquet_rows(path: Path, batch_size: int = 128, max_rows: int = 0) -> Iterator[dict[str, Any]]:
    if pq is None:
        raise RuntimeError("pyarrow is required for parquet input")
    seen = 0
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            if max_rows and seen >= max_rows:
                return
            seen += 1
            if isinstance(row, dict):
                yield row


def iter_normalized_examples(
    raw_root: Path,
    *,
    skip_roots: Iterable[Path] = (),
    max_rows_per_file: int = 0,
    parquet_batch_size: int = 128,
) -> Iterator[dict[str, Any]]:
    stats = {"rows_in": 0, "rows_out": 0, "rows_dropped": 0, "bad_json": 0}
    for path in discover_raw_files(raw_root, skip_roots):
        if path.suffix == ".jsonl":
            normalized_event, handled_as_event_log = prep.normalize_event_log_jsonl(
                path, stats, max_rows_per_file
            )
            if handled_as_event_log:
                if normalized_event is not None:
                    yield normalized_event
                continue
            rows = _iter_jsonl_rows(path, max_rows=max_rows_per_file)
        elif path.suffix == ".parquet":
            rows = _iter_parquet_rows(
                path, batch_size=parquet_batch_size, max_rows=max_rows_per_file
            )
        else:
            continue

        for row in rows:
            stats["rows_in"] += 1
            normalized = prep.normalize_row(row)
            if normalized is None:
                stats["rows_dropped"] += 1
                continue
            stats["rows_out"] += 1
            yield normalized


def build_open_instruct_tokenizer(tokenizer_path: Path):
    dataset_transformation = get_dataset_transformation()
    tc = dataset_transformation.TokenizerConfig(
        tokenizer_name_or_path=str(tokenizer_path),
        chat_template_name="olmo",
    )
    return tc.tokenizer


def tokenize_example(
    example: dict[str, Any],
    tokenizer,
    *,
    sequence_length: int,
    strict_length: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    dataset_transformation = get_dataset_transformation()
    row = {"messages": example["messages"]}
    try:
        if strict_length:
            row = dataset_transformation.sft_tulu_tokenize_v1(
                row, tokenizer, max_seq_length=sequence_length
            )
            keep = dataset_transformation.sft_tulu_strict_filter_v1(
                row, tokenizer, max_seq_length=sequence_length
            )
        else:
            row = dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
                row, tokenizer, max_seq_length=sequence_length
            )
            keep = dataset_transformation.sft_tulu_filter_v1(row, tokenizer)
    except Exception as exc:
        print(f"[online-tokenize] dropped row after tokenization error: {exc}", flush=True)
        return None
    if not keep:
        return None
    input_ids = np.asarray(row[dataset_transformation.INPUT_IDS_KEY], dtype=np.int64)
    labels = np.asarray(row[dataset_transformation.LABELS_KEY], dtype=np.int64)
    label_mask = labels != dataset_transformation.MASKED_TOKEN_VALUE
    if input_ids.size == 0 or not bool(label_mask.any()):
        return None
    if input_ids.size > sequence_length:
        return None
    return input_ids, label_mask.astype(np.bool_)


@dataclass
class PackedInstance:
    input_ids: np.ndarray
    label_mask: np.ndarray
    doc_lens: list[int]


class StreamingPacker:
    def __init__(self, *, sequence_length: int, pad_token_id: int):
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self._tokens: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._doc_lens: list[int] = []
        self._used = 0

    def add(self, input_ids: np.ndarray, label_mask: np.ndarray) -> list[PackedInstance]:
        if input_ids.size > self.sequence_length:
            input_ids = input_ids[: self.sequence_length]
            label_mask = label_mask[: self.sequence_length]
        out: list[PackedInstance] = []
        if self._used and self._used + input_ids.size > self.sequence_length:
            out.append(self.flush())
        self._tokens.append(input_ids)
        self._masks.append(label_mask)
        self._doc_lens.append(int(input_ids.size))
        self._used += int(input_ids.size)
        if self._used == self.sequence_length:
            out.append(self.flush())
        return out

    def flush(self) -> PackedInstance:
        if not self._tokens:
            raise RuntimeError("cannot flush empty packer")
        input_ids = np.concatenate(self._tokens)
        label_mask = np.concatenate(self._masks)
        pad = self.sequence_length - input_ids.size
        if pad:
            input_ids = np.pad(input_ids, (0, pad), constant_values=self.pad_token_id)
            label_mask = np.pad(label_mask, (0, pad), constant_values=False)
        instance = PackedInstance(input_ids=input_ids, label_mask=label_mask, doc_lens=self._doc_lens)
        self._tokens = []
        self._masks = []
        self._doc_lens = []
        self._used = 0
        return instance

    @property
    def has_partial(self) -> bool:
        return bool(self._tokens)


def _save_npy_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, value)
    os.replace(tmp, path)


def write_cache_part(
    cache_dir: Path,
    part_idx: int,
    instances: list[PackedInstance],
    *,
    sequence_length: int,
    token_dtype: np.dtype,
) -> None:
    if not instances:
        return
    part_prefix = cache_dir / "parts" / f"part_{part_idx:06d}"
    input_ids = np.stack([inst.input_ids for inst in instances]).astype(token_dtype, copy=False)
    label_mask = np.stack([inst.label_mask for inst in instances]).astype(np.bool_, copy=False)
    max_docs = max(len(inst.doc_lens) for inst in instances)
    doc_lens = np.zeros((len(instances), max_docs), dtype=np.int32)
    for idx, inst in enumerate(instances):
        doc_lens[idx, : len(inst.doc_lens)] = inst.doc_lens

    _save_npy_atomic(part_prefix.with_suffix(".input_ids.npy"), input_ids)
    _save_npy_atomic(part_prefix.with_suffix(".label_mask.npy"), label_mask)
    _save_npy_atomic(part_prefix.with_suffix(".doc_lens.npy"), doc_lens)
    atomic_write_json(
        part_prefix.with_suffix(".ready.json"),
        {
            "part": part_idx,
            "num_instances": len(instances),
            "sequence_length": sequence_length,
            "token_dtype": str(np.dtype(token_dtype)),
            "max_docs": max_docs,
            "created_at": time.time(),
        },
    )


def produce_cache(
    *,
    raw_root: Path,
    cache_dir: Path,
    tokenizer_path: Path,
    sequence_length: int,
    part_instances: int,
    max_rows_per_file: int,
    max_examples: int,
    strict_length: bool,
    skip_root: list[Path],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_open_instruct_tokenizer(tokenizer_path)
    token_dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
    pad_token_id = int(tokenizer.pad_token_id)
    metadata = {
        "raw_root": str(raw_root),
        "tokenizer": str(tokenizer_path),
        "sequence_length": sequence_length,
        "part_instances": part_instances,
        "strict_length": strict_length,
        "started_at": time.time(),
    }
    atomic_write_json(cache_dir / "producer.json", metadata)

    existing_parts = ready_parts(cache_dir)
    part_idx = existing_parts[-1]["part"] + 1 if existing_parts else 0
    packer = StreamingPacker(sequence_length=sequence_length, pad_token_id=pad_token_id)
    pending: list[PackedInstance] = []
    seen = kept = emitted = 0
    start = time.time()
    for example in iter_normalized_examples(
        raw_root,
        skip_roots=skip_root,
        max_rows_per_file=max_rows_per_file,
    ):
        seen += 1
        tokenized = tokenize_example(
            example,
            tokenizer,
            sequence_length=sequence_length,
            strict_length=strict_length,
        )
        if tokenized is None:
            continue
        kept += 1
        for instance in packer.add(*tokenized):
            pending.append(instance)
        while len(pending) >= part_instances:
            write_cache_part(
                cache_dir,
                part_idx,
                pending[:part_instances],
                sequence_length=sequence_length,
                token_dtype=token_dtype,
            )
            emitted += part_instances
            pending = pending[part_instances:]
            elapsed = max(1e-9, time.time() - start)
            print(
                f"[online-tokenize] part={part_idx:06d} seen={seen:,} kept={kept:,} "
                f"instances={emitted:,} rate={kept / elapsed:,.2f} examples/s",
                flush=True,
            )
            part_idx += 1
        if max_examples and kept >= max_examples:
            break

    if packer.has_partial:
        pending.append(packer.flush())
    if pending:
        write_cache_part(
            cache_dir,
            part_idx,
            pending,
            sequence_length=sequence_length,
            token_dtype=token_dtype,
        )
        emitted += len(pending)
    metadata.update({"finished_at": time.time(), "seen": seen, "kept": kept, "instances": emitted})
    atomic_write_json(cache_dir / "producer.done.json", metadata)
    print(
        f"[online-tokenize] done seen={seen:,} kept={kept:,} instances={emitted:,}",
        flush=True,
    )


def ready_parts(cache_dir: Path) -> list[dict[str, Any]]:
    parts = []
    for ready_path in sorted((cache_dir / "parts").glob("part_*.ready.json")):
        try:
            data = json.loads(ready_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        prefix = ready_path.with_suffix("").with_suffix("")
        if (
            prefix.with_suffix(".input_ids.npy").exists()
            and prefix.with_suffix(".label_mask.npy").exists()
            and prefix.with_suffix(".doc_lens.npy").exists()
        ):
            data["prefix"] = str(prefix)
            parts.append(data)
    parts.sort(key=lambda item: int(item["part"]))
    return parts


class OnlinePackedSFTDataLoader(TextDataLoaderBase):
    """Read fixed-length packed cache parts while a tokenizer process fills them."""

    def __init__(
        self,
        *,
        cache_dir: PathOrStr,
        tokenizer: OLMoTokenizerConfig,
        sequence_length: int,
        global_batch_size: int,
        work_dir: PathOrStr,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        min_ready_batches: int = 1,
        poll_interval: float = 5.0,
        include_doc_lens: bool = True,
    ):
        if TextDataLoaderBase is object or DataCollator is None:
            raise RuntimeError("OLMo-core is required to construct OnlinePackedSFTDataLoader")
        super().__init__(
            collator=DataCollator(
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.padded_vocab_size(),
            ),
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )
        self.cache_dir = Path(cache_dir)
        self.sequence_length = sequence_length
        self.min_ready_batches = min_ready_batches
        self.poll_interval = poll_interval
        self.include_doc_lens = include_doc_lens
        self._parts: list[dict[str, Any]] = []
        self._part_arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.wait_seconds = 0.0

    @property
    def instances_per_global_batch(self) -> int:
        return self.global_batch_size // self.sequence_length

    @property
    def instances_per_rank_batch(self) -> int:
        return self.rank_batch_size // self.sequence_length

    @property
    def total_batches(self) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {
            "batches_processed": self.batches_processed,
            "tokens_processed": self.tokens_processed,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.batches_processed = int(state_dict.get("batches_processed", 0))
        self.tokens_processed = int(state_dict.get("tokens_processed", 0))
        self._epoch = state_dict.get("epoch", self._epoch)

    def reshuffle(self, epoch: int | None = None, **kwargs) -> None:
        del kwargs
        self._epoch = 1 if epoch is None and self._epoch is None else (epoch or self._epoch + 1)

    def get_mock_batch(self) -> dict[str, Any]:
        g = torch.Generator(device="cpu")
        g.manual_seed(33333 + self.dp_rank)
        input_ids = torch.randint(
            0,
            100,
            (self.instances_per_rank_batch, self.sequence_length),
            generator=g,
            device="cpu",
        )
        label_mask = torch.ones_like(input_ids, dtype=torch.bool)
        out: dict[str, Any] = {"input_ids": input_ids, "label_mask": label_mask}
        if self.include_doc_lens:
            out["doc_lens"] = torch.full(
                (self.instances_per_rank_batch, 1), self.sequence_length, dtype=torch.long
            )
            out["max_doc_lens"] = [self.sequence_length] * self.instances_per_rank_batch
        return out

    def _refresh_parts(self) -> None:
        self._parts = ready_parts(self.cache_dir)

    def _available_instances(self) -> int:
        return sum(int(part["num_instances"]) for part in self._parts)

    def _wait_for_instance(self, global_instance_idx: int) -> None:
        while True:
            self._refresh_parts()
            if self._available_instances() > global_instance_idx:
                return
            done = self.cache_dir / "producer.done.json"
            if done.exists():
                raise StopIteration
            start = time.time()
            print(
                f"[online-loader rank={self.dp_rank}] waiting for cache instance "
                f"{global_instance_idx:,}; available={self._available_instances():,}",
                flush=True,
            )
            time.sleep(self.poll_interval)
            self.wait_seconds += time.time() - start

    def _locate_instance(self, global_instance_idx: int) -> tuple[dict[str, Any], int]:
        self._wait_for_instance(global_instance_idx)
        offset = 0
        for part in self._parts:
            count = int(part["num_instances"])
            if offset <= global_instance_idx < offset + count:
                return part, global_instance_idx - offset
            offset += count
        raise StopIteration

    def _load_part_arrays(self, part: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        part_idx = int(part["part"])
        arrays = self._part_arrays.get(part_idx)
        if arrays is not None:
            return arrays
        prefix = Path(part["prefix"])
        arrays = (
            np.load(prefix.with_suffix(".input_ids.npy"), mmap_mode="r"),
            np.load(prefix.with_suffix(".label_mask.npy"), mmap_mode="r"),
            np.load(prefix.with_suffix(".doc_lens.npy"), mmap_mode="r"),
        )
        self._part_arrays[part_idx] = arrays
        if len(self._part_arrays) > 4:
            oldest = sorted(self._part_arrays)[0]
            self._part_arrays.pop(oldest, None)
        return arrays

    def _get_instance(self, global_instance_idx: int) -> dict[str, Any]:
        part, local_idx = self._locate_instance(global_instance_idx)
        input_ids, label_mask, doc_lens = self._load_part_arrays(part)
        item: dict[str, Any] = {
            "input_ids": np.asarray(input_ids[local_idx], dtype=np.int64),
            "label_mask": np.asarray(label_mask[local_idx], dtype=np.bool_),
        }
        if self.include_doc_lens:
            lens = np.asarray(doc_lens[local_idx], dtype=np.int64)
            lens = lens[lens > 0]
            item["doc_lens"] = torch.from_numpy(lens)
        return item

    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        global_batch_idx = self.batches_processed
        while True:
            start = global_batch_idx * self.instances_per_global_batch
            local_indices = range(
                start + self.dp_rank,
                start + self.instances_per_global_batch,
                self.dp_world_size,
            )
            try:
                instances = [self._get_instance(idx) for idx in local_indices]
            except StopIteration:
                return
            yield self.collator(instances)
            global_batch_idx += 1


def start_background_producer(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    cmd = [
        str(args.online_python),
        str(SCRIPT_DIR / "online_sft_cache.py"),
        "produce-cache",
        "--raw-root",
        str(args.raw_root),
        "--cache-dir",
        str(args.online_cache_dir),
        "--tokenizer",
        str(args.online_tokenizer),
        "--sequence-length",
        str(args.sequence_length),
        "--part-instances",
        str(args.online_cache_part_instances),
        "--max-rows-per-file",
        str(args.online_max_rows_per_file),
        "--max-examples",
        str(args.online_max_examples),
    ]
    if not args.online_truncate_overlength:
        cmd.append("--strict-length")
    for skip_root in args.online_skip_root:
        cmd.extend(["--skip-root", str(skip_root)])
    Path(args.online_cache_dir).mkdir(parents=True, exist_ok=True)
    print("[online-tokenize] starting producer: " + " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, text=True)


def wait_for_ready_batches(
    cache_dir: Path,
    *,
    min_ready_batches: int,
    global_batch_size: int,
    sequence_length: int,
    poll_interval: float,
) -> None:
    required_instances = min_ready_batches * (global_batch_size // sequence_length)
    while True:
        available = sum(int(part["num_instances"]) for part in ready_parts(cache_dir))
        if available >= required_instances:
            return
        done = cache_dir / "producer.done.json"
        if done.exists() and available > 0:
            return
        print(
            f"[online-loader] warmup waiting: {available:,}/{required_instances:,} "
            "instances ready",
            flush=True,
        )
        time.sleep(poll_interval)


def build_smoke_raw_dataset(
    *,
    raw_root: Path,
    output_root: Path,
    rows_per_dataset: int,
    max_event_log_lines: int,
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        copied = 0
        target_dataset = output_root / dataset_dir.name
        for path in discover_raw_files(dataset_dir):
            if copied >= rows_per_dataset:
                break
            rel = path.relative_to(dataset_dir)
            out_path = target_dataset / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".jsonl":
                lines: list[str] = []
                with path.open("r", encoding="utf-8") as handle:
                    for line_idx, line in enumerate(handle):
                        if line_idx >= max_event_log_lines:
                            break
                        if line.strip():
                            lines.append(line)
                        if len(lines) >= rows_per_dataset:
                            break
                if not lines:
                    continue
                out_path.write_text("".join(lines), encoding="utf-8")
                copied += len(lines)
            elif path.suffix == ".parquet" and pq is not None and pa is not None:
                rows = list(_iter_parquet_rows(path, max_rows=rows_per_dataset - copied))
                if not rows:
                    continue
                pq.write_table(pa.Table.from_pylist(rows), out_path)
                copied += len(rows)
        print(f"[smoke] {dataset_dir.name}: copied approximately {copied} rows", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    produce = subparsers.add_parser("produce-cache")
    produce.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    produce.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    produce.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    produce.add_argument("--sequence-length", type=int, default=65536)
    produce.add_argument("--part-instances", type=int, default=128)
    produce.add_argument("--max-rows-per-file", type=int, default=0)
    produce.add_argument("--max-examples", type=int, default=0)
    produce.add_argument("--strict-length", action="store_true")
    produce.add_argument("--skip-root", type=Path, action="append", default=[])

    smoke = subparsers.add_parser("build-smoke-raw")
    smoke.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--rows-per-dataset", type=int, default=3)
    smoke.add_argument("--max-event-log-lines", type=int, default=200)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "produce-cache":
        produce_cache(
            raw_root=args.raw_root,
            cache_dir=args.cache_dir,
            tokenizer_path=args.tokenizer,
            sequence_length=args.sequence_length,
            part_instances=args.part_instances,
            max_rows_per_file=args.max_rows_per_file,
            max_examples=args.max_examples,
            strict_length=args.strict_length,
            skip_root=args.skip_root,
        )
        return 0
    if args.command == "build-smoke-raw":
        build_smoke_raw_dataset(
            raw_root=args.raw_root,
            output_root=args.output_root,
            rows_per_dataset=args.rows_per_dataset,
            max_event_log_lines=args.max_event_log_lines,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
