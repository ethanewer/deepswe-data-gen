#!/usr/bin/env python3
"""Build an eval-aligned mini-swe-agent raw view of the 260612 synthetic data."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_swebench_ml_sft_mix import BASH_TOOL, adapt_to_bash_tool, json_dumps
from qwen_agentic_sft.data import discover_raw_files, iter_jsonl_rows, normalize_row


DEFAULT_INPUT_JSONL = Path(
    "/wbl-fast/usrs/ee/code-swe-data/data/new-synthetic-data/260612/"
    "highquality-1x-duplicate-reasoning-90pct-30k-full/data/train.jsonl"
)


def open_shards(output_dir: Path, shards: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        (output_dir / f"shard-{idx:03d}.jsonl").open("w", encoding="utf-8")
        for idx in range(shards)
    ]


def aggregate_stats(total: dict[str, int], stats: dict[str, int]) -> None:
    for key, value in stats.items():
        total[key] = total.get(key, 0) + int(value)


def is_trueish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "passed", "pass"}
    return False


def is_positive_reward(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        try:
            return float(value.strip()) > 0
        except ValueError:
            return is_trueish(value)
    return False


def load_uuid_allowlist(path: Path) -> set[str]:
    allowed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("{"):
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
                uuid = row.get("uuid")
            else:
                uuid = text.split()[0]
            if not uuid:
                raise ValueError(f"missing uuid in {path}:{line_number}")
            allowed.add(str(uuid))
    return allowed


def load_line_number_allowlist(path: Path) -> set[int]:
    allowed: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for file_line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("{"):
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{file_line_number}") from exc
                value = row.get("line_number")
            else:
                value = text.split()[0]
            try:
                line_number = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid line_number in {path}:{file_line_number}") from exc
            if line_number < 0:
                raise ValueError(f"negative line_number in {path}:{file_line_number}")
            allowed.add(line_number)
    return allowed


def iter_jsonl_lines(path: Path) -> Iterator[str]:
    if path.name.endswith(".jsonl.zst"):
        process = subprocess.Popen(
            ["zstd", "-dc", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        assert process.stdout is not None
        completed_naturally = False
        try:
            for line in process.stdout:
                yield line.rstrip("\n")
            completed_naturally = True
        finally:
            if not completed_naturally and process.poll() is None:
                process.terminate()
            process.stdout.close()
            return_code = process.wait()
            if return_code not in (0, -15, -13, 141):
                stderr = ""
                if process.stderr is not None:
                    stderr = process.stderr.read().strip()
                raise RuntimeError(f"zstd failed while reading {path}: {stderr}")
            if process.stderr is not None:
                process.stderr.close()
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)

    allowed_uuids = load_uuid_allowlist(args.allow_uuid_file) if args.allow_uuid_file else None
    allowed_line_numbers = (
        load_line_number_allowlist(args.allow_line_number_file) if args.allow_line_number_file else None
    )
    data_dir = args.output_root / "data"
    handles = open_shards(data_dir, args.shards)
    rows_seen = 0
    rows_written = 0
    rows_skipped = 0
    transform_stats: dict[str, int] = {}
    if args.input_root is None:
        input_paths = [args.input_jsonl]
    else:
        search_root = args.input_root / "data" if (args.input_root / "data").is_dir() else args.input_root
        input_paths = [
            path
            for path in discover_raw_files(search_root)
            if path.name.endswith((".jsonl", ".jsonl.zst"))
        ]
        if not input_paths:
            raise FileNotFoundError(f"no JSONL shards found under {args.input_root}")
    try:
        global_line_number = 0

        def process_row(row: dict[str, Any]) -> None:
            nonlocal rows_skipped, rows_written
            if allowed_uuids is not None and str(row.get("uuid")) not in allowed_uuids:
                rows_skipped += 1
                transform_stats["rows_filtered_allow_uuid"] = (
                    transform_stats.get("rows_filtered_allow_uuid", 0) + 1
                )
                return
            if not args.include_failed and not is_trueish(row.get("passed")):
                rows_skipped += 1
                transform_stats["rows_filtered_not_passed"] = (
                    transform_stats.get("rows_filtered_not_passed", 0) + 1
                )
                return
            if not args.include_nonpositive_reward and not is_positive_reward(row.get("reward")):
                rows_skipped += 1
                transform_stats["rows_filtered_nonpositive_reward"] = (
                    transform_stats.get("rows_filtered_nonpositive_reward", 0) + 1
                )
                return
            example = normalize_row(row)
            if example is None:
                rows_skipped += 1
                transform_stats["rows_filtered_normalization"] = (
                    transform_stats.get("rows_filtered_normalization", 0) + 1
                )
                return
            transformed, stats = adapt_to_bash_tool(
                example,
                reasoning_tool_boundary=True,
                strict_prompt=True,
                require_submit=True,
                tool_observation_roles=True,
                single_tool_calls=args.single_tool_calls,
            )
            aggregate_stats(transform_stats, stats)
            if transformed is None:
                rows_skipped += 1
                return
            emitted = {
                "messages": transformed["messages"],
                "tools": transformed.get("tools", BASH_TOOL),
                "source": "swerebench_highquality_miniswe_aligned_passed",
                "source_note": (
                    "Passed high-quality synthetic SWE traces normalized to the "
                    "strict mini-swe-agent prompt, XML tool observations, and "
                    "reasoning/tool-call assistant format."
                ),
                "source_outcome": {
                    "passed": row.get("passed"),
                    "reward": row.get("reward"),
                    "task_id": row.get("task_id"),
                    "uuid": row.get("uuid"),
                },
            }
            handles[rows_written % args.shards].write(json_dumps(emitted) + "\n")
            rows_written += 1

        for input_path in input_paths:
            if allowed_line_numbers is not None:
                row_iter: Iterator[dict[str, Any]] = iter(())
                for line in iter_jsonl_lines(input_path):
                    if args.max_rows and rows_seen >= args.max_rows:
                        break
                    current_line_number = global_line_number
                    global_line_number += 1
                    rows_seen += 1
                    if current_line_number not in allowed_line_numbers:
                        rows_skipped += 1
                        transform_stats["rows_filtered_allow_line_number"] = (
                            transform_stats.get("rows_filtered_allow_line_number", 0) + 1
                        )
                        continue
                    text = line.strip()
                    if not text:
                        rows_skipped += 1
                        transform_stats["rows_filtered_blank_line"] = (
                            transform_stats.get("rows_filtered_blank_line", 0) + 1
                        )
                        continue
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError:
                        rows_skipped += 1
                        transform_stats["rows_filtered_bad_json"] = (
                            transform_stats.get("rows_filtered_bad_json", 0) + 1
                        )
                        continue
                    if not isinstance(row, dict):
                        rows_skipped += 1
                        transform_stats["rows_filtered_non_object_json"] = (
                            transform_stats.get("rows_filtered_non_object_json", 0) + 1
                        )
                        continue
                    process_row(row)
                    if args.log_every and rows_seen % args.log_every == 0:
                        print(
                            json.dumps(
                                {
                                    "rows_seen": rows_seen,
                                    "rows_written": rows_written,
                                    "rows_skipped": rows_skipped,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                if args.max_rows and rows_seen >= args.max_rows:
                    break
                continue

            for row in iter_jsonl_rows(input_path):
                if args.max_rows and rows_seen >= args.max_rows:
                    break
                rows_seen += 1
                process_row(row)
                if args.log_every and rows_seen % args.log_every == 0:
                    print(
                        json.dumps(
                            {
                                "rows_seen": rows_seen,
                                "rows_written": rows_written,
                                "rows_skipped": rows_skipped,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if args.max_rows and rows_seen >= args.max_rows:
                break
    finally:
        for handle in handles:
            handle.close()

    nonempty_shards = 0
    for path in data_dir.glob("shard-*.jsonl"):
        if path.stat().st_size:
            nonempty_shards += 1
        else:
            path.unlink()

    manifest = {
        "input_jsonl": str(args.input_jsonl) if args.input_root is None else None,
        "input_root": str(args.input_root) if args.input_root is not None else None,
        "input_files": [str(path) for path in input_paths],
        "allow_uuid_file": str(args.allow_uuid_file) if args.allow_uuid_file else None,
        "allowed_uuid_count": len(allowed_uuids) if allowed_uuids is not None else None,
        "allow_line_number_file": str(args.allow_line_number_file) if args.allow_line_number_file else None,
        "allowed_line_number_count": len(allowed_line_numbers) if allowed_line_numbers is not None else None,
        "output_root": str(args.output_root),
        "transform": (
            "reasoning_tool_boundary_strict_miniswe_toolobs"
            + ("_single" if args.single_tool_calls else "")
            + "_submit"
        ),
        "outcome_filter": {
            "include_failed": args.include_failed,
            "include_nonpositive_reward": args.include_nonpositive_reward,
        },
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "rows_skipped": rows_skipped,
        "shards": nonempty_shards,
        "requested_shards": args.shards,
        "transform_stats": transform_stats,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--parquet-batch-size", type=int, default=128)
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="include rows where passed is false; disabled by default",
    )
    parser.add_argument(
        "--include-nonpositive-reward",
        action="store_true",
        help="include rows with reward <= 0; disabled by default",
    )
    parser.add_argument("--single-tool-calls", action="store_true")
    parser.add_argument(
        "--allow-uuid-file",
        type=Path,
        default=None,
        help=(
            "optional newline-delimited UUID allowlist; rows not present in this file "
            "are skipped before outcome and format filtering"
        ),
    )
    parser.add_argument(
        "--allow-line-number-file",
        type=Path,
        default=None,
        help=(
            "optional newline-delimited global 0-based raw row line-number allowlist; "
            "unlisted rows are not JSON-parsed"
        ),
    )
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
