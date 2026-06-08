#!/usr/bin/env python3
"""Build raw mini-swe SFT rows from public SWE-bench Multilingual trajectories.

The local split-table corpora are useful as broad repair anchors, but they do
not contain mini-swe-agent-v2 trajectories for the multilingual target harness.
This builder keeps the public source explicit in the manifest so those rows are
not confused with the local split-table data.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from datasets import load_dataset


BASH_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the shell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
]

MINI_SWE_SYSTEM = "You are a helpful assistant that can interact with a computer shell to solve programming tasks."
MINI_SWE_OUTPUT_LIMIT_CHARS = 10_000
MINI_SWE_OUTPUT_HEAD_CHARS = 5_000
MINI_SWE_OUTPUT_TAIL_CHARS = 5_000
MINI_SWE_LONG_OUTPUT_WARNING = """The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file."""

PUBLIC_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("alienkevin_glm5_multilingual", "AlienKevin/SWE-bench-multilingual-glm-5-trajectories", "train"),
    (
        "alienkevin_minimax_m25_multilingual",
        "AlienKevin/SWE-bench-multilingual-minimax-m2.5-trajectories",
        "train",
    ),
)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def normalize_text(value: str | None) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def extract_problem_description(messages: list[dict[str, Any]] | None) -> str:
    if not messages or len(messages) < 2:
        return ""
    content = str(messages[1].get("content") or "")
    match = re.search(
        r"<pr_description>\s*Consider the following PR description:\s*(.*?)\s*</pr_description>",
        content,
        flags=re.S,
    )
    return match.group(1).strip() if match else content.strip()


def load_problem_to_id() -> dict[str, str]:
    mapping: dict[str, str] = {}
    dataset = load_dataset("SWE-bench/SWE-bench_Multilingual", split="test", streaming=True)
    for row in dataset:
        problem = normalize_text(str(row.get("problem_statement") or ""))
        instance_id = str(row.get("instance_id") or "")
        if problem and instance_id:
            mapping[problem] = instance_id
    return mapping


def normalize_tool_output(content: str) -> str:
    text = content.strip("\n")
    if text.startswith("<returncode>"):
        return text

    returncode = "0"
    match = re.search(r"\n?\[exit code (-?\d+)\]\s*$", text)
    if match:
        returncode = match.group(1)
        text = text[: match.start()].strip("\n")

    parts = [f"<returncode>{returncode}</returncode>"]
    if len(text) < MINI_SWE_OUTPUT_LIMIT_CHARS:
        parts.append(f"<output>\n{text}\n</output>")
    else:
        elided = len(text) - MINI_SWE_OUTPUT_LIMIT_CHARS
        parts.extend(
            [
                f"<warning>\n{MINI_SWE_LONG_OUTPUT_WARNING}\n</warning>",
                f"<output_head>\n{text[:MINI_SWE_OUTPUT_HEAD_CHARS]}\n</output_head>",
                f"<elided_chars>\n{elided} characters elided\n</elided_chars>",
                f"<output_tail>\n{text[-MINI_SWE_OUTPUT_TAIL_CHARS:]}\n</output_tail>",
            ]
        )
    return "\n".join(parts)


def normalize_call(call: Any) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None
    name = ""
    arguments: Any = {}
    function = call.get("function")
    if isinstance(function, dict):
        name = str(function.get("name") or "")
        arguments = function.get("arguments", {})
    elif isinstance(function, str):
        name = function
        arguments = call.get("arguments", {})
    else:
        name = str(call.get("name") or "")
        arguments = call.get("arguments", {})

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"command": arguments}
    if not isinstance(arguments, dict):
        return None
    if name != "bash":
        return None
    command = str(arguments.get("command") or "").rstrip()
    if not command:
        return None
    return {"function": {"name": "bash", "arguments": {"command": command}}}


def normalize_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]] | None, dict[str, int]]:
    out: list[dict[str, Any]] = []
    stats = {
        "assistant_turns": 0,
        "assistant_with_bash": 0,
        "tool_turns": 0,
        "submit_turns": 0,
    }
    for raw in messages:
        role = str(raw.get("role") or "").lower()
        if role == "system":
            out.append({"role": "system", "content": MINI_SWE_SYSTEM})
            continue
        if role == "user":
            out.append({"role": "user", "content": str(raw.get("content") or "")})
            continue
        if role == "tool":
            stats["tool_turns"] += 1
            out.append({"role": "tool", "content": normalize_tool_output(str(raw.get("content") or ""))})
            continue
        if role != "assistant":
            continue

        stats["assistant_turns"] += 1
        calls = [call for call in (normalize_call(item) for item in (raw.get("tool_calls") or [])) if call]
        if not calls:
            return None, stats
        stats["assistant_with_bash"] += 1
        if any("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in call["function"]["arguments"]["command"] for call in calls):
            stats["submit_turns"] += 1
        message = {
            "role": "assistant",
            "content": str(raw.get("content") or ""),
            "tool_calls": calls,
        }
        out.append(message)

    if not any(message.get("role") == "assistant" for message in out):
        return None, stats
    if stats["submit_turns"] == 0:
        return None, stats
    return out, stats


def fill_empty_assistant_content(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        if str(message.get("content") or "").strip():
            continue
        command = ""
        first_call = message["tool_calls"][0]
        function = first_call.get("function") if isinstance(first_call, dict) else {}
        arguments = function.get("arguments", {}) if isinstance(function, dict) else {}
        if isinstance(arguments, dict):
            command = str(arguments.get("command") or "")
        text = command.strip().lower()
        if "complete_task_and_submit_final_output" in text:
            thought = "THOUGHT: The patch is ready. I will submit the final patch output."
        elif "git diff" in text:
            thought = "THOUGHT: I will inspect the current source diff before deciding the next step."
        elif any(marker in text for marker in ("cat >", "python", "apply_patch", "sed -i", "perl -")):
            thought = "THOUGHT: I have enough context to edit the source code and will apply the change now."
        elif any(marker in text for marker in ("test", "pytest", "mvn", "gradle", "npm", "cargo", "go test")):
            thought = "THOUGHT: I will run a focused check to verify the change."
        else:
            thought = "THOUGHT: I will inspect the repository with a focused shell command."
        message["content"] = thought


def iter_rows(
    *,
    source_name: str,
    dataset_name: str,
    split: str,
    parquet_path: Path | None,
    problem_to_id: dict[str, str],
    predictive_ids: set[str],
    max_rows: int,
    fill_empty_content: bool,
) -> Iterable[dict[str, Any]]:
    emitted = 0
    rows: Iterable[dict[str, Any]]
    if parquet_path is not None and parquet_path.exists():
        rows = pq.read_table(parquet_path).to_pylist()
    else:
        rows = load_dataset(dataset_name, split=split, streaming=True)

    for row_number, row in enumerate(rows, 1):
        if not boolish(row.get("resolved")):
            continue
        messages, stats = normalize_messages(list(row.get("messages") or []))
        if messages is None:
            continue
        if fill_empty_content:
            fill_empty_assistant_content(messages)

        problem = normalize_text(extract_problem_description(messages))
        canonical_id = problem_to_id.get(problem, "")
        metadata = {
            "source": source_name,
            "dataset_name": dataset_name,
            "row_number": row_number,
            "source_instance_id": row.get("instance_id"),
            "canonical_instance_id": canonical_id,
            "in_predictive_30": canonical_id in predictive_ids,
            "model": row.get("model"),
            "traj_id": row.get("traj_id"),
            **stats,
        }
        yield {"messages": messages, "tools": BASH_TOOL, "metadata": metadata}
        emitted += 1
        if max_rows > 0 and emitted >= max_rows:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--predictive-ids", type=Path, default=Path("eval/benchmarks/swebench_multilingual/predictive_30_instance_ids.txt"))
    parser.add_argument("--parquet-root", type=Path)
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--fill-empty-assistant-content", action="store_true")
    parser.add_argument("--seed", type=int, default=60608)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    predictive_ids = set()
    if args.predictive_ids.exists():
        predictive_ids = {line.strip() for line in args.predictive_ids.read_text().splitlines() if line.strip()}
    problem_to_id = load_problem_to_id()

    summaries: list[dict[str, Any]] = []
    total_rows = 0
    total_predictive = 0
    for source_name, dataset_name, split in PUBLIC_SOURCES:
        source_dir = args.output_root / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = None
        if args.parquet_root is not None:
            parquet_path = args.parquet_root / f"{source_name}.parquet"
        rows = list(
            iter_rows(
                source_name=source_name,
                dataset_name=dataset_name,
                split=split,
                parquet_path=parquet_path,
                problem_to_id=problem_to_id,
                predictive_ids=predictive_ids,
                max_rows=args.max_rows_per_source,
                fill_empty_content=args.fill_empty_assistant_content,
            )
        )
        random.Random(args.seed).shuffle(rows)
        out_path = source_dir / "data.jsonl"
        predictive_count = 0
        canonical_counts: dict[str, int] = {}
        with out_path.open("w", encoding="utf-8") as out:
            for item in rows:
                cid = item["metadata"].get("canonical_instance_id") or ""
                if item["metadata"].get("in_predictive_30"):
                    predictive_count += 1
                if cid:
                    canonical_counts[cid] = canonical_counts.get(cid, 0) + 1
                out.write(json_dumps(item) + "\n")
        summaries.append(
            {
                "source": source_name,
                "dataset_name": dataset_name,
                "split": split,
                "rows": len(rows),
                "predictive_30_rows": predictive_count,
                "canonical_ids": len(canonical_counts),
                "output": str(out_path),
            }
        )
        total_rows += len(rows)
        total_predictive += predictive_count

    manifest = {
        "output_root": str(args.output_root),
        "rows": total_rows,
        "predictive_30_rows": total_predictive,
        "sources": summaries,
        "selection": "resolved public SWE-bench Multilingual mini-swe trajectories with real bash tool calls",
        "note": "These rows are public target-harness augmentation, not part of split_datasets.csv.",
        "fill_empty_assistant_content": args.fill_empty_assistant_content,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
