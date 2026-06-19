#!/usr/bin/env python3
"""Inspect candidate SWE-style task sources for benchmark-aligned datagen.

The script is intentionally read-mostly. It gathers enough metadata to decide
whether a source can be adapted to the pinned mini-swe-agent/SWE-bench harness
without changing prompts or runtime behavior.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import re
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from datasets import IterableDataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download


CURRENT_INDEX = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swerebench-traces-highquality-2x-duplicate-reasoning-90pct/metadata/index.jsonl"
)
DEFAULT_ENV_FILE = Path("/wbl-fast/usrs/ee/code-swe-data/.env")


@dataclass(frozen=True)
class CurrentTaskIndex:
    task_ids: set[str]
    repos: set[str]
    repo_task_keys: set[str]
    repo_issue_keys: set[str]


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_env_file(path: Path) -> None:
    """Load local credentials without printing them or overriding live env."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def hub_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_READ_TOKEN")
    )


def current_task_index(path: Path = CURRENT_INDEX) -> CurrentTaskIndex:
    task_ids: set[str] = set()
    repos: set[str] = set()
    repo_task_keys: set[str] = set()
    repo_issue_keys: set[str] = set()
    for row in read_jsonl(path):
        task_id = str(row.get("task_id") or row.get("instance_id") or "")
        repo = str(row.get("repo") or "")
        if task_id:
            task_ids.add(task_id)
        if repo:
            repos.add(repo)
        if repo and task_id:
            repo_task_keys.add(f"{repo}::{task_id}")
            issue = issue_number_from_instance_id(task_id)
            if issue:
                repo_issue_keys.add(f"{repo}::{issue}")
    return CurrentTaskIndex(task_ids, repos, repo_task_keys, repo_issue_keys)


def issue_number_from_instance_id(instance_id: str) -> str:
    match = re.search(r"-(\d+)$", instance_id)
    return match.group(1) if match else ""


def file_summary(api: HfApi, repo_id: str, limit: int = 1000) -> dict[str, Any]:
    info = api.dataset_info(repo_id, files_metadata=True)
    siblings = list(info.siblings or [])
    suffix_counts: Counter[str] = Counter()
    top_files = []
    total_size = 0
    for sibling in siblings:
        name = sibling.rfilename
        size = getattr(sibling, "size", None) or 0
        total_size += size
        suffix = Path(name).suffix or "<none>"
        suffix_counts[suffix] += 1
        if len(top_files) < limit:
            top_files.append({"path": name, "size": size})
    return {
        "repo_id": repo_id,
        "sha": getattr(info, "sha", None),
        "card_data": getattr(info, "card_data", None),
        "sibling_count": len(siblings),
        "total_size_bytes": total_size,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "sample_files": top_files[:50],
        "all_files": top_files,
    }


def split_names(repo_id: str) -> list[str]:
    try:
        from datasets import get_dataset_split_names

        return list(get_dataset_split_names(repo_id))
    except Exception as exc:  # noqa: BLE001 - recorded in output
        return [f"<error:{type(exc).__name__}:{exc}>"]


def take_stream_rows(repo_id: str, split: str, n: int = 20) -> tuple[list[dict[str, Any]], str | None]:
    try:
        dataset = load_dataset(repo_id, split=split, streaming=True)
        assert isinstance(dataset, IterableDataset)
        rows = []
        for row in dataset:
            rows.append(row)
            if len(rows) >= n:
                break
        return rows, None
    except Exception as exc:  # noqa: BLE001 - caller records exact failure
        return [], f"{type(exc).__name__}: {exc}"


def field_lengths(row: dict[str, Any], fields: Iterable[str]) -> dict[str, int]:
    lengths = {}
    for field in fields:
        value = row.get(field)
        if value is None:
            lengths[field] = -1
        elif isinstance(value, str):
            lengths[field] = len(value)
        else:
            lengths[field] = len(json.dumps(value, default=str))
    return lengths


def patch_touches_non_test(patch: str) -> bool:
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3].removeprefix("b/")
        lowered = path.lower()
        if not any(part in lowered for part in ("/test", "tests/", "test_", "_test.", ".snap", "fixture", "golden")):
            return True
    return False


def summarize_swe_rebench_rows(
    rows: Iterable[dict[str, Any]],
    current: CurrentTaskIndex,
) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    exact_task_overlap = 0
    repo_issue_overlap = 0
    repo_overlap = 0
    nonempty_patch = 0
    nonempty_test_patch = 0
    non_test_gold = 0
    with_image = 0
    with_test_cmd = 0
    samples = []
    total = 0
    for row in rows:
        total += 1
        instance_id = str(row.get("instance_id") or "")
        repo = str(row.get("repo") or "")
        patch = str(row.get("patch") or "")
        test_patch = str(row.get("test_patch") or "")
        install_config = row.get("install_config") or {}
        meta = row.get("meta") or {}
        llm_meta = meta.get("llm_metadata") if isinstance(meta, dict) else None
        difficulty = ""
        if isinstance(llm_meta, dict):
            difficulty = str(llm_meta.get("difficulty") or "")
        counters["difficulty"][difficulty or "<missing>"] += 1
        counters["repo"][repo or "<missing>"] += 1
        if instance_id in current.task_ids:
            exact_task_overlap += 1
        issue = issue_number_from_instance_id(instance_id)
        if repo and issue and f"{repo}::{issue}" in current.repo_issue_keys:
            repo_issue_overlap += 1
        if repo in current.repos:
            repo_overlap += 1
        if patch.strip():
            nonempty_patch += 1
        if test_patch.strip():
            nonempty_test_patch += 1
        if patch_touches_non_test(patch):
            non_test_gold += 1
        if row.get("image_name") or row.get("docker_image"):
            with_image += 1
        if isinstance(install_config, dict) and install_config.get("test_cmd"):
            with_test_cmd += 1
        if len(samples) < 8:
            samples.append(
                {
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": row.get("base_commit"),
                    "difficulty": difficulty,
                    "problem_statement_chars": len(str(row.get("problem_statement") or "")),
                    "patch_chars": len(patch),
                    "test_patch_chars": len(test_patch),
                    "has_image": bool(row.get("image_name") or row.get("docker_image")),
                    "has_test_cmd": isinstance(install_config, dict) and bool(install_config.get("test_cmd")),
                    "patch_head": patch[:600],
                    "test_cmd": install_config.get("test_cmd") if isinstance(install_config, dict) else None,
                }
            )
    return {
        "rows_scanned": total,
        "difficulty_counts": dict(counters["difficulty"]),
        "exact_task_overlap": exact_task_overlap,
        "repo_issue_overlap": repo_issue_overlap,
        "repo_overlap": repo_overlap,
        "nonempty_patch": nonempty_patch,
        "nonempty_test_patch": nonempty_test_patch,
        "gold_patch_touches_non_test": non_test_gold,
        "with_image": with_image,
        "with_test_cmd": with_test_cmd,
        "top_repos": counters["repo"].most_common(20),
        "samples": samples,
    }


def scan_stream_swe_rebench(repo_id: str, split: str, current: CurrentTaskIndex, max_rows: int = 0) -> dict[str, Any]:
    dataset = load_dataset(repo_id, split=split, streaming=True)
    assert isinstance(dataset, IterableDataset)
    iterator = iter(dataset)
    if max_rows > 0:
        rows = []
        for _ in range(max_rows):
            try:
                rows.append(next(iterator))
            except StopIteration:
                break
        return summarize_swe_rebench_rows(rows, current)
    return summarize_swe_rebench_rows(iterator, current)


def read_parquet_schema_from_hub(repo_id: str, filename: str) -> dict[str, Any]:
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=hub_token())
    parquet_file = pq.ParquetFile(path)
    return {
        "local_path": path,
        "num_row_groups": parquet_file.num_row_groups,
        "metadata_num_rows": parquet_file.metadata.num_rows if parquet_file.metadata else None,
        "schema": str(parquet_file.schema_arrow),
    }


def sample_tasktrove_parquet(
    repo_id: str,
    filename: str,
    sample_rows: int,
    current: CurrentTaskIndex,
) -> dict[str, Any]:
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=hub_token())
    parquet_file = pq.ParquetFile(path)
    rows = []
    task_summaries = []
    total_rows = parquet_file.metadata.num_rows if parquet_file.metadata else None
    for batch in parquet_file.iter_batches(batch_size=1):
        record = batch.to_pylist()[0]
        rows.append(record)
        task_summaries.append(inspect_harbor_task_binary(record, current))
        if len(rows) >= sample_rows:
            break
    return {
        "filename": filename,
        "local_path": path,
        "metadata_num_rows": total_rows,
        "schema": str(parquet_file.schema_arrow),
        "sample_task_summaries": task_summaries,
    }


def scan_parquet_path_overlap(repo_id: str, filename: str, current: CurrentTaskIndex) -> dict[str, Any]:
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=hub_token())
    parquet_file = pq.ParquetFile(path)
    current_issue_numbers = {key.rsplit("::", 1)[1] for key in current.repo_issue_keys}
    total = 0
    exact = 0
    repo_issue = 0
    path_examples = []
    for batch in parquet_file.iter_batches(columns=["path"], batch_size=4096):
        for record in batch.to_pylist():
            task_path = str(record.get("path") or "")
            total += 1
            if task_path in current.task_ids:
                exact += 1
                if len(path_examples) < 8:
                    path_examples.append(task_path)
            issue = issue_number_from_instance_id(task_path)
            if issue and issue in current_issue_numbers:
                repo_issue += 1
    return {
        "filename": filename,
        "metadata_num_rows": parquet_file.metadata.num_rows if parquet_file.metadata else None,
        "rows_scanned": total,
        "exact_path_overlap": exact,
        "loose_issue_number_overlap": repo_issue,
        "exact_path_overlap_examples": path_examples,
    }


def inspect_harbor_task_binary(record: dict[str, Any], current: CurrentTaskIndex) -> dict[str, Any]:
    binary = record.get("task_binary")
    task_path = str(record.get("path") or "")
    if binary is None:
        return {"path": task_path, "error": "no task_binary field"}
    if isinstance(binary, list):
        binary = bytes(binary)
    elif isinstance(binary, memoryview):
        binary = binary.tobytes()
    elif not isinstance(binary, (bytes, bytearray)):
        return {"path": task_path, "error": f"unexpected task_binary type {type(binary).__name__}"}
    data = bytes(binary)
    try:
        data = gzip.decompress(data)
    except OSError:
        pass
    names = []
    text_files: dict[str, str] = {}
    file_sizes: dict[str, int] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            for member in tf.getmembers():
                names.append(member.name)
                file_sizes[member.name] = member.size
                base_name = member.name.removeprefix("./")
                if is_interesting_task_file(base_name):
                    extracted = tf.extractfile(member)
                    if extracted is not None:
                        text_files[base_name] = extracted.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - record exact failure
        return {"path": task_path, "error": f"{type(exc).__name__}: {exc}"}
    return inspect_task_files(task_path, names, text_files, file_sizes, current)


def first_present(files: dict[str, str], names: list[str]) -> str:
    for name in names:
        if files.get(name):
            return files[name]
    return ""


def is_interesting_task_file(base_name: str) -> bool:
    if base_name in {
        "instruction.md",
        "task.toml",
        "metadata.json",
        "tests/test.sh",
        "tests/test.patch",
        "tests/test_patch.diff",
        "tests/gold.patch",
        "solution/solution.patch",
        "solution/model.patch",
        "solution/gold.patch",
        "solution/patch.diff",
        "solution/fix.patch",
        "solution/solve.sh",
        "solution/solution.sh",
    }:
        return True
    if base_name.startswith("solution/") and base_name.endswith((".patch", ".diff", ".sh")):
        return True
    if base_name.startswith("tests/") and base_name.endswith((".patch", ".diff", ".sh")):
        return True
    return False


def inspect_task_files(
    task_path: str,
    names: list[str],
    text_files: dict[str, str],
    file_sizes: dict[str, int],
    current: CurrentTaskIndex,
) -> dict[str, Any]:
    task_toml = text_files.get("task.toml", "")
    instruction = text_files.get("instruction.md", "")
    solution_patch = first_present(
        text_files,
        [
            "solution/solution.patch",
            "solution/model.patch",
            "solution/gold.patch",
            "solution/patch.diff",
            "solution/fix.patch",
        ],
    )
    test_sh = text_files.get("tests/test.sh", "")
    test_patch = first_present(text_files, ["tests/test.patch", "tests/test_patch.diff", "tests/gold.patch"])
    solve_sh = first_present(text_files, ["solution/solve.sh", "solution/solution.sh"])
    embedded_solution_patch = extract_embedded_patch(solve_sh)
    metadata_json = text_files.get("metadata.json", "")
    parsed_metadata = parse_json_object(metadata_json)
    metadata_task_id = extract_toml_string(task_toml, "task_id")
    repo_url = extract_toml_string(task_toml, "repository_url")
    metadata_instance_id = str(parsed_metadata.get("instance_id") or "")
    metadata_repo = str(parsed_metadata.get("repo") or "")
    solution_repo = extract_github_repo(solve_sh)
    inferred_task_id = metadata_task_id or metadata_instance_id or (task_path if "__" in task_path else "")
    inferred_repo = repo_url.removeprefix("https://github.com/") if repo_url else metadata_repo or solution_repo
    issue = issue_number_from_instance_id(inferred_task_id or task_path)
    return {
        "path": task_path,
        "file_count": len(names),
        "has_instruction": bool(instruction.strip()),
        "has_task_toml": bool(task_toml.strip()),
        "has_tests_test_sh": bool(test_sh.strip()),
        "has_tests_test_patch": bool(test_patch.strip()),
        "has_solution_patch": bool(solution_patch.strip()),
        "has_solution_script": bool(solve_sh.strip()),
        "has_embedded_solution_patch": bool(embedded_solution_patch.strip()),
        "has_gold_git_diff": patch_touches_non_test(solution_patch) or patch_touches_non_test(embedded_solution_patch),
        "solution_patch_chars": len(solution_patch),
        "solution_patch_touches_non_test": patch_touches_non_test(solution_patch),
        "solution_script_chars": len(solve_sh),
        "embedded_solution_patch_chars": len(embedded_solution_patch),
        "embedded_solution_patch_touches_non_test": patch_touches_non_test(embedded_solution_patch),
        "instruction_chars": len(instruction),
        "metadata_task_id": metadata_task_id,
        "metadata_instance_id": metadata_instance_id,
        "metadata_repo": metadata_repo,
        "metadata_base_commit": parsed_metadata.get("base_commit"),
        "inferred_task_id": inferred_task_id,
        "inferred_repo": inferred_repo,
        "repository_url": repo_url,
        "metadata_json_head": metadata_json[:1000],
        "exact_current_task_overlap": inferred_task_id in current.task_ids if inferred_task_id else False,
        "repo_issue_current_overlap": bool(inferred_repo and issue and f"{inferred_repo}::{issue}" in current.repo_issue_keys),
        "test_sh_head": test_sh[:600],
        "solution_patch_head": solution_patch[:600],
        "solution_script_head": solve_sh[:600],
        "embedded_solution_patch_head": embedded_solution_patch[:600],
        "file_names_sample": names[:80],
        "top_level_files": sorted(n for n in names if "/" not in n.strip("./"))[:20],
        "interesting_file_sizes": {
            key: file_sizes.get(key) or file_sizes.get(f"./{key}")
            for key in [
                "instruction.md",
                "task.toml",
                "tests/test.sh",
                "tests/test.patch",
                "tests/test_patch.diff",
                "solution/solution.patch",
                "solution/solve.sh",
                "solution/solution.sh",
                "metadata.json",
            ]
        },
    }


def parse_json_object(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_github_repo(text: str) -> str:
    match = re.search(r"https://github\.com/([^/\s'\".]+/[^/\s'\".]+?)(?:\.git)?(?:[\s'\"]|$)", text)
    return match.group(1) if match else ""


def extract_embedded_patch(text: str) -> str:
    if "diff --git " not in text:
        return ""
    chunks = []
    for match in re.finditer(r"(?m)^diff --git ", text):
        start = match.start()
        tail = text[start:]
        marker_match = re.search(r"(?m)^(__SOLUTION__|PATCH_EOF|EOF)\s*$", tail)
        chunk = tail[: marker_match.start()] if marker_match else tail
        if chunk.strip():
            chunks.append(chunk.strip() + "\n")
    for chunk in chunks:
        if patch_touches_non_test(chunk):
            return chunk
    return chunks[0] if chunks else ""


def extract_toml_string(text: str, key: str) -> str:
    match = re.search(rf'(?m)^\s*{re.escape(key)}\s*=\s*"((?:\\.|[^"])*)"', text)
    if not match:
        return ""
    return bytes(match.group(1), "utf-8").decode("unicode_escape", errors="replace")


def choose_interesting_tasktrove_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for item in files:
        path = item["path"]
        lowered = path.lower()
        if not path.endswith(".parquet"):
            continue
        if any(token in lowered for token in ("swe", "code", "repo", "terminal", "nemotron")):
            candidates.append(item)
    return sorted(candidates, key=lambda item: (0 if "swegym" in item["path"].lower() else 1, item["path"]))[:80]


def choose_nemotron_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for item in files:
        path = item["path"]
        if not path.endswith(".tar.gz"):
            continue
        lowered = path.lower()
        rank = 10
        if "easy" in lowered:
            rank = 0
        elif "debugging" in lowered or "file_operations" in lowered:
            rank = 1
        elif "data_processing" in lowered or "scientific_computing" in lowered:
            rank = 2
        elif "medium" in lowered:
            rank = 5
        candidates.append((rank, path, item))
    return [item for _, _, item in sorted(candidates)[:12]]


def sample_harbor_tarball(
    repo_id: str,
    filename: str,
    sample_tasks: int,
    current: CurrentTaskIndex,
) -> dict[str, Any]:
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=hub_token())
    selected_roots: list[str] = []
    total_members = 0
    with tarfile.open(path, mode="r:gz") as tf:
        for member in tf:
            total_members += 1
            clean_name = member.name.strip("./")
            if clean_name.endswith("/instruction.md") or clean_name == "instruction.md":
                root = clean_name.removesuffix("/instruction.md")
                if clean_name == "instruction.md":
                    root = ""
                selected_roots.append(root)
                if len(selected_roots) >= sample_tasks:
                    break

    samples = []
    with tarfile.open(path, mode="r:gz") as tf:
        members = tf.getmembers()
        for root in selected_roots:
            names: list[str] = []
            text_files: dict[str, str] = {}
            file_sizes: dict[str, int] = {}
            prefix = f"{root}/" if root else ""
            for member in members:
                clean_name = member.name.strip("./")
                if prefix:
                    if not clean_name.startswith(prefix):
                        continue
                    relative = clean_name[len(prefix):]
                else:
                    relative = clean_name
                if not relative:
                    continue
                names.append(relative)
                file_sizes[relative] = member.size
                if not member.isfile() or not is_interesting_task_file(relative):
                    continue
                extracted = tf.extractfile(member)
                if extracted is not None:
                    text_files[relative] = extracted.read().decode("utf-8", errors="replace")
            inspected = inspect_task_files(root, names, text_files, file_sizes, current)
            inspected["archive_root"] = root
            samples.append(inspected)

    return {
        "filename": filename,
        "local_path": path,
        "sample_task_count": len(samples),
        "members_until_samples_found": total_members,
        "sample_task_summaries": samples,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--swe-rebench-max-rows", type=int, default=0)
    parser.add_argument("--tasktrove-sample-rows", type=int, default=3)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/wbl-fast/usrs/ee/code-swe-data/cache/hf")
    load_env_file(args.env_file)
    current = current_task_index()
    api = HfApi(token=hub_token())
    report: dict[str, Any] = {
        "current_dataset": {
            "index_path": str(CURRENT_INDEX),
            "unique_task_ids": len(current.task_ids),
            "unique_repos": len(current.repos),
            "repo_issue_keys": len(current.repo_issue_keys),
        },
        "sources": {},
    }

    for repo_id in [
        "nebius/SWE-rebench",
        "nvidia/Nemotron-Terminal-Synthetic-Tasks",
        "open-thoughts/TaskTrove",
    ]:
        source = {
            "file_summary": file_summary(api, repo_id),
            "split_names": split_names(repo_id),
        }
        report["sources"][repo_id] = source
        write_json(args.output_dir / f"{repo_id.replace('/', '__')}.files.json", source["file_summary"])

    swe_source = report["sources"]["nebius/SWE-rebench"]
    swe_source["stream_sample_errors"] = {}
    for split in ("test", "filtered"):
        rows, error = take_stream_rows("nebius/SWE-rebench", split=split, n=8)
        if error:
            swe_source["stream_sample_errors"][split] = error
        else:
            swe_source[f"{split}_sample_summary"] = summarize_swe_rebench_rows(rows, current)
    if not args.sample_only:
        for split in ("test", "filtered"):
            try:
                swe_source[f"{split}_full_scan"] = scan_stream_swe_rebench(
                    "nebius/SWE-rebench", split, current, args.swe_rebench_max_rows
                )
            except Exception as exc:  # noqa: BLE001 - record exact failure
                swe_source[f"{split}_full_scan_error"] = f"{type(exc).__name__}: {exc}"

    nemotron = report["sources"]["nvidia/Nemotron-Terminal-Synthetic-Tasks"]
    nemotron["interesting_files"] = choose_nemotron_files(nemotron["file_summary"]["all_files"])
    nemotron_samples = []
    for item in nemotron["interesting_files"][:5]:
        try:
            nemotron_samples.append(
                sample_harbor_tarball(
                    "nvidia/Nemotron-Terminal-Synthetic-Tasks",
                    item["path"],
                    args.tasktrove_sample_rows,
                    current,
                )
            )
        except Exception as exc:  # noqa: BLE001
            nemotron_samples.append({"filename": item["path"], "error": f"{type(exc).__name__}: {exc}"})
    nemotron["tarball_samples"] = nemotron_samples

    tasktrove = report["sources"]["open-thoughts/TaskTrove"]
    tasktrove["interesting_files"] = choose_interesting_tasktrove_files(
        tasktrove["file_summary"]["all_files"]
    )
    # Prefer SWE-specific Harbor binaries first, then a small terminal/code sample.
    preferred = []
    for item in tasktrove["interesting_files"]:
        lowered = item["path"].lower()
        if "swegym" in lowered or "swe" in lowered:
            preferred.append(item)
    for item in tasktrove["interesting_files"]:
        if item not in preferred:
            preferred.append(item)
    tasktrove_samples = []
    tasktrove_overlap_scans = []
    for item in preferred[:10]:
        try:
            tasktrove_samples.append(
                sample_tasktrove_parquet(
                    "open-thoughts/TaskTrove",
                    item["path"],
                    args.tasktrove_sample_rows,
                    current,
                )
            )
            tasktrove_overlap_scans.append(
                scan_parquet_path_overlap(
                    "open-thoughts/TaskTrove",
                    item["path"],
                    current,
                )
            )
        except Exception as exc:  # noqa: BLE001
            tasktrove_samples.append({"filename": item["path"], "error": f"{type(exc).__name__}: {exc}"})
    tasktrove["parquet_samples"] = tasktrove_samples
    tasktrove["path_overlap_scans"] = tasktrove_overlap_scans

    write_json(args.output_dir / "task_source_research_report.json", report)
    print(json.dumps({
        "output": str(args.output_dir / "task_source_research_report.json"),
        "sources": list(report["sources"]),
    }, indent=2))


if __name__ == "__main__":
    main()
