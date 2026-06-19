#!/usr/bin/env python3
"""Select targeted tasks for the next SWE trace-generation round.

The selector is intentionally metadata-first.  It reads local metadata indexes
from an existing raw trace dataset, computes strict/pass/duplicate coverage,
then emits a planning JSONL plus a runnable packed Pyxis TSV only for rows that
already have a local task directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BASE_RAW_DATASET = Path(
    "/wbl-fast/usrs/ee/code-swe-data/runtime/hf_upload/"
    "swerebench-traces-raw-source-plus-all-local-generated-plus-other-sources-exact-20260616-0745"
)
DEFAULT_LANGUAGES = ("c", "cpp", "php", "java", "rust", "go", "ts", "js")
SWE_REBENCH_V2 = "nebius/SWE-rebench-V2"
SWE_REBENCH_SPLIT = "train"
STRICT_REASONING_THRESHOLD = 0.9
PATCH_BYTES_SOFT_CAP = 12_000
TRAJECTORY_CHARS_SOFT_CAP = 800_000
MODEL = "local qwen3.6-35b-a3b-fp8"
MODEL_SETTINGS = {
    "litellm_model": "openai/qwen3.6-35b-a3b-fp8",
    "api_key_env": "OPENAI_API_KEY",
    "api_base": "http://l40s-8gpu-dy-l40s-8gpu-cr-0-2.integrated.pcluster:20010/v1",
    "extra_body_json": json.dumps(
        {"chat_template_kwargs": {"enable_thinking": True}},
        separators=(",", ":"),
    ),
}
PACKED_TSV_FIELDS = [
    "index",
    "rollout_id",
    "instance_id",
    "task_dir",
    "workspace",
    "image",
    "model",
    "litellm_model",
    "api_key_env",
    "api_base",
    "extra_body_json",
    "difficulty",
    "language",
    "instruction_style",
    "repo",
    "outside_original_high_quality_set",
]


@dataclass
class ExistingStats:
    raw_rows: int = 0
    pass_rows: int = 0
    strict_rows: int = 0
    raw_by_language: Counter[str] = field(default_factory=Counter)
    pass_by_language: Counter[str] = field(default_factory=Counter)
    strict_by_language: Counter[str] = field(default_factory=Counter)
    raw_by_task: Counter[str] = field(default_factory=Counter)
    pass_by_task: Counter[str] = field(default_factory=Counter)
    strict_by_task: Counter[str] = field(default_factory=Counter)
    strict_by_task_language: Counter[tuple[str, str]] = field(default_factory=Counter)
    raw_task_ids: set[str] = field(default_factory=set)
    strict_task_ids: set[str] = field(default_factory=set)
    index_paths: list[str] = field(default_factory=list)
    strict_patch_bytes: list[int] = field(default_factory=list)
    strict_trajectory_chars: list[int] = field(default_factory=list)
    drop_reasons: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-raw-dataset", type=Path, default=DEFAULT_BASE_RAW_DATASET)
    parser.add_argument("--max-swerebench-v2", type=int, default=200)
    parser.add_argument("--max-other-source", type=int, default=200)
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated target languages, ordered only for tie-breaking.",
    )
    parser.add_argument("--seed", type=int, default=260616)
    parser.add_argument(
        "--swerebench-v2-scan-limit",
        type=int,
        default=0,
        help="Optional cap on streamed SWE-rebench V2 rows; 0 means no explicit cap.",
    )
    parser.add_argument(
        "--skip-swerebench-v2",
        action="store_true",
        help="Use only local metadata-derived candidates; useful offline.",
    )
    return parser.parse_args()


def normalize_language(value: Any) -> str:
    language = str(value or "").strip().lower()
    aliases = {
        "javascript": "js",
        "typescript": "ts",
        "c++": "cpp",
        "csharp": "cs",
    }
    return aliases.get(language, language)


def parse_languages(value: str) -> list[str]:
    languages = [normalize_language(part) for part in re.split(r"[,\s]+", value) if part.strip()]
    if not languages:
        raise SystemExit("--languages must include at least one language")
    return list(dict.fromkeys(languages))


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "passed", "submitted"}
    return False


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def intish(value: Any, default: int = 0) -> int:
    return int(numeric(value, float(default)))


def task_slug(instance_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", instance_id).strip("-").lower()


def safe_model(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", model).strip("-")


def stable_random(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def metadata_index_paths(base_raw_dataset: Path) -> list[Path]:
    metadata_dir = base_raw_dataset / "metadata"
    preferred = [
        metadata_dir / "parent_index.jsonl",
        metadata_dir / "appended_index.jsonl",
        metadata_dir / "other_sources_exact_index.jsonl",
    ]
    paths = [path for path in preferred if path.exists()]
    if paths:
        return paths
    return sorted(metadata_dir.glob("*index.jsonl"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def task_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("instance_id") or "").strip()


def is_pass_row(row: dict[str, Any]) -> bool:
    return boolish(row.get("passed")) and numeric(row.get("reward"), 0.0) > 0.0


def strict_drop_reason(row: dict[str, Any]) -> str:
    if not is_pass_row(row):
        return "not_passed"
    if str(row.get("agent_exit_status") or "").strip() != "Submitted":
        return "not_submitted"
    if intish(row.get("model_patch_bytes")) <= 0:
        return "empty_patch"
    percent = numeric(row.get("percent_messages_with_reasoning"), 0.0)
    if percent < STRICT_REASONING_THRESHOLD and not boolish(row.get("has_all_assistant_reasoning")):
        return "low_reasoning_coverage"
    if intish(row.get("api_calls")) <= 0:
        return "api_calls_zero"
    return ""


def is_strict_row(row: dict[str, Any]) -> bool:
    return strict_drop_reason(row) == ""


def load_existing_stats(base_raw_dataset: Path) -> tuple[ExistingStats, list[dict[str, Any]]]:
    stats = ExistingStats()
    rows: list[dict[str, Any]] = []
    for path in metadata_index_paths(base_raw_dataset):
        stats.index_paths.append(str(path))
        source_index = path.name.removesuffix(".jsonl")
        for row in iter_jsonl(path):
            row = dict(row)
            row["_source_index"] = source_index
            rows.append(row)
            tid = task_id(row)
            language = normalize_language(row.get("language"))
            if not tid:
                continue
            stats.raw_rows += 1
            stats.raw_task_ids.add(tid)
            stats.raw_by_task[tid] += 1
            if language:
                stats.raw_by_language[language] += 1
            if is_pass_row(row):
                stats.pass_rows += 1
                stats.pass_by_task[tid] += 1
                if language:
                    stats.pass_by_language[language] += 1
            reason = strict_drop_reason(row)
            if reason:
                stats.drop_reasons[reason] += 1
            else:
                stats.strict_rows += 1
                stats.strict_task_ids.add(tid)
                stats.strict_by_task[tid] += 1
                if language:
                    stats.strict_by_language[language] += 1
                    stats.strict_by_task_language[(tid, language)] += 1
                patch_bytes = intish(row.get("model_patch_bytes"))
                trajectory_chars = intish(row.get("trajectory_chars"))
                if patch_bytes > 0:
                    stats.strict_patch_bytes.append(patch_bytes)
                if trajectory_chars > 0:
                    stats.strict_trajectory_chars.append(trajectory_chars)
    return stats, rows


def percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def summarize_counts(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def source_priority(source: str) -> int:
    if source == "other_sources_exact_rejected_candidates":
        return 0
    if source == "appended_index":
        return 1
    if source == "other_sources_exact_index":
        return 2
    return 3


def infer_task_dir(row: dict[str, Any]) -> Path | None:
    explicit = row.get("task_dir")
    if explicit:
        path = Path(str(explicit))
        if path.exists():
            return path.resolve()

    source_run_root = row.get("source_run_root")
    tid = task_id(row)
    if not source_run_root or not tid:
        return None
    root = Path(str(source_run_root))
    style = str(row.get("instruction_style") or "").strip()
    slug = task_slug(tid)
    candidates = []
    if style:
        candidates.extend([root / f"tasks-{style}" / slug, root / "tasks" / slug])
    candidates.extend(
        [
            root / "tasks-original" / slug,
            root / "tasks-deepswe" / slug,
            root / "tasks-full" / slug,
            root / "tasks" / slug,
        ]
    )
    for candidate in candidates:
        if (candidate / "task.toml").exists():
            return candidate.resolve()
    for task_root in sorted(root.glob("tasks-*")):
        candidate = task_root / slug
        if (candidate / "task.toml").exists():
            return candidate.resolve()
    return None


def read_task_image(task_dir: Path | None) -> str:
    if not task_dir:
        return ""
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return ""
    try:
        with task_toml.open("rb") as handle:
            data = tomllib.load(handle)
        return str(data.get("environment", {}).get("docker_image") or "")
    except (tomllib.TOMLDecodeError, OSError):
        text = task_toml.read_text(encoding="utf-8", errors="replace")
        match = re.search(r'(?m)^docker_image\s*=\s*"((?:\\.|[^"])*)"', text)
        if not match:
            return ""
        return bytes(match.group(1), "utf-8").decode("unicode_escape", errors="replace")


def candidate_base(
    *,
    row: dict[str, Any],
    source_dataset: str,
    source_status: str,
    selection_reason: str,
    stats: ExistingStats,
    seed: int,
) -> dict[str, Any]:
    tid = task_id(row)
    language = normalize_language(row.get("language"))
    difficulty = str(row.get("difficulty") or row.get("quality_tier") or "").strip().lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "easy" if source_dataset != SWE_REBENCH_V2 else ""
    patch_bytes = intish(row.get("model_patch_bytes") or row.get("patch_bytes") or row.get("patch_len"))
    trajectory_chars = intish(row.get("trajectory_chars"))
    strict_task_count = stats.strict_by_task.get(tid, 0)
    pass_task_count = stats.pass_by_task.get(tid, 0)
    return {
        "task_id": tid,
        "instance_id": tid,
        "repo": str(row.get("repo") or ""),
        "language": language,
        "difficulty": difficulty,
        "source_dataset": source_dataset,
        "source_index": row.get("_source_index", ""),
        "source_status": source_status,
        "selection_reason": selection_reason,
        "strict_language_count": int(stats.strict_by_language.get(language, 0)),
        "pass_language_count": int(stats.pass_by_language.get(language, 0)),
        "raw_language_count": int(stats.raw_by_language.get(language, 0)),
        "strict_task_count": int(strict_task_count),
        "pass_task_count": int(pass_task_count),
        "raw_task_count": int(stats.raw_by_task.get(tid, 0)),
        "duplicate_risk": "high" if strict_task_count >= 3 else "medium" if strict_task_count else "low",
        "patch_bytes": patch_bytes,
        "trajectory_chars": trajectory_chars,
        "agent_exit_status": row.get("agent_exit_status", ""),
        "reward": row.get("reward", ""),
        "passed": boolish(row.get("passed")),
        "teacher": row.get("teacher", ""),
        "source_run_root": row.get("source_run_root", ""),
        "prompt_chars": intish(row.get("prompt_chars")),
        "small_patch_preference": patch_bytes == 0 or patch_bytes <= PATCH_BYTES_SOFT_CAP,
        "compact_trace_preference": trajectory_chars == 0 or trajectory_chars <= TRAJECTORY_CHARS_SOFT_CAP,
        "selection_jitter": stable_random(seed, f"{source_dataset}:{tid}:{row.get('line_number', '')}"),
    }


def candidate_sort_key(candidate: dict[str, Any], language_order: dict[str, int]) -> tuple[Any, ...]:
    difficulty = candidate.get("difficulty") or ""
    patch_bytes = intish(candidate.get("patch_bytes"))
    trajectory_chars = intish(candidate.get("trajectory_chars"))
    return (
        int(candidate.get("strict_language_count") or 0),
        int(candidate.get("strict_task_count") or 0),
        source_priority(str(candidate.get("source_index") or candidate.get("source_dataset") or "")),
        difficulty != "easy",
        difficulty not in {"easy", "medium"},
        patch_bytes == 0,
        patch_bytes if patch_bytes else 10**12,
        trajectory_chars == 0,
        trajectory_chars if trajectory_chars else 10**12,
        language_order.get(str(candidate.get("language") or ""), 999),
        float(candidate.get("selection_jitter") or 0.0),
        candidate.get("task_id") or "",
    )


def choose_balanced(
    candidates: list[dict[str, Any]],
    limit: int,
    languages: list[str],
    language_order: dict[str, int],
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        buckets[str(candidate.get("language") or "")].append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda candidate: candidate_sort_key(candidate, language_order))

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    ordered_languages = sorted(
        languages,
        key=lambda language: (
            buckets[language][0].get("strict_language_count", 10**12) if buckets.get(language) else 10**12,
            language_order.get(language, 999),
        ),
    )
    while len(selected) < limit:
        progressed = False
        for language in ordered_languages:
            bucket = buckets.get(language)
            while bucket:
                candidate = bucket.pop(0)
                key = f"{candidate.get('source_dataset')}::{candidate.get('task_id')}"
                if key in seen:
                    continue
                seen.add(key)
                selected.append(candidate)
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    for index, candidate in enumerate(selected):
        candidate["selection_rank"] = index
    return selected


def local_other_source_candidates(
    rows: list[dict[str, Any]],
    stats: ExistingStats,
    languages: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    target_languages = set(languages)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        tid = task_id(row)
        language = normalize_language(row.get("language"))
        if not tid or language not in target_languages:
            continue
        if is_strict_row(row):
            continue
        if stats.strict_by_task.get(tid, 0) >= 2:
            continue
        difficulty = str(row.get("difficulty") or "").lower()
        if difficulty and difficulty not in {"easy", "medium"}:
            continue
        source_index = str(row.get("_source_index") or "")
        if source_index not in {"appended_index", "other_sources_exact_index"}:
            continue
        task_dir = infer_task_dir(row)
        image = read_task_image(task_dir)
        source_status = "ready_existing_task_dir" if task_dir and image else "planned_missing_task_dir_or_image"
        candidate = candidate_base(
            row=row,
            source_dataset="local_raw_rerun",
            source_status=source_status,
            selection_reason="rerun_non_strict_underrepresented_language_small_patch_preferred",
            stats=stats,
            seed=seed,
        )
        candidate.update(
            {
                "task_dir": str(task_dir) if task_dir else "",
                "image": image,
                "strict_drop_reason": strict_drop_reason(row),
                "line_number": row.get("line_number", ""),
                "appended_line_number": row.get("appended_line_number", ""),
            }
        )
        candidates.append(candidate)
    return candidates


def rejected_other_source_candidates(
    base_raw_dataset: Path,
    stats: ExistingStats,
    languages: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    path = base_raw_dataset / "metadata" / "other_sources_exact_rejected_candidates.jsonl"
    if not path.exists():
        return []
    target_languages = set(languages)
    candidates: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        tid = task_id(row)
        language = normalize_language(row.get("language"))
        if not tid or language not in target_languages:
            continue
        if stats.strict_by_task.get(tid, 0) >= 2:
            continue
        candidate = candidate_base(
            row={**row, "_source_index": "other_sources_exact_rejected_candidates"},
            source_dataset="other_sources_exact_rejected_candidates",
            source_status="planned_regenerate_previous_strict_audit_rejected",
            selection_reason="regenerate_with_real_unified_diff_clean_apply_allowed_paths",
            stats=stats,
            seed=seed,
        )
        candidate.update(
            {
                "workspace": row.get("workspace", ""),
                "strict_audit_path": row.get("strict_audit_path", ""),
                "reject_reasons": row.get("reject_reasons", []),
                "task_dir": "",
                "image": "",
            }
        )
        candidates.append(candidate)
    return candidates


def gate_failures(metadata: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if metadata.get("code") not in (None, "", "A"):
        failures.append("code")
    if metadata.get("intent_completeness") not in (None, "", "complete"):
        failures.append("intent")
    if metadata.get("test_alignment_issues"):
        failures.append("test_alignment")
    if any((metadata.get("detected_issues") or {}).values()):
        failures.append("detected_issues")
    return failures


def patch_size_from_row(row: dict[str, Any]) -> int:
    for key in ("patch", "gold_patch", "test_patch"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return len(value.encode("utf-8"))
    return 0


def load_swerebench_v2_candidates(
    stats: ExistingStats,
    languages: list[str],
    seed: int,
    scan_limit: int,
) -> tuple[list[dict[str, Any]], str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        return [], f"datasets import failed: {exc}"

    target_languages = set(languages)
    candidates: list[dict[str, Any]] = []
    scanned = 0
    try:
        dataset = load_dataset(SWE_REBENCH_V2, split=SWE_REBENCH_SPLIT, streaming=True)
        for row in dataset:
            scanned += 1
            if scan_limit and scanned > scan_limit:
                break
            tid = task_id(row)
            language = normalize_language(row.get("language"))
            if not tid or language not in target_languages:
                continue
            if tid in stats.raw_task_ids:
                continue
            metadata = ((row.get("meta") or {}).get("llm_metadata") or {}) if isinstance(row.get("meta"), dict) else {}
            difficulty = str(metadata.get("difficulty") or row.get("difficulty") or "").lower()
            if difficulty not in {"easy", "medium"}:
                continue
            failures = gate_failures(metadata)
            synthetic_row = {
                "task_id": tid,
                "repo": row.get("repo", ""),
                "language": language,
                "difficulty": difficulty,
                "patch_bytes": patch_size_from_row(row),
                "_source_index": "swerebench_v2_hf_stream",
            }
            candidate = candidate_base(
                row=synthetic_row,
                source_dataset=SWE_REBENCH_V2,
                source_status="planned_hf_metadata_only",
                selection_reason=(
                    "new_swerebench_v2_easy_medium_strict_metadata_gate"
                    if not failures
                    else "new_swerebench_v2_easy_medium_gate_relaxed"
                ),
                stats=stats,
                seed=seed,
            )
            candidate.update(
                {
                    "confidence": metadata.get("confidence"),
                    "quality_gate_pass": not failures,
                    "quality_gate_failures": failures,
                    "annotation_code": metadata.get("code", ""),
                    "intent_completeness": metadata.get("intent_completeness", ""),
                    "task_dir": "",
                    "image": str(row.get("image_name") or ""),
                }
            )
            candidates.append(candidate)
    except Exception as exc:  # noqa: BLE001 - keep CLI useful offline.
        return candidates, f"load_dataset failed after scanning {scanned} rows: {exc}"
    return candidates, ""


def build_packed_records(selected: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in selected:
        task_dir = str(candidate.get("task_dir") or "")
        image = str(candidate.get("image") or "")
        if candidate.get("source_status") != "ready_existing_task_dir" or not task_dir or not image:
            continue
        index = len(records)
        instruction_style = str(candidate.get("instruction_style") or "original")
        instance_id = str(candidate["instance_id"])
        workspace = (
            output_dir.resolve()
            / "pyxis-traces"
            / instruction_style
            / safe_model(MODEL)
            / "r00"
            / task_slug(instance_id)
        )
        records.append(
            {
                "index": index,
                "rollout_id": "r00",
                "instance_id": instance_id,
                "task_dir": task_dir,
                "workspace": str(workspace),
                "image": image,
                "model": MODEL,
                "litellm_model": MODEL_SETTINGS["litellm_model"],
                "api_key_env": MODEL_SETTINGS["api_key_env"],
                "api_base": MODEL_SETTINGS["api_base"],
                "extra_body_json": MODEL_SETTINGS["extra_body_json"],
                "difficulty": candidate.get("difficulty") or "easy",
                "language": candidate.get("language") or "",
                "instruction_style": instruction_style,
                "repo": candidate.get("repo") or "",
                "outside_original_high_quality_set": "true",
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def write_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            values = []
            for field in PACKED_TSV_FIELDS:
                value = str(record.get(field, ""))
                values.append(value.replace("\t", " ").replace("\r", " ").replace("\n", " "))
            handle.write("\t".join(values) + "\n")


def main() -> None:
    args = parse_args()
    languages = parse_languages(args.languages)
    language_order = {language: index for index, language in enumerate(languages)}
    random.seed(args.seed)

    if not args.base_raw_dataset.exists():
        raise SystemExit(f"base raw dataset does not exist: {args.base_raw_dataset}")

    stats, rows = load_existing_stats(args.base_raw_dataset)
    other_candidates = local_other_source_candidates(rows, stats, languages, args.seed)
    other_candidates.extend(rejected_other_source_candidates(args.base_raw_dataset, stats, languages, args.seed))
    other_selected = choose_balanced(other_candidates, args.max_other_source, languages, language_order)

    swerebench_error = ""
    swerebench_candidates: list[dict[str, Any]] = []
    if not args.skip_swerebench_v2 and args.max_swerebench_v2 > 0:
        swerebench_candidates, swerebench_error = load_swerebench_v2_candidates(
            stats,
            languages,
            args.seed,
            args.swerebench_v2_scan_limit,
        )
    swerebench_selected = choose_balanced(
        swerebench_candidates,
        args.max_swerebench_v2,
        languages,
        language_order,
    )

    selected = other_selected + swerebench_selected
    for rank, record in enumerate(selected):
        record["selection_rank"] = rank

    packed_records = build_packed_records(selected, args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_jsonl = args.output_dir / "source_selection_manifest.jsonl"
    summary_json = args.output_dir / "source_selection_summary.json"
    manifest_tsv = args.output_dir / "source_selection_manifest.tsv"
    write_jsonl(manifest_jsonl, selected)
    write_tsv(manifest_tsv, packed_records)

    duplicate_counts = Counter(stats.strict_by_task.values())
    strict_duplicate_tasks = sum(1 for count in stats.strict_by_task.values() if count > 1)
    summary = {
        "base_raw_dataset": str(args.base_raw_dataset),
        "metadata_index_paths": stats.index_paths,
        "languages": languages,
        "seed": args.seed,
        "strict_filter": {
            "passed": True,
            "reward_positive": True,
            "agent_exit_status": "Submitted",
            "model_patch_bytes_gt": 0,
            "reasoning_threshold": STRICT_REASONING_THRESHOLD,
            "api_calls_gt": 0,
        },
        "existing": {
            "raw_rows": stats.raw_rows,
            "pass_rows": stats.pass_rows,
            "strict_rows": stats.strict_rows,
            "raw_by_language": summarize_counts(stats.raw_by_language),
            "pass_by_language": summarize_counts(stats.pass_by_language),
            "strict_by_language": summarize_counts(stats.strict_by_language),
            "strict_unique_tasks": len(stats.strict_task_ids),
            "strict_duplicate_tasks": strict_duplicate_tasks,
            "strict_duplicate_count_histogram": summarize_counts(duplicate_counts),
            "strict_drop_reasons": summarize_counts(stats.drop_reasons),
            "strict_patch_bytes": {
                "median": int(statistics.median(stats.strict_patch_bytes)) if stats.strict_patch_bytes else None,
                "p90": percentile(stats.strict_patch_bytes, 0.90),
                "p99": percentile(stats.strict_patch_bytes, 0.99),
                "max": max(stats.strict_patch_bytes) if stats.strict_patch_bytes else None,
            },
            "strict_trajectory_chars": {
                "median": int(statistics.median(stats.strict_trajectory_chars))
                if stats.strict_trajectory_chars
                else None,
                "p90": percentile(stats.strict_trajectory_chars, 0.90),
                "p99": percentile(stats.strict_trajectory_chars, 0.99),
                "max": max(stats.strict_trajectory_chars) if stats.strict_trajectory_chars else None,
            },
        },
        "candidates": {
            "other_source_considered": len(other_candidates),
            "other_source_selected": len(other_selected),
            "swerebench_v2_considered": len(swerebench_candidates),
            "swerebench_v2_selected": len(swerebench_selected),
            "swerebench_v2_error": swerebench_error,
        },
        "selected": {
            "total": len(selected),
            "by_language": summarize_counts(Counter(record.get("language", "") for record in selected)),
            "by_source_dataset": summarize_counts(Counter(record.get("source_dataset", "") for record in selected)),
            "by_source_status": summarize_counts(Counter(record.get("source_status", "") for record in selected)),
            "ready_packed_tsv_rows": len(packed_records),
        },
        "outputs": {
            "source_selection_manifest_jsonl": str(manifest_jsonl),
            "source_selection_summary_json": str(summary_json),
            "source_selection_manifest_tsv": str(manifest_tsv),
            "packed_tsv_fields": PACKED_TSV_FIELDS,
            "packed_tsv_note": "Headerless TSV; only rows with ready_existing_task_dir and docker image are included.",
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
