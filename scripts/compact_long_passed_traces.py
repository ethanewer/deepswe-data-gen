#!/usr/bin/env python3
"""Prepare or run long passed-trace compaction requests.

The script only reads source metadata indexes and trajectory JSON files. It
writes compaction artifacts under --output-dir and never edits source datasets.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import signal
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ENV_FILE = Path("/wbl-fast/usrs/ee/code-swe-data/.env")
DEFAULT_MODEL = "xiaomi/mimo-v2.5"
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B-Thinking-2507"
COMPACTION_SCHEMA_VERSION = 2
REASONING_KEYS = {"reasoning_content", "reasoning", "reasoning_details", "thinking"}
PROMPT_PROCESS_MARKER_RE = re.compile(
    r"("
    r"create a new initial mini-swe task prompt|"
    r"compacted successful trajectory|"
    r"discarded[- ]prefix|"
    r"retained suffix|"
    r"cut turn index|"
    r"replacement user message|"
    r"replacement prompt|"
    r"original row id:|"
    r"compaction variant:|"
    r"full discarded prefix|"
    r"mini_swe_task_prompt"
    r")",
    re.IGNORECASE,
)
PROMPT_STATE_CONTEXT_RE = re.compile(
    r"("
    r"current(?:ly)?|investigat|examined|found|identified|file|test|failure|error|"
    r"command|diff|patch|implementation|behavior|module|function|class|method"
    r")",
    re.IGNORECASE,
)
PROMPT_TASK_INTENT_RE = re.compile(
    r"(fix|implement|update|change|add|remove|ensure|make|support|handle|debug|resolve|issue|bug|failing)",
    re.IGNORECASE,
)
PROMPT_CONTINUATION_MARKER_RE = re.compile(
    r"("
    r"\bcontinue\b|\balready\b|previous attempt|progress so far|what has already been done|"
    r"current repository state|already modified|already applied|fixes already applied|"
    r"has been (?:addressed|updated|changed|modified|implemented|added|removed|fixed|partially)|"
    r"have been (?:addressed|updated|changed|modified|implemented|added|removed|fixed|partially)|"
    r"was (?:addressed|updated|changed|modified|implemented|added|removed|fixed)|"
    r"were (?:addressed|updated|changed|modified|implemented|added|removed|fixed)|"
    r"implementation includes|changes? (?:made|introduced|include)|remaining (?:work|task|issue|step)|"
    r"next step|finish(?:es|ing)? (?:the )?(?:remaining )?(?:work|task)|"
    r"just verify|only need(?:s)? to verify|generate (?:the )?patch|submit (?:the )?(?:final )?(?:answer|output)|"
    r"required interface|interface notes|recommended workflow|command execution rules|finish by issuing|"
    r"complete_task_and_submit_final_output|##\s*boundaries"
    r")",
    re.IGNORECASE,
)
PROMPT_PATH_CONTRADICTION_RE = re.compile(
    r"(/testbed\s+(?:does not|doesn't)\s+exist|repo(?:sitory)?\s+(?:is\s+)?(?:located|checked out)\s+at\s+/(?!testbed\b))",
    re.IGNORECASE,
)
PROMPT_IMPERATIVE_TEST_EDIT_RE = re.compile(
    r"("
    r"(?<!non-)(?:add|create|modify|update|change|write)\s+(?:a\s+|new\s+|the\s+)?(?:unit\s+|integration\s+)?tests?\b|"
    r"(?:add|create|modify|update|change|write).{0,80}\btests?/"
    r")",
    re.IGNORECASE | re.DOTALL,
)
FIRST_ACTION_REJECT_CATEGORIES = {"edit", "test", "diff_status", "submit"}
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a noninteractive bash command in the task environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}


class SuffixSelectionError(ValueError):
    pass


class PrefixRequestTooLongError(ValueError):
    pass


def prompt_requests_test_edits(prompt: str) -> bool:
    test_edit_match = PROMPT_IMPERATIVE_TEST_EDIT_RE.search(prompt)
    if not test_edit_match:
        return False
    negative_test_boundary = re.search(
        r"do not (?:add|create|modify|update|change|write) (?:a |new |the )?(?:unit |integration )?tests?|"
        r"\bnon-test (?:source )?files?\b",
        prompt,
        re.IGNORECASE,
    )
    return negative_test_boundary is None


@dataclass(frozen=True)
class IndexCandidate:
    row: dict[str, Any]
    index_path: Path
    index_line_number: int
    original_row_id: str
    original_row_path: str
    trajectory_path: Path
    index_estimated_tokens: int
    index_estimation_source: str
    strict_source: str


@dataclass(frozen=True)
class SuffixSelection:
    cut_turn_index: int
    retained_messages: list[dict[str, Any]]
    original_length: int
    retained_length: int
    request_length: int
    prefix_messages: list[dict[str, Any]]
    model_messages: list[dict[str, str]]
    first_retained_command_category: str
    first_retained_command_preview: str
    boundary_previous_role: str
    boundary_next_role: str


@dataclass(frozen=True)
class QualitySignals:
    raw_by_uuid: dict[str, dict[str, Any]]
    aligned_by_uuid: dict[str, dict[str, Any]]
    excluded_uuids: set[str]


class LengthEstimator:
    def __init__(
        self,
        *,
        tokenizer_name: str,
        token_budget: int,
        char_budget: int | None,
        chars_per_token: float,
        no_tokenizer: bool,
    ) -> None:
        self.token_budget = token_budget
        self.chars_per_token = chars_per_token
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        self.mode = "char_fallback"
        self.detail = "tokenizer_disabled" if no_tokenizer else "tokenizer_unavailable"
        if char_budget is None:
            char_budget = int(token_budget * chars_per_token)
        self.char_budget = char_budget

        if no_tokenizer:
            return

        try:
            from transformers import AutoTokenizer  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            self.mode = "qwen3_tokenizer"
            self.detail = tokenizer_name
        except Exception as exc:  # noqa: BLE001 - fallback is an explicit feature
            self.tokenizer = None
            self.mode = "char_fallback"
            self.detail = f"{type(exc).__name__}: {exc}"

    @property
    def budget(self) -> int:
        if self.tokenizer is not None:
            return self.token_budget
        return self.char_budget

    @property
    def length_unit(self) -> str:
        if self.tokenizer is not None:
            return "tokens"
        return "chars"

    def count_text(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        if self.tokenizer is not None:
            try:
                return len(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
            except Exception:
                pass
        return self.count_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))

    def token_estimate_from_length(self, length: int) -> int:
        if self.tokenizer is not None:
            return length
        return max(1, int(round(length / self.chars_per_token)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        "--metadata-index",
        dest="indexes",
        type=Path,
        action="append",
        required=True,
        help="Metadata index JSONL. May be passed more than once.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-original-tokens",
        type=int,
        default=65_536,
        help="Minimum index-estimated original trajectory size to consider long.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum selected traces. 0 means no limit.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many sorted candidates before applying --limit.")
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=0,
        help="Maximum selected source rows per task ID within this invocation. 0 disables the cap.",
    )
    parser.add_argument(
        "--compaction-variant",
        choices=("reasoning_stripped", "reasoning_retained"),
        default="reasoning_stripped",
    )
    parser.add_argument(
        "--qwen3-token-budget",
        type=int,
        default=55_000,
        help="Maximum retained suffix size in Qwen3 tokens.",
    )
    parser.add_argument(
        "--model-request-token-budget",
        type=int,
        default=60_000,
        help="Maximum compaction model request size in Qwen3 tokens. The request contains the discarded prefix.",
    )
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER)
    parser.add_argument("--no-tokenizer", action="store_true")
    parser.add_argument(
        "--char-budget",
        type=int,
        help="Fallback retained suffix budget in characters. Defaults to qwen3-token-budget * chars-per-token.",
    )
    parser.add_argument(
        "--model-request-char-budget",
        type=int,
        help="Fallback compaction request budget in characters. Defaults to model-request-token-budget * chars-per-token.",
    )
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument(
        "--include-unknown-length",
        action="store_true",
        help="Select rows with no index length estimate; trajectory is read only after selection.",
    )
    parser.add_argument(
        "--max-trajectory-chars",
        type=int,
        default=5_000_000,
        help="Skip index rows with trajectory_chars/trajectory_bytes above this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--raw-quality-signals",
        type=Path,
        action="append",
        default=[],
        help="raw_index_row_quality_signals.jsonl path. May be supplied more than once.",
    )
    parser.add_argument(
        "--aligned-quality-signals",
        type=Path,
        action="append",
        default=[],
        help="aligned_passed_row_quality_signals.jsonl path. May be supplied more than once.",
    )
    parser.add_argument(
        "--require-quality-signal",
        action="store_true",
        help="Skip rows without a matching raw or aligned quality-signal UUID.",
    )
    parser.add_argument(
        "--require-strict-trainable-submit",
        action="store_true",
        help="Require aligned trainable_submit_patch_available_strict=true.",
    )
    parser.add_argument(
        "--require-good-submit",
        action="store_true",
        help="Require aligned good_submission_behavior_available=true.",
    )
    parser.add_argument(
        "--exclude-manual-patch-targets",
        action="store_true",
        help="Require aligned manual_patch_target_turns=0 when aligned quality is available.",
    )
    parser.add_argument(
        "--exclude-huge-patch",
        action="store_true",
        help="Require raw huge_patch_flag=false when raw quality is available.",
    )
    parser.add_argument(
        "--exclude-task-overrepresented",
        action="store_true",
        help="Require raw task_overrepresented_strict=false when raw quality is available.",
    )
    parser.add_argument(
        "--language",
        action="append",
        default=[],
        help="Restrict candidates to this language label. May be supplied more than once.",
    )
    parser.add_argument(
        "--exclude-uuid-file",
        type=Path,
        action="append",
        default=[],
        help="Text file containing source row UUIDs/original row IDs to skip. May be supplied more than once.",
    )
    parser.add_argument(
        "--longest-first",
        action="store_true",
        help="Select largest qualifying rows first. Default selects shortest qualifying long rows first.",
    )
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--extra-body-json",
        help="JSON object to pass as provider-specific extra_body on chat completion calls.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument(
        "--request-timeout-seconds",
        "--model-timeout-seconds",
        dest="request_timeout_seconds",
        type=float,
        default=120.0,
        help="Hard timeout for each compaction model attempt.",
    )
    parser.add_argument(
        "--prefix-head-messages",
        type=int,
        default=4,
        help="Always include this many earliest discarded-prefix messages in model requests.",
    )
    parser.add_argument(
        "--max-prefix-message-chars",
        type=int,
        default=24_000,
        help="Middle-truncate each discarded-prefix message to this size before request fitting.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--max-json-repair-retries",
        type=int,
        default=1,
        help="If the model returns invalid JSON, ask the model to repair the response this many times.",
    )
    parser.add_argument(
        "--first-action-policy",
        choices=("any", "prefer_inspect", "require_inspect"),
        default="prefer_inspect",
        help=(
            "Boundary policy for the first retained assistant action. prefer_inspect chooses the first inspect/read/search "
            "boundary when available and otherwise avoids edit/test/diff/submit starts. require_inspect rejects rows "
            "without an inspect/read/search retained start."
        ),
    )
    parser.add_argument(
        "--allow-user-boundary",
        action="store_true",
        help="Allow retained suffixes whose previous source message is user. Default requires a tool->assistant boundary.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON in {path}:{line_number}: {exc}", file=sys.stderr)
                continue
            if isinstance(row, dict):
                yield line_number, row


def load_quality_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Quality signals file does not exist: {path}")
        for _, row in iter_jsonl(path):
            uuid = first_nonempty(row.get("uuid"))
            if uuid:
                rows[uuid] = row
    return rows


def load_excluded_uuids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Exclude UUID file does not exist: {path}")
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                text = raw.strip()
                if text and not text.startswith("#"):
                    excluded.add(text)
    return excluded


def load_quality_signals(args: argparse.Namespace) -> QualitySignals:
    return QualitySignals(
        raw_by_uuid=load_quality_rows(args.raw_quality_signals),
        aligned_by_uuid=load_quality_rows(args.aligned_quality_signals),
        excluded_uuids=load_excluded_uuids(args.exclude_uuid_file),
    )


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "passed", "success", "submitted"}
    return False


def int_from_any(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def float_from_any(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


def nested_values(row: dict[str, Any], keys: Iterable[str]) -> Iterable[tuple[str, Any]]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in keys:
        if key in row:
            yield key, row.get(key)
        if key in metadata:
            yield f"metadata.{key}", metadata.get(key)


def strict_pass_source(row: dict[str, Any]) -> str | None:
    reward = next((value for _, value in nested_values(row, ("reward",))), None)
    reward_int = int_from_any(reward)
    if reward_int != 1:
        return None

    passed_values = list(nested_values(row, ("passed", "strict_audit_accepted", "accepted")))
    passed_fields = [(name, value) for name, value in passed_values if value is not None]
    if passed_fields and not any(bool_from_any(value) for _, value in passed_fields):
        return None

    strict_fields = (
        "benchmark_profile",
        "instruction_style",
        "mini_swe_agent_config_file",
        "source_manifest",
        "row_source",
        "strict_audit_path",
    )
    for name, value in nested_values(row, strict_fields):
        if "strict" in str(value).lower():
            return f"{name}=strict"

    for name, value in nested_values(row, ("strict_audit_accepted", "accepted")):
        if bool_from_any(value):
            return f"{name}=true"

    return "reward_1_passed"


def index_length_estimate(row: dict[str, Any], chars_per_token: float) -> tuple[int | None, str]:
    token_fields = (
        "trajectory_tokens",
        "message_tokens",
        "messages_tokens",
        "token_count",
        "tokens",
        "total_tokens",
        "num_tokens",
        "estimated_tokens",
        "estimated_token_count",
    )
    for name, value in nested_values(row, token_fields):
        parsed = int_from_any(value)
        if parsed and parsed > 0:
            return parsed, name

    char_fields = (
        "trajectory_chars",
        "message_chars",
        "messages_chars",
        "input_chars",
        "trajectory_bytes",
    )
    for name, value in nested_values(row, char_fields):
        parsed = int_from_any(value)
        if parsed and parsed > 0:
            return max(1, int(round(parsed / chars_per_token))), f"{name}/chars_per_token"

    return None, "missing"


def index_char_length(row: dict[str, Any]) -> int | None:
    for _, value in nested_values(row, ("trajectory_chars", "trajectory_bytes")):
        parsed = int_from_any(value)
        if parsed and parsed > 0:
            return parsed
    return None


def resolve_path(raw_path: str, index_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    for base in (index_path.parent, Path.cwd()):
        candidate = base / path
        if candidate.exists():
            return candidate
    return index_path.parent / path


def resolve_optional_path(raw_path: Any, index_path: Path) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path)
    if not text:
        return None
    path = resolve_path(text, index_path)
    return path


def read_text_if_small(path: Path | None, max_bytes: int = 8_000_000) -> str:
    if path is None:
        return ""
    try:
        if path.exists() and path.stat().st_size <= max_bytes:
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return ""


def trajectory_path_for_row(row: dict[str, Any], index_path: Path) -> Path | None:
    for _, value in nested_values(row, ("trajectory_path", "trajectory_json", "trajectory_file")):
        if value:
            return resolve_path(str(value), index_path)
    return None


def original_row_id(row: dict[str, Any], index_line_number: int) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    task_id = first_nonempty(row.get("task_id"), row.get("instance_id"), metadata.get("task_id"), metadata.get("instance_id"))
    rollout_id = first_nonempty(row.get("rollout_id"), metadata.get("rollout_id"))
    return first_nonempty(row.get("uuid"), row.get("id"), metadata.get("uuid"), f"{task_id}:{rollout_id}" if task_id else "", index_line_number)


def original_row_path(row: dict[str, Any], trajectory_path: Path) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return first_nonempty(
        row.get("row_path"),
        row.get("source_path"),
        row.get("path"),
        metadata.get("row_path"),
        metadata.get("source_path"),
        metadata.get("path"),
        trajectory_path,
    )


def quality_for_row(row: dict[str, Any], quality: QualitySignals) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    uuid = first_nonempty(row.get("uuid"), row.get("id"), metadata.get("uuid"))
    return quality.raw_by_uuid.get(uuid), quality.aligned_by_uuid.get(uuid)


def quality_row_value(row: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    return row.get(key, default)


def passes_quality_filters(
    row: dict[str, Any],
    args: argparse.Namespace,
    quality: QualitySignals,
    stats: Counter[str],
) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    uuid = first_nonempty(row.get("uuid"), row.get("id"), metadata.get("uuid"))
    if uuid and uuid in quality.excluded_uuids:
        stats["skipped_excluded_uuid"] += 1
        return False
    allowed_languages = {str(language).lower() for language in args.language if str(language)}
    language = first_nonempty(row.get("language"), metadata.get("language")).lower()
    if allowed_languages and language not in allowed_languages:
        stats["skipped_language_filter"] += 1
        return False

    raw_quality, aligned_quality = quality_for_row(row, quality)
    if args.require_quality_signal and raw_quality is None and aligned_quality is None:
        stats["skipped_missing_quality_signal"] += 1
        return False
    if args.require_strict_trainable_submit and not bool_from_any(
        quality_row_value(aligned_quality, "trainable_submit_patch_available_strict", False)
    ):
        stats["skipped_not_strict_trainable_submit"] += 1
        return False
    if args.require_good_submit and not bool_from_any(
        quality_row_value(aligned_quality, "good_submission_behavior_available", False)
    ):
        stats["skipped_not_good_submit"] += 1
        return False
    if args.exclude_manual_patch_targets and int_from_any(
        quality_row_value(aligned_quality, "manual_patch_target_turns", 0)
    ):
        stats["skipped_manual_patch_targets"] += 1
        return False
    if args.exclude_huge_patch and bool_from_any(quality_row_value(raw_quality, "huge_patch_flag", False)):
        stats["skipped_huge_patch"] += 1
        return False
    if args.exclude_task_overrepresented and bool_from_any(
        quality_row_value(raw_quality, "task_overrepresented_strict", False)
    ):
        stats["skipped_task_overrepresented"] += 1
        return False
    return True


def select_candidates(args: argparse.Namespace) -> tuple[list[IndexCandidate], Counter[str]]:
    candidates: list[IndexCandidate] = []
    stats: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    quality = load_quality_signals(args)
    for index_path in args.indexes:
        for line_number, row in iter_jsonl(index_path):
            stats["index_rows"] += 1
            if not passes_quality_filters(row, args, quality, stats):
                continue
            strict_source = strict_pass_source(row)
            if strict_source is None:
                stats["skipped_not_strict_passed"] += 1
                continue

            estimate, estimate_source = index_length_estimate(row, args.chars_per_token)
            if estimate is None and not args.include_unknown_length:
                stats["skipped_missing_index_length"] += 1
                continue
            if estimate is not None and estimate < args.min_original_tokens:
                stats["skipped_not_long"] += 1
                continue
            char_length = index_char_length(row)
            if args.max_trajectory_chars and char_length and char_length > args.max_trajectory_chars:
                stats["skipped_too_large"] += 1
                continue

            trajectory_path = trajectory_path_for_row(row, index_path)
            if trajectory_path is None:
                stats["skipped_missing_trajectory_path"] += 1
                continue

            provenance_index_path = index_path
            provenance_line_number = line_number
            source_index_path_text = first_nonempty(row.get("source_index_path"), row.get("original_index_path"))
            if source_index_path_text:
                provenance_index_path = Path(source_index_path_text)
            source_line = int_from_any(first_nonempty(row.get("source_index_line_number"), row.get("original_index_line_number")))
            if source_line:
                provenance_line_number = source_line

            row_id = original_row_id(row, line_number)
            key = (row_id, str(trajectory_path))
            if key in seen:
                stats["skipped_duplicate"] += 1
                continue
            seen.add(key)
            candidates.append(
                IndexCandidate(
                    row=row,
                    index_path=provenance_index_path,
                    index_line_number=provenance_line_number,
                    original_row_id=row_id,
                    original_row_path=original_row_path(row, trajectory_path),
                    trajectory_path=trajectory_path,
                    index_estimated_tokens=estimate or 0,
                    index_estimation_source=estimate_source,
                    strict_source=strict_source,
                )
            )
            stats["candidate_rows"] += 1

    if args.longest_first:
        candidates.sort(key=lambda item: (-item.index_estimated_tokens, str(item.index_path), item.index_line_number))
    else:
        candidates.sort(key=lambda item: (item.index_estimated_tokens, str(item.index_path), item.index_line_number))
    if args.max_per_task and args.max_per_task > 0:
        capped: list[IndexCandidate] = []
        task_counts: Counter[str] = Counter()
        for candidate in candidates:
            metadata = candidate.row.get("metadata") if isinstance(candidate.row.get("metadata"), dict) else {}
            task_id = first_nonempty(candidate.row.get("task_id"), candidate.row.get("instance_id"), metadata.get("task_id"), metadata.get("instance_id"))
            key = task_id or candidate.original_row_id
            if task_counts[key] >= args.max_per_task:
                stats["skipped_max_per_task"] += 1
                continue
            task_counts[key] += 1
            capped.append(candidate)
        candidates = capped
    if args.offset and args.offset > 0:
        candidates = candidates[args.offset :]
        stats["offset_rows"] = args.offset
    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]
    stats["selected_rows"] = len(candidates)
    return candidates, stats


def extract_messages(trajectory: Any) -> list[dict[str, Any]]:
    if isinstance(trajectory, dict):
        for key in ("messages", "trajectory", "conversation"):
            value = trajectory.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    if isinstance(trajectory, list):
        return [item for item in trajectory if isinstance(item, dict)]
    return []


def strip_reasoning(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in REASONING_KEYS:
                continue
            if key == "provider_specific_fields" and isinstance(item, dict):
                stripped_provider = {k: strip_reasoning(v) for k, v in item.items() if k not in REASONING_KEYS}
                if stripped_provider:
                    out[key] = stripped_provider
                continue
            out[key] = strip_reasoning(item)
        return out
    if isinstance(value, list):
        return [strip_reasoning(item) for item in value]
    return value


def messages_for_variant(messages: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    cloned = copy.deepcopy(messages)
    if variant == "reasoning_stripped":
        return [strip_reasoning(message) for message in cloned]
    return cloned


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def safe_id_fragment(text: str, max_chars: int = 48) -> str:
    fragment = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._")
    return (fragment or "run")[:max_chars]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = f"\n...[truncated {len(text) - max_chars} chars]...\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def truncate_message(message: dict[str, Any], max_chars: int) -> dict[str, Any]:
    cloned = copy.deepcopy(message)
    content = cloned.get("content")
    if isinstance(content, str):
        truncated = truncate_middle(content, max_chars)
        if truncated != content:
            cloned["content"] = truncated
            cloned["content_truncated_from_chars"] = len(content)
    elif content is not None:
        serialized = stable_json(content)
        cloned["content"] = truncate_middle(serialized, max_chars)
        cloned["content_serialized_for_compaction"] = True
        if len(serialized) > max_chars:
            cloned["content_truncated_from_chars"] = len(serialized)
    return cloned


def truncate_messages(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    return [truncate_message(message, max_chars) for message in messages]


def model_request_budget(args: argparse.Namespace, estimator: LengthEstimator) -> int:
    if estimator.tokenizer is not None:
        return args.model_request_token_budget
    if args.model_request_char_budget is not None:
        return args.model_request_char_budget
    return int(args.model_request_token_budget * args.chars_per_token)


def parse_model_json_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = text.strip()
    if not stripped:
        return None, "empty_response"
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None, "no_json_object_found"
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return None, f"json_decode_error:{exc}"
    if not isinstance(parsed, dict):
        return None, f"json_root_not_object:{type(parsed).__name__}"
    required = {
        "mini_swe_task_prompt",
        "original_goal",
        "current_repo_state",
        "actions_already_taken",
        "uncertainty_notes",
        "compaction_model_reasoning",
    }
    missing = sorted(required.difference(parsed))
    if missing:
        return parsed, "missing_required_keys:" + ",".join(missing)
    prompt = parsed.get("mini_swe_task_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return parsed, "empty_mini_swe_task_prompt"
    if PROMPT_PROCESS_MARKER_RE.search(prompt):
        return parsed, "mini_swe_task_prompt_contains_compaction_process_marker"
    stripped_prompt = prompt.strip()
    if len(stripped_prompt) < 250:
        return parsed, "mini_swe_task_prompt_too_short"
    if not PROMPT_TASK_INTENT_RE.search(stripped_prompt):
        return parsed, "mini_swe_task_prompt_missing_task_intent"
    if PROMPT_CONTINUATION_MARKER_RE.search(stripped_prompt):
        return parsed, "mini_swe_task_prompt_contains_continuation_marker"
    if PROMPT_PATH_CONTRADICTION_RE.search(stripped_prompt):
        return parsed, "mini_swe_task_prompt_contains_path_contradiction"
    if prompt_requests_test_edits(stripped_prompt):
        return parsed, "mini_swe_task_prompt_requests_test_edits"
    reasoning = parsed.get("compaction_model_reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return parsed, "empty_compaction_model_reasoning"
    return parsed, None


def context_anchor_terms(
    *,
    candidate: IndexCandidate,
    original_user_prompt: str,
    prefix_messages: list[dict[str, Any]],
    max_terms: int = 80,
) -> list[str]:
    context = candidate_context(candidate)
    raw_text = original_user_prompt + "\n" + "\n".join(text_from_content(message.get("content")) for message in prefix_messages[-8:])
    candidates: list[str] = []
    for value in (context.get("task_id"), context.get("repo")):
        if isinstance(value, str):
            candidates.extend(part for part in re.split(r"[^A-Za-z0-9_.-]+", value) if len(part) >= 4)
    candidates.extend(re.findall(r"`([^`]{4,120})`", raw_text))
    candidates.extend(
        re.findall(
            r"\b[A-Za-z0-9_./-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|php|c|cc|cpp|cxx|h|hpp|rb|sh|md|json|yaml|yml|toml)\b",
            raw_text,
        )
    )
    candidates.extend(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b", raw_text))
    seen: set[str] = set()
    out: list[str] = []
    for term in candidates:
        cleaned = str(term).strip().strip("'\".,:;()[]{}")
        if len(cleaned) < 4 or len(cleaned) > 120:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= max_terms:
            break
    return out


def validate_prompt_faithfulness(
    *,
    parsed: dict[str, Any],
    candidate: IndexCandidate,
    original_user_prompt: str,
    prefix_messages: list[dict[str, Any]],
) -> str | None:
    prompt = str(parsed.get("mini_swe_task_prompt") or "")
    if not prompt.strip():
        return "faithfulness_empty_prompt"
    anchors = context_anchor_terms(
        candidate=candidate,
        original_user_prompt=original_user_prompt,
        prefix_messages=prefix_messages,
    )
    lower_prompt = prompt.lower()
    matched = [term for term in anchors if term.lower() in lower_prompt]
    if anchors and not matched:
        return "faithfulness_no_context_anchor_in_prompt"
    if re.search(r"\b(src/main\.c|include/config\.h|undefined reference|main application module|missing configuration files)\b", prompt, re.IGNORECASE):
        context_text = (original_user_prompt + "\n" + stable_json(prefix_messages[-8:])).lower()
        if not any(token in context_text for token in ("src/main.c", "include/config.h", "undefined reference", "main application module", "missing configuration files")):
            return "faithfulness_generic_build_hallucination"
    return None


class RequestTimeoutError(TimeoutError):
    pass


@contextmanager
def request_timeout(seconds: float):
    if seconds <= 0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def handler(signum: int, frame: Any) -> None:
        raise RequestTimeoutError(f"model request exceeded {seconds:.1f}s")

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def message_role(message: dict[str, Any]) -> str:
    return str(message.get("role") or "").lower()


def safe_suffix_starts(messages: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, message in enumerate(messages)
        if index > 0 and message_role(message) == "assistant"
    ]


def command_from_assistant(message: dict[str, Any] | None) -> str:
    if not isinstance(message, dict):
        return ""
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError:
                    return arguments
                if isinstance(parsed, dict) and isinstance(parsed.get("command"), str):
                    return parsed["command"]
            if isinstance(arguments, dict) and isinstance(arguments.get("command"), str):
                return arguments["command"]
    content = text_from_content(message.get("content"))
    if "complete_task_and_submit_final_output" in content.lower():
        return "complete_task_and_submit_final_output"
    return content.strip()


def command_category(command: str) -> str:
    stripped = command.strip()
    lower = stripped.lower()
    if not lower:
        return "no_command"
    if "complete_task_and_submit_final_output" in lower:
        return "submit"
    if re.search(r"(^|\s)(pytest|tox|npm test|pnpm test|yarn test|go test|cargo test|mvn test|gradle test|make test|ctest|rspec|phpunit)\b", lower):
        return "test"
    if re.search(r"(^|\s)(apply_patch|sed -i|perl -pi|patch\b|cat\s+>|tee\s+|printf\s+.*>)", lower):
        return "edit"
    if re.search(r"(^|\s)(python[0-9.]*|node|ruby|php|perl|go run|cargo run)\s+(- <<|<<|'|\"|-c)", lower) and re.search(r"(>|write_text|open\(.+['\"]w|sed -i)", lower):
        return "edit"
    if re.search(r"(^|\s)git\s+(diff|status|show)\b", lower):
        return "diff_status"
    if re.search(r"(^|\s)(rg|grep|find|ls|pwd|sed -n|cat|head|tail|awk|tree)\b", lower):
        return "inspect"
    if re.search(r"(^|\s)(python[0-9.]*|node|ruby|php|perl|go run|cargo run)\s+(- <<|<<|'|\"|-c)", lower):
        return "probe"
    if re.search(r"(^|\s)(pip install|npm install|pnpm install|yarn install|go mod|cargo fetch|composer install)\b", lower):
        return "setup"
    return "other"


def first_safe_suffix_start_at_or_after(
    messages: list[dict[str, Any]],
    start: int,
    *,
    first_action_policy: str,
    allow_user_boundary: bool,
) -> int | None:
    candidates: list[int] = []
    for candidate in safe_suffix_starts(messages):
        if candidate < start:
            continue
        previous_role = message_role(messages[candidate - 1]) if candidate > 0 else ""
        if previous_role != "tool" and not allow_user_boundary:
            continue
        candidates.append(candidate)
    if not candidates:
        return None
    if first_action_policy == "any":
        return candidates[0]
    inspect_candidates = [
        candidate
        for candidate in candidates
        if command_category(command_from_assistant(messages[candidate])) == "inspect"
    ]
    if inspect_candidates:
        return inspect_candidates[0]
    if first_action_policy == "require_inspect":
        return None
    non_rejected = [
        candidate
        for candidate in candidates
        if command_category(command_from_assistant(messages[candidate])) not in FIRST_ACTION_REJECT_CATEGORIES
    ]
    if non_rejected:
        return non_rejected[0]
    return None


def text_from_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(text_from_content(item.get("text", item.get("content", item))))
            else:
                parts.append(text_from_content(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value:
                return text_from_content(value[key])
    return str(value)


def message_reasoning_content(message: dict[str, Any]) -> list[str]:
    contents: list[str] = []
    candidates: list[Any] = []
    for key in ("reasoning_content", "reasoning", "reasoning_details", "thinking", "thought"):
        candidates.append(message.get(key))
    provider = message.get("provider_specific_fields")
    if isinstance(provider, dict):
        for key in ("reasoning_content", "reasoning", "reasoning_details", "thinking", "thought"):
            candidates.append(provider.get(key))
    extra = message.get("extra")
    response = extra.get("response") if isinstance(extra, dict) else None
    choices = response.get("choices") if isinstance(response, dict) else None
    if isinstance(choices, list):
        for choice in choices:
            choice_message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(choice_message, dict):
                for key in ("reasoning_content", "reasoning", "reasoning_details", "thinking", "thought"):
                    candidates.append(choice_message.get(key))
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type", "")).lower()
            if part_type in {"thinking", "reasoning"}:
                candidates.append(part.get("text", part.get("content", part.get("thinking", part.get("reasoning")))))
    for value in candidates:
        text = text_from_content(value).strip()
        if text:
            contents.append(text)
    return contents


def collect_message_reasoning(messages: list[Any]) -> list[dict[str, Any]]:
    reasoning: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        for content in message_reasoning_content(message):
            key = (index, content)
            if key in seen:
                continue
            seen.add(key)
            reasoning.append({"message_index": index, "path": f"messages[{index}]", "content": content})
    return reasoning


def reasoning_metrics(messages: list[Any], reasoning: list[dict[str, Any]]) -> dict[str, Any]:
    top_indices = {
        item.get("message_index")
        for item in reasoning
        if isinstance(item.get("message_index"), int) and str(item.get("content") or "").strip()
    }
    assistant_count = 0
    with_reasoning = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message_role(message) != "assistant":
            continue
        assistant_count += 1
        if index in top_indices or any(message_reasoning_content(message)):
            with_reasoning += 1
    return {
        "assistant_message_count": assistant_count,
        "assistant_messages_with_reasoning": with_reasoning,
        "assistant_messages_without_reasoning": assistant_count - with_reasoning,
        "percent_messages_with_reasoning": (with_reasoning / assistant_count) if assistant_count else 0.0,
    }


def extract_submit_marker(messages: list[Any]) -> bool:
    marker = "complete_task_and_submit_final_output"
    for message in messages:
        if not isinstance(message, dict) or message_role(message) != "assistant":
            continue
        if marker in text_from_content(message.get("content")).lower():
            return True
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if marker in json.dumps(call, ensure_ascii=False, default=str).lower():
                    return True
    return False


def leading_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message_role(message)
        if role == "system":
            out.append(copy.deepcopy(message))
            continue
        if role == "user":
            break
        break
    return out


def prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message_role(message) == "user":
            return text_from_content(message.get("content"))
    return ""


def candidate_context(candidate: IndexCandidate) -> dict[str, Any]:
    metadata = candidate.row.get("metadata") if isinstance(candidate.row.get("metadata"), dict) else {}
    return {
        "task_id": first_nonempty(candidate.row.get("task_id"), candidate.row.get("instance_id"), metadata.get("task_id")),
        "repo": first_nonempty(candidate.row.get("repo"), metadata.get("repo")),
        "language": first_nonempty(candidate.row.get("language"), metadata.get("language")),
        "difficulty": first_nonempty(candidate.row.get("difficulty"), metadata.get("difficulty")),
        "teacher": first_nonempty(candidate.row.get("teacher"), candidate.row.get("model"), metadata.get("teacher")),
        "rollout_id": first_nonempty(candidate.row.get("rollout_id"), metadata.get("rollout_id")),
        "result_path": first_nonempty(candidate.row.get("result_path"), metadata.get("result_path")),
        "patch_path": first_nonempty(candidate.row.get("patch_path"), metadata.get("patch_path")),
    }


def request_user_content(
    *,
    candidate: IndexCandidate,
    original_row_id: str,
    cut_turn_index: int,
    original_message_count: int,
    prefix_messages: list[dict[str, Any]],
    original_user_prompt: str,
    retained_message_count: int,
    retained_estimated_length: int,
    retained_estimated_token_length: int,
    estimator: LengthEstimator,
    variant: str,
) -> str:
    prefix_json = json.dumps(prefix_messages, ensure_ascii=False, sort_keys=True, indent=2)
    context = candidate_context(candidate)
    original_prompt = original_user_prompt or prompt_from_messages(prefix_messages)
    omitted_prefix_message_count = max(0, cut_turn_index - len(prefix_messages))
    truncated_prefix_message_count = sum(
        1 for message in prefix_messages if isinstance(message, dict) and message.get("content_truncated_from_chars")
    )
    prefix_scope = (
        "The JSON below is the included discarded-prefix excerpt. It contains the full discarded prefix."
        if omitted_prefix_message_count == 0
        else (
            "The JSON below is an included discarded-prefix excerpt. It preserves the earliest prefix messages and "
            "the latest prefix messages that fit the request budget; some middle prefix messages are omitted."
        )
    )
    return (
        "Create a new initial Mini-SWE task prompt for a compacted successful trajectory.\n\n"
        f"Original row id: {original_row_id}\n"
        f"Task id: {context['task_id']}\n"
        f"Repo: {context['repo']}\n"
        f"Language: {context['language']}\n"
        f"Difficulty: {context['difficulty']}\n"
        f"Teacher: {context['teacher']}\n"
        f"Rollout id: {context['rollout_id']}\n"
        f"Compaction variant: {variant}\n"
        f"Cut turn index: {cut_turn_index}\n"
        f"Original message count: {original_message_count}\n"
        f"Full discarded prefix message count: {cut_turn_index}\n"
        f"Included discarded prefix message count: {len(prefix_messages)}\n"
        f"Omitted middle discarded prefix message count: {omitted_prefix_message_count}\n"
        f"Truncated included prefix message count: {truncated_prefix_message_count}\n"
        f"Retained suffix message count: {retained_message_count}\n"
        f"Retained suffix estimated length: {retained_estimated_length} {estimator.length_unit}\n"
        f"Retained suffix estimated Qwen3 tokens: {retained_estimated_token_length}\n\n"
        f"{prefix_scope} The retained suffix starts immediately after the full discarded prefix and will be "
        "appended after the new user prompt you write. Do not request more information and do not summarize the "
        "retained suffix; you are not receiving it. Use the discarded prefix to reconstruct a standalone initial "
        "task prompt that states the original goal plus the current repository state, actions already taken, "
        "files touched, commands run, failures observed, and any constraints needed for the following retained "
        "assistant turn to continue naturally. If content is omitted or truncated, do not invent exact details; "
        "include any uncertainty in uncertainty_notes while still writing a useful replacement prompt.\n\n"
        "Return only strict JSON, with no markdown fence or prose before or after it. Use keys "
        "mini_swe_task_prompt, original_goal, current_repo_state, "
        "actions_already_taken, failures_observed, files_touched, uncertainty_notes, compaction_model_reasoning. "
        "The mini_swe_task_prompt value must be the exact replacement user message for the compacted row. Write it "
        "as a natural Mini-SWE benchmark task against the current checkout: first state the issue or requested "
        "source-code change, then add a short natural context paragraph only if needed for continuity. The prompt "
        "must not read like a resumed session or progress report. Put actions already taken in the JSON metadata "
        "fields, not in the replacement prompt. If the checkout contains partial work, describe it as current code "
        "or current behavior, for example `The current implementation in FILE ...`, not as something already done "
        "by a previous agent. Do not use progress-report headers such as "
        "Progress So Far, Current Repository State, Required Interface, or Interface Notes. Do not say please "
        "continue, continue by, previous attempt, already, has been updated, has been addressed, remaining work, "
        "next step, just verify, generate a patch, or submit. "
        "Do not include Mini-SWE harness scaffolding such as Boundaries, Recommended Workflow, Command Execution "
        "Rules, bash tool rules, or the final submit command; the chat template supplies those separately. Do not "
        "instruct the agent to add, modify, or update tests. Assume the repository is the normal Mini-SWE "
        "working tree; do not claim that /testbed does not exist or that the repo lives at another absolute path. "
        "Keep mini_swe_task_prompt under 7000 characters unless essential continuity details require more. The "
        "mini_swe_task_prompt must read like a normal benchmark task prompt and must not mention this compaction "
        "pipeline, discarded prefixes, retained suffixes, cut turns, trajectories, replacement prompts, or this "
        "JSON conversion process. If the benchmark task itself is about a domain concept called compaction, keep "
        "that domain wording.\n\n"
        "Original user prompt excerpt from the prefix:\n"
        f"{original_prompt[:8000]}\n\n"
        "Discarded prefix messages JSON:\n"
        f"{prefix_json}"
    )


def build_model_messages(
    *,
    candidate: IndexCandidate,
    original_row_id: str,
    cut_turn_index: int,
    original_message_count: int,
    prefix_messages: list[dict[str, Any]],
    original_user_prompt: str,
    retained_message_count: int,
    retained_estimated_length: int,
    retained_estimated_token_length: int,
    estimator: LengthEstimator,
    variant: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You compact long successful SWE agent trajectories by reading the discarded prefix and writing "
                "a replacement Mini-SWE initial task prompt. Be faithful, concrete, and explicit about uncertainty "
                "in the JSON metadata. The replacement prompt itself must read like an ordinary source-code issue "
                "from the benchmark, not like a progress report or resumed-session note, and must not reveal that "
                "this compaction pipeline, trajectory editing, or prefix/suffix reconstruction happened."
            ),
        },
        {
            "role": "user",
            "content": request_user_content(
                candidate=candidate,
                original_row_id=original_row_id,
                cut_turn_index=cut_turn_index,
                original_message_count=original_message_count,
                prefix_messages=prefix_messages,
                original_user_prompt=original_user_prompt,
                retained_message_count=retained_message_count,
                retained_estimated_length=retained_estimated_length,
                retained_estimated_token_length=retained_estimated_token_length,
                estimator=estimator,
                variant=variant,
            ),
        },
    ]


def build_json_repair_messages(response_text: str, error: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair malformed JSON for a compaction metadata response. Return only one strict JSON object. "
                "Do not add markdown or prose."
            ),
        },
        {
            "role": "user",
            "content": (
                "The previous response could not be parsed as the required JSON object.\n"
                f"Parse error: {error}\n\n"
                "Return strict JSON with these keys: mini_swe_task_prompt, original_goal, current_repo_state, "
                "actions_already_taken, failures_observed, files_touched, uncertainty_notes, "
                "compaction_model_reasoning.\n"
                "The mini_swe_task_prompt must read like an ordinary benchmark task prompt and must not mention "
                "this compaction pipeline, discarded prefixes, retained suffixes, cut turns, trajectories, "
                "replacement prompts, or JSON repair. It must not say please continue, continue by, previous "
                "attempt, already, has been updated, has been addressed, remaining work, next step, just verify, "
                "generate a patch, submit, Current Repository State, "
                "Progress So Far, Required Interface, Interface Notes, Boundaries, Recommended Workflow, Command "
                "Execution Rules, or the final submit command. It must not instruct the agent to add, modify, or "
                "update tests. If the task itself is about a domain concept called "
                "compaction, keep that domain wording. The compaction_model_reasoning field must be non-empty and "
                "should explain the faithfulness/continuity choices made while constructing the replacement "
                "prompt.\n\n"
                "Previous response:\n"
                f"{response_text}"
            ),
        },
    ]


def compacted_messages(
    *,
    original_messages: list[dict[str, Any]],
    retained_messages: list[dict[str, Any]],
    replacement_prompt: str,
) -> list[dict[str, Any]]:
    messages = leading_system_messages(original_messages)
    messages.append({"role": "user", "content": replacement_prompt})
    messages.extend(copy.deepcopy(retained_messages))
    return messages


def patch_text_for_candidate(candidate: IndexCandidate, trajectory: Any) -> tuple[str, str]:
    patch_path_text = first_nonempty(candidate.row.get("patch_path"), candidate.row.get("model_patch_path"))
    patch_path = resolve_optional_path(patch_path_text, candidate.index_path) if patch_path_text else None
    text = read_text_if_small(patch_path)
    if text:
        return text, str(patch_path)

    result_path_text = first_nonempty(candidate.row.get("result_path"))
    result_path = resolve_optional_path(result_path_text, candidate.index_path) if result_path_text else None
    result = load_json(result_path) if result_path is not None and result_path.exists() else {}
    if isinstance(result, dict):
        for key in ("model_patch", "patch", "submission"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value, patch_path_text
        result_patch = resolve_optional_path(
            first_nonempty(result.get("patch_path"), result.get("model_patch_path")),
            candidate.index_path,
        )
        text = read_text_if_small(result_patch)
        if text:
            return text, str(result_patch)

    workspace = candidate.trajectory_path.parent.parent if candidate.trajectory_path.parent.name == "agent" else candidate.trajectory_path.parent
    for candidate_path in (
        workspace / "model.patch",
        workspace / "logs" / "artifacts" / "model.patch",
        workspace / "patch.txt",
    ):
        text = read_text_if_small(candidate_path)
        if text:
            return text, str(candidate_path)

    if isinstance(trajectory, dict):
        info = trajectory.get("info") if isinstance(trajectory.get("info"), dict) else {}
        submission = info.get("submission")
        if isinstance(submission, str) and submission.strip():
            return submission, patch_path_text
    return "", patch_path_text


def row_field(candidate: IndexCandidate, name: str, default: Any = "") -> Any:
    metadata = candidate.row.get("metadata") if isinstance(candidate.row.get("metadata"), dict) else {}
    if name in candidate.row and candidate.row.get(name) is not None:
        return candidate.row.get(name)
    if name in metadata and metadata.get(name) is not None:
        return metadata.get(name)
    return default


def build_compacted_trajectory(
    *,
    original_trajectory: Any,
    messages: list[dict[str, Any]],
    request_id: str,
    parsed_model_response: dict[str, Any],
    source_trajectory_path: Path,
) -> dict[str, Any]:
    info = copy.deepcopy(original_trajectory.get("info")) if isinstance(original_trajectory, dict) else {}
    if not isinstance(info, dict):
        info = {}
    return {
        "trajectory_format": first_nonempty(
            original_trajectory.get("trajectory_format") if isinstance(original_trajectory, dict) else "",
            "mini-swe-agent-1.1",
        ),
        "info": info,
        "messages": messages,
    }


def write_compacted_trajectory(
    *,
    output_dir: Path,
    request_id: str,
    trajectory: dict[str, Any],
) -> Path:
    path = output_dir / "compacted_trajectories" / request_id / "agent" / "mini-swe-agent.trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_compacted_raw_record(
    *,
    candidate: IndexCandidate,
    original_trajectory: Any,
    original_messages: list[dict[str, Any]],
    compacted_messages_value: list[dict[str, Any]],
    compacted_trajectory_path: Path,
    request_id: str,
    parsed_model_response: dict[str, Any],
    cut_turn_index: int,
    retained_message_count: int,
    original_estimated_tokens: int,
    compacted_estimated_length: int,
    compacted_estimated_tokens: int,
    request_estimated_tokens: int,
    estimator: LengthEstimator,
    variant: str,
    compaction_model: str,
    patch_text: str,
    patch_path_text: str,
    source_trajectory_sha256: str,
    full_prefix_sha256: str,
    retained_suffix_sha256: str,
    boundary_previous_message_sha256: str,
    boundary_next_message_sha256: str,
    boundary_previous_role: str,
    boundary_next_role: str,
    first_retained_command_category: str,
    first_retained_command_preview: str,
) -> dict[str, Any]:
    reasoning = collect_message_reasoning(compacted_messages_value)
    metrics = reasoning_metrics(compacted_messages_value, reasoning)
    replacement_prompt = str(parsed_model_response.get("mini_swe_task_prompt") or "")
    compaction_model_reasoning = str(parsed_model_response.get("compaction_model_reasoning") or "").strip()
    task_id = first_nonempty(row_field(candidate, "task_id"), row_field(candidate, "instance_id"))
    teacher = first_nonempty(row_field(candidate, "teacher"), row_field(candidate, "model"))
    reward = int_from_any(row_field(candidate, "reward", 1)) or 0
    passed = bool_from_any(row_field(candidate, "passed", reward == 1))
    agent_exit_status = first_nonempty(row_field(candidate, "agent_exit_status"), "Submitted" if passed else "")
    model_stats = original_trajectory.get("info", {}).get("model_stats", {}) if isinstance(original_trajectory, dict) else {}
    api_calls = int_from_any(row_field(candidate, "api_calls", model_stats.get("api_calls", 0))) or 0
    cost_usd = float_from_any(row_field(candidate, "cost_usd", model_stats.get("instance_cost", 0.0)))
    compacted_messages_json = json.dumps(compacted_messages_value, ensure_ascii=False, separators=(",", ":"))
    trajectory_bytes = len(
        json.dumps(
            build_compacted_trajectory(
                original_trajectory=original_trajectory,
                messages=compacted_messages_value,
                request_id=request_id,
                parsed_model_response=parsed_model_response,
                source_trajectory_path=candidate.trajectory_path,
            ),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    uuid = sha256_text(stable_json([candidate.original_row_id, request_id, cut_turn_index, replacement_prompt]))
    metadata = {
        "dataset": "",
        "source": "deepswe-compaction",
        "row_source": "compaction_prefix_v2",
        "compaction_schema_version": COMPACTION_SCHEMA_VERSION,
        "compaction_request_id": request_id,
        "compaction_variant": variant,
        "compaction_model": compaction_model,
        "compaction_original_row_id": candidate.original_row_id,
        "compaction_original_row_path": candidate.original_row_path,
        "compaction_original_index_path": str(candidate.index_path),
        "compaction_original_index_line_number": candidate.index_line_number,
        "source_trajectory_sha256": source_trajectory_sha256,
        "full_prefix_sha256": full_prefix_sha256,
        "retained_suffix_sha256": retained_suffix_sha256,
        "boundary_previous_message_sha256": boundary_previous_message_sha256,
        "boundary_next_message_sha256": boundary_next_message_sha256,
        "boundary_previous_role": boundary_previous_role,
        "boundary_next_role": boundary_next_role,
        "first_retained_command_category": first_retained_command_category,
        "first_retained_command_preview": first_retained_command_preview,
        "source_trajectory_path": str(candidate.trajectory_path),
        "trajectory_path": str(compacted_trajectory_path),
        "result_path": first_nonempty(row_field(candidate, "result_path")),
        "patch_path": patch_path_text,
        "task_id": task_id,
        "instance_id": task_id,
        "repo": first_nonempty(row_field(candidate, "repo")),
        "difficulty": first_nonempty(row_field(candidate, "difficulty")),
        "language": first_nonempty(row_field(candidate, "language")),
        "instruction_style": first_nonempty(row_field(candidate, "instruction_style")),
        "benchmark_profile": first_nonempty(row_field(candidate, "benchmark_profile")),
        "teacher": teacher,
        "model": teacher,
        "passed": passed,
        "reward": reward,
        "agent_exit_status": agent_exit_status,
        "api_calls": api_calls,
        "cost_usd": cost_usd,
        "message_count": len(compacted_messages_value),
        "assistant_message_count": metrics["assistant_message_count"],
        "assistant_messages_with_reasoning": metrics["assistant_messages_with_reasoning"],
        "assistant_messages_without_reasoning": metrics["assistant_messages_without_reasoning"],
        "percent_messages_with_reasoning": metrics["percent_messages_with_reasoning"],
        "has_any_reasoning": metrics["assistant_messages_with_reasoning"] > 0,
        "has_all_assistant_reasoning": metrics["assistant_message_count"] > 0
        and metrics["assistant_messages_with_reasoning"] == metrics["assistant_message_count"],
        "reasoning_turns": len(reasoning),
        "reasoning_chars": sum(len(str(item.get("content", ""))) for item in reasoning),
        "trajectory_chars": len(compacted_messages_json),
        "trajectory_bytes": trajectory_bytes,
        "original_message_count": len(original_messages),
        "original_estimated_token_length": original_estimated_tokens,
        "compacted_estimated_token_length": compacted_estimated_tokens,
        "compacted_estimated_length": compacted_estimated_length,
        "compaction_request_estimated_token_length": request_estimated_tokens,
        "cut_turn_index": cut_turn_index,
        "retained_message_count": retained_message_count,
        "length_estimator": estimator.mode,
        "length_estimator_detail": estimator.detail,
        "length_unit": estimator.length_unit,
        "prompt": replacement_prompt,
        "prompt_chars": len(replacement_prompt),
        "prompt_sha256": sha256_text(replacement_prompt) if replacement_prompt else "",
        "prompt_has_task_intent": bool(PROMPT_TASK_INTENT_RE.search(replacement_prompt)),
        "prompt_has_state_context": bool(PROMPT_STATE_CONTEXT_RE.search(replacement_prompt)),
        "prompt_contains_process_marker": bool(PROMPT_PROCESS_MARKER_RE.search(replacement_prompt)),
        "prompt_contains_continuation_marker": bool(PROMPT_CONTINUATION_MARKER_RE.search(replacement_prompt)),
        "prompt_contains_path_contradiction": bool(PROMPT_PATH_CONTRADICTION_RE.search(replacement_prompt)),
        "prompt_requests_test_edits": prompt_requests_test_edits(replacement_prompt),
        "compaction_model_reasoning": compaction_model_reasoning,
        "compaction_model_reasoning_chars": len(compaction_model_reasoning),
        "compaction_model_reasoning_sha256": sha256_text(compaction_model_reasoning) if compaction_model_reasoning else "",
        "has_compaction_model_reasoning": bool(compaction_model_reasoning),
        "has_submit_marker": extract_submit_marker(compacted_messages_value),
        "model_patch_bytes": len(patch_text.encode("utf-8")) if patch_text else int_from_any(row_field(candidate, "model_patch_bytes", 0)) or 0,
        "model_patch_sha256": sha256_text(patch_text) if patch_text else first_nonempty(row_field(candidate, "model_patch_sha256"), row_field(candidate, "code_diff_sha256")),
        "code_diff_sha256": sha256_text(patch_text) if patch_text else first_nonempty(row_field(candidate, "code_diff_sha256"), row_field(candidate, "model_patch_sha256")),
        "compaction_inherits_original_pass": True,
        "execution_validated_after_compaction": False,
    }
    record: dict[str, Any] = {
        "uuid": uuid,
        "task_id": task_id,
        "teacher": teacher,
        "reward": reward,
        "passed": passed,
        "percent_messages_with_reasoning": metrics["percent_messages_with_reasoning"],
        "deepswe_prompt_augmentation": bool_from_any(row_field(candidate, "deepswe_prompt_augmentation", False)),
        "prompt": replacement_prompt,
        "compaction_model_reasoning": compaction_model_reasoning,
        "messages": compacted_messages_value,
        "tools": [BASH_TOOL],
        "reasoning": reasoning,
        "metadata": metadata,
        "compaction_model_response": parsed_model_response,
    }
    if patch_text:
        record["model_patch"] = patch_text
    return record


def fit_prefix_model_request(
    *,
    args: argparse.Namespace,
    candidate: IndexCandidate,
    original_row_id: str,
    cut_turn_index: int,
    original_message_count: int,
    prefix_messages: list[dict[str, Any]],
    original_user_prompt: str,
    retained_messages: list[dict[str, Any]],
    retained_length: int,
    retained_estimated_tokens: int,
    estimator: LengthEstimator,
    request_budget: int,
    variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    head_count = max(0, min(args.prefix_head_messages, len(prefix_messages)))
    head_max_chars = args.max_prefix_message_chars
    best_prefix: list[dict[str, Any]] = []
    best_request: list[dict[str, str]] = []
    best_length = 0
    while True:
        prefix_head = truncate_messages(prefix_messages[:head_count], head_max_chars)
        best_prefix = prefix_head
        best_request = build_model_messages(
            candidate=candidate,
            original_row_id=original_row_id,
            cut_turn_index=cut_turn_index,
            original_message_count=original_message_count,
            prefix_messages=best_prefix,
            original_user_prompt=original_user_prompt,
            retained_message_count=len(retained_messages),
            retained_estimated_length=retained_length,
            retained_estimated_token_length=retained_estimated_tokens,
            estimator=estimator,
            variant=variant,
        )
        best_length = estimator.count_messages(best_request)
        if best_length <= request_budget:
            break
        if head_max_chars > 2_000:
            head_max_chars = max(2_000, int(head_max_chars * 0.6))
            continue
        if head_count > 1:
            head_count -= 1
            head_max_chars = args.max_prefix_message_chars
            continue
        if head_count == 1:
            head_count = 0
            head_max_chars = args.max_prefix_message_chars
            continue
        raise PrefixRequestTooLongError(
            f"prefix request with {len(best_prefix)} head messages truncated to {head_max_chars} chars each is "
            f"{best_length} {estimator.length_unit}, above model request budget {request_budget}"
        )

    prefix_tail_source = prefix_messages[head_count:]
    low = 0
    high = len(prefix_tail_source)
    while low <= high:
        tail_count = (low + high) // 2
        tail = truncate_messages(prefix_tail_source[-tail_count:], args.max_prefix_message_chars) if tail_count else []
        included_prefix = prefix_head + tail
        model_messages = build_model_messages(
            candidate=candidate,
            original_row_id=original_row_id,
            cut_turn_index=cut_turn_index,
            original_message_count=original_message_count,
            prefix_messages=included_prefix,
            original_user_prompt=original_user_prompt,
            retained_message_count=len(retained_messages),
            retained_estimated_length=retained_length,
            retained_estimated_token_length=retained_estimated_tokens,
            estimator=estimator,
            variant=variant,
        )
        request_length = estimator.count_messages(model_messages)
        if request_length <= request_budget:
            best_prefix = included_prefix
            best_request = model_messages
            best_length = request_length
            low = tail_count + 1
        else:
            high = tail_count - 1
    return best_prefix, best_request, best_length


def select_suffix(
    *,
    args: argparse.Namespace,
    candidate: IndexCandidate,
    messages: list[dict[str, Any]],
    estimator: LengthEstimator,
    suffix_budget: int,
    request_budget: int,
    original_row_id: str,
    variant: str,
) -> SuffixSelection:
    original_length = estimator.count_messages(messages)
    low = 0
    high = len(messages)
    best_start = len(messages)
    while low <= high:
        mid = (low + high) // 2
        suffix = messages[mid:]
        suffix_length = estimator.count_messages(suffix)
        if suffix_length <= suffix_budget:
            best_start = mid
            high = mid - 1
        else:
            low = mid + 1

    safe_start = first_safe_suffix_start_at_or_after(
        messages,
        best_start,
        first_action_policy=args.first_action_policy,
        allow_user_boundary=args.allow_user_boundary,
    )
    if safe_start is None:
        raise ValueError(
            "no retained suffix boundary fits the suffix budget and first-action policy "
            f"{args.first_action_policy}"
        )

    retained_messages = copy.deepcopy(messages[safe_start:])
    if not retained_messages:
        raise ValueError("selected retained suffix is empty")

    retained_length = estimator.count_messages(retained_messages)
    full_prefix_messages = messages[:safe_start]
    original_user_prompt = prompt_from_messages(full_prefix_messages)
    prefix_messages = messages_for_variant(full_prefix_messages, variant)
    retained_estimated_tokens = estimator.token_estimate_from_length(retained_length)
    included_prefix_messages, model_messages, request_length = fit_prefix_model_request(
        args=args,
        candidate=candidate,
        original_row_id=original_row_id,
        cut_turn_index=safe_start,
        original_message_count=len(messages),
        prefix_messages=prefix_messages,
        original_user_prompt=original_user_prompt,
        retained_messages=retained_messages,
        retained_length=retained_length,
        retained_estimated_tokens=retained_estimated_tokens,
        estimator=estimator,
        request_budget=request_budget,
        variant=variant,
    )
    first_command = command_from_assistant(retained_messages[0] if retained_messages else None)
    return SuffixSelection(
        cut_turn_index=safe_start,
        retained_messages=retained_messages,
        original_length=original_length,
        retained_length=retained_length,
        request_length=request_length,
        prefix_messages=included_prefix_messages,
        model_messages=model_messages,
        first_retained_command_category=command_category(first_command),
        first_retained_command_preview=first_command[:500],
        boundary_previous_role=message_role(messages[safe_start - 1]) if safe_start > 0 else "",
        boundary_next_role=message_role(messages[safe_start]) if safe_start < len(messages) else "",
    )


def parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    try:
        parsed = shlex.split(value, posix=True)
        if len(parsed) == 1:
            value = parsed[0]
    except ValueError:
        value = value.strip("\"'")
    return key, value


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parsed = parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)


def configure_optional_ca_bundle() -> None:
    try:
        from eval.paths import configure_ca_bundle

        configure_ca_bundle(os.environ)
    except Exception:
        return


def extra_body_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.extra_body_json:
        return None
    try:
        parsed = json.loads(args.extra_body_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --extra-body-json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--extra-body-json must decode to a JSON object")
    return parsed


def call_chat_model(args: argparse.Namespace, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    api_key = os.environ.get(args.api_key_env)
    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if not api_base and (str(args.model).startswith("openrouter/") or args.api_key_env == "OPENROUTER_API_KEY"):
        api_base = "https://openrouter.ai/api/v1"
    extra_body = extra_body_from_args(args)
    last_exc: Exception | None = None

    def call_openai_compatible(attempt: int) -> tuple[str, dict[str, Any]]:
        from openai import OpenAI  # type: ignore

        client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=api_base,
            timeout=args.request_timeout_seconds if args.request_timeout_seconds > 0 else None,
        )
        request_kwargs: dict[str, Any] = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_output_tokens,
        }
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        content = message.content
        metadata: dict[str, Any] = {"client": "openai", "attempt": attempt, "extra_body": extra_body or {}}
        message_dump = message.model_dump() if hasattr(message, "model_dump") else {}
        provider_reasoning = (
            message_dump.get("reasoning")
            or message_dump.get("reasoning_content")
            or message_dump.get("thinking")
        )
        if provider_reasoning:
            metadata["provider_reasoning"] = provider_reasoning
            metadata["provider_reasoning_chars"] = len(str(provider_reasoning))
            metadata["provider_reasoning_sha256"] = sha256_text(str(provider_reasoning))
        return content or "", metadata

    for attempt in range(1, args.max_retries + 1):
        try:
            if api_base:
                with request_timeout(args.request_timeout_seconds):
                    return call_openai_compatible(attempt)
            try:
                import litellm  # type: ignore

                kwargs: dict[str, Any] = {
                    "model": args.model,
                    "messages": messages,
                    "temperature": args.temperature,
                    "max_tokens": args.max_output_tokens,
                    "drop_params": True,
                }
                if args.request_timeout_seconds > 0:
                    kwargs["timeout"] = args.request_timeout_seconds
                if api_base:
                    kwargs["api_base"] = api_base
                if api_key:
                    kwargs["api_key"] = api_key
                if extra_body:
                    kwargs["extra_body"] = extra_body
                with request_timeout(args.request_timeout_seconds):
                    response = litellm.completion(**kwargs)
                message = response["choices"][0]["message"]
                content = message.get("content")
                metadata = {"client": "litellm", "attempt": attempt, "extra_body": extra_body or {}}
                provider_reasoning = (
                    message.get("reasoning")
                    or message.get("reasoning_content")
                    or message.get("thinking")
                )
                if provider_reasoning:
                    metadata["provider_reasoning"] = provider_reasoning
                    metadata["provider_reasoning_chars"] = len(str(provider_reasoning))
                    metadata["provider_reasoning_sha256"] = sha256_text(str(provider_reasoning))
                return content or "", metadata
            except ImportError:
                with request_timeout(args.request_timeout_seconds):
                    return call_openai_compatible(attempt)
            except Exception:
                raise
        except Exception as exc:  # noqa: BLE001 - retries must cover provider exceptions
            last_exc = exc
            if attempt == args.max_retries:
                break
            time.sleep(min(30.0, 2.0**attempt))
    assert last_exc is not None
    raise last_exc


def index_record(
    *,
    candidate: IndexCandidate,
    request_id: str,
    request_line_number: int,
    cut_turn_index: int | None,
    retained_message_count: int,
    original_message_count: int,
    original_estimated_token_length: int,
    retained_estimated_token_length: int,
    request_estimated_length: int,
    request_estimated_token_length: int,
    estimator: LengthEstimator,
    variant: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "compaction_schema_version": COMPACTION_SCHEMA_VERSION,
        "request_id": request_id,
        "request_line_number": request_line_number,
        "original_row_id": candidate.original_row_id,
        "original_row_path": candidate.original_row_path,
        "trajectory_path": str(candidate.trajectory_path),
        "original_index_path": str(candidate.index_path),
        "original_index_line_number": candidate.index_line_number,
        "cut_turn_index": cut_turn_index,
        "retained_message_count": retained_message_count,
        "original_message_count": original_message_count,
        "original_estimated_token_length": original_estimated_token_length,
        "retained_estimated_token_length": retained_estimated_token_length,
        "request_estimated_length": request_estimated_length,
        "request_estimated_token_length": request_estimated_token_length,
        "index_estimated_token_length": candidate.index_estimated_tokens,
        "index_estimation_source": candidate.index_estimation_source,
        "length_unit": estimator.length_unit,
        "length_estimator": estimator.mode,
        "length_estimator_detail": estimator.detail,
        "length_budget": estimator.budget,
        "qwen3_token_budget": estimator.token_budget,
        "char_budget": estimator.char_budget,
        "compaction_variant": variant,
        "strict_pass_source": candidate.strict_source,
        "status": status,
    }
    if extra:
        record.update(extra)
    return record


def write_outputs(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    request_path = output_dir / "compaction_request.jsonl"
    index_path = output_dir / "compaction_index.jsonl"
    records_path = output_dir / "compaction_records.jsonl"
    summary_path = output_dir / "compaction_summary.json"
    if output_dir.exists() and not args.overwrite and (request_path.exists() or index_path.exists() or records_path.exists()):
        raise SystemExit(f"Output files already exist under {output_dir}; pass --overwrite to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_resolved = output_dir.resolve()
    request_id_prefix = (
        f"{safe_id_fragment(output_resolved.parent.name)}-"
        f"{safe_id_fragment(output_resolved.name)}-"
        f"{sha256_text(str(output_resolved))[:10]}"
    )

    estimator = LengthEstimator(
        tokenizer_name=args.tokenizer_name,
        token_budget=args.qwen3_token_budget,
        char_budget=args.char_budget,
        chars_per_token=args.chars_per_token,
        no_tokenizer=args.no_tokenizer,
    )
    candidates, stats = select_candidates(args)
    request_budget = model_request_budget(args, estimator)
    run_started = time.time()
    if args.run_model:
        load_env_file(args.env_file)
        configure_optional_ca_bundle()

    statuses: Counter[str] = Counter()
    record_rows = 0
    with (
        request_path.open("w", encoding="utf-8") as requests,
        index_path.open("w", encoding="utf-8") as index,
        records_path.open("w", encoding="utf-8") as records,
    ):
        for request_line_number, candidate in enumerate(candidates, start=1):
            request_id = f"{request_id_prefix}-{request_line_number:06d}"
            try:
                trajectory = load_json(candidate.trajectory_path)
                messages = extract_messages(trajectory)
                if not messages:
                    status = "error_no_messages"
                    request_record = {
                        "compaction_schema_version": COMPACTION_SCHEMA_VERSION,
                        "request_id": request_id,
                        "original_row_id": candidate.original_row_id,
                        "original_row_path": candidate.original_row_path,
                        "trajectory_path": str(candidate.trajectory_path),
                        "original_index_path": str(candidate.index_path),
                        "original_index_line_number": candidate.index_line_number,
                        "cut_turn_index": None,
                        "retained_message_count": 0,
                        "original_message_count": 0,
                        "original_estimated_token_length": candidate.index_estimated_tokens,
                        "retained_estimated_token_length": 0,
                        "request_estimated_length": 0,
                        "request_estimated_token_length": 0,
                        "length_unit": estimator.length_unit,
                        "length_estimator": estimator.mode,
                        "length_estimator_detail": estimator.detail,
                        "length_budget": estimator.budget,
                        "compaction_variant": args.compaction_variant,
                        "status": status,
                    }
                    idx = index_record(
                        candidate=candidate,
                        request_id=request_id,
                        request_line_number=request_line_number,
                        cut_turn_index=None,
                        retained_message_count=0,
                        original_message_count=0,
                        original_estimated_token_length=candidate.index_estimated_tokens,
                        retained_estimated_token_length=0,
                        request_estimated_length=0,
                        request_estimated_token_length=0,
                        estimator=estimator,
                        variant=args.compaction_variant,
                        status=status,
                    )
                    requests.write(json.dumps(request_record, sort_keys=True) + "\n")
                    index.write(json.dumps(idx, sort_keys=True) + "\n")
                    requests.flush()
                    index.flush()
                    statuses[status] += 1
                    continue

                selection = select_suffix(
                    args=args,
                    candidate=candidate,
                    messages=messages,
                    estimator=estimator,
                    suffix_budget=estimator.budget,
                    request_budget=request_budget,
                    original_row_id=candidate.original_row_id,
                    variant=args.compaction_variant,
                )
                cut_turn_index = selection.cut_turn_index
                retained_messages = selection.retained_messages
                original_length = selection.original_length
                retained_length = selection.retained_length
                request_length = selection.request_length
                original_estimated_tokens = estimator.token_estimate_from_length(original_length)
                retained_estimated_tokens = estimator.token_estimate_from_length(retained_length)
                request_estimated_tokens = estimator.token_estimate_from_length(request_length)
                model_messages = selection.model_messages
                full_prefix_message_count = cut_turn_index
                included_prefix_message_count = len(selection.prefix_messages)
                omitted_prefix_message_count = max(0, full_prefix_message_count - included_prefix_message_count)
                full_prefix_sha256 = sha256_text(stable_json(messages[:cut_turn_index]))
                retained_suffix_sha256 = sha256_text(stable_json(retained_messages))
                included_prefix_sha256 = sha256_text(stable_json(selection.prefix_messages))
                boundary_previous_message_sha256 = (
                    sha256_text(stable_json(messages[cut_turn_index - 1])) if cut_turn_index > 0 else ""
                )
                boundary_next_message_sha256 = (
                    sha256_text(stable_json(messages[cut_turn_index])) if cut_turn_index < len(messages) else ""
                )
                source_trajectory_sha256 = sha256_file(candidate.trajectory_path)
                status = "dry_run"
                model_response = None
                model_responses: list[dict[str, Any]] = []
                model_metadata: dict[str, Any] = {}
                parsed_model_response = None
                model_json_error = None
                raw_record = None
                compacted_trajectory_path = ""
                compacted_estimated_length = 0
                compacted_estimated_tokens = 0
                original_user_prompt_for_validation = prompt_from_messages(messages[:cut_turn_index])
                if args.run_model:
                    try:
                        model_response, model_metadata = call_chat_model(args, model_messages)
                        parsed_model_response, model_json_error = parse_model_json_response(model_response)
                        if parsed_model_response is not None and not model_json_error:
                            model_json_error = validate_prompt_faithfulness(
                                parsed=parsed_model_response,
                                candidate=candidate,
                                original_user_prompt=original_user_prompt_for_validation,
                                prefix_messages=selection.prefix_messages,
                            )
                        model_responses.append(
                            {
                                "kind": "initial",
                                "response": model_response,
                                "metadata": model_metadata,
                                "json_error": model_json_error,
                            }
                        )
                        if model_json_error:
                            repair_errors: list[str] = [model_json_error]
                            for repair_attempt in range(1, max(0, args.max_json_repair_retries) + 1):
                                repair_messages = build_json_repair_messages(model_response, model_json_error)
                                repair_response, repair_metadata = call_chat_model(args, repair_messages)
                                repaired_response, repair_error = parse_model_json_response(repair_response)
                                if repaired_response is not None and not repair_error:
                                    repair_error = validate_prompt_faithfulness(
                                        parsed=repaired_response,
                                        candidate=candidate,
                                        original_user_prompt=original_user_prompt_for_validation,
                                        prefix_messages=selection.prefix_messages,
                                    )
                                model_responses.append(
                                    {
                                        "kind": "json_repair",
                                        "repair_attempt": repair_attempt,
                                        "response": repair_response,
                                        "metadata": repair_metadata,
                                        "json_error": repair_error,
                                    }
                                )
                                if repair_error:
                                    repair_errors.append(repair_error)
                                    model_response = repair_response
                                    model_json_error = repair_error
                                    parsed_model_response = None
                                    continue
                                model_response = repair_response
                                parsed_model_response = repaired_response
                                model_json_error = None
                                model_metadata = {
                                    **model_metadata,
                                    "json_repair_used": True,
                                    "json_repair_attempts": repair_attempt,
                                    "json_repair_metadata": repair_metadata,
                                    "json_repair_errors": repair_errors,
                                }
                                break
                        if model_json_error:
                            status = "model_completed_invalid_json"
                            model_metadata["model_json_error"] = model_json_error
                        else:
                            replacement_prompt = str(parsed_model_response.get("mini_swe_task_prompt") or "").strip()
                            new_messages = compacted_messages(
                                original_messages=messages,
                                retained_messages=retained_messages,
                                replacement_prompt=replacement_prompt,
                            )
                            compacted_estimated_length = estimator.count_messages(new_messages)
                            compacted_estimated_tokens = estimator.token_estimate_from_length(compacted_estimated_length)
                            compacted_trajectory = build_compacted_trajectory(
                                original_trajectory=trajectory,
                                messages=new_messages,
                                request_id=request_id,
                                parsed_model_response=parsed_model_response,
                                source_trajectory_path=candidate.trajectory_path,
                            )
                            compacted_trajectory_path_obj = write_compacted_trajectory(
                                output_dir=output_dir,
                                request_id=request_id,
                                trajectory=compacted_trajectory,
                            )
                            compacted_trajectory_path = str(compacted_trajectory_path_obj)
                            patch_text, patch_path_text = patch_text_for_candidate(candidate, trajectory)
                            raw_record = build_compacted_raw_record(
                                candidate=candidate,
                                original_trajectory=trajectory,
                                original_messages=messages,
                                compacted_messages_value=new_messages,
                                compacted_trajectory_path=compacted_trajectory_path_obj,
                                request_id=request_id,
                                parsed_model_response=parsed_model_response,
                                cut_turn_index=cut_turn_index,
                                retained_message_count=len(retained_messages),
                                original_estimated_tokens=original_estimated_tokens,
                                compacted_estimated_length=compacted_estimated_length,
                                compacted_estimated_tokens=compacted_estimated_tokens,
                                request_estimated_tokens=request_estimated_tokens,
                                estimator=estimator,
                                variant=args.compaction_variant,
                                compaction_model=args.model,
                                patch_text=patch_text,
                                patch_path_text=patch_path_text,
                                source_trajectory_sha256=source_trajectory_sha256,
                                full_prefix_sha256=full_prefix_sha256,
                                retained_suffix_sha256=retained_suffix_sha256,
                                boundary_previous_message_sha256=boundary_previous_message_sha256,
                                boundary_next_message_sha256=boundary_next_message_sha256,
                                boundary_previous_role=selection.boundary_previous_role,
                                boundary_next_role=selection.boundary_next_role,
                                first_retained_command_category=selection.first_retained_command_category,
                                first_retained_command_preview=selection.first_retained_command_preview,
                            )
                            status = "model_completed"
                    except Exception as exc:  # noqa: BLE001 - keep per-row failures in the index
                        status = "model_error"
                        model_metadata = {"error_type": type(exc).__name__, "error_message": str(exc)}

                request_record = {
                    "compaction_schema_version": COMPACTION_SCHEMA_VERSION,
                    "request_id": request_id,
                    "original_row_id": candidate.original_row_id,
                    "original_row_path": candidate.original_row_path,
                    "trajectory_path": str(candidate.trajectory_path),
                    "original_index_path": str(candidate.index_path),
                    "original_index_line_number": candidate.index_line_number,
                    "cut_turn_index": cut_turn_index,
                    "retained_message_count": len(retained_messages),
                    "original_message_count": len(messages),
                    "original_estimated_token_length": original_estimated_tokens,
                    "original_estimated_length": original_length,
                    "retained_estimated_token_length": retained_estimated_tokens,
                    "retained_estimated_length": retained_length,
                    "request_estimated_length": request_length,
                    "request_estimated_token_length": request_estimated_tokens,
                    "model_request_budget": request_budget,
                    "full_prefix_message_count": full_prefix_message_count,
                    "included_prefix_message_count": included_prefix_message_count,
                    "omitted_prefix_message_count": omitted_prefix_message_count,
                    "source_trajectory_sha256": source_trajectory_sha256,
                    "full_prefix_sha256": full_prefix_sha256,
                    "included_prefix_sha256": included_prefix_sha256,
                    "retained_suffix_sha256": retained_suffix_sha256,
                    "boundary_previous_message_sha256": boundary_previous_message_sha256,
                    "boundary_next_message_sha256": boundary_next_message_sha256,
                    "boundary_previous_role": selection.boundary_previous_role,
                    "boundary_next_role": selection.boundary_next_role,
                    "first_retained_command_category": selection.first_retained_command_category,
                    "first_retained_command_preview": selection.first_retained_command_preview,
                    "compacted_trajectory_path": compacted_trajectory_path,
                    "compacted_estimated_length": compacted_estimated_length,
                    "compacted_estimated_token_length": compacted_estimated_tokens,
                    "length_unit": estimator.length_unit,
                    "length_estimator": estimator.mode,
                    "length_estimator_detail": estimator.detail,
                    "length_budget": estimator.budget,
                    "compaction_variant": args.compaction_variant,
                    "status": status,
                    "model": args.model if args.run_model else None,
                    "messages": model_messages,
                }
                if model_response is not None:
                    request_record["model_response"] = model_response
                if len(model_responses) > 1:
                    request_record["model_responses"] = model_responses
                if parsed_model_response is not None:
                    request_record["parsed_model_response"] = parsed_model_response
                if model_json_error is not None:
                    request_record["model_json_error"] = model_json_error
                if model_metadata:
                    request_record["model_metadata"] = model_metadata

                idx = index_record(
                    candidate=candidate,
                    request_id=request_id,
                    request_line_number=request_line_number,
                    cut_turn_index=cut_turn_index,
                    retained_message_count=len(retained_messages),
                    original_message_count=len(messages),
                    original_estimated_token_length=original_estimated_tokens,
                    retained_estimated_token_length=retained_estimated_tokens,
                    request_estimated_length=request_length,
                    request_estimated_token_length=request_estimated_tokens,
                    estimator=estimator,
                    variant=args.compaction_variant,
                    status=status,
                    extra={
                        "compaction_schema_version": COMPACTION_SCHEMA_VERSION,
                        "model": args.model if args.run_model else None,
                        "model_request_budget": request_budget,
                        "full_prefix_message_count": full_prefix_message_count,
                        "included_prefix_message_count": included_prefix_message_count,
                        "omitted_prefix_message_count": omitted_prefix_message_count,
                        "source_trajectory_sha256": source_trajectory_sha256,
                        "full_prefix_sha256": full_prefix_sha256,
                        "included_prefix_sha256": included_prefix_sha256,
                        "retained_suffix_sha256": retained_suffix_sha256,
                        "boundary_previous_message_sha256": boundary_previous_message_sha256,
                        "boundary_next_message_sha256": boundary_next_message_sha256,
                        "boundary_previous_role": selection.boundary_previous_role,
                        "boundary_next_role": selection.boundary_next_role,
                        "first_retained_command_category": selection.first_retained_command_category,
                        "first_retained_command_preview": selection.first_retained_command_preview,
                        "compacted_trajectory_path": compacted_trajectory_path,
                        "compacted_estimated_length": compacted_estimated_length,
                        "compacted_estimated_token_length": compacted_estimated_tokens,
                        **model_metadata,
                    },
                )
                requests.write(json.dumps(request_record, ensure_ascii=False, sort_keys=True) + "\n")
                index.write(json.dumps(idx, ensure_ascii=False, sort_keys=True) + "\n")
                requests.flush()
                index.flush()
                if raw_record is not None:
                    records.write(
                        json.dumps(
                            {
                                "compaction_schema_version": COMPACTION_SCHEMA_VERSION,
                                "request_id": request_id,
                                "raw_record": raw_record,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    records.flush()
                    record_rows += 1
                statuses[status] += 1
                if args.run_model and args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
            except Exception as exc:  # noqa: BLE001 - one bad trajectory should not lose the batch
                if isinstance(exc, PrefixRequestTooLongError):
                    status = "prefix_request_too_long"
                elif isinstance(exc, json.JSONDecodeError):
                    status = "trajectory_json_error"
                elif isinstance(exc, OSError):
                    status = "source_file_error"
                elif "first-action policy" in str(exc):
                    status = "no_policy_compliant_suffix"
                else:
                    status = "error_read_or_prepare"
                request_record = {
                    "compaction_schema_version": COMPACTION_SCHEMA_VERSION,
                    "request_id": request_id,
                    "original_row_id": candidate.original_row_id,
                    "original_row_path": candidate.original_row_path,
                    "trajectory_path": str(candidate.trajectory_path),
                    "original_index_path": str(candidate.index_path),
                    "original_index_line_number": candidate.index_line_number,
                    "cut_turn_index": None,
                    "retained_message_count": 0,
                    "original_message_count": 0,
                    "original_estimated_token_length": candidate.index_estimated_tokens,
                    "retained_estimated_token_length": 0,
                    "request_estimated_length": 0,
                    "request_estimated_token_length": 0,
                    "length_unit": estimator.length_unit,
                    "length_estimator": estimator.mode,
                    "length_estimator_detail": estimator.detail,
                    "length_budget": estimator.budget,
                    "status": status,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "compaction_variant": args.compaction_variant,
                }
                idx = index_record(
                    candidate=candidate,
                    request_id=request_id,
                    request_line_number=request_line_number,
                    cut_turn_index=None,
                    retained_message_count=0,
                    original_message_count=0,
                    original_estimated_token_length=candidate.index_estimated_tokens,
                    retained_estimated_token_length=0,
                    request_estimated_length=0,
                    request_estimated_token_length=0,
                    estimator=estimator,
                    variant=args.compaction_variant,
                    status=status,
                    extra={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
                requests.write(json.dumps(request_record, ensure_ascii=False, sort_keys=True) + "\n")
                index.write(json.dumps(idx, ensure_ascii=False, sort_keys=True) + "\n")
                requests.flush()
                index.flush()
                statuses[status] += 1

    summary = {
        "index_paths": [str(path) for path in args.indexes],
        "output_dir": str(output_dir),
        "request_path": str(request_path),
        "index_path": str(index_path),
        "records_path": str(records_path),
        "record_rows": record_rows,
        "run_model": args.run_model,
        "model": args.model if args.run_model else None,
        "compaction_schema_version": COMPACTION_SCHEMA_VERSION,
        "compaction_variant": args.compaction_variant,
        "length_estimator": estimator.mode,
        "length_estimator_detail": estimator.detail,
        "length_unit": estimator.length_unit,
        "length_budget": estimator.budget,
        "model_request_budget": request_budget,
        "model_request_token_budget": args.model_request_token_budget,
        "model_request_char_budget": request_budget if estimator.tokenizer is None else args.model_request_char_budget,
        "qwen3_token_budget": estimator.token_budget,
        "char_budget": estimator.char_budget,
        "first_action_policy": args.first_action_policy,
        "allow_user_boundary": args.allow_user_boundary,
        "extra_body_json": args.extra_body_json or "",
        "stats": dict(sorted(stats.items())),
        "statuses": dict(sorted(statuses.items())),
        "elapsed_seconds": round(time.time() - run_started, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    write_outputs(args)


if __name__ == "__main__":
    main()
