#!/usr/bin/env python3
"""Build a retry manifest for direct Docker datagen.

Rows are skipped only when the existing workspace contains a real model trace
with at least one API call. Zero-call setup failures remain retryable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-id", required=True)
    parser.add_argument(
        "--servers-file",
        type=Path,
        required=True,
        help="JSON file containing a list or {'servers': [...]} with base_url entries.",
    )
    parser.add_argument("--include-real-traces", action="store_true")
    return parser.parse_args()


def load_servers(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("servers", [])
    base_urls = [str(item["base_url"]).rstrip("/") for item in raw if item.get("base_url")]
    if not base_urls:
        raise SystemExit(f"no base_url entries found in {path}")
    return base_urls


def result_has_model_trace(workspace: Path) -> bool:
    result_path = workspace / "result.json"
    if not result_path.exists():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(result.get("api_calls") or 0) > 0


def rewrite_workspace(workspace: str, old_rollout_id: str, new_rollout_id: str) -> str:
    path = Path(workspace)
    parts = list(path.parts)
    for i, part in enumerate(parts):
        if part == old_rollout_id:
            parts[i] = new_rollout_id
            return str(Path(*parts))
    return str(path.parent.parent / new_rollout_id / path.name)


def main() -> None:
    args = parse_args()
    base_urls = load_servers(args.servers_file)
    rows_out: list[str] = []
    skipped_real = 0
    source_rows = 0
    for line in args.source_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source_rows += 1
        fields = line.split("\t")
        if len(fields) < 16:
            raise ValueError(f"manifest row has {len(fields)} fields, expected 16: {line[:200]}")
        if not args.include_real_traces and result_has_model_trace(Path(fields[4])):
            skipped_real += 1
            continue
        fields[1] = args.rollout_id
        fields[4] = rewrite_workspace(fields[4], old_rollout_id=line.split("\t")[1], new_rollout_id=args.rollout_id)
        fields[9] = base_urls[len(rows_out) % len(base_urls)]
        rows_out.append("\t".join(fields))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows_out) + ("\n" if rows_out else ""), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_rows": source_rows,
                "output_rows": len(rows_out),
                "skipped_real_model_traces": skipped_real,
                "servers": len(base_urls),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
