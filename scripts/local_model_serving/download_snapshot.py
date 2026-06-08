#!/usr/bin/env python3
import os
import pathlib
import sys

from huggingface_hub import snapshot_download


def refresh_symlink(link_path: pathlib.Path, target_path: pathlib.Path) -> None:
    tmp_link = link_path.with_name(f".{link_path.name}.tmp")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    os.symlink(target_path, tmp_link, target_is_directory=True)
    if link_path.exists() and not link_path.is_symlink():
        raise RuntimeError(f"{link_path} exists and is not a symlink; refusing to replace it")
    os.replace(tmp_link, link_path)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: download_snapshot.py <repo_id> <stable_symlink>", file=sys.stderr)
        return 2

    repo_id = sys.argv[1]
    link_path = pathlib.Path(sys.argv[2])
    root = pathlib.Path(os.environ["LOCAL_MODEL_SERVING_ROOT"])
    cache_dir = pathlib.Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    max_workers = int(os.environ.get("HF_MAX_WORKERS", "8"))

    link_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = pathlib.Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            cache_dir=cache_dir,
            max_workers=max_workers,
            local_dir=None,
        )
    )
    if not snapshot_path.is_relative_to(root):
        raise RuntimeError(f"snapshot path escaped /wbl-fast runtime: {snapshot_path}")
    refresh_symlink(link_path, snapshot_path)
    print(snapshot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
