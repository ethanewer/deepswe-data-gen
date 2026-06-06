#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f .venv/bin/activate ]; then
  echo "Missing .venv. Run ./scripts/setup_nemo_automodel_env.sh first." >&2
  exit 1
fi

source .venv/bin/activate

FA3_REPO="${FA3_REPO:-kernels-community/vllm-flash-attn3}"
FA3_REVISION="${FA3_REVISION:-3cca6264464f83a229b78b19a5af84aa5c3b78c0}"
FA3_BUILD_NAME="${FA3_BUILD_NAME:-torch210-cxx11-cu128-x86_64-linux}"
export FA3_REPO FA3_REVISION FA3_BUILD_NAME

python - <<'PY'
import importlib.metadata as metadata
import importlib.util
import os
import shutil
import sysconfig
from pathlib import Path

from huggingface_hub import snapshot_download

repo = os.environ["FA3_REPO"]
revision = os.environ["FA3_REVISION"]
build_name = os.environ["FA3_BUILD_NAME"]

snapshot = Path(
    snapshot_download(
        repo,
        revision=revision,
        allow_patterns=[f"build/{build_name}/*"],
    )
)
src = snapshot / "build" / build_name
site = Path(sysconfig.get_paths()["purelib"])
pkg = site / "flash_attn_interface"
dist = site / "flash_attn_3-3.0.0.dist-info"

required = [
    "__init__.py",
    "flash_attn_interface.py",
    "_ops.py",
    "_vllm_flash_attn3_cuda_9aa33c0.abi3.so",
    "metadata.json",
]
missing = [name for name in required if not (src / name).exists()]
if missing:
    raise SystemExit(f"Missing files in {src}: {missing}")

shutil.rmtree(pkg, ignore_errors=True)
shutil.rmtree(dist, ignore_errors=True)
pkg.mkdir(parents=True, exist_ok=True)
dist.mkdir(parents=True, exist_ok=True)

for name in required:
    shutil.copy2(src / name, pkg / name, follow_symlinks=True)

(dist / "METADATA").write_text(
    "\n".join(
        [
            "Metadata-Version: 2.1",
            "Name: flash-attn-3",
            "Version: 3.0.0",
            f"Summary: Prebuilt FlashAttention 3 interface from {repo}",
            "Requires-Python: >=3.8",
            "",
        ]
    )
)
(dist / "top_level.txt").write_text("flash_attn_interface\n")
(dist / "WHEEL").write_text(
    "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: install_flash_attn3_prebuilt.sh",
            "Root-Is-Purelib: false",
            "Tag: cp310-abi3-linux_x86_64",
            "",
        ]
    )
)
(dist / "RECORD").write_text(
    "\n".join(
        [
            "flash_attn_interface/__init__.py,,",
            "flash_attn_interface/flash_attn_interface.py,,",
            "flash_attn_interface/_ops.py,,",
            "flash_attn_interface/_vllm_flash_attn3_cuda_9aa33c0.abi3.so,,",
            "flash_attn_interface/metadata.json,,",
            "flash_attn_3-3.0.0.dist-info/METADATA,,",
            "flash_attn_3-3.0.0.dist-info/top_level.txt,,",
            "flash_attn_3-3.0.0.dist-info/WHEEL,,",
            "flash_attn_3-3.0.0.dist-info/RECORD,,",
            "",
        ]
    )
)

spec = importlib.util.find_spec("flash_attn_interface")
print(f"flash_attn_interface: {bool(spec)} {spec.origin if spec else ''}")
print(f"flash-attn-3: {metadata.version('flash-attn-3')}")
PY
