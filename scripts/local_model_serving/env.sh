#!/usr/bin/env bash
set -euo pipefail

export LOCAL_MODEL_SERVING_ROOT="${LOCAL_MODEL_SERVING_ROOT:-/scratch/local_model_serving}"

# Keep every cache/temp path off /home. SGLang/FlashInfer/TVM caches on NFS can
# add minutes of startup latency or hang while compiling kernels.
export HOME="$LOCAL_MODEL_SERVING_ROOT/home"
export HF_HOME="$LOCAL_MODEL_SERVING_ROOT/hf_home"
export HUGGINGFACE_HUB_CACHE="$LOCAL_MODEL_SERVING_ROOT/hf_cache"
export HF_HUB_CACHE="$LOCAL_MODEL_SERVING_ROOT/hf_cache/hub"
export HF_XET_CACHE="$LOCAL_MODEL_SERVING_ROOT/hf_xet_cache"
export TRANSFORMERS_CACHE="$LOCAL_MODEL_SERVING_ROOT/transformers_cache"
export XDG_CACHE_HOME="$LOCAL_MODEL_SERVING_ROOT/xdg_cache"
export PIP_CACHE_DIR="$LOCAL_MODEL_SERVING_ROOT/pip_cache"
export TORCH_HOME="$LOCAL_MODEL_SERVING_ROOT/torch_home"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_MODEL_SERVING_ROOT/torchinductor_cache"
export TRITON_CACHE_DIR="$LOCAL_MODEL_SERVING_ROOT/triton_cache"
export CUDA_CACHE_PATH="$LOCAL_MODEL_SERVING_ROOT/cuda_cache"
export TVM_FFI_CACHE_DIR="$LOCAL_MODEL_SERVING_ROOT/tvm_ffi_cache"
export FLASHINFER_WORKSPACE_BASE="$LOCAL_MODEL_SERVING_ROOT/flashinfer_workspace"
export PYTHONPYCACHEPREFIX="$LOCAL_MODEL_SERVING_ROOT/pycache"
export TMPDIR="$LOCAL_MODEL_SERVING_ROOT/tmp"

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export SGLANG_VENV="${SGLANG_VENV:-$LOCAL_MODEL_SERVING_ROOT/venvs/venv-sglang}"

export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

export KIMI_MODEL_PATH="${KIMI_MODEL_PATH:-$LOCAL_MODEL_SERVING_ROOT/models/moonshotai_Kimi-K2.6.snapshot}"
export MIMO_MODEL_PATH="${MIMO_MODEL_PATH:-$LOCAL_MODEL_SERVING_ROOT/models/XiaomiMiMo_MiMo-V2.5.snapshot}"
export QWEN36_MODEL_PATH="${QWEN36_MODEL_PATH:-$LOCAL_MODEL_SERVING_ROOT/models/Qwen_Qwen3.6-27B.snapshot}"

if [[ -d "$SGLANG_VENV/lib/python3.12/site-packages/torch/lib" ]]; then
  sglang_ld_paths=("$SGLANG_VENV/lib/python3.12/site-packages/torch/lib")
  for nvidia_lib_dir in "$SGLANG_VENV"/lib/python3.12/site-packages/nvidia/*/lib; do
    [[ -d "$nvidia_lib_dir" ]] && sglang_ld_paths+=("$nvidia_lib_dir")
  done
  export LD_LIBRARY_PATH="$(IFS=:; echo "${sglang_ld_paths[*]}"):${LD_LIBRARY_PATH:-}"
fi

mkdir -p \
  "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_XET_CACHE" "$TRANSFORMERS_CACHE" \
  "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
  "$TVM_FFI_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$PYTHONPYCACHEPREFIX" \
  "$TMPDIR" "$LOCAL_MODEL_SERVING_ROOT/models" "$LOCAL_MODEL_SERVING_ROOT/logs"

if [[ -f /wbl-fast/usrs/ee/code-swe-data/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /wbl-fast/usrs/ee/code-swe-data/.env
  set +a
fi
