#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL 8B defaults for the shared SWE-bench multilingual eval runner.
# CHECKPOINT_DIR, CHECKPOINT_STEP_DIR, CHECKPOINT_LABEL, RUN_SUFFIX, and
# BENCHMARK may be overridden by the caller.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

export EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-8}"
export EVAL_ACCELERATOR_LABEL="${EVAL_ACCELERATOR_LABEL:-l40s}"
export CONTEXT_LABEL="${CONTEXT_LABEL:-65k}"

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Thinking}"
export MODEL_SOURCE_MODEL_NAME="${MODEL_SOURCE_MODEL_NAME:-Qwen/Qwen3-VL-8B-Thinking}"
export MODEL_VARIANT_LABEL="${MODEL_VARIANT_LABEL:-qwen3-vl-8b-thinking}"
export LITELLM_MODEL="${LITELLM_MODEL:-openai/$MODEL_NAME}"

export MERGE_QWEN3_VL_TEXT_OVERLAY="${MERGE_QWEN3_VL_TEXT_OVERLAY:-true}"
export QWEN3_VL_BASE_MODEL="${QWEN3_VL_BASE_MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
export VLLM_PREWARM_REGISTRY_MODEL="${VLLM_PREWARM_REGISTRY_MODEL:-Qwen3VLForConditionalGeneration}"
# Match the 4B Qwen thinking eval path. The qwen3 parser treats output without a
# generated </think> as reasoning-only, which can hide otherwise valid
# <tool_call> XML from the tool parser when a checkpoint is still shaky.
export REASONING_PARSER="${REASONING_PARSER:-deepseek_r1}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
export MAX_TOKENS="${MAX_TOKENS:-16384}"

export DESCRIPTION="${DESCRIPTION:-Qwen3-VL 8B SWE260612 checkpoint on ${EVAL_GPU_COUNT} ${EVAL_ACCELERATOR_LABEL} GPUs.}"

exec "$SCRIPT_DIR/run_qwen3_4b_swe260612_step50_l40s_eval.sh" "$@"
