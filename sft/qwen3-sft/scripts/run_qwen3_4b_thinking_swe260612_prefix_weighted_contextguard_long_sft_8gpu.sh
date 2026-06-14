#!/usr/bin/env bash
set -euo pipefail

# Longer clean-SFT candidate after dropping no-reasoning tool-call contexts.
# Keeps the v15 data/policy fixed and tests whether 4B benefits from more
# clean tool-use signal at a lower learning rate.
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen3_4b_thinking_swe260612_prefix_weighted_v17_contextguard_long400_lr1e6_65k_assistant_h200_8gpu_sft/}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_thinking_swe260612_prefix_weighted_v17_contextguard_long400_lr1e6_65k_assistant_h200_8gpu_sft}"

export MAX_STEPS="${MAX_STEPS:-400}"
export CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-50}"
export LR="${LR:-1.0e-6}"
export MIN_LR="${MIN_LR:-1.0e-7}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-20}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_qwen3_4b_thinking_swe260612_prefix_weighted_balanced_sft_8gpu.sh" "$@"
