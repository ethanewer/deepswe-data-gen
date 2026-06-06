#!/usr/bin/env python3
"""Packed SFT benchmark for OLMo-3 7B Think-style runs."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyPackedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.layer_norm import LayerNormType
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import AdamWConfig, LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    GarbageCollectorCallback,
    MonkeyPatcherCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

from online_sft_cache import (
    OnlinePackedSFTDataLoader,
    start_background_producer,
    wait_for_ready_batches,
)


DEFAULT_DATASET_DIR = (
    "/wbl-fast/usrs/ee/code-swe-data/data/tokenized/"
    "code-swe-terminal-agentic-sft-olmo3-65k"
)
DEFAULT_TOKEN_IDS_GLOB = f"{DEFAULT_DATASET_DIR}/**/token_ids_part_*.npy"
DEFAULT_LABEL_MASK_GLOB = f"{DEFAULT_DATASET_DIR}/**/labels_mask_*.npy"
DEFAULT_LOAD_PATH = (
    "/wbl-fast/usrs/mk/data/checkpoints/"
    "Olmo-3-1025-7B-stage3-step11921/model_and_optim"
)
DEFAULT_RAW_ROOT = "/wbl-fast/usrs/ee/code-swe-data/data/code-swe-terminal-agentic-sft"
DEFAULT_ONLINE_CACHE_DIR = (
    "/wbl-fast/usrs/ee/code-swe-data/sft/olmo3-sft/work/online-cache"
)
DEFAULT_ONLINE_TOKENIZER = "/wbl-fast/usrs/mk/data/tokenizers/Olmo-3-7B-Think-SFT"
DEFAULT_ONLINE_PYTHON = "/wbl-fast/usrs/mk/open-instruct/.venv/bin/python"


@dataclass
class JsonlMetricsCallback(Callback):
    path: str

    def post_attach(self) -> None:
        if get_rank() == 0:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("")

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        if get_rank() != 0:
            return
        record = {
            "step": step,
            "time": time.time(),
            "metrics": {k: float(v) for k, v in metrics.items() if _is_number(v)},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def patch_ddp_settings(args: argparse.Namespace) -> None:
    if args.ddp_bucket_cap_mb == 100 and args.ddp_optimize_mode == "default":
        return

    import torch
    from olmo_core.nn.transformer.model import Transformer

    original_apply_ddp = Transformer.apply_ddp

    def apply_ddp_with_settings(
        self: Transformer,
        dp_mesh=None,
        param_dtype=None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
    ):
        from torch.distributed._composable.replicate import replicate

        target_dtype = param_dtype or self.dtype
        if target_dtype != self.dtype:
            self.to(dtype=target_dtype)

        if compile_enabled:
            if args.ddp_optimize_mode == "default":
                if autograd_compile_enabled:
                    torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"  # type: ignore[attr-defined]
                else:
                    torch._dynamo.config.optimize_ddp = "ddp_optimizer"  # type: ignore[attr-defined]
            else:
                torch._dynamo.config.optimize_ddp = args.ddp_optimize_mode  # type: ignore[attr-defined]

        replicate(self, device_mesh=dp_mesh, bucket_cap_mb=args.ddp_bucket_cap_mb)
        self.register_forward_pre_hook(
            original_apply_ddp.__globals__["_hide_cpu_inputs_from_torch"],
            prepend=True,
            with_kwargs=True,
        )
        self.register_forward_pre_hook(
            original_apply_ddp.__globals__["_unhide_cpu_inputs_from_torch"],
            prepend=False,
            with_kwargs=True,
        )

    Transformer.apply_ddp = apply_ddp_with_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run packed full-SFT for OLMo-3 7B with the measured 8-GPU 65k default recipe."
    )
    parser.add_argument("--save-folder", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--sequence-length", type=int, default=65536)
    parser.add_argument("--local-batch-seqs", type=int, default=4)
    parser.add_argument("--global-batch-seqs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=33333)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--optim",
        choices=["skip_step_adamw", "adamw"],
        default="adamw",
    )
    parser.add_argument("--adamw-fused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adamw-foreach", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-step-foreach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--packing-workers", type=int, default=0)
    parser.add_argument("--source-group-size", type=int, default=1)
    parser.add_argument(
        "--attn-backend",
        choices=["flash_2", "flash_3", "flash_4", "torch"],
        default="flash_2",
    )
    parser.add_argument(
        "--activation-checkpointing",
        choices=["full", "feed_forward", "selected_ops", "budget", "none"],
        default="full",
    )
    parser.add_argument("--activation-memory-budget", type=float, default=0.5)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-optim", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-ops", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--layer-norm",
        choices=["default", "fused_rms", "cute_rms"],
        default="default",
    )
    parser.add_argument("--data-parallel", choices=["none", "ddp", "fsdp", "hsdp"], default="fsdp")
    parser.add_argument("--dp-shard-degree", type=int, default=None)
    parser.add_argument("--dp-num-replicas", type=int, default=None)
    parser.add_argument(
        "--reduce-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Gradient reduction dtype for DDP/FSDP.",
    )
    parser.add_argument(
        "--fsdp-wrapping-strategy",
        choices=["full", "blocks", "fine_grained"],
        default="full",
    )
    parser.add_argument("--fsdp-prefetch-factor", type=int, default=0)
    parser.add_argument(
        "--context-parallel-style",
        choices=["none", "llama3", "zig_zag", "ulysses"],
        default="none",
    )
    parser.add_argument("--context-parallel-degree", type=int, default=1)
    parser.add_argument("--context-parallel-head-stride", type=int, default=4)
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=100)
    parser.add_argument(
        "--ddp-optimize-mode",
        choices=["default", "ddp_optimizer", "python_reducer_without_compiled_forward"],
        default="default",
    )
    parser.add_argument("--load-path", default=DEFAULT_LOAD_PATH)
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--token-ids-glob", default=DEFAULT_TOKEN_IDS_GLOB)
    parser.add_argument("--label-mask-glob", default=DEFAULT_LABEL_MASK_GLOB)
    parser.add_argument("--data-mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--online-cache-dir", default=DEFAULT_ONLINE_CACHE_DIR)
    parser.add_argument("--online-tokenizer", default=DEFAULT_ONLINE_TOKENIZER)
    parser.add_argument("--online-python", default=DEFAULT_ONLINE_PYTHON)
    parser.add_argument("--online-min-ready-batches", type=int, default=2)
    parser.add_argument("--online-cache-part-instances", type=int, default=128)
    parser.add_argument("--online-poll-interval", type=float, default=5.0)
    parser.add_argument("--online-max-rows-per-file", type=int, default=0)
    parser.add_argument("--online-max-examples", type=int, default=0)
    parser.add_argument("--online-truncate-overlength", action="store_true")
    parser.add_argument("--online-skip-root", action="append", default=[])
    parser.add_argument("--no-online-doc-lens", action="store_true")
    return parser.parse_args()


def build_components(args: argparse.Namespace):
    seq_len = args.sequence_length
    world_size = get_world_size()
    local_batch_tokens = args.local_batch_seqs * seq_len

    if args.context_parallel_style == "none":
        cp_degree = 1
    else:
        cp_degree = args.context_parallel_degree
        if cp_degree <= 1:
            raise ValueError("--context-parallel-degree must be > 1 when CP is enabled")
        if world_size % cp_degree != 0:
            raise ValueError("--context-parallel-degree must divide world size")
        if seq_len % cp_degree != 0:
            raise ValueError("--sequence-length must be divisible by --context-parallel-degree")
    dp_world_size = world_size // cp_degree

    global_batch_seqs = args.global_batch_seqs or (args.local_batch_seqs * dp_world_size)
    global_batch_tokens = global_batch_seqs * seq_len

    if global_batch_tokens % (local_batch_tokens * dp_world_size) != 0:
        raise ValueError(
            "--global-batch-seqs must be a multiple of "
            "--local-batch-seqs * data_parallel_world_size for clean grad accumulation"
        )

    tokenizer_config = TokenizerConfig.dolma2()
    model_kwargs = {}
    if args.layer_norm != "default":
        model_kwargs["layer_norm_name"] = LayerNormType(args.layer_norm)
    model_config = TransformerConfig.olmo3_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName(args.attn_backend),
        dtype=DType.bfloat16,
        fused_ops=args.fused_ops,
        **model_kwargs,
    ).with_rope_scaling(
        YaRNRoPEScalingConfig(
            factor=8,
            beta_fast=32,
            beta_slow=1,
            old_context_len=8192,
        )
    )
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    if args.data_mode == "offline":
        dataset_config = NumpyPackedFSLDatasetConfig(
            tokenizer=tokenizer_config,
            paths=[args.token_ids_glob],
            label_mask_paths=[args.label_mask_glob],
            expand_glob=True,
            generate_doc_lengths=True,
            long_doc_strategy=LongDocStrategy.truncate,
            sequence_length=seq_len,
            work_dir=args.work_dir,
            source_group_size=args.source_group_size,
            packing_workers=args.packing_workers,
        )

        data_loader_config = NumpyDataLoaderConfig(
            global_batch_size=global_batch_tokens,
            seed=34521,
            num_workers=args.num_workers,
        )
    else:
        dataset_config = None
        data_loader_config = None

    if args.activation_checkpointing == "full":
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        )
    elif args.activation_checkpointing == "feed_forward":
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )
    elif args.activation_checkpointing == "selected_ops":
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_ops,
        )
    elif args.activation_checkpointing == "budget":
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=args.activation_memory_budget,
        )
    else:
        ac_config = None
    dp_config = None
    reduce_dtype = DType.float32 if args.reduce_dtype == "float32" else DType.bfloat16
    if args.data_parallel in ("ddp", "fsdp", "hsdp"):
        dp_config = TransformerDataParallelConfig(
            name=DataParallelType(args.data_parallel),
            param_dtype=DType.bfloat16,
            reduce_dtype=reduce_dtype,
            shard_degree=args.dp_shard_degree,
            num_replicas=args.dp_num_replicas,
            wrapping_strategy=TransformerDataParallelWrappingStrategy(
                args.fsdp_wrapping_strategy
            ),
            prefetch_factor=args.fsdp_prefetch_factor,
        )
    elif world_size > 1:
        raise ValueError("--data-parallel must be set when running with more than one rank")

    if args.context_parallel_style == "llama3":
        cp_config = TransformerContextParallelConfig.llama3(
            degree=cp_degree, head_stride=args.context_parallel_head_stride
        )
    elif args.context_parallel_style == "zig_zag":
        cp_config = TransformerContextParallelConfig.zig_zag(
            degree=cp_degree, head_stride=args.context_parallel_head_stride
        )
    elif args.context_parallel_style == "ulysses":
        cp_config = TransformerContextParallelConfig.ulysses(degree=cp_degree)
    else:
        cp_config = None

    if args.optim == "adamw":
        optim_config = AdamWConfig(
            lr=args.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            compile=args.compile_optim,
            fused=args.adamw_fused,
            foreach=args.adamw_foreach,
        )
    else:
        optim_config = SkipStepAdamWConfig(
            lr=args.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            compile=args.compile_optim,
            foreach=args.skip_step_foreach,
        )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=local_batch_tokens,
        max_sequence_length=seq_len,
        optim=optim_config,
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=args.compile_model,
        dp_config=dp_config,
        cp_config=cp_config,
        ac_config=ac_config,
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=args.save_folder,
            work_dir=args.work_dir,
            checkpointer=CheckpointerConfig(save_thread_count=1, load_thread_count=32),
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=1000,
            max_duration=Duration.steps(args.max_steps),
            load_path=None if args.skip_load else args.load_path,
            load_strategy=LoadStrategy.never if args.skip_load else LoadStrategy.if_available,
            load_trainer_state=False,
            load_optim_state=False,
        )
        .with_callback("checkpointer", CheckpointerCallback(enabled=False))
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("monkey_patcher", MonkeyPatcherCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("jsonl_metrics", JsonlMetricsCallback(path=args.metrics_jsonl))
    )

    return (
        model_config,
        dataset_config,
        data_loader_config,
        train_module_config,
        trainer_config,
        tokenizer_config,
        global_batch_tokens,
    )


def main() -> None:
    args = parse_args()

    patch_ddp_settings(args)
    producer = None
    prepare_training_environment(shared_filesystem=True)
    try:
        seed_all(args.seed)
        (
            model_config,
            dataset_config,
            data_loader_config,
            train_module_config,
            trainer_config,
            tokenizer_config,
            global_batch_tokens,
        ) = build_components(args)

        model = model_config.build(init_device="meta")
        train_module = train_module_config.build(model)
        if args.data_mode == "online":
            producer = start_background_producer(args)
            wait_for_ready_batches(
                Path(args.online_cache_dir),
                min_ready_batches=args.online_min_ready_batches,
                global_batch_size=global_batch_tokens,
                sequence_length=args.sequence_length,
                poll_interval=args.online_poll_interval,
            )
            data_loader = OnlinePackedSFTDataLoader(
                cache_dir=args.online_cache_dir,
                tokenizer=tokenizer_config,
                sequence_length=args.sequence_length,
                global_batch_size=global_batch_tokens,
                work_dir=args.work_dir,
                dp_world_size=get_world_size(train_module.dp_process_group),
                dp_rank=get_rank(train_module.dp_process_group),
                min_ready_batches=args.online_min_ready_batches,
                poll_interval=args.online_poll_interval,
                include_doc_lens=not args.no_online_doc_lens,
            )
        else:
            assert dataset_config is not None
            assert data_loader_config is not None
            dataset = dataset_config.build()
            data_loader = data_loader_config.build(
                dataset, dp_process_group=train_module.dp_process_group
            )
        trainer = trainer_config.build(train_module, data_loader)

        if args.skip_load:
            if get_rank() == 0:
                print("[olmo-bench] skip-load enabled; benchmarking initialized weights")
        elif not trainer.maybe_load_checkpoint() and args.load_path:
            if get_rank() == 0:
                print(f"[olmo-bench] loading checkpoint from {args.load_path}")
            trainer.load_checkpoint(args.load_path, load_trainer_state=False, load_optim_state=False)

        trainer.fit()
    finally:
        if producer is not None and producer.poll() is None:
            producer.terminate()
        teardown_training_environment()


if __name__ == "__main__":
    main()
