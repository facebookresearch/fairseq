# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional

import torch

from fairseq.dataclass.constants import (
    DATASET_IMPL_CHOICES,
    DDP_BACKEND_CHOICES,
    DDP_COMM_HOOK_CHOICES,
    GENERATION_CONSTRAINTS_CHOICES,
    GENERATION_DECODING_FORMAT_CHOICES,
    LOG_FORMAT_CHOICES,
    PIPELINE_CHECKPOINT_CHOICES,
    PRINT_ALIGNMENT_CHOICES,
    ZERO_SHARDING_CHOICES,
)

from omegaconf import II, MISSING


@dataclass
class FairseqDataclass:
    """fairseq base dataclass that supported fetching attributes and metas"""

    _name: Optional[str] = None

    @staticmethod
    def name():
        return None

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(
        self, attribute_name: str, meta: str, default: Optional[Any] = None
    ) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith(
                "${"
            ):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif (
                getattr(self, attribute_name)
                != self.__dataclass_fields__[attribute_name].default
            ):
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")

    def _get_argparse_const(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_const")

    def _get_argparse_alias(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_alias")

    def _get_choices(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "choices")


@dataclass
class CommonConfig(FairseqDataclass):
    # This is the core dataclass including common parameters shared by all different jobs. Please append your params to other dataclasses if they were
    # used for a particular purpose or task, such as those dedicated for `distributed training`, `optimization`, etc.
    no_progress_bar: bool = field(
        default=False, metadata={"help": "disable progress bar"}
    )
    log_interval: int = field(
        default=100,
        metadata={
            "help": "log progress every N batches (when progress bar is disabled)"
        },
    )
    log_format: Optional[LOG_FORMAT_CHOICES] = field(
        default=None, metadata={"help": "log format to use"}
    )
    log_file: Optional[str] = field(
        default=None, metadata={"help": "log file to copy metrics to."}
    )
    tensorboard_logdir: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to save logs for tensorboard, should match --logdir "
            "of running tensorboard (default: no tensorboard logging)"
        },
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights and Biases project name to use for logging"},
    )
    azureml_logging: Optional[bool] = field(
        default=False, metadata={"help": "Log scalars to AzureML context"},
    )
    seed: int = field(
        default=1, metadata={"help": "pseudo random number generator seed"}
    )
    cpu: bool = field(default=False, metadata={"help": "use CPU instead of CUDA"})
    tpu: bool = field(default=False, metadata={"help": "use TPU instead of CUDA"})
    bf16: bool = field(default=False, metadata={"help": "use bfloat16; implies --tpu"})
    memory_efficient_bf16: bool = field(
        default=False,
        metadata={
            "help": "use a memory-efficient version of BF16 training; implies --bf16"
        },
    )
    fp16: bool = field(default=False, metadata={"help": "use FP16"})
    memory_efficient_fp16: bool = field(
        default=False,
        metadata={
            "help": "use a memory-efficient version of FP16 training; implies --fp16"
        },
    )
    fp16_no_flatten_grads: bool = field(
        default=False, metadata={"help": "don't flatten FP16 grads tensor"}
    )
    fp16_init_scale: int = field(
        default=2 ** 7, metadata={"help": "default FP16 loss scale"}
    )
    fp16_scale_window: Optional[int] = field(
        default=None,
        metadata={"help": "number of updates before increasing loss scale"},
    )
    fp16_scale_tolerance: float = field(
        default=0.0,
        metadata={
            "help": "pct of updates that can overflow before decreasing the loss scale"
        },
    )
    min_loss_scale: float = field(
        default=1e-4,
        metadata={"help": "minimum FP16/AMP loss scale, after which training is stopped"},
    )
    threshold_loss_scale: Optional[float] = field(
        default=None, metadata={"help": "threshold FP16 loss scale from below"}
    )
    amp: bool = field(default=False, metadata={"help": "use automatic mixed precision"})
    amp_batch_retries: int = field(
        default=2,
        metadata={"help": "number of retries of same batch after reducing loss scale with AMP"},
    )
    amp_init_scale: int = field(
        default=2 ** 7, metadata={"help": "default AMP loss scale"}
    )
    amp_scale_window: Optional[int] = field(
        default=None,
        metadata={"help": "number of updates before increasing AMP loss scale"},
    )
    user_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to a python module containing custom extensions (tasks and/or architectures)"
        },
    )
    empty_cache_freq: int = field(
        default=0,
        metadata={"help": "how often to clear the PyTorch CUDA cache (0 to disable)"},
    )
    all_gather_list_size: int = field(
        default=16384,
        metadata={"help": "number of bytes reserved for gathering stats from workers"},
    )
    model_parallel_size: int = field(
        default=1, metadata={"help": "total number of GPUs to parallelize model over"}
    )
    quantization_config_path: Optional[str] = field(
        default=None, metadata={"help": "path to quantization config file"}
    )
    profile: bool = field(
        default=False, metadata={"help": "enable autograd profiler emit_nvtx"}
    )
    reset_logging: bool = field(
        default=False,
        metadata={
            "help": "when using Hydra, reset the logging at the beginning of training"
        },
    )
    suppress_crashes: bool = field(
        default=False,
        metadata={
            "help": "suppress crashes when training with the hydra_train entry point so that the "
                    "main method can return a value (useful for sweeps)"
        },
    )
    use_plasma_view: bool = field(
        default=False, metadata={"help": "Store indices and sizes in shared memory"}
    )
    plasma_path: Optional[str] = field(
        default="/tmp/plasma",
        metadata={
            "help": "path to run plasma_store, defaults to /tmp/plasma. Paths outside /tmp tend to fail."
        },
    )


@dataclass
class DistributedTrainingConfig(FairseqDataclass):
    distributed_world_size: int = field(
        default=max(1, torch.cuda.device_count()),
        metadata={
            "help": "total number of GPUs across all nodes (default: all visible GPUs)"
        },
    )
    distributed_rank: Optional[int] = field(
        default=0, metadata={"help": "rank of the current worker"}
    )
    distributed_backend: str = field(
        default="nccl", metadata={"help": "distributed backend"}
    )
    distributed_init_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "typically tcp://hostname:port that will be used to "
            "establish initial connetion"
        },
    )
    distributed_port: int = field(
        default=-1,
        metadata={
            "help": "port number (not required if using --distributed-init-method)"
        },
    )
    device_id: int = field(
        default=0,
        metadata={
            "help": "which GPU to use (usually configured automatically)",
            "argparse_alias": "--local_rank",
        },
    )
    distributed_no_spawn: bool = field(
        default=False,
        metadata={
            "help": "do not spawn multiple processes even if multiple GPUs are visible"
        },
    )
    ddp_backend: DDP_BACKEND_CHOICES = field(
        default="pytorch_ddp", metadata={"help": "DistributedDataParallel backend"}
    )
    ddp_comm_hook: DDP_COMM_HOOK_CHOICES = field(
        default="none", metadata={"help": "communication hook"}
    )
    bucket_cap_mb: int = field(
        default=25, metadata={"help": "bucket size for reduction"}
    )
    fix_batches_to_gpus: bool = field(
        default=False,
        metadata={
            "help": "don't shuffle batches between GPUs; this reduces overall "
            "randomness and may affect precision but avoids the cost of re-reading the data"
        },
    )
    find_unused_parameters: bool = field(
        default=False,
        metadata={
            "help": "disable unused parameter detection (not applicable to "
            "--ddp-backend=legacy_ddp)"
        },
    )
    fast_stat_sync: bool = field(
        default=False,
        metadata={"help": "[deprecated] this is now defined per Criterion"},
    )
    heartbeat_timeout: int = field(
        default=-1,
        metadata={
            "help": "kill the job if no progress is made in N seconds; "
            "set to -1 to disable"
        },
    )
    broadcast_buffers: bool = field(
        default=False,
        metadata={
            "help": "Copy non-trainable parameters between GPUs, such as "
            "batchnorm population statistics"
        },
    )
    slowmo_momentum: Optional[float] = field(
        default=None,
        metadata={
            "help": "SlowMo momentum term; by default use 0.0 for 16 GPUs, "
            "0.2 for 32 GPUs; 0.5 for 64 GPUs, 0.6 for > 64 GPUs"
        },
    )
    slowmo_algorithm: str = field(
        default="LocalSGD", metadata={"help": "whether to use LocalSGD or SGP"}
    )
    localsgd_frequency: int = field(
        default=3, metadata={"help": "Local SGD allreduce frequency"}
    )
    nprocs_per_node: int = field(
        default=max(1, torch.cuda.device_count()),
        metadata={
            "help": "number of GPUs in each node. An allreduce operation across GPUs in "
            "a node is very fast. Hence, we do allreduce across GPUs in a node, "
            "and gossip across different nodes"
        },
    )
    pipeline_model_parallel: bool = field(
        default=False,
        metadata={"help": "if set, use pipeline model parallelism across GPUs"},
    )
    pipeline_balance: Optional[str] = field(
        default=None,
        metadata={
            "help": "partition the model into N_K pieces, where each piece "
            "contains N_i layers. The sum(args.pipeline_balance) "
            "should equal the total number of layers in the model"
        },
    )
    pipeline_devices: Optional[str] = field(
        default=None,
        metadata={
            "help": "a list of device indices indicating which device to place "
            "each of the N_K partitions. The length of this list should "
            "equal the length of the --pipeline-balance argument"
        },
    )
    pipeline_chunks: Optional[int] = field(
        default=0, metadata={"help": "microbatch count for pipeline model parallelism"}
    )
    pipeline_encoder_balance: Optional[str] = field(
        default=None,
        metadata={
            "help": "partition the pipeline parallel encoder into N_K pieces, where each piece "
            "contains N_i layers. The sum(args.pipeline_encoder_balance) "
            "should equal the total number of encoder layers in the model"
        },
    )
    pipeline_encoder_devices: Optional[str] = field(
        default=None,
        metadata={
            "help": "a list of device indices indicating which device to place "
            "each of the N_K partitions. The length of this list should "
            "equal the length of the --pipeline-encoder-balance argument"
        },
    )
    pipeline_decoder_balance: Optional[str] = field(
        default=None,
        metadata={
            "help": "partition the pipeline parallel decoder into N_K pieces, where each piece "
            "contains N_i layers. The sum(args.pipeline_decoder_balance) "
            "should equal the total number of decoder layers in the model"
        },
    )
    pipeline_decoder_devices: Optional[str] = field(
        default=None,
        metadata={
            "help": "a list of device indices indicating which device to place "
            "each of the N_K partitions. The length of this list should "
            "equal the length of the --pipeline-decoder-balance argument"
        },
    )
    pipeline_checkpoint: PIPELINE_CHECKPOINT_CHOICES = field(
        default="never",
        metadata={"help": "checkpointing mode for pipeline model parallelism"},
    )
    zero_sharding: ZERO_SHARDING_CHOICES = field(
        default="none", metadata={"help": "ZeRO sharding"}
    )
    fp16: bool = II("common.fp16")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    tpu: bool = II("common.tpu")
    # configuration for --ddp-backend=fully_sharded
    no_reshard_after_forward: bool = field(
        default=False, metadata={"help": "don't reshard parameters after forward pass"},
    )
    fp32_reduce_scatter: bool = field(
        default=False, metadata={"help": "reduce-scatter grads in FP32"},
    )
    cpu_offload: bool = field(
        default=False, metadata={"help": "offload FP32 params to CPU"}
    )
    use_sharded_state: bool = field(
        default=False, metadata={"help": "use sharded checkpoint files"},
    )


@dataclass
class DatasetConfig(FairseqDataclass):
    num_workers: int = field(
        default=1, metadata={"help": "how many subprocesses to use for data loading"}
    )
    skip_invalid_size_inputs_valid_test: bool = field(
        default=False,
        metadata={"help": "ignore too long or too short lines in valid and test set"},
    )
    max_tokens: Optional[int] = field(
        default=None, metadata={"help": "maximum number of tokens in a batch"}
    )
    batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "number of examples in a batch",
            "argparse_alias": "--max-sentences",
        },
    )
    required_batch_size_multiple: int = field(
        default=8, metadata={"help": "batch size will be a multiplier of this value"}
    )
    required_seq_len_multiple: int = field(
        default=1,
        metadata={
            "help": "maximum sequence length in batch will be a multiplier of this value"
        },
    )
    dataset_impl: Optional[DATASET_IMPL_CHOICES] = field(
        default=None, metadata={"help": "output dataset implementation"}
    )
    data_buffer_size: int = field(
        default=10, metadata={"help": "Number of batches to preload"}
    )
    train_subset: str = field(
        default="train",
        metadata={"help": "data subset to use for training (e.g. train, valid, test)"},
    )
    valid_subset: str = field(
        default="valid",
        metadata={
            "help": "comma separated list of data subsets to use for validation"
            " (e.g. train, valid, test)"
        },
    )
    combine_valid_subsets: Optional[bool] = field(
        default=None,
        metadata={
            "help": "comma separated list of data subsets to use for validation"
                    " (e.g. train, valid, test)",
            "argparse_alias": "--combine-val",
        },
    )
    ignore_unused_valid_subsets: Optional[bool] = field(
        default=False,
        metadata={"help": "do not raise error if valid subsets are ignored"},
    )

    validate_interval: int = field(
        default=1, metadata={"help": "validate every N epochs"}
    )
    validate_interval_updates: int = field(
        default=0, metadata={"help": "validate every N updates"}
    )
    validate_after_updates: int = field(
        default=0, metadata={"help": "dont validate until reaching this many updates"}
    )
    fixed_validation_seed: Optional[int] = field(
        default=None, metadata={"help": "specified random seed for validation"}
    )
    disable_validation: bool = field(
        default=False, metadata={"help": "disable validation"}
    )
    max_tokens_valid: Optional[int] = field(
        default=II("dataset.max_tokens"),
        metadata={
            "help": "maximum number of tokens in a validation batch"
            " (defaults to --max-tokens)"
        },
    )
    batch_size_valid: Optional[int] = field(
        default=II("dataset.batch_size"),
        metadata={
            "help": "batch size of the validation batch (defaults to --batch-size)",
            "argparse_alias": "--max-sentences-valid",
        },
    )
    max_valid_steps: Optional[int] = field(default=None, metadata={'help': 'How many batches to evaluate',
                                                                   "argparse_alias": "--nval"})
    curriculum: int = field(
        default=0, metadata={"help": "don't shuffle batches for first N epochs"}
    )
    gen_subset: str = field(
        default="test",
        metadata={"help": "data subset to generate (train, valid, test)"},
    )
    num_shards: int = field(
        default=1, metadata={"help": "shard generation over N shards"}
    )
    shard_id: int = field(
        default=0, metadata={"help": "id of the shard to generate (id < num_shards)"}
    )


@dataclass
class OptimizationConfig(FairseqDataclass):
    max_epoch: int = field(
        default=0, metadata={"help": "force stop training at specified epoch"}
    )
    max_update: int = field(
        default=0, metadata={"help": "force stop training at specified update"}
    )
    stop_time_hours: float = field(
        default=0,
        metadata={
            "help": "force stop training after specified cumulative time (if >0)"
        },
    )
    clip_norm: float = field(
        default=0.0, metadata={"help": "clip threshold of gradients"}
    )
    sentence_avg: bool = field(
        default=False,
        metadata={
            "help": "normalize gradients by the number of sentences in a batch"
            " (default is to normalize by number of tokens)"
        },
    )
    update_freq: List[int] = field(
        default_factory=lambda: [1],
        metadata={"help": "update parameters every N_i batches, when in epoch i"},
    )
    lr: List[float] = field(
        default_factory=lambda: [0.25],
        metadata={
            "help": "learning rate for the first N epochs; all epochs >N using LR_N"
            " (note: this may be interpreted differently depending on --lr-scheduler)"
        },
    )
    stop_min_lr: float = field(
        default=-1.0,
        metadata={"help": "stop training when the learning rate reaches this minimum"},
    )
    use_bmuf: bool = field(
        default=False,
        metadata={
            "help": "specify global optimizer for syncing models on different GPUs/shards"
        },
    )


@dataclass
class CheckpointConfig(FairseqDataclass):
    save_dir: str = field(
        default="checkpoints", metadata={"help": "path to save checkpoints"}
    )
    restore_file: str = field(
        default="checkpoint_last.pt",
        metadata={
            "help": "filename from which to load checkpoint "
            "(default: <save-dir>/checkpoint_last.pt"
        },
    )
    finetune_from_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "finetune from a pretrained model; note that meters and lr scheduler will be reset"
        },
    )
    reset_dataloader: bool = field(
        default=False,
        metadata={
            "help": "if set, does not reload dataloader state from the checkpoint"
        },
    )
    reset_lr_scheduler: bool = field(
        default=False,
        metadata={
            "help": "if set, does not load lr scheduler state from the checkpoint"
        },
    )
    reset_meters: bool = field(
        default=False,
        metadata={"help": "if set, does not load meters from the checkpoint"},
    )
    reset_optimizer: bool = field(
        default=False,
        metadata={"help": "if set, does not load optimizer state from the checkpoint"},
    )
    optimizer_overrides: str = field(
        default="{}",
        metadata={
            "help": "a dictionary used to override optimizer args when loading a checkpoint"
        },
    )
    save_interval: int = field(
        default=1, metadata={"help": "save a checkpoint every N epochs"}
    )
    save_interval_updates: int = field(
        default=0, metadata={"help": "save a checkpoint (and validate) every N updates"}
    )
    keep_interval_updates: int = field(
        default=-1,
        metadata={
            "help": "keep the last N checkpoints saved with --save-interval-updates"
        },
    )
    keep_interval_updates_pattern: int = field(
        default=-1,
        metadata={
            "help": "when used with --keep-interval-updates, skips deleting "
                    "any checkpoints with update X where "
                    "X %% keep_interval_updates_pattern == 0"
        },
    )
    keep_last_epochs: int = field(
        default=-1, metadata={"help": "keep last N epoch checkpoints"}
    )
    keep_best_checkpoints: int = field(
        default=-1, metadata={"help": "keep best N checkpoints based on scores"}
    )
    no_save: bool = field(
        default=False, metadata={"help": "don't save models or checkpoints"}
    )
    no_epoch_checkpoints: bool = field(
        default=False, metadata={"help": "only store last and best checkpoints"}
    )
    no_last_checkpoints: bool = field(
        default=False, metadata={"help": "don't store last checkpoints"}
    )
    no_save_optimizer_state: bool = field(
        default=False,
        metadata={"help": "don't save optimizer-state as part of checkpoint"},
    )
    best_checkpoint_metric: str = field(
        default="loss", metadata={"help": 'metric to use for saving "best" checkpoints'}
    )
    maximize_best_checkpoint_metric: bool = field(
        default=False,
        metadata={
            "help": 'select the largest metric value for saving "best" checkpoints'
        },
    )
    patience: int = field(
        default=-1,
        metadata={
            "help": (
                "early stop training if valid performance doesn't "
                "improve for N consecutive validation runs; note "
                "that this is influenced by --validate-interval"
            )
        },
    )
    checkpoint_suffix: str = field(
        default="", metadata={"help": "suffix to add to the checkpoint file name"}
    )
    checkpoint_shard_count: int = field(
        default=1,
        metadata={
            "help": "Number of shards containing the checkpoint - "
            "if the checkpoint is over 300GB, it is preferable "
            "to split it into shards to prevent OOM on CPU while loading "
            "the checkpoint"
        },
    )
    load_checkpoint_on_all_dp_ranks: bool = field(
        default=False,
        metadata={
            "help": "load checkpoints on all data parallel devices "
            "(default: only load on rank 0 and broadcast to other devices)"
        },
    )
    write_checkpoints_asynchronously: bool = field(
        default=False,
        metadata={
            "help": (
                "Write checkpoints asynchronously in a separate "
                "thread. NOTE: This feature is currently being tested."
            ),
            "argparse_alias": "--save-async",
        },
    )
    model_parallel_size: int = II("common.model_parallel_size")


@dataclass
class FairseqBMUFConfig(FairseqDataclass):
    block_lr: float = field(
        default=1, metadata={"help": "block learning rate for bmuf"}
    )
    block_momentum: float = field(
        default=0.875, metadata={"help": "block momentum for bmuf"}
    )
    global_sync_iter: int = field(
        default=50, metadata={"help": "Iteration for syncing global model"}
    )
    warmup_iterations: int = field(
        default=500, metadata={"help": "warmup iterations for model to broadcast"}
    )
    use_nbm: bool = field(
        default=False,
        metadata={"help": "Specify whether you want to use classical BM / Nesterov BM"},
    )
    average_sync: bool = field(
        default=False,
        metadata={
            "help": "Specify whether you want to average the local momentum after each sync"
        },
    )
    distributed_world_size: int = II("distributed_training.distributed_world_size")


@dataclass
class GenerationConfig(FairseqDataclass):
    beam: int = field(
        default=5, metadata={"help": "beam size"},
    )
    nbest: int = field(
        default=1, metadata={"help": "number of hypotheses to output"},
    )
    max_len_a: float = field(
        default=0,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length"
        },
    )
    max_len_b: int = field(
        default=200,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length"
        },
    )
    min_len: int = field(
        default=1, metadata={"help": "minimum generation length"},
    )
    match_source_len: bool = field(
        default=False, metadata={"help": "generations should match the source length"},
    )
    unnormalized: bool = field(
        default=False, metadata={"help": "compare unnormalized hypothesis scores"},
    )
    no_early_stop: bool = field(
        default=False, metadata={"help": "deprecated"},
    )
    no_beamable_mm: bool = field(
        default=False, metadata={"help": "don't use BeamableMM in attention layers"},
    )
    lenpen: float = field(
        default=1,
        metadata={
            "help": "length penalty: <1.0 favors shorter, >1.0 favors longer sentences"
        },
    )
    unkpen: float = field(
        default=0,
        metadata={
            "help": "unknown word penalty: <0 produces more unks, >0 produces fewer"
        },
    )
    replace_unk: Optional[str] = field(
        default=None,
        metadata={
            "help": "perform unknown replacement (optionally with alignment dictionary)",
            "argparse_const": "@@ ",
        },
    )
    sacrebleu: bool = field(
        default=False, metadata={"help": "score with sacrebleu"},
    )
    score_reference: bool = field(
        default=False, metadata={"help": "just score the reference translation"},
    )
    prefix_size: int = field(
        default=0,
        metadata={"help": "initialize generation by target prefix of given length"},
    )
    no_repeat_ngram_size: int = field(
        default=0,
        metadata={
            "help": "ngram blocking such that this size ngram cannot be repeated in the generation"
        },
    )
    sampling: bool = field(
        default=False,
        metadata={"help": "sample hypotheses instead of using beam search"},
    )
    sampling_topk: int = field(
        default=-1,
        metadata={"help": "sample from top K likely next words instead of all words"},
    )
    sampling_topp: float = field(
        default=-1.0,
        metadata={
            "help": "sample from the smallest set whose cumulative probability mass exceeds p for next words"
        },
    )
    constraints: Optional[GENERATION_CONSTRAINTS_CHOICES] = field(
        default=None,
        metadata={
            "help": "enables lexically constrained decoding",
            "argparse_const": "ordered",
        },
    )
    temperature: float = field(
        default=1.0, metadata={"help": "temperature for generation"},
    )
    diverse_beam_groups: int = field(
        default=-1, metadata={"help": "number of groups for Diverse Beam Search"},
    )
    diverse_beam_strength: float = field(
        default=0.5,
        metadata={"help": "strength of diversity penalty for Diverse Beam Search"},
    )
    diversity_rate: float = field(
        default=-1.0,
        metadata={"help": "strength of diversity penalty for Diverse Siblings Search"},
    )
    print_alignment: Optional[PRINT_ALIGNMENT_CHOICES] = field(
        default=None,
        metadata={
            "help": "if set, uses attention feedback to compute and print alignment to source tokens "
            "(valid options are: hard, soft, otherwise treated as hard alignment)",
            "argparse_const": "hard",
        },
    )
    print_step: bool = field(
        default=False, metadata={"help": "print steps"},
    )
    lm_path: Optional[str] = field(
        default=None, metadata={"help": "path to lm checkpoint for lm fusion"},
    )
    lm_weight: float = field(
        default=0.0, metadata={"help": "weight for lm probs for lm fusion"},
    )

    # arguments for iterative refinement generator
    iter_decode_eos_penalty: float = field(
        default=0.0,
        metadata={"help": "if > 0.0, it penalized early-stopping in decoding."},
    )
    iter_decode_max_iter: int = field(
        default=10, metadata={"help": "maximum iterations for iterative refinement."},
    )
    iter_decode_force_max_iter: bool = field(
        default=False,
        metadata={
            "help": "if set, run exact the maximum number of iterations without early stop"
        },
    )
    iter_decode_with_beam: int = field(
        default=1,
        metadata={
            "help": "if > 1, model will generate translations varying by the lengths."
        },
    )
    iter_decode_with_external_reranker: bool = field(
        default=False,
        metadata={
            "help": "if set, the last checkpoint are assumed to be a reranker to rescore the translations"
        },
    )
    retain_iter_history: bool = field(
        default=False,
        metadata={
            "help": "if set, decoding returns the whole history of iterative refinement"
        },
    )
    retain_dropout: bool = field(
        default=False, metadata={"help": "Use dropout at inference time"},
    )
    # temporarily set to Any until https://github.com/facebookresearch/hydra/issues/1117 is fixed
    # retain_dropout_modules: Optional[List[str]] = field(
    retain_dropout_modules: Any = field(
        default=None,
        metadata={
            "help": "if set, only retain dropout for the specified modules; "
            "if not set, then dropout will be retained for all modules"
        },
    )
    # special decoding format for advanced decoding.
    decoding_format: Optional[GENERATION_DECODING_FORMAT_CHOICES] = field(
        default=None,
        metadata={"help": "special decoding format for advanced decoding."},
    )
    no_seed_provided: bool = field(
        default=False,
        metadata={"help": "if set, dont use seed for initializing random generators"},
    )


@dataclass
class CommonEvalConfig(FairseqDataclass):
    path: Optional[str] = field(
        default=None, metadata={"help": "path(s) to model file(s), colon separated"},
    )
    post_process: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "post-process text by removing BPE, letter segmentation, etc. "
                "Valid options can be found in fairseq.data.utils.post_process."
            ),
            "argparse_const": "subword_nmt",
            "argparse_alias": "--remove-bpe",
        },
    )
    quiet: bool = field(default=False, metadata={"help": "only print final scores"})
    model_overrides: str = field(
        default="{}",
        metadata={
            "help": "a dictionary used to override model args at generation that were used during model training"
        },
    )
    results_path: Optional[str] = field(
        default=None, metadata={"help": "path to save eval results (optional)"}
    )


@dataclass
class EvalLMConfig(FairseqDataclass):
    output_word_probs: bool = field(
        default=False,
        metadata={
            "help": "if set, outputs words and their predicted log probabilities to standard output"
        },
    )
    output_word_stats: bool = field(
        default=False,
        metadata={
            "help": "if set, outputs word statistics such as word count, average probability, etc"
        },
    )
    context_window: int = field(
        default=0,
        metadata={
            "help": "ensures that every evaluated token has access to a context of at least this size, if possible"
        },
    )
    softmax_batch: int = field(
        default=sys.maxsize,
        metadata={
            "help": "if BxT is more than this, will batch the softmax over vocab to this amount of tokens, in order to fit into GPU memory"
        },
    )


@dataclass
class InteractiveConfig(FairseqDataclass):
    buffer_size: int = field(
        default=0,
        metadata={
            "help": "read this many sentences into a buffer before processing them"
        },
    )
    input: str = field(
        default="-", metadata={"help": "file to read from; use - for stdin"},
    )


@dataclass
class FairseqConfig(FairseqDataclass):
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    bmuf: FairseqBMUFConfig = FairseqBMUFConfig()
    generation: GenerationConfig = GenerationConfig()
    eval_lm: EvalLMConfig = EvalLMConfig()
    interactive: InteractiveConfig = InteractiveConfig()
    model: Any = MISSING
    task: Any = None
    criterion: Any = None
    optimizer: Any = None
    lr_scheduler: Any = None
    scoring: Any = None
    bpe: Any = None
    tokenizer: Any = None
