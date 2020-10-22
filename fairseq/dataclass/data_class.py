# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from argparse import Namespace
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass.constants import (
    DDP_BACKEND_CHOICES,
    DISTRIBUTED_WRAPPER_CHOICES,
    GENERATION_CONSTRAINTS_CHOICES,
    GENERATION_DECODING_FORMAT_CHOICES,
    LOG_FORMAT_CHOICES,
    PIPELINE_CHECKPOINT_CHOICES,
    ZERO_SHARDING_CHOICES,
)
from fairseq.dataclass.utils import ChoiceEnum, FairseqDataclass
from fairseq.models import ARCH_MODEL_REGISTRY, MODEL_DATACLASS_REGISTRY
from fairseq.optim.bmuf import FairseqBMUFConfig
from fairseq.registry import REGISTRIES
from fairseq.tasks import TASK_DATACLASS_REGISTRY
from hydra.core.config_store import ConfigStore
from omegaconf import II


logger = logging.getLogger(__name__)


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
    tensorboard_logdir: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to save logs for tensorboard, should match --logdir "
            "of running tensorboard (default: no tensorboard logging)"
        },
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
        metadata={"help": "minimum FP16 loss scale, after which training is stopped"},
    )
    threshold_loss_scale: Optional[float] = field(
        default=None, metadata={"help": "threshold FP16 loss scale from below"}
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
        metadata={"help": "which GPU to use (usually configured automatically)"},
    )
    local_rank: int = field(
        default=0,
        metadata={"help": "which GPU to use (usually configured automatically)"},
    )
    distributed_no_spawn: bool = field(
        default=False,
        metadata={
            "help": "do not spawn multiple processes even if multiple GPUs are visible"
        },
    )
    ddp_backend: DDP_BACKEND_CHOICES = field(
        default="c10d", metadata={"help": "DistributedDataParallel backend"}
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
            "no_c10d ddp-backend"
        },
    )
    fast_stat_sync: bool = field(
        default=False,
        metadata={"help": "[deprecated] this is now defined per Criterion"},
    )
    broadcast_buffers: bool = field(
        default=False,
        metadata={
            "help": "Copy non-trainable parameters between GPUs, such as "
            "batchnorm population statistics"
        },
    )
    distributed_wrapper: DISTRIBUTED_WRAPPER_CHOICES = field(
        default="DDP", metadata={"help": "DistributedDataParallel backend"}
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
    tpu: bool = II("common.tpu")


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
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = field(
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
        default=None,
        metadata={
            "help": "maximum number of tokens in a validation batch"
            " (defaults to --max-tokens)"
        },
    )
    batch_size_valid: Optional[int] = field(
        default=None,
        metadata={
            "help": "batch size of the validation batch" " (defaults to --batch-size)",
            "argparse_alias": "--max-sentences-valid",
        },
    )
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
        default=25.0, metadata={"help": "clip threshold of gradients"}
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
    min_lr: float = field(
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
    model_parallel_size: int = II("common.model_parallel_size")
    distributed_rank: int = II("distributed_training.distributed_rank")


@dataclass
class GenerationConfig(FairseqDataclass):
    beam: int = field(
        default=5,
        metadata={"help": "beam size"},
    )
    nbest: int = field(
        default=1,
        metadata={"help": "number of hypotheses to output"},
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
        default=1,
        metadata={"help": "minimum generation length"},
    )
    match_source_len: bool = field(
        default=False,
        metadata={"help": "generations should match the source length"},
    )
    unnormalized: bool = field(
        default=False,
        metadata={"help": "compare unnormalized hypothesis scores"},
    )
    no_early_stop: bool = field(
        default=False,
        metadata={"help": "deprecated"},
    )
    no_beamable_mm: bool = field(
        default=False,
        metadata={"help": "don't use BeamableMM in attention layers"},
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
        default=False,
        metadata={"help": "score with sacrebleu"},
    )
    score_reference: bool = field(
        default=False,
        metadata={"help": "just score the reference translation"},
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
        default=1.0,
        metadata={"help": "temperature for generation"},
    )
    diverse_beam_groups: int = field(
        default=-1,
        metadata={"help": "number of groups for Diverse Beam Search"},
    )
    diverse_beam_strength: float = field(
        default=0.5,
        metadata={"help": "strength of diversity penalty for Diverse Beam Search"},
    )
    diversity_rate: float = field(
        default=-1.0,
        metadata={"help": "strength of diversity penalty for Diverse Siblings Search"},
    )
    print_alignment: bool = field(
        default=False,
        metadata={
            "help": "if set, uses attention feedback to compute and print alignment to source tokens"
        },
    )
    print_step: bool = field(
        default=False,
        metadata={"help": "print steps"},
    )
    lm_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to lm checkpoint for lm fusion"},
    )
    lm_weight: float = field(
        default=0.0,
        metadata={"help": "weight for lm probs for lm fusion"},
    )

    # arguments for iterative refinement generator
    iter_decode_eos_penalty: float = field(
        default=0.0,
        metadata={"help": "if > 0.0, it penalized early-stopping in decoding."},
    )
    iter_decode_max_iter: int = field(
        default=10,
        metadata={"help": "maximum iterations for iterative refinement."},
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
        default=False,
        metadata={"help": "Use dropout at inference time"},
    )
    retain_dropout_modules: Optional[List[str]] = field(
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
        default=None,
        metadata={"help": "path(s) to model file(s), colon separated"},
    )
    post_process: Optional[str] = field(
        default=None,
        metadata={
            "help": "post-process text by removing pre-processing such as BPE, letter segmentation, etc "
            "(valid options are: sentencepiece, wordpiece, letter, _EOW, none, otherwise treated as BPE symbol)",
            "argparse_const": "@@ ",
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
        default="-",
        metadata={"help": "file to read from; use - for stdin"},
    )


CONFIGS = {
    "common": CommonConfig,
    "common_eval": CommonEvalConfig,
    "distributed_training": DistributedTrainingConfig,
    "dataset": DatasetConfig,
    "optimization": OptimizationConfig,
    "checkpoint": CheckpointConfig,
    "bmuf": FairseqBMUFConfig,
    "generation": GenerationConfig,
    "eval_lm": EvalLMConfig,
    "interactive": InteractiveConfig,
}


def register_module_dataclass(
    cs: ConfigStore, registry: Dict[str, Any], group: str
) -> None:
    """register dataclasses defined in modules in config store, for example, in migrated tasks, models, etc."""
    # note that if `group == model`, we register all model archs, not the model name.
    for k, v in registry.items():
        node_ = v()
        node_._name = k
        cs.store(name=k, group=group, node=node_, provider="fairseq")


def register_hydra_cfg(cs: ConfigStore, name: str = "default") -> None:
    """cs: config store instance, register common training configs"""

    for k, v in CONFIGS.items():
        try:
            cs.store(name=k, node=v())
        except BaseException:
            logger.error(f"{k} - {v()}")
            raise

    register_module_dataclass(cs, TASK_DATACLASS_REGISTRY, "task")
    register_module_dataclass(cs, MODEL_DATACLASS_REGISTRY, "model")

    for k, v in REGISTRIES.items():
        register_module_dataclass(cs, v["dataclass_registry"], k)


def _override_attr(
    sub_node: str, data_class: Type[FairseqDataclass], args: Namespace
) -> List[str]:
    overrides = []

    def get_default(f):
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    for k, v in data_class.__dataclass_fields__.items():
        if k.startswith("_"):
            # private member, skip
            continue

        val = get_default(v) if not hasattr(args, k) else getattr(args, k)

        if val is None:
            overrides.append("{}.{}=null".format(sub_node, k))
        elif val == "":
            overrides.append("{}.{}=''".format(sub_node, k))
        elif isinstance(val, str):
            overrides.append("{}.{}='{}'".format(sub_node, k, val))
        else:
            overrides.append("{}.{}={}".format(sub_node, k, val))
    return overrides


def migrate_registry(
    name, value, registry, args, overrides, deletes, use_name_as_val=False
):
    if value in registry:
        overrides.append("{}={}".format(name, value))
        overrides.append("{}._name={}".format(name, value))
        overrides.extend(_override_attr(name, registry[value], args))
    elif use_name_as_val and value is not None:
        overrides.append("{}={}".format(name, value))
    else:
        deletes.append(name)


def override_module_args(args: Namespace) -> Tuple[List[str], List[str]]:
    """use the field in args to overrides those in cfg"""
    overrides = []
    deletes = []

    for k, v in CONFIGS.items():
        overrides.extend(_override_attr(k, v, args))

    if args is not None:
        if hasattr(args, "task"):
            migrate_registry(
                "task", args.task, TASK_DATACLASS_REGISTRY, args, overrides, deletes
            )
        else:
            deletes.append("task")

        # these options will be set to "None" if they have not yet been migrated
        # so we can populate them with the entire flat args
        CORE_REGISTRIES = {"criterion", "optimizer", "lr_scheduler"}

        for k, v in REGISTRIES.items():
            if hasattr(args, k):
                migrate_registry(
                    k,
                    getattr(args, k),
                    v["dataclass_registry"],
                    args,
                    overrides,
                    deletes,
                    use_name_as_val=k not in CORE_REGISTRIES,
                )
            else:
                deletes.append(k)

        no_dc = True
        if hasattr(args, "arch"):
            if args.arch in ARCH_MODEL_REGISTRY:
                m_cls = ARCH_MODEL_REGISTRY[args.arch]
                dc = getattr(m_cls, "__dataclass", None)
                if dc is not None:
                    overrides.append("model={}".format(args.arch))
                    overrides.append("model._name={}".format(args.arch))
                    # override model params with those exist in args
                    overrides.extend(_override_attr("model", dc, args))
                    no_dc = False
        if no_dc:
            deletes.append("model")

    return overrides, deletes
