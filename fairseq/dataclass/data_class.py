# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Type, Tuple
import torch
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass.utils import FairseqDataclass, ChoiceEnum
import sys
from fairseq.tasks import TASK_DATACLASS_REGISTRY
from fairseq.models import ARCH_DATACLASS_REGISTRY
from fairseq.criterions import CRITERION_DATACLASS_REGISTRY
from fairseq.optim import OPTIMIZER_DATACLASS_REGISTRY
from fairseq.optim.bmuf import FairseqBMUFConfig
from fairseq.optim.lr_scheduler import LR_SCHEDULER_DATACLASS_REGISTRY
from hydra.core.config_store import ConfigStore
from argparse import Namespace


@dataclass
class CommonParams(FairseqDataclass):
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
    log_format: Optional[ChoiceEnum(["json", "none", "simple", "tqdm"])] = field(
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
    checkpoint_suffix: str = field(
        default="", metadata={"help": "suffix to add to the checkpoint file name"}
    )
    quantization_config_path: Optional[str] = field(
        default=None, metadata={"help": "path to quantization config file"}
    )
    profile: bool = field(
        default=False, metadata={"help": "enable autograd profiler emit_nvtx"}
    )


@dataclass
class DistributedTrainingParams(FairseqDataclass):
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
    ddp_backend: ChoiceEnum(["c10d", "no_c10d"]) = field(
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
    distributed_wrapper: ChoiceEnum(["DDP", "SlowMo"]) = field(
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


@dataclass
class DatasetParams(FairseqDataclass):
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
    max_sentences: Optional[int] = field(
        default=None, metadata={"help": "maximum number of sentences in a batch"}
    )
    batch_size: Optional[int] = field(
        default=None, metadata={"help": "maximum number of sentences in a batch"}
    )
    required_batch_size_multiple: int = field(
        default=8, metadata={"help": "batch size will be a multiplier of this value"}
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
    max_sentences_valid: Optional[int] = field(
        default=None,
        metadata={
            "help": "maximum number of sentences in a validation batch"
            " (defaults to --max-sentences)"
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
class OptimizationParams(FairseqDataclass):
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
class CheckpointParams(FairseqDataclass):
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


@dataclass
class CommonEvalParams(FairseqDataclass):
    path: Optional[str] = field(
        default=None, metadata={"help": "path(s) to model file(s), colon separated"}
    )
    remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE tokens before scoring (can be set to sentencepiece)"
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
class EvalLMParams(FairseqDataclass):
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
class TrainingConfig(FairseqDataclass):
    """Config for training, a composition of training params"""

    common: CommonParams = CommonParams()
    distributed_training: DistributedTrainingParams = DistributedTrainingParams()
    dataset: DatasetParams = DatasetParams()
    optimization: OptimizationParams = OptimizationParams()
    checkpoint: CheckpointParams = CheckpointParams()
    bmuf: FairseqBMUFConfig = FairseqBMUFConfig()


@dataclass
class EvalLMConfig(FairseqDataclass):
    """Config for eval lm, a composition of eval_lm params"""

    common: CommonParams = CommonParams()
    distributed_training: DistributedTrainingParams = DistributedTrainingParams()
    dataset: DatasetParams = DatasetParams()
    optimization: OptimizationParams = OptimizationParams()
    checkpoint: CheckpointParams = CheckpointParams()
    bmuf: FairseqBMUFConfig = FairseqBMUFConfig()
    common_eval: CommonEvalParams = CommonEvalParams()
    eval_lm: EvalLMParams = EvalLMParams()


def register_params_dataclass(
    cs: ConfigStore, name: str, group: str, data_class: Type[FairseqDataclass]
) -> None:
    """register params dataclass in config store"""
    node_ = data_class(_name=data_class.name())
    cs.store(name=name, group=group, node=node_)


def register_module_dataclass(
    cs: ConfigStore, registry: Dict[str, Any], group: str
) -> None:
    """register dataclasses defined in modules in config store, for example, in migrated tasks, models, etc."""
    # note that if `group == model`, we register all model archs, not the model name.
    for k, v in registry.items():
        if v is not None:
            node_ = v(_name=k)
            cs.store(name=k, group=group, node=node_)


def register_training_hydra_cfg(cs: ConfigStore, name: str = "default") -> None:
    """cs: config store instance, register common training configs"""

    register_params_dataclass(
        cs, name="training_params", group="params", data_class=TrainingConfig
    )

    register_module_dataclass(cs, TASK_DATACLASS_REGISTRY, "task")
    register_module_dataclass(cs, ARCH_DATACLASS_REGISTRY, "model")
    register_module_dataclass(cs, CRITERION_DATACLASS_REGISTRY, "criterion")
    register_module_dataclass(cs, OPTIMIZER_DATACLASS_REGISTRY, "optimizer")
    register_module_dataclass(cs, LR_SCHEDULER_DATACLASS_REGISTRY, "lr_scheduler")


def register_eval_lm_hydra_cfg(cs: ConfigStore, name: str = "default") -> None:
    """cs: config store instance, register common training configs"""

    register_params_dataclass(
        cs, name="eval_lm_params", group="params", data_class=EvalLMConfig
    )

    register_module_dataclass(cs, TASK_DATACLASS_REGISTRY, "task")
    register_module_dataclass(cs, CRITERION_DATACLASS_REGISTRY, "criterion")
    register_module_dataclass(cs, OPTIMIZER_DATACLASS_REGISTRY, "optimizer")
    register_module_dataclass(cs, LR_SCHEDULER_DATACLASS_REGISTRY, "lr_scheduler")


def _override_attr(
    sub_node: str, data_class: Type[FairseqDataclass], args: Namespace
) -> List[str]:
    overrides = []
    for k in data_class.__dataclass_fields__.keys():
        if k == "_name":
            # private member, skip
            continue
        if not hasattr(args, k):
            # print(f"cannot override {sub_node}.{k} since args does not have attribute {k}")
            continue
        if getattr(args, k) is None:
            overrides.append("{}.{}=null".format(sub_node, k))
        elif getattr(args, k) == "":
            overrides.append("{}.{}=''".format(sub_node, k))
        elif isinstance(getattr(args, k), str):
            if (
                getattr(args, k).startswith("[")
                or getattr(args, k).startswith("(")
                or getattr(args, k).startswith("{")
                or ("," in getattr(args, k))
            ):
                overrides.append("{}.{}='{}'".format(sub_node, k, getattr(args, k)))
            else:
                overrides.append("{}.{}={}".format(sub_node, k, getattr(args, k)))
        else:
            overrides.append("{}.{}={}".format(sub_node, k, getattr(args, k)))
    return overrides


def override_training_args(args: Namespace) -> Tuple[List[str], List[str]]:
    overrides = []

    overrides.extend(_override_attr("params.common", CommonParams, args))
    overrides.extend(_override_attr("params.dataset", DatasetParams, args))
    overrides.extend(
        _override_attr("params.distributed_training", DistributedTrainingParams, args)
    )
    overrides.extend(_override_attr("params.optimization", OptimizationParams, args))
    overrides.extend(_override_attr("params.checkpoint", CheckpointParams, args))
    overrides.extend(_override_attr("params.bmuf", FairseqBMUFConfig, args))
    module_overrides, module_deletes = override_module_args(args)
    overrides.extend(module_overrides)

    return overrides, module_deletes


def override_eval_lm_args(args: Namespace) -> Tuple[List[str], List[str]]:
    overrides = []

    overrides.extend(_override_attr("params.common", CommonParams, args))
    overrides.extend(_override_attr("params.dataset", DatasetParams, args))
    overrides.extend(
        _override_attr("params.distributed_training", DistributedTrainingParams, args)
    )
    overrides.extend(_override_attr("params.common_eval", CommonEvalParams, args))
    overrides.extend(_override_attr("params.eval_lm", EvalLMParams, args))
    overrides.extend(_override_attr("params.bmuf", FairseqBMUFConfig, args))
    module_overrides, module_deletes = override_module_args(args)
    overrides.extend(module_overrides)

    return overrides, module_deletes


def override_module_args(args: Namespace) -> Tuple[List[str], List[str]]:
    """use the field in args to overrides those in cfg"""
    overrides = []
    deletes = []

    if args is not None:
        assert (
            hasattr(args, "task")
            and hasattr(args, "criterion")
            and hasattr(args, "optimizer")
            and hasattr(args, "lr_scheduler")
        )
        if args.task in TASK_DATACLASS_REGISTRY:
            overrides.append("task={}".format(args.task))
            overrides.append("task._name={}".format(args.task))
            overrides.extend(
                _override_attr("task", TASK_DATACLASS_REGISTRY[args.task], args)
            )
        else:
            deletes.append("task")
        if args.criterion in CRITERION_DATACLASS_REGISTRY:
            overrides.append("criterion={}".format(args.criterion))
            overrides.append("criterion._name={}".format(args.criterion))
            overrides.extend(
                _override_attr(
                    "criterion", CRITERION_DATACLASS_REGISTRY[args.criterion], args
                )
            )
        else:
            deletes.append("criterion")
        if args.optimizer in OPTIMIZER_DATACLASS_REGISTRY:
            overrides.append("optimizer={}".format(args.optimizer))
            overrides.append("optimizer._name={}".format(args.optimizer))
            overrides.extend(
                _override_attr(
                    "optimizer", OPTIMIZER_DATACLASS_REGISTRY[args.optimizer], args
                )
            )
        else:
            deletes.append("optimizer")
        if args.lr_scheduler in LR_SCHEDULER_DATACLASS_REGISTRY:
            overrides.append("lr_scheduler={}".format(args.lr_scheduler))
            overrides.append("lr_scheduler._name={}".format(args.lr_scheduler))
            overrides.extend(
                _override_attr(
                    "lr_scheduler",
                    LR_SCHEDULER_DATACLASS_REGISTRY[args.lr_scheduler],
                    args,
                )
            )
        else:
            deletes.append("lr_scheduler")

        if hasattr(args, "arch"):
            if args.arch in ARCH_DATACLASS_REGISTRY:
                overrides.append("model={}".format(args.arch))
                overrides.append("model._name={}".format(args.arch))
                # override model params with those exist in args
                overrides.extend(
                    _override_attr("model", ARCH_DATACLASS_REGISTRY[args.arch], args)
                )
            else:
                deletes.append("model")

    return overrides, deletes
