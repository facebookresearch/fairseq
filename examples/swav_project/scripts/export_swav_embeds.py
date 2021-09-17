#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from cmath import log
from collections import Counter
from ctypes import util
import shutil
from fairseq.data import indexed_dataset
from multiprocessing import Pool, Value
from fairseq.binarizer import Binarizer
import functools
import time

from importlib_metadata import metadata
import numpy as np
from fairseq.dataclass.configs import FairseqConfig
import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
# from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.dataclass import utils as dataclass_utils
from fairseq.dataclass import configs
from typing import Any, Dict, List, Optional, Tuple, Type

from fairseq.logging import metrics, progress_bar
from omegaconf import DictConfig


from fairseq.options import get_parser, add_dataset_args, add_distributed_training_args
from fairseq.options import gen_parser_from_dataclass
from fairseq.options import CommonEvalConfig
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional
from fairseq.dataclass import FairseqDataclass
from sklearn.cluster import KMeans
from torch.nn import CosineSimilarity
import pickle

# from fairseq.swav_utils import custom_convert_namespace_to_omegaconf, custom_override_module_args

# NOTE ---- MISCELLANEOUS ----------------
def custom_override_module_args(args: Namespace, config_class=FairseqConfig) -> Tuple[List[str], List[str]]:
    """use the field in args to overrides those in cfg"""
    overrides = []
    deletes = []

    for k in config_class.__dataclass_fields__.keys():
        overrides.extend(
            dataclass_utils._override_attr(k, config_class.__dataclass_fields__[k].type, args)
        )

    if args is not None:
        if hasattr(args, "task"):
            from fairseq.tasks import TASK_DATACLASS_REGISTRY

            dataclass_utils.migrate_registry(
                "task", args.task, TASK_DATACLASS_REGISTRY, args, overrides, deletes
            )
        else:
            deletes.append("task")

        # these options will be set to "None" if they have not yet been migrated
        # so we can populate them with the entire flat args
        CORE_REGISTRIES = {"criterion", "optimizer", "lr_scheduler"}

        from fairseq.registry import REGISTRIES

        for k, v in REGISTRIES.items():
            if hasattr(args, k):
                dataclass_utils.migrate_registry(
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
            from fairseq.models import ARCH_MODEL_REGISTRY, ARCH_MODEL_NAME_REGISTRY

            if args.arch in ARCH_MODEL_REGISTRY:
                m_cls = ARCH_MODEL_REGISTRY[args.arch]
                dc = getattr(m_cls, "__dataclass", None)
                if dc is not None:
                    m_name = ARCH_MODEL_NAME_REGISTRY[args.arch]
                    overrides.append("model={}".format(m_name))
                    overrides.append("model._name={}".format(args.arch))
                    # override model params with those exist in args
                    overrides.extend(dataclass_utils._override_attr("model", dc, args))
                    no_dc = False
        if no_dc:
            deletes.append("model")

    return overrides, deletes


def custom_convert_namespace_to_omegaconf(args: Namespace, override_module_args=None) -> DictConfig:
    # Here we are using field values provided in args to override counterparts inside config object
    if override_module_args is None:
        override_module_args = dataclass_utils.override_module_args
        # override_module_args = functools.partial(custom_override_module_args, config_class=GatherParaDataFairseqConfig)
    overrides, deletes = override_module_args(args)

    # configs will be in fairseq/config after installation
    config_path = os.path.join("..", "config")
    # config_path = os.path.join(".", "config")

    dataclass_utils.GlobalHydra.instance().clear()

    with dataclass_utils.initialize(config_path=config_path):
        try:
            composed_cfg = dataclass_utils.compose("config", overrides=overrides, strict=False)
        except Exception:
            logger.error("Error when composing. Overrides: " + str(overrides))
            raise

        for k in deletes:
            composed_cfg[k] = None

    cfg = dataclass_utils.OmegaConf.create(
        dataclass_utils.OmegaConf.to_container(composed_cfg, resolve=True, enum_to_str=True)
    )

    # hack to be able to set Namespace in dict config. this should be removed when we update to newer
    # omegaconf version that supports object flags, or when we migrate all existing models
    from omegaconf import _utils

    old_primitive = _utils.is_primitive_type
    _utils.is_primitive_type = lambda _: True

    if cfg.task is None and getattr(args, "task", None):
        cfg.task = Namespace(**vars(args))
        from fairseq.tasks import TASK_REGISTRY

        dataclass_utils._set_legacy_defaults(cfg.task, TASK_REGISTRY[args.task])
        cfg.task._name = args.task
    if cfg.model is None and getattr(args, "arch", None):
        cfg.model = Namespace(**vars(args))
        from fairseq.models import ARCH_MODEL_REGISTRY

        dataclass_utils._set_legacy_defaults(cfg.model, ARCH_MODEL_REGISTRY[args.arch])
        cfg.model._name = args.arch
    if cfg.optimizer is None and getattr(args, "optimizer", None):
        cfg.optimizer = Namespace(**vars(args))
        from fairseq.optim import OPTIMIZER_REGISTRY

        dataclass_utils._set_legacy_defaults(cfg.optimizer, OPTIMIZER_REGISTRY[args.optimizer])
        cfg.optimizer._name = args.optimizer
    if cfg.lr_scheduler is None and getattr(args, "lr_scheduler", None):
        cfg.lr_scheduler = Namespace(**vars(args))
        from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY

        dataclass_utils._set_legacy_defaults(cfg.lr_scheduler, LR_SCHEDULER_REGISTRY[args.lr_scheduler])
        cfg.lr_scheduler._name = args.lr_scheduler
    if cfg.criterion is None and getattr(args, "criterion", None):
        cfg.criterion = Namespace(**vars(args))
        from fairseq.criterions import CRITERION_REGISTRY

        dataclass_utils._set_legacy_defaults(cfg.criterion, CRITERION_REGISTRY[args.criterion])
        cfg.criterion._name = args.criterion

    _utils.is_primitive_type = old_primitive
    dataclass_utils.OmegaConf.set_struct(cfg, True)
    return cfg





logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("export_sent_embeddings")



logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("gather_swav_pseudo_data")


@dataclass
class AnalysisConfig(FairseqDataclass):
    swav_langs: str = field(
        default=None, metadata={"help": "langs"},
    )
    analyze_name: Optional[str] = field(
        default="aly", metadata={"help": "path to lm checkpoint for lm fusion"},
    )
    analyze_max_step: int = field(
        default=500,
        metadata={
            "help": "ensures that every evaluated token has access to a context of at least this size, if possible"
        },
    )
    no_train_subset_shuffle: bool = field(
        default=False, metadata={"help": "default to shuffle train set"}
    )
    aly_para: bool = field(
        default=False, metadata={"help": "parallel data analysis, instead of mono data"}
    )
    no_aly_save: bool = field(
        default=False, metadata={"help": "save the data"}
    )
    aly_subsets: str = field(
        default="train", metadata={"help": "subset to run analysis"}
    )
    export_flush_steps: int = field(
        default=-1, metadata={"help": "Export Flush into multiple .pth data after x steps"}
    )



@dataclass
class ExportSwavEmbedsFairseqConfig(FairseqConfig):
    # swav_clustering: SwavClusteringConfig = SwavClusteringConfig()
    analysis: AnalysisConfig = AnalysisConfig()


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, lang, "bin"),
        impl=args.dataset_impl,
        vocab_size=len(vocab),
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(
        filename, vocab, consumer, append_eos=append_eos, offset=offset, end=end
    )
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    if lang is not None:
        lang_part = ".{}-{}.{}".format(args.source_lang, args.target_lang, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}-{}".format(args.source_lang, args.target_lang)

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def export_text_data(src_texts, tgt_texts, src_path, tgt_path):
    with open(src_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(src_texts))
    
    with open(tgt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tgt_texts))



def mono_obtain_clustered_para_data(
    cfg, task, model, criterion, saved_cfg, models, 
    data_parallel_world_size,
    data_parallel_rank, **kwargs):
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    assert isinstance(cfg.analysis.swav_langs, str)
    swav_langs = cfg.analysis.swav_langs.split(",")
    # export_only = cfg.swav_clustering.export_only
    export_flush_steps = cfg.analysis.export_flush_steps
    dictionary = task.source_dictionary

    rank_reprs = f'[{data_parallel_rank}/{data_parallel_world_size}]'

    raw_key = "raw"

    def export_prototype_dataset(_subset, dataset, shuffle, lang):
        # Initialize data iterator
        model.cuda()
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
            # quick fix
            epoch=1,
            disable_iterator_cache=False,
        ).next_epoch_itr(shuffle=shuffle)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{_subset}' subset",
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )
        log_outputs = []
        cur_step = 0
        max_step = cfg.analysis.analyze_max_step
        dummy_batch = None
        analyze_name = f'{cfg.analysis.analyze_name}.limit{max_step}.{_subset}.r{data_parallel_rank}'
        for i, sample in enumerate(progress):
            is_dummy = False
            if sample is None or not bool(sample) or len(sample) == 0:
                logger.warning(f'{rank_reprs} encounter empty sample at step {i}, convert to dummy batch')
                assert dummy_batch is not None
                is_dummy = True
                sample = dummy_batch
            if dummy_batch is None:
                dummy_batch = sample

            sample = utils.move_to_cuda(sample) if use_cuda else sample
            aly_output = task.analyze_step(sample, model, criterion, cfg, is_dummy=is_dummy)
            if export_flush_steps > 1 and i > 1 and i % export_flush_steps == 0:
                tmp_analyze_name = f'{analyze_name}i{i}'
                logger.warning(f'{rank_reprs} flushing pth data: {tmp_analyze_name}')
                aly_obj = task.analyze_done(cfg, 
                    infer_name=tmp_analyze_name, 
                    save_path=os.path.join(cfg.common_eval.results_path, f'{lang}_part{data_parallel_rank}'), 
                    save=True, 
                    world_size=data_parallel_world_size, 
                    rank=data_parallel_rank, 
                )        
            if i > max_step:
                break
        logger.warning(f'{rank_reprs} flushing final pth.data: {analyze_name}')
        aly_obj = task.analyze_done(cfg,
            infer_name=analyze_name, 
            save_path=os.path.join(cfg.common_eval.results_path, f'{lang}_part{data_parallel_rank}'),
            save=True, 
            world_size=data_parallel_world_size, 
            rank=data_parallel_rank, 
        )
            
        del progress
        del itr
        torch.cuda.empty_cache()
        # offload the model to CUDA to free up memory for other stuff
        model.cpu()
        # if export_only:
        #     return None
        # return aly_obj

    valid_subsets = getattr(cfg.dataset, "valid_subset").split(",")
    for vsubset in valid_subsets:
        shuffle = not cfg.analysis.no_train_subset_shuffle if "train" in vsubset else False
        logger.warning(f'mono analysis on [{vsubset}] {rank_reprs} with shuffle={shuffle}')
        # objs = []
        for lang_id, lang in enumerate(swav_langs):
            try:
                task.load_dataset_for_analysis(vsubset, combine=False, epoch=1, 
                    task_cfg=saved_cfg.task, train_lang_sep=True)
                subset = f'{vsubset}_{lang}'
                dataset = task.dataset(subset)
            except KeyError:
                raise Exception(f"Cannot find dataset: {rank_reprs}" + subset)
            export_prototype_dataset(subset, dataset, shuffle, lang)
            # objs.append(obj)
            keys = list(task.datasets.keys())
            for k in keys:
                logger.warning(f'Removing dataset to avoid OOM: {k}')
                del task.datasets[k]
            logger.warning(f'DEBUG: STOP at 1st lang')
            continue
        # if export_only:
        #     continue
        


script_convert_namespace_to_omegaconf = functools.partial(custom_convert_namespace_to_omegaconf,
    override_module_args=functools.partial(custom_override_module_args, config_class=ExportSwavEmbedsFairseqConfig)
)
def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = script_convert_namespace_to_omegaconf(cfg),

    utils.import_user_module(cfg.common)

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None
    
    logger.warning(f'overrides: {overrides}')
    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    task = tasks.setup_task(cfg.task)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
        task=task,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()

    logger.warning(f'Analyze on distributed     : {cfg.distributed_training}')
    logger.warning(f'Analyze on common_eval     : {cfg.common_eval}')
    logger.warning(f'Analyze on analysis        : {cfg.analysis}')

    mono_obtain_clustered_para_data(cfg, task, model, criterion, saved_cfg, 
        models, data_parallel_world_size, data_parallel_rank
    )


def get_validation_parser(default_task=None):
    parser = get_parser("Validation", default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser, default_world_size=1)
    group = parser.add_argument_group("Evaluation")
    gen_parser_from_dataclass(group, CommonEvalConfig())
    group = parser.add_argument_group("ExportSwavEmbeds")
    gen_parser_from_dataclass(group, ExportSwavEmbedsFairseqConfig())
    return parser


def cli_main():
    parser = get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = get_validation_parser()
    override_args = options.parse_args_and_arch(
        override_parser, suppress_defaults=True
    )
    print("Overwrite: {}".format(override_args))

    distributed_utils.call_main(
        script_convert_namespace_to_omegaconf(args), main, override_args=override_args
    )


if __name__ == "__main__":
    cli_main()






