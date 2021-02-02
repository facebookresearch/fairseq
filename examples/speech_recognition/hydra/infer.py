#!/usr/bin/env python -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import hashlib
import logging
import os
import shutil
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import editdistance
import torch
import torch.distributed as dist
from examples.speech_recognition.hydra.decoder import Decoder, DecoderConfig
from fairseq import (checkpoint_utils, distributed_utils, progress_bar, tasks,
                     utils)
from fairseq.data.data_utils import post_process
from fairseq.dataclass.configs import (CheckpointConfig, CommonConfig,
                                       CommonEvalConfig, DatasetConfig,
                                       DistributedTrainingConfig,
                                       FairseqDataclass, GenerationConfig)
from fairseq.dataclass.initialize import hydra_init
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.logging.progress_bar import BaseProgressBar
from fairseq.models.fairseq_model import FairseqModel
from omegaconf import MISSING, OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"


@dataclass
class DecodingConfig(FairseqDataclass):
    exp_dir: str = field(
        default=MISSING,
        metadata={"help": "Path to the experiment directory"},
    )
    unique_wer_file: bool = field(
        default=False,
        metadata={"help": "If set, use a unique file for storing WER"},
    )
    write_sentences: bool = field(
        default=True,
        metadata={"help": "If set, write hypothesis and reference sentences"},
    )
    decoder: DecoderConfig = DecoderConfig()


@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    decoding: DecodingConfig = DecodingConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    generation: GenerationConfig = GenerationConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()


def reset_logging():
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)
    root.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)


class InferenceProcessor:
    def __init__(self, cfg: InferConfig) -> None:
        self.cfg = cfg
        self.task = tasks.setup_task(cfg.task)
        self.tgt_dict = self.task.target_dictionary

        models, saved_cfg = self.load_model_ensemble()
        self.models = models
        self.saved_cfg = saved_cfg

        self.task.load_dataset(
            self.cfg.dataset.gen_subset,
            task_cfg=saved_cfg.task,
        )
        self.generator = Decoder(cfg.decoding.decoder, self.tgt_dict)
        self.gen_timer = StopwatchMeter()
        self.wps_meter = TimeMeter()
        self.num_sentences = 0
        self.total_errors = 0
        self.total_length = 0

        self.hypo_words_file = None
        self.hypo_units_file = None
        self.ref_words_file = None
        self.ref_units_file = None

        self.progress_bar = self.build_progress_bar()

    def __enter__(self) -> "InferenceProcessor":
        if self.cfg.decoding.write_sentences:
            self.hypo_words_file = self.get_res_file("hypo.word")
            self.hypo_units_file = self.get_res_file("hypo.units")
            self.ref_words_file = self.get_res_file("ref.word")
            self.ref_units_file = self.get_res_file("ref.units")
        return self

    def __exit__(self, *exc) -> bool:
        if self.cfg.decoding.write_sentences:
            self.hypo_words_file.close()
            self.hypo_units_file.close()
            self.ref_words_file.close()
            self.ref_units_file.close()
        return False

    def __iter__(self) -> Any:
        for sample in self.progress_bar:
            if not self.cfg.common.cpu:
                sample = utils.move_to_cuda(sample)

            # Happens on the last batch.
            if "net_input" not in sample:
                continue
            yield sample

    def log(self, *args, **kwargs):
        self.progress_bar.log(*args, **kwargs)

    def print(self, *args, **kwargs):
        self.progress_bar.print(*args, **kwargs)

    def get_res_file(self, fname: str) -> None:
        if self.data_parallel_world_size > 1:
            fname = f"{fname}.{self.data_parallel_rank}"
        return open(fname, "w", buffering=1)

    def merge_shards(self) -> None:
        """Merges all shard files into shard 0, then removes shard suffix."""

        shard_id = self.data_parallel_rank
        num_shards = self.data_parallel_world_size

        def merge_shards_with_root(fname: str) -> None:
            logger.info("Merging %s on shard %d", fname, shard_id)
            base_fpath = Path(f"{fname}.0")
            with open(base_fpath, "a") as out_file:
                for s in range(1, num_shards):
                    shard_fpath = Path(f"{fname}.{s}")
                    with open(shard_fpath, "r") as in_file:
                        for line in in_file:
                            out_file.write(line)
                    shard_fpath.unlink()
            shutil.move(f"{fname}.0", fname)

        if shard_id == (0 % num_shards):
            merge_shards_with_root("hypo.word")
        if shard_id == (1 % num_shards):
            merge_shards_with_root("hypo.units")
        if shard_id == (2 % num_shards):
            merge_shards_with_root("ref.word")
        if shard_id == (3 % num_shards):
            merge_shards_with_root("ref.units")
        dist.barrier()

    def optimize_model(self, model: FairseqModel) -> None:
        gcfg = self.cfg.generation
        model.make_generation_fast_(
            beamable_mm_beam_size=None if gcfg.no_beamable_mm else gcfg.beam,
            need_attn=gcfg.print_alignment,
        )
        if self.cfg.common.fp16:
            model.half()
        if not self.cfg.common.cpu:
            model.cuda()

    def load_model_ensemble(self) -> Tuple[List[FairseqModel], FairseqDataclass]:
        arg_overrides = ast.literal_eval(self.cfg.common_eval.model_overrides)
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.cfg.common_eval.path),
            arg_overrides=arg_overrides,
            task=self.task,
            suffix=self.cfg.checkpoint.checkpoint_suffix,
            strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=self.cfg.checkpoint.checkpoint_shard_count,
        )
        for model in models:
            self.optimize_model(model)
        return models, saved_cfg

    def get_dataset_itr(self, disable_iterator_cache: bool = False) -> None:
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.cfg.dataset.gen_subset),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=(sys.maxsize, sys.maxsize),
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.cfg.dataset.num_workers,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        ).next_epoch_itr(shuffle=False)

    def build_progress_bar(
        self,
        epoch: Optional[int] = None,
        prefix: Optional[str] = None,
        default_log_format: str = "tqdm",
    ) -> BaseProgressBar:
        return progress_bar.progress_bar(
            iterator=self.get_dataset_itr(),
            log_format=self.cfg.common.log_format,
            log_interval=self.cfg.common.log_interval,
            epoch=epoch,
            prefix=prefix,
            tensorboard_logdir=self.cfg.common.tensorboard_logdir,
            default_log_format=default_log_format,
        )

    @property
    def data_parallel_world_size(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_rank(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    def process_sentence(
        self,
        sample: Dict[str, Any],
        hypo: Dict[str, Any],
        sid: int,
        batch_id: int,
    ) -> Tuple[int, int]:
        speaker = None  # Speaker can't be parsed from dataset.

        if "target_label" in sample:
            toks = sample["target_label"]
        else:
            toks = sample["target"]
        toks = toks[batch_id, :]

        # Processes hypothesis.
        hyp_pieces = self.tgt_dict.string(hypo["tokens"].int().cpu())
        if "words" in hypo:
            hyp_words = " ".join(hypo["words"])
        else:
            hyp_words = post_process(hyp_pieces,
                                     self.cfg.common_eval.post_process)

        # Processes target.
        target_tokens = utils.strip_pad(toks, self.tgt_dict.pad())
        tgt_pieces = self.tgt_dict.string(target_tokens.int().cpu())
        tgt_words = post_process(tgt_pieces,
                                 self.cfg.common_eval.post_process)

        if self.cfg.decoding.write_sentences:
            print(f"{hyp_pieces} ({speaker}-{sid})", file=self.hypo_units_file)
            print(f"{hyp_words} ({speaker}-{sid})", file=self.hypo_words_file)
            print(f"{tgt_pieces} ({speaker}-{sid})", file=self.ref_units_file)
            print(f"{tgt_words} ({speaker}-{sid})", file=self.ref_words_file)

        hyp_words, tgt_words = hyp_words.split(), tgt_words.split()

        return editdistance.eval(hyp_words, tgt_words), len(tgt_words)

    def process_sample(self, sample: Dict[str, Any]) -> None:
        self.gen_timer.start()
        hypos = self.task.inference_step(
            generator=self.generator,
            models=self.models,
            sample=sample,
        )
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        self.gen_timer.stop(num_generated_tokens)
        self.wps_meter.update(num_generated_tokens)

        for batch_id, sample_id in enumerate(sample["id"].tolist()):
            errs, length = self.process_sentence(
                sample=sample,
                sid=sample_id,
                batch_id=batch_id,
                hypo=hypos[batch_id][0],
            )
            self.total_errors += errs
            self.total_length += length

        self.log({"wps": round(self.wps_meter.avg)})
        if "nsentences" in sample:
            self.num_sentences += sample["nsentences"]
        else:
            self.num_sentences += sample["id"].numel()

    def log_generation_time(self) -> None:
        logger.info("Processed %d sentences (%d tokens) in %.1fs %.2f "
                    "sentences per second, %.2f tokens per second)",
                    self.num_sentences, self.gen_timer.n, self.gen_timer.sum,
                    self.num_sentences / self.gen_timer.sum,
                    1.0 / self.gen_timer.avg)


def parse_wer(wer_file: Path) -> float:
    with open(wer_file, "r") as f:
        return float(f.readline().strip().split(" ")[1])


def get_wer_file(cfg: InferConfig) -> Path:
    """Hashes the decoding parameters to a unique file ID."""
    if cfg.decoding.unique_wer_file:
        yaml_str = OmegaConf.to_yaml(cfg.decoding)
        fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
        return Path(f"wer.{fid % 1000000}")
    else:
        return Path("wer")


def main(cfg: InferConfig) -> float:
    """Entry point for main processing logic.

    Args:
        cfg: The inferance configuration to use.
        wer: Optional shared memory pointer for returning the WER. If not None,
            the final WER value will be written here instead of being returned.

    Returns:
        The final WER if `wer` is None, otherwise None.
    """

    yaml_str, wer_file = OmegaConf.to_yaml(cfg.decoding), get_wer_file(cfg)

    # Validates the provided configuration.
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 4000000
    if not cfg.common.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA not found; set `cpu=True` to run without CUDA")
    if cfg.generation.nbest > 1:
        raise ValueError("`nbest > 1` not implemented yet")

    with InferenceProcessor(cfg) as processor:
        for sample in processor:
            processor.process_sample(sample)

        processor.log_generation_time()

        if cfg.decoding.write_sentences:
            processor.merge_shards()

        errs_t, leng_t = processor.total_errors, processor.total_length

        if cfg.common.cpu:
            logger.warning("Merging WER requires CUDA.")
        else:
            stats = torch.LongTensor([errs_t, leng_t]).cuda()
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            errs_t, leng_t = stats[0].item(), stats[1].item()

        wer = errs_t * 100.0 / leng_t

        if distributed_utils.is_master(cfg.distributed_training):
            with open(wer_file, "w") as f:
                f.write(f"WER: {wer}\n\n{yaml_str}")

        return wer


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> None:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Working directory: %s", Path.cwd())
    wer = float("inf")

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

        wer = parse_wer(get_wer_file(cfg))
    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))

    logger.info("Word error rate: %.4f", wer)
    return wer


def cli_main() -> None:
    try:
        from hydra._internal.utils import \
            get_args  # pylint: disable=import-outside-toplevel
        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        v = InferConfig.__dataclass_fields__[k].default
        try:
            cs.store(name=k, node=v)
        except BaseException:
            logger.error(f"{k} - {v}")
            raise

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()
