#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import os
import pathlib
import re
import sys
from argparse import Namespace

import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.utils import safe_getattr

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.weight_viz")


@torch.inference_mode
def add_histograms(
    model: torch.nn.Module,
    writer: SummaryWriter,
    global_step: int,
):
    for k, p in model.named_parameters():
        if not p.requires_grad or model._is_non_quant_param(k):
            continue

        writer.add_histogram(k, p, global_step=global_step)


def main(cfg: DictConfig, **unused_kwargs):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    logger.info(cfg)

    if cfg.eval_lm.context_window > 0:
        # reduce tokens per sample by the required context window size
        cfg.task.tokens_per_sample -= cfg.eval_lm.context_window

    # Initialize the task using the current *cfg*
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))

    writer = SummaryWriter(log_dir=cfg.common_eval.path)
    for fname in pathlib.Path(cfg.common_eval.path).glob("*[0-9].pt"):
        fname = str(fname)
        digits = re.findall(r"\d+", fname)
        assert digits, f"Invalid checkpoint path: {fname}"
        global_step = int(digits.pop())
        print(f"Processing {fname}")

        state = checkpoint_utils.load_checkpoint_to_cpu(
            checkpoint_utils.get_maybe_sharded_checkpoint_filename(
                filename=fname,
                suffix=cfg.checkpoint.checkpoint_suffix,
                shard_idx=0,
                num_shards=cfg.checkpoint.checkpoint_shard_count,
            ),
            arg_overrides=eval(cfg.common_eval.model_overrides),
        )
        _models, model_args, _ = checkpoint_utils.load_model_ensemble_and_task(
            [cfg.common_eval.path],
            num_shards=cfg.checkpoint.checkpoint_shard_count,
            task=task,
            state=state,
        )

        model = _models.pop()
        is_madgrad_par = (
            model_args["optimizer"]["_name"] == "madgrad"
            and safe_getattr(model_args["optimizer"], "quant_method") == "parq"
            and safe_getattr(model_args["optimizer"], "quant_bits", 32) < 32
        )
        if is_madgrad_par:
            optimizer_state = state["last_optimizer_state"]["state"]
            model.copy_par_optimizer_z(optimizer_state)
        else:
            cfg["optimizer"] = model_args["optimizer"]
            cfg["model"] = model_args["model"]
            model.prepare_for_inference_(cfg)
        add_histograms(model, writer, global_step)


def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()
