# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import torch.fb.rendezvous.zeus  # noqa: F401
from fairseq import options
from fairseq.file_io import PathManager
from fairseq_cli.train import distributed_main, main
from fairseq_latte_prod import tasks  # noqa
from fvcore.fb.manifold import ManifoldPathHandler


def get_fb_training_parser():
    parser = options.get_training_parser()
    parser.add_argument(
        "--tensorboard-manifold",
        action="store_true",
        help="[FB only] send tensorboard plots to manifold",
    )
    parser.add_argument(
        "--log-dir",
        metavar="LOG",
        default=None,
        help="[FB only] Dir to store log in addition to stdout. If this "
        "is not set, it will be set to args.save_dir",
    )
    return parser


def fb_main(
    device_id,
    args,
    start_rank,
    log_path=None,
    after_distributed_init_fn: Optional[
        Callable[[argparse.Namespace], argparse.Namespace]
    ] = None,
):
    """[FB] entry point for each worker process."""

    args.distributed_rank = start_rank + device_id

    def add_handler(handler):
        for root in ["fairseq", "fairseq_cli"]:
            logger = logging.getLogger(root)
            logger.propagate = False  # don't propagate to parent loggers
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)

    # write fairseq logs to stdout
    add_handler(logging.StreamHandler(sys.stdout))

    # support Manifold for checkpoints
    # For latte_training use case, we have separate NMTManifoldPathHandler registered in
    # https://fburl.com/wurd7t70. So if parameters need to be updated the right place
    # is ~/fbsource/fbcode/fblearner/flow/projects/fairseq/latte_training/manifold_file_io.py
    try:
        PathManager.register_handler(ManifoldPathHandler(max_parallel=16, timeout_sec=1800))
    except KeyError:
        logging.warning("ManifoldPathHandler already registered.")

    def train_main():
        distributed_main(
            device_id, args, after_distributed_init_fn=after_distributed_init_fn
        )

    if log_path is not None and args.distributed_rank == 0:
        # write logs from worker 0 to train.log
        PathManager.mkdirs(args.save_dir)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        Path(log_path).touch(0o777, exist_ok=True)
        add_handler(logging.FileHandler(log_path))
        train_main()
    else:
        train_main()


if __name__ == "__main__":
    parser = get_fb_training_parser()
    args = options.parse_args_and_arch(parser)

    log_dir = args.log_dir if args.log_dir is not None else args.save_dir
    log_path = os.path.join(log_dir, "train.log")

    if args.distributed_init_method is not None and torch.cuda.device_count() > 1:
        start_rank = args.distributed_rank
        args.distributed_rank = None
        torch.multiprocessing.spawn(
            fn=fb_main,
            args=(args, start_rank, log_path),
            nprocs=torch.cuda.device_count(),
        )
    else:
        # single GPU training
        fb_main(args.device_id, args, 0, log_path)
