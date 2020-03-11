# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
import sys
from pathlib import Path

import torch.fb.rendezvous.zeus  # noqa: F401
from fairseq import options
from fairseq.file_io import PathManager
from fairseq_cli.train import distributed_main, main
from fvcore.fb.manifold import ManifoldPathHandler


def get_fb_training_parser():
    parser = options.get_training_parser()
    parser.add_argument(
        "--tensorboard-manifold",
        action="store_true",
        help="[FB only] send tensorboard plots to manifold",
    )
    parser.add_argument(
        "--manifold-ttl", type=int, help="[FB only] Set object ttl for manifold storage"
    )
    parser.add_argument(
        "--manifold-has-user-data",
        type=bool,
        default=False,
        help="[FB only] Set has-user-data-flag for manifold storage",
    )
    return parser


def fb_main(device_id, args, start_rank, log_path=None):
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
    PathManager.register_handler(ManifoldPathHandler(max_parallel=16, timeout_sec=1800))

    def train_main():
        if args.distributed_world_size > 1:
            distributed_main(device_id, args)
        else:
            main(args)

    if log_path is not None and args.distributed_rank == 0:
        # write logs from worker 0 to train.log
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        Path(log_path).touch(0o777, exist_ok=True)
        add_handler(logging.FileHandler(log_path))
        train_main()
    else:
        train_main()


if __name__ == "__main__":
    parser = get_fb_training_parser()
    args = options.parse_args_and_arch(parser)

    log_path = os.path.join(args.save_dir, "train.log")

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
