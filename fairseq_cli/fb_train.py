# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import torch.fb.rendezvous.zeus  # noqa: F401
from fairseq import distributed_utils, options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.file_io import PathManager
from fairseq_cli.train import main as fairseq_train_main
from fvcore.fb.manifold import ManifoldPathHandler


logger = logging.getLogger(__file__)


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

    # For latte_training use case, we have separate NMTManifoldPathHandler registered in
    # https://fburl.com/wurd7t70. So if parameters need to be updated the right place
    # is ~/fbsource/fbcode/fblearner/flow/projects/fairseq/latte_training/manifold_file_io.py
    # for manifold
    parser.add_argument(
        "--manifold-max-parallel",
        default=8,
        type=int,
        help="set ManifoldPathHandler max_parallel download number",
    )
    parser.add_argument(
        "--manifold-timeout-sec",
        default=1800,
        type=int,
        help="set ManifoldPathHandler timeout seconds",
    )
    parser.add_argument(
        "--manifold-has-user-data",
        default=True,
        type=lambda x: x.lower() not in ("no", "false", "f", "n", "0")
        if x is not None
        else None,
        help="set ManifoldPathHandler has_user_data option",
    )
    parser.add_argument(
        "--manifold-num-retries",
        default=15,
        type=int,
        help="set ManifoldPathHandler num_retries option",
    )
    parser.add_argument(
        "--manifold-ttl",
        default=None,
        type=int,
        help="A manifold  resource's time-to-live, applied to all manifold written resources. By default, there is no TTL.",
    )
    # for manifold
    return parser


def init_manifold(args):
    # support Manifold for checkpoints
    # For latte_training use case, we use a separate NMTManifoldPathHandler
    # registered in https://fburl.com/diffusion/djgz9bwx.
    try:
        PathManager.register_handler(
            ManifoldPathHandler(
                max_parallel=args.manifold_max_parallel,
                timeout_sec=args.manifold_timeout_sec,
                has_user_data=args.manifold_has_user_data,
                num_retries=args.manifold_num_retries,
                ttl=args.manifold_ttl,
            )
        )
        logger.info(
            f"ManifoldPathHandler is set: max_parallel={args.manifold_max_parallel}, "
            f"timeout_sec={args.manifold_timeout_sec}; has_user_data={args.manifold_has_user_data}"
        )
    except KeyError:
        logging.warning("ManifoldPathHandler already registered.")


def fb_main(
    device_id,
    args,
    start_rank,
    log_path=None,
    manifold_log_uri=None,
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

    init_manifold(args)

    def train_main():
        cfg = convert_namespace_to_omegaconf(args)
        distributed_utils.distributed_main(
            device_id,
            fairseq_train_main,
            cfg,
            kwargs={
                "after_distributed_init_fn": after_distributed_init_fn,
            },
        )

    if manifold_log_uri is not None and log_path is None:
        log_path = tempfile.mktemp()

    if log_path is not None and args.distributed_rank == 0:
        # write logs from worker 0 to train.log
        PathManager.mkdirs(args.save_dir)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        Path(log_path).touch(0o777, exist_ok=True)
        add_handler(logging.FileHandler(log_path))
        train_main()
        if manifold_log_uri is not None:
            PathManager.copy_from_local(
                local_path=log_path, dst_path=manifold_log_uri, overwrite=True
            )
    else:
        train_main()


if __name__ == "__main__":
    parser = get_fb_training_parser()
    args = options.parse_args_and_arch(parser)

    log_dir = args.log_dir if args.log_dir is not None else args.save_dir
    log_path = os.path.join(log_dir, "train.log")

    distributed_utils.infer_init_method(
        convert_namespace_to_omegaconf(args), force_distributed=True
    )

    start_rank = args.distributed_rank
    args.distributed_rank = None  # assign automatically
    torch.multiprocessing.spawn(
        fn=fb_main,
        args=(args, start_rank, log_path),
        nprocs=min(
            torch.cuda.device_count(),
            args.distributed_world_size,
        ),
    )
