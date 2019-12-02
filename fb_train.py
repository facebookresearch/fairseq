# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import sys
from pathlib import Path

import torch.fb.rendezvous.zeus  # noqa: F401
from fairseq import options
from fairseq.fb_pathmgr import fb_pathmgr
from fvcore.common.file_io import PathManager
from train import distributed_main, main


def get_fb_training_parser():
    parser = options.get_training_parser()
    parser.add_argument(
        "--tensorboard-manifold",
        action="store_true",
        help="[FB only] send tensorboard plots to manifold",
    )
    return parser


def fb_main(device_id, args, start_rank, log_path=None):
    """[FB] entry point for each worker process."""

    args.distributed_rank = start_rank + device_id

    # support Manifold for checkpoints
    fb_pathmgr.register()

    def train_main():
        if args.distributed_world_size > 1:
            distributed_main(device_id, args)
        else:
            main(args)

    if log_path is not None and args.distributed_rank == 0:
        # write logs from worker 0 to train.log
        PathManager.mkdirs(os.path.dirname(log_path))
        Path(log_path).touch(0o777, exist_ok=True)
        with PathManager.open(log_path, "a") as h:
            with contextlib.redirect_stdout(tee(sys.stdout, h)):
                with contextlib.redirect_stderr(tee(sys.stderr, h)):
                    train_main()
    else:
        train_main()


class tee(object):
    """Source: http://shallowsky.com/blog/programming/python-tee.html"""

    def __init__(self, _fd1, _fd2):
        self.fd1 = _fd1
        self.fd2 = _fd2

    def __del__(self):
        if self.fd1 != sys.stdout and self.fd1 != sys.stderr:
            self.fd1.close()
        if self.fd2 != sys.stdout and self.fd2 != sys.stderr:
            self.fd2.close()

    def write(self, text):
        self.fd1.write(text)
        self.fd1.flush()
        self.fd2.write(text)
        self.fd2.flush()

    def flush(self):
        self.fd1.flush()
        self.fd2.flush()


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
