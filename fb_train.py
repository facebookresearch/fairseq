# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import contextlib
import os
from pathlib import Path
import sys

import torch.fb.rendezvous.zeus  # noqa: F401

from fairseq import options

from train import distributed_main, main


def zeus_distributed_main(device_id, args, start_rank, log_path=None):
    args.distributed_rank = start_rank + device_id
    if log_path is not None and start_rank == 0 and device_id == 0:
        with open(log_path, "a") as h:
            with contextlib.redirect_stdout(tee(sys.stdout, h)):
                with contextlib.redirect_stderr(tee(sys.stderr, h)):
                    distributed_main(device_id, args)
    else:
        distributed_main(device_id, args)


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


if __name__ == '__main__':
    parser = options.get_training_parser()
    parser.add_argument("--tensorboard-manifold", action="store_true",
                        help="[FB only] send tensorboard plots to manifold")
    args = options.parse_args_and_arch(parser)
    if args.tensorboard_logdir and args.tensorboard_manifold:
        raise ValueError(
            "Invalid Args: --tensorboard_logdir and --tensorboard_manifold are both specified."
        )
    if args.tensorboard_manifold:
        args.tbmf_wrapper = True
    log_path = os.path.join(args.save_dir, 'train.log')
    Path(log_path).touch(0o777, exist_ok=True)
    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1:
            start_rank = args.distributed_rank
            args.distributed_rank = None
            torch.multiprocessing.spawn(
                fn=zeus_distributed_main,
                args=(args, start_rank, log_path),
                nprocs=torch.cuda.device_count(),
            )
        else:
            zeus_distributed_main(args.device_id, args, 0, log_path)
    else:
        with open(log_path, "a") as h:
            with contextlib.redirect_stdout(tee(sys.stdout, h)):
                with contextlib.redirect_stderr(tee(sys.stderr, h)):
                    # single GPU training
                    main(args)
