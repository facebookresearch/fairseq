#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from fairseq import distributed_utils, options

from train import main as single_process_main


def main(args):
    if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
        print('| WARNING: when using --update-freq on a single machine, you '
              'will get better performance with --ddp-backend=no_c10d')

    # Train with multiprocessing.
    base_rank = args.distributed_rank
    torch.multiprocessing.spawn(
        fn=run,
        args=(args, base_rank, ),
        nprocs=torch.cuda.device_count(),
        daemon=True,
    )


def run(i, args, base_rank):
    args.distributed_rank = base_rank + i
    args.device_id = i
    args.distributed_rank = distributed_utils.distributed_init(args)
    single_process_main(args)


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
