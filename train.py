#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq import options

from distributed_train import main as distributed_main
from multiprocessing_train import main as multiprocessing_main
from singleprocess_train import main as singleprocess_main


def main(args):
    if args.distributed_port > 0 \
            or args.distributed_init_method is not None:
        distributed_main(args)
    elif args.distributed_world_size > 1:
        multiprocessing_main(args)
    else:
        singleprocess_main(args)


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
