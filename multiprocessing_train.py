# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
import random
import torch

from torch import multiprocessing
from train import main as single_process_main, parse_train_args
from fairseq.distributed_utils import distributed_init, supress_output

def main():

    args = parse_train_args()
    if args.distributed_port == -1:
        args.distributed_port = random.randint(10000, 20000)

    args.distributed_world_size = torch.cuda.device_count()
    args.distributed_master_host = 'localhost'
    args.distributed_init_method = f'tcp://{args.distributed_master_host}:{args.distributed_port + 1}'

    mp = multiprocessing.get_context("spawn")
    procs = []

    for i in range(args.distributed_world_size):
        args.device_id = i
        args.distributed_rank = i
        procs.append(mp.Process(target=run, args=(args, )))
        procs[i].start()

    for p in procs:
        p.join()
        print(f'Process {p} complete')


def run(args):
    distributed_init(args)

    if args.device_id != 0:
        supress_output()

    single_process_main(args)

if __name__ == '__main__':
    main()
