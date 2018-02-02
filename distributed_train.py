# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
import os
import socket
import subprocess

from train import main as single_process_main, parse_train_args
from fairseq.distributed_utils import distributed_init, supress_output

def main():
    args = parse_train_args()
    node_list = subprocess.check_output(['scontrol', 'show', 'hostnames',
                                         os.environ.get("SLURM_JOB_NODELIST")])
    args.distributed_master_host = node_list.split()[0].decode('utf-8')

    if args.distributed_port == -1:
        raise ValueError("--distributed-port must be specified for distributed training")

    if args.distributed_init_method is None:
        args.distributed_init_method = f'tcp://{args.distributed_master_host}:{args.distributed_port + 1}'

    args.device_id = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID")) * \
           int(os.environ.get("SLURM_NTASKS_PER_NODE")) + \
           args.device_id
    print("Rank: {}, host: {}, local rank {} ".format(rank, socket.gethostname(), args.device_id))
    args.distributed_rank = rank

    distributed_init(args)
    single_process_main(args)


if __name__ == '__main__':
    main()
