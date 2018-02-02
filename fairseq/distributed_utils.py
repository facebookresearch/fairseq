# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
import torch.distributed

def distributed_init(args):
    if args.distributed_world_size == 1:
        pass
    print(f'Distributed init {args.distributed_init_method}')
    if args.distributed_init_method.startswith("tcp://"):
        torch.distributed.init_process_group(backend=args.distributed_backend,
                                             init_method=args.distributed_init_method,
                                             world_size=args.distributed_world_size,
                                             rank=args.distributed_rank)
    else:
        torch.distributed.init_process_group(backend=args.distributed_backend,
                                             init_method=args.distributed_init_method,
                                             world_size=args.distributed_world_size)



def supress_output():
    import builtins as __builtin__
    # Supress printing for all but 0th device.
    # print(str, force=True) will force it print
    _print = __builtin__.print
    def print(*args, **kwargs):
        if 'force' in kwargs:
            force = kwargs.pop('force')
            if force:
                _print(*args, **kwargs)
    __builtin__.print = print
