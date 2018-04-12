# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import pickle

import torch.distributed

from fairseq import utils


def is_master(args):
    return args.distributed_rank == 0


def distributed_init(args):
    if args.distributed_world_size == 1:
        raise ValueError('Cannot initialize distributed with distributed_world_size=1')

    print('| distributed init (rank {}): {}'.format(
        args.distributed_rank, args.distributed_init_method), flush=True)
    if args.distributed_init_method.startswith('tcp://'):
        torch.distributed.init_process_group(
            backend=args.distributed_backend, init_method=args.distributed_init_method,
            world_size=args.distributed_world_size, rank=args.distributed_rank)
    else:
        torch.distributed.init_process_group(
            backend=args.distributed_backend, init_method=args.distributed_init_method,
            world_size=args.distributed_world_size)

    args.distributed_rank = torch.distributed.get_rank()
    if not is_master(args):
        suppress_output()

    return args.distributed_rank


def suppress_output():
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        if 'force' in kwargs:
            force = kwargs.pop('force')
            if force:
                builtin_print(*args, **kwargs)

    __builtin__.print = print


def all_gather_list(data, max_size=4096):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if not hasattr(all_gather_list, '_in_buffer') or \
            max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [
            torch.cuda.ByteTensor(max_size)
            for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255*256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2:enc_size+2] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    result = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * utils.item(out_buffer[0])) + utils.item(out_buffer[1])
        result.append(
            pickle.loads(bytes(out_buffer[2:size+2].tolist()))
        )
    return result
