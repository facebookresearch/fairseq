# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import namedtuple
import os
import pickle
import subprocess

import torch
from torch import nn

from fairseq import utils


def is_master(args):
    return args.distributed_rank == 0


_use_c10d = [True]


C10dStatus = namedtuple('C10dStatus', ['has_c10d', 'is_default'])


if hasattr(nn.parallel, 'deprecated'):
    c10d_status = C10dStatus(has_c10d=True, is_default=True)
elif hasattr(torch.distributed, 'c10d') and hasattr(torch.distributed.c10d, 'init_process_group'):
    c10d_status = C10dStatus(has_c10d=True, is_default=False)
else:
    c10d_status = C10dStatus(has_c10d=False, is_default=False)


if c10d_status.is_default:
    import torch.distributed as dist_c10d
    import torch.distributed.deprecated as dist_no_c10d
elif c10d_status.has_c10d:
    import torch.distributed.c10d as dist_c10d
    import torch.distributed as dist_no_c10d
else:
    import torch.distributed as dist_no_c10d


def infer_init_method(args):
    if args.distributed_init_method is not None:
        return

    # support torch.distributed.launch
    if all(key in os.environ for key in [
        'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK'
    ]):
        args.distributed_init_method = 'tcp://{addr}:{port}'.format(
            addr=os.environ['MASTER_ADDR'],
            port=os.environ['MASTER_PORT'],
        )
        args.distributed_world_size = int(os.environ['WORLD_SIZE'])
        args.distributed_rank = int(os.environ['RANK'])

    # we can determine the init method automatically for Slurm
    elif args.distributed_port > 0:
        node_list = os.environ.get('SLURM_JOB_NODELIST')
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list])
                args.distributed_init_method = 'tcp://{host}:{port}'.format(
                    host=hostnames.split()[0].decode('utf-8'),
                    port=args.distributed_port)
                args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
                args.device_id = int(os.environ.get('SLURM_LOCALID'))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass


def distributed_init(args):
    if args.distributed_world_size == 1:
        raise ValueError('Cannot initialize distributed with distributed_world_size=1')

    if args.ddp_backend == 'no_c10d' or not c10d_status.has_c10d:
        args.ddp_backend = 'no_c10d'
        _use_c10d[0] = False

    print('| distributed init (rank {}): {}'.format(
        args.distributed_rank, args.distributed_init_method), flush=True)

    if _use_c10d[0]:
        init_fn = dist_c10d.init_process_group
    else:
        init_fn = dist_no_c10d.init_process_group

    init_fn(
        backend=args.distributed_backend,
        init_method=args.distributed_init_method,
        world_size=args.distributed_world_size,
        rank=args.distributed_rank,
    )

    suppress_output(is_master(args))

    return args.distributed_rank


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_rank():
    if _use_c10d[0]:
        return dist_c10d.get_rank()
    else:
        return dist_no_c10d.get_rank()


def get_world_size():
    if _use_c10d[0]:
        return dist_c10d.get_world_size()
    else:
        return dist_no_c10d.get_world_size()


def get_default_group():
    if _use_c10d[0]:
        return dist_c10d.group.WORLD
    else:
        return dist_no_c10d.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    if _use_c10d[0]:
        return dist_c10d.all_reduce(tensor, group=group)
    else:
        return dist_no_c10d.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    rank = get_rank()
    world_size = get_world_size()

    buffer_size = max_size * world_size
    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
    buffer = all_gather_list._buffer
    buffer.zero_()

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255*256

    buffer_rank = buffer[rank * max_size : (rank + 1) * max_size]
    buffer_rank[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer_rank[1] = enc_size % 255
    buffer_rank[2:enc_size+2] = torch.ByteTensor(list(enc))

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            size = (255 * utils.item(out_buffer[0])) + utils.item(out_buffer[1])
            if size > 0:
                result.append(
                    pickle.loads(bytes(out_buffer[2:size+2].tolist()))
                )
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )
