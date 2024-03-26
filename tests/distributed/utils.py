# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import tempfile

import torch


def spawn_and_init(fn, world_size, args=None):
    if args is None:
        args = ()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        torch.multiprocessing.spawn(
            fn=functools.partial(init_and_run, fn, args),
            args=(
                world_size,
                tmp_file.name,
            ),
            nprocs=world_size,
            join=True,
        )


def distributed_init(rank, world_size, tmp_file):
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="file://{}".format(tmp_file),
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)


def init_and_run(fn, args, rank, world_size, tmp_file):
    distributed_init(rank, world_size, tmp_file)
    group = torch.distributed.new_group()
    fn(rank, group, *args)


def objects_are_equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        for k in a.keys():
            if not objects_are_equal(a[k], b[k]):
                return False
        return True
    elif isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            return False
        return all(objects_are_equal(x, y) for x, y in zip(a, b))
    elif torch.is_tensor(a):
        return (
            a.size() == b.size()
            and a.dtype == b.dtype
            and a.device == b.device
            and torch.all(a == b)
        )
    else:
        return a == b
