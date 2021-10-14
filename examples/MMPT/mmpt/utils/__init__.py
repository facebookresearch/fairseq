# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import numpy as np
import torch

from .shardedtensor import *
from .load_config import *


def set_seed(seed=43211):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_world_size():
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


def get_local_rank():
    return torch.distributed.get_rank() \
        if torch.distributed.is_initialized() else 0


def print_on_rank0(func):
    local_rank = get_local_rank()
    if local_rank == 0:
        print("[INFO]", func)


class RetriMeter(object):
    """
    Statistics on whether retrieval yields a better pair.
    """
    def __init__(self, freq=1024):
        self.freq = freq
        self.total = 0
        self.replace = 0
        self.updates = 0

    def __call__(self, data):
        if isinstance(data, np.ndarray):
            self.replace += data.shape[0] - int((data[:, 0] == -1).sum())
            self.total += data.shape[0]
        elif torch.is_tensor(data):
            self.replace += int(data.sum())
            self.total += data.size(0)
        else:
            raise ValueError("unsupported RetriMeter data type.", type(data))

        self.updates += 1
        if get_local_rank() == 0 and self.updates % self.freq == 0:
            print("[INFO]", self)

    def __repr__(self):
        return "RetriMeter (" + str(self.replace / self.total) \
            + "/" + str(self.replace) + "/" + str(self.total) + ")"
