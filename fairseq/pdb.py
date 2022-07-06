# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os
import pdb
import sys
from time import sleep

from torch import distributed

__all__ = ["set_trace", "distributed_set_trace"]


_stdin = [None]
_stdin_lock = multiprocessing.Lock()
try:
    _stdin_fd = sys.stdin.fileno()
except Exception:
    _stdin_fd = None


class MultiprocessingPdb(pdb.Pdb):
    """A Pdb wrapper that works in a multiprocessing environment.

    Usage: `from fairseq import pdb; pdb.set_trace()`
    """

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        stdin_bak = sys.stdin
        with _stdin_lock:
            try:
                if _stdin_fd is not None:
                    if not _stdin[0]:
                        _stdin[0] = os.fdopen(_stdin_fd)
                    sys.stdin = _stdin[0]
                self.cmdloop()
            finally:
                sys.stdin = stdin_bak


def set_trace():
    pdb = MultiprocessingPdb()
    pdb.set_trace(sys._getframe().f_back)


def distributed_set_trace(rank=0, sleep_time=10000):
    """
    In distributed training, `set_trace()` allows user to interact
    with the code but there will be `world_size`(multiple) printed output.

    This methods make the debugging run only on *one* process
    while other processes are sleeping. If we are not using
    distributed training, the behavior is the same as `set_trace`.
    Args:
        rank (int):
            rank of the current process. 0 <= rank <= `world_size`
        sleep_time (int):
            sleep time (in second) of all other processes.
    """
    if not distributed.is_initialized() or distributed.get_rank() == rank:
        pdb = MultiprocessingPdb()
        pdb.set_trace(sys._getframe().f_back)
    else:
        sleep(sleep_time)
