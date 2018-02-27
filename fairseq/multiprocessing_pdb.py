# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import multiprocessing
import os
import pdb
import sys


class MultiprocessingPdb(pdb.Pdb):
    """A Pdb wrapper that works in a multiprocessing environment.

    Usage: `from fairseq import pdb; pdb.set_trace()`
    """

    _stdin_fd = sys.stdin.fileno()
    _stdin = None
    _stdin_lock = multiprocessing.Lock()

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        stdin_bak = sys.stdin
        with self._stdin_lock:
            try:
                if not self._stdin:
                    self._stdin = os.fdopen(self._stdin_fd)
                sys.stdin = self._stdin
                self.cmdloop()
            finally:
                sys.stdin = stdin_bak


pdb = MultiprocessingPdb()
