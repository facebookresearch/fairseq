# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

def multiprocessing_pdb():
    """A Pdb wrapper that works in a multiprocessing environment.

    Usage: `from fairseq import pdb; pdb.set_trace()`
    """
    import pdb
    import sys
    class MultiprocessingPdb(pdb.Pdb):
        def interaction(self, *args, **kwargs):
            orig_stdin = sys.stdin
            try:
                sys.stdin = open('/dev/stdin')
                pdb.Pdb.interaction(self, *args, **kwargs)
            finally:
                sys.stdin = orig_stdin
    return MultiprocessingPdb()


pdb = multiprocessing_pdb()
