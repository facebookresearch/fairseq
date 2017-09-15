# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

"""
Progress bar wrapper around tqdm which handles non-tty outputs
"""

import sys

from tqdm import tqdm


class progress_bar(tqdm):
    enabled = sys.stderr.isatty()
    print_interval = 1000

    def __new__(cls, *args, **kwargs):
        if cls.enabled:
            return tqdm(*args, **kwargs)
        else:
            return simple_progress_bar(cls.print_interval, *args, **kwargs)


class simple_progress_bar(tqdm):

    def __init__(self, print_interval, *args, **kwargs):
        super(simple_progress_bar, self).__init__(*args, **kwargs)
        self.print_interval = print_interval

    def __iter__(self):
        size = len(self.iterable)
        for i, obj in enumerate(self.iterable):
            yield obj
            if i > 0 and i % self.print_interval == 0:
                msg = '{} {:5d} / {:d} {}\n'.format(self.desc, i, size, self.postfix)
                sys.stdout.write(msg)
                sys.stdout.flush()

    @classmethod
    def write(cls, s, file=None, end="\n"):
        fp = file if file is not None else sys.stdout
        fp.write(s)
        fp.write(end)
        fp.flush()

    @staticmethod
    def status_printer(file):
        def print_status(s):
            pass
        return print_status
