# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

"""
Progress bar wrapper around tqdm which handles non-TTY outputs.
"""

from collections import OrderedDict
from numbers import Number
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


class simple_progress_bar(object):
    """A minimal replacement for tqdm in non-TTY environments."""

    def __init__(self, print_interval, iterable, desc, *_args, **_kwargs):
        super().__init__()
        self.print_interval = print_interval
        self.iterable = iterable
        self.desc = desc

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        size = len(self.iterable)
        for i, obj in enumerate(self.iterable):
            yield obj
            if i > 0 and i % self.print_interval == 0:
                msg = '{}:  {:5d} / {:d} {}\n'.format(self.desc, i, size, self.postfix)
                sys.stdout.write(msg)
                sys.stdout.flush()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        # Sort in alphabetical order to be more deterministic
        postfix = OrderedDict([] if ordered_dict is None else ordered_dict)
        for key in sorted(kwargs.keys()):
            postfix[key] = kwargs[key]
        # Preprocess stats according to datatype
        for key in postfix.keys():
            # Number: limit the length of the string
            if isinstance(postfix[key], Number):
                postfix[key] = '{0:2.3g}'.format(postfix[key])
            # Else for any other type, try to get the string conversion
            elif not isinstance(postfix[key], str):
                postfix[key] = str(postfix[key])
            # Else if it's a string, don't need to preprocess anything
        # Stitch together to get the final postfix
        self.postfix = ', '.join(key + '=' + postfix[key].strip()
                                 for key in postfix.keys())

    @classmethod
    def write(cls, s, file=None, end="\n"):
        fp = file if file is not None else sys.stdout
        fp.write(s)
        fp.write(end)
        fp.flush()
