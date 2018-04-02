# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Wrapper around various loggers and progress bars (e.g., tqdm).
"""

from collections import OrderedDict
import json
from numbers import Number
import sys

from tqdm import tqdm

from fairseq.meters import AverageMeter


def build_progress_bar(args, iterator, epoch=None, prefix=None, default='tqdm', no_progress_bar='none'):
    if args.log_format is None:
        args.log_format = no_progress_bar if args.no_progress_bar else default

    if args.log_format == 'tqdm' and not sys.stderr.isatty():
        args.log_format = 'simple'

    if args.log_format == 'json':
        bar = json_progress_bar(iterator, epoch, prefix, args.log_interval)
    elif args.log_format == 'none':
        bar = noop_progress_bar(iterator, epoch, prefix)
    elif args.log_format == 'simple':
        bar = simple_progress_bar(iterator, epoch, prefix, args.log_interval)
    elif args.log_format == 'tqdm':
        bar = tqdm_progress_bar(iterator, epoch, prefix)
    else:
        raise ValueError('Unknown log format: {}'.format(args.log_format))
    return bar


class progress_bar(object):
    """Abstract class for progress bars."""
    def __init__(self, iterable, epoch=None, prefix=None):
        self.iterable = iterable
        self.epoch = epoch
        self.prefix = ''
        if epoch is not None:
            self.prefix += '| epoch {:03d}'.format(epoch)
        if prefix is not None:
            self.prefix += ' | {}'.format(prefix)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        raise NotImplementedError

    def log(self, stats):
        """Log intermediate stats according to log_interval."""
        raise NotImplementedError

    def print(self, stats):
        """Print end-of-epoch stats."""
        raise NotImplementedError

    def _str_commas(self, stats):
        return ', '.join(key + '=' + stats[key].strip()
                         for key in stats.keys())

    def _str_pipes(self, stats):
        return ' | '.join(key + ' ' + stats[key].strip()
                          for key in stats.keys())

    def _format_stats(self, stats):
        postfix = OrderedDict(stats)
        # Preprocess stats according to datatype
        for key in postfix.keys():
            # Number: limit the length of the string
            if isinstance(postfix[key], Number):
                postfix[key] = '{:g}'.format(postfix[key])
            # Meter: display both current and average value
            elif isinstance(postfix[key], AverageMeter):
                postfix[key] = '{:.2f} ({:.2f})'.format(
                    postfix[key].val, postfix[key].avg)
            # Else for any other type, try to get the string conversion
            elif not isinstance(postfix[key], str):
                postfix[key] = str(postfix[key])
            # Else if it's a string, don't need to preprocess anything
        return postfix


class json_progress_bar(progress_bar):
    """Log output in JSON format."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix)
        self.log_interval = log_interval
        self.stats = None

    def __iter__(self):
        size = float(len(self.iterable))
        for i, obj in enumerate(self.iterable):
            yield obj
            if self.stats is not None and i > 0 and \
                    self.log_interval is not None and i % self.log_interval == 0:
                update = self.epoch - 1 + float(i / size) if self.epoch is not None else None
                stats = self._format_stats(self.stats, epoch=self.epoch, update=update)
                print(json.dumps(stats), flush=True)

    def log(self, stats):
        """Log intermediate stats according to log_interval."""
        self.stats = stats

    def print(self, stats):
        """Print end-of-epoch stats."""
        stats = self._format_stats(self.stats, epoch=self.epoch)
        print("sweep_log: " + json.dumps(stats), flush=True)

    def _format_stats(self, stats, epoch=None, update=None):
        postfix = OrderedDict()
        if epoch is not None:
            postfix['epoch'] = epoch
        if update is not None:
            postfix['update'] = update
        # Preprocess stats according to datatype
        for key in stats.keys():
            # Meter: display both current and average value
            if isinstance(stats[key], AverageMeter):
                postfix[key] = stats[key].val
                postfix[key + '_avg'] = stats[key].avg
            else:
                postfix[key] = stats[key]
        return postfix


class noop_progress_bar(progress_bar):
    """No logging."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)

    def __iter__(self):
        for obj in self.iterable:
            yield obj

    def log(self, stats):
        """Log intermediate stats according to log_interval."""
        pass

    def print(self, stats):
        """Print end-of-epoch stats."""
        pass


class simple_progress_bar(progress_bar):
    """A minimal logger for non-TTY environments."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix)
        self.log_interval = log_interval
        self.stats = None

    def __iter__(self):
        size = len(self.iterable)
        for i, obj in enumerate(self.iterable):
            yield obj
            if self.stats is not None and i > 0 and \
                    self.log_interval is not None and i % self.log_interval == 0:
                postfix = self._str_commas(self.stats)
                print('{}:  {:5d} / {:d} {}'.format(self.prefix, i, size, postfix),
                      flush=True)

    def log(self, stats):
        """Log intermediate stats according to log_interval."""
        self.stats = self._format_stats(stats)

    def print(self, stats):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        print('{} | {}'.format(self.prefix, postfix), flush=True)


class tqdm_progress_bar(progress_bar):
    """Log to tqdm."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)
        self.tqdm = tqdm(iterable, self.prefix, leave=False)

    def __iter__(self):
        return iter(self.tqdm)

    def log(self, stats):
        """Log intermediate stats according to log_interval."""
        self.tqdm.set_postfix(self._format_stats(stats), refresh=False)

    def print(self, stats):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        self.tqdm.write('{} | {}'.format(self.tqdm.desc, postfix))
