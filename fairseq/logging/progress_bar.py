# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper around various loggers and progress bars (e.g., tqdm).
"""

import atexit
import json
import logging
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager
from numbers import Number
from typing import Optional

import torch

from .meters import AverageMeter, StopwatchMeter, TimeMeter


logger = logging.getLogger(__name__)


def progress_bar(
    iterator,
    log_format: Optional[str] = None,
    log_interval: int = 100,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
    tensorboard_logdir: Optional[str] = None,
    default_log_format: str = 'tqdm',
):
    if log_format is None:
        log_format = default_log_format
    if log_format == 'tqdm' and not sys.stderr.isatty():
        log_format = 'simple'

    if log_format == 'json':
        bar = JsonProgressBar(iterator, epoch, prefix, log_interval)
    elif log_format == 'none':
        bar = NoopProgressBar(iterator, epoch, prefix)
    elif log_format == 'simple':
        bar = SimpleProgressBar(iterator, epoch, prefix, log_interval)
    elif log_format == 'tqdm':
        bar = TqdmProgressBar(iterator, epoch, prefix)
    else:
        raise ValueError('Unknown log format: {}'.format(log_format))

    if tensorboard_logdir:
        try:
            # [FB only] custom wrapper for TensorBoard
            import palaas  # noqa
            from .fb_tbmf_wrapper import FbTbmfWrapper
            bar = FbTbmfWrapper(bar, log_interval)
        except ImportError:
            bar = TensorboardProgressBarWrapper(bar, tensorboard_logdir)

    return bar


def build_progress_bar(
    args,
    iterator,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
    default: str = 'tqdm',
    no_progress_bar: str = 'none',
):
    """Legacy wrapper that takes an argparse.Namespace."""
    if getattr(args, 'no_progress_bar', False):
        default = no_progress_bar
    if getattr(args, 'distributed_rank', 0) == 0:
        tensorboard_logdir = getattr(args, 'tensorboard_logdir', None)
    else:
        tensorboard_logdir = None
    return progress_bar(
        iterator,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch,
        prefix=prefix,
        tensorboard_logdir=tensorboard_logdir,
        default_log_format=default,
    )


def format_stat(stat):
    if isinstance(stat, Number):
        stat = '{:g}'.format(stat)
    elif isinstance(stat, AverageMeter):
        stat = '{:.3f}'.format(stat.avg)
    elif isinstance(stat, TimeMeter):
        stat = '{:g}'.format(round(stat.avg))
    elif isinstance(stat, StopwatchMeter):
        stat = '{:g}'.format(round(stat.sum))
    elif torch.is_tensor(stat):
        stat = stat.tolist()
    return stat


class BaseProgressBar(object):
    """Abstract class for progress bars."""
    def __init__(self, iterable, epoch=None, prefix=None):
        self.iterable = iterable
        self.offset = getattr(iterable, 'offset', 0)
        self.epoch = epoch
        self.prefix = ''
        if epoch is not None:
            self.prefix += 'epoch {:03d}'.format(epoch)
        if prefix is not None:
            self.prefix += ' | {}'.format(prefix)

    def __len__(self):
        return len(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        raise NotImplementedError

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        raise NotImplementedError

    def print(self, stats, tag=None, step=None):
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
            postfix[key] = str(format_stat(postfix[key]))
        return postfix


@contextmanager
def rename_logger(logger, new_name):
    old_name = logger.name
    if new_name is not None:
        logger.name = new_name
    yield logger
    logger.name = old_name


class JsonProgressBar(BaseProgressBar):
    """Log output in JSON format."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix)
        self.log_interval = log_interval
        self.i = None
        self.size = None

    def __iter__(self):
        self.size = len(self.iterable)
        for i, obj in enumerate(self.iterable, start=self.offset):
            self.i = i
            yield obj

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        step = step or self.i or 0
        if (
            step > 0
            and self.log_interval is not None
            and step % self.log_interval == 0
        ):
            update = (
                self.epoch - 1 + (self.i + 1) / float(self.size)
                if self.epoch is not None
                else None
            )
            stats = self._format_stats(stats, epoch=self.epoch, update=update)
            with rename_logger(logger, tag):
                logger.info(json.dumps(stats))

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self.stats = stats
        if tag is not None:
            self.stats = OrderedDict([(tag + '_' + k, v) for k, v in self.stats.items()])
        stats = self._format_stats(self.stats, epoch=self.epoch)
        with rename_logger(logger, tag):
            logger.info(json.dumps(stats))

    def _format_stats(self, stats, epoch=None, update=None):
        postfix = OrderedDict()
        if epoch is not None:
            postfix['epoch'] = epoch
        if update is not None:
            postfix['update'] = round(update, 3)
        # Preprocess stats according to datatype
        for key in stats.keys():
            postfix[key] = format_stat(stats[key])
        return postfix


class NoopProgressBar(BaseProgressBar):
    """No logging."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)

    def __iter__(self):
        for obj in self.iterable:
            yield obj

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        pass

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        pass


class SimpleProgressBar(BaseProgressBar):
    """A minimal logger for non-TTY environments."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix)
        self.log_interval = log_interval
        self.i = None
        self.size = None

    def __iter__(self):
        self.size = len(self.iterable)
        for i, obj in enumerate(self.iterable, start=self.offset):
            self.i = i
            yield obj

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        step = step or self.i or 0
        if (
            step > 0
            and self.log_interval is not None
            and step % self.log_interval == 0
        ):
            stats = self._format_stats(stats)
            postfix = self._str_commas(stats)
            with rename_logger(logger, tag):
                logger.info(
                    '{}:  {:5d} / {:d} {}'
                    .format(self.prefix, self.i + 1, self.size, postfix)
                )

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        with rename_logger(logger, tag):
            logger.info('{} | {}'.format(self.prefix, postfix))


class TqdmProgressBar(BaseProgressBar):
    """Log to tqdm."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)
        from tqdm import tqdm
        self.tqdm = tqdm(iterable, self.prefix, leave=False)

    def __iter__(self):
        return iter(self.tqdm)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        self.tqdm.set_postfix(self._format_stats(stats), refresh=False)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        self.tqdm.write('{} | {}'.format(self.tqdm.desc, postfix))


try:
    _tensorboard_writers = {}
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def _close_writers():
    for w in _tensorboard_writers.values():
        w.close()


atexit.register(_close_writers)


class TensorboardProgressBarWrapper(BaseProgressBar):
    """Log to tensorboard."""

    def __init__(self, wrapped_bar, tensorboard_logdir):
        self.wrapped_bar = wrapped_bar
        self.tensorboard_logdir = tensorboard_logdir

        if SummaryWriter is None:
            logger.warning(
                "tensorboard or required dependencies not found, please see README "
                "for using tensorboard. (e.g. pip install tensorboardX)"
            )

    def _writer(self, key):
        if SummaryWriter is None:
            return None
        _writers = _tensorboard_writers
        if key not in _writers:
            _writers[key] = SummaryWriter(os.path.join(self.tensorboard_logdir, key))
            _writers[key].add_text('sys.argv', " ".join(sys.argv))
        return _writers[key]

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats to tensorboard."""
        self._log_to_tensorboard(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self._log_to_tensorboard(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def _log_to_tensorboard(self, stats, tag=None, step=None):
        writer = self._writer(tag or '')
        if writer is None:
            return
        if step is None:
            step = stats['num_updates']
        for key in stats.keys() - {'num_updates'}:
            if isinstance(stats[key], AverageMeter):
                writer.add_scalar(key, stats[key].val, step)
            elif isinstance(stats[key], Number):
                writer.add_scalar(key, stats[key], step)
        writer.flush()
