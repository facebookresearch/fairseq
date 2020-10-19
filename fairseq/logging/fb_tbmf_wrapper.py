# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Wrapper around tbwriter api for for writing to manifold-tensorboard.
FB Internal (not to be open-sourced)
"""

import datetime
import os
from numbers import Number

from .meters import AverageMeter
from .progress_bar import BaseProgressBar


try:
    from palaas import tbwriter, register_manifold
except ImportError:
    pass


class LogCounter:
    def __init__(self, interval):
        self.log_interval = interval
        self.log_counter = 1

    def advance(self):
        self.log_counter += 1
        return self.log_counter % self.log_interval == 0


class FbTbmfWrapper(BaseProgressBar):
    """Log to tensorboard."""

    # manifold sub folder to used by all instances.
    manifold_job_path = ""

    def _get_job_path(self):
        # get slurm job name
        job_id = os.environ.get("SLURM_JOB_NAME")
        if job_id is None:
            # TODO
            # try to get fb learner job name
            job_id = ""

        if job_id is not None and job_id != "":
            return job_id
        else:
            # get date-time str
            time = datetime.datetime.now()
            time_str = "{}-{}-{}-{}:{}".format(
                time.year, time.month, time.day, time.hour, time.minute
            )
            return time_str

    def __init__(self, wrapped_bar, log_interval):
        self.wrapped_bar = wrapped_bar
        if FbTbmfWrapper.manifold_job_path == "":
            FbTbmfWrapper.manifold_job_path = self._get_job_path()
        self.log_interval = log_interval
        # We need a log counter for every variable.
        self.counters = {}
        self.counter_disabled_list = []

        self.log_counter = 1
        self._tbwriter = None
        try:
            self._tbwriter = tbwriter.get_tbwriter(FbTbmfWrapper.manifold_job_path)
            register_manifold()
        except Exception:
            pass

        self.disable_buffering("valid")

        self._writers = {}

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag="", step=None):
        """Log intermediate stats to tensorboard."""
        self._log_to_tensorboard(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag="", step=None):
        """Print end-of-epoch stats."""
        self._log_to_tensorboard(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def __exit__(self, *exc):
        if self._tbwriter is not None:
            self._tbwriter.close()
        return False

    def disable_buffering(self, tag):
        if tag is not None:
            self.counter_disabled_list.append(tag)

    def _log_to_tensorboard(self, stats, tag="", step=None):
        writer = self._tbwriter
        if writer is None:
            return

        # Get LogCounter for this variable
        if tag not in self.counter_disabled_list:
            if tag not in self.counters:
                self.counters[tag] = LogCounter(self.log_interval)
            if not self.counters[tag].advance():
                return

        if step is None:
            step = stats["num_updates"]
        for key in stats.keys() - {"num_updates"}:
            if isinstance(stats[key], AverageMeter):
                writer.add_scalar(tag, key, stats[key].val, step)
            elif isinstance(stats[key], Number):
                writer.add_scalar(tag, key, stats[key], step)
        writer.flush()
