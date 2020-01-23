# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect
from collections import OrderedDict
import time
from typing import Dict, Optional


class Meter(object):
    """Base class for Meters."""

    def __init__(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def reset(self):
        raise NotImplementedError

    @property
    def smoothed_value(self) -> float:
        """Smoothed value used for logging."""
        raise NotImplementedError


def safe_round(number, ndigits):
    if hasattr(number, '__round__'):
        return round(number, ndigits)
    else:
        return number


class AverageMeter(Meter):
    """Computes and stores the average and current value"""

    def __init__(self, round: Optional[int] = None):
        self.round = round
        self.reset()

    def reset(self):
        self.val = None  # most recent update
        self.sum = 0  # sum from all updates
        self.count = 0  # total n from all updates

    def update(self, val, n=1):
        if val is not None:
            self.val = val
            if n > 0:
                self.sum += val * n
                self.count += n

    def state_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'round': self.round,
        }

    def load_state_dict(self, state_dict):
        self.val = state_dict['val']
        self.sum = state_dict['sum']
        self.count = state_dict['count']
        self.round = state_dict.get('round', None)

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else self.val

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val


class TimeMeter(Meter):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init: int = 0, n: int = 0, round: Optional[int] = None):
        self.round = round
        self.reset(init, n)

    def reset(self, init=0, n=0):
        self.init = init
        self.start = time.time()
        self.n = n

    def update(self, val=1):
        self.n += val

    def state_dict(self):
        return {
            'init': self.elapsed_time,
            'n': self.n,
            'round': self.round,
        }

    def load_state_dict(self, state_dict):
        if 'start' in state_dict:
            # backwards compatibility for old state_dicts
            self.reset(init=state_dict['init'])
        else:
            self.reset(init=state_dict['init'], n=state_dict['n'])
            self.round = state_dict.get('round', None)

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val


class StopwatchMeter(Meter):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self, round: Optional[int] = None):
        self.round = round
        self.sum = 0
        self.n = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n

    def reset(self):
        self.sum = 0  # cumulative time during which stopwatch was active
        self.n = 0  # total n across all start/stop
        self.start()

    def state_dict(self):
        return {
            'sum': self.sum,
            'n': self.n,
            'round': self.round,
        }

    def load_state_dict(self, state_dict):
        self.sum = state_dict['sum']
        self.n = state_dict['n']
        self.start_time = None
        self.round = state_dict.get('round', None)

    @property
    def avg(self):
        return self.sum / self.n if self.n > 0 else self.sum

    @property
    def elapsed_time(self):
        if self.start_time is None:
            return 0.
        return time.time() - self.start_time

    @property
    def smoothed_value(self) -> float:
        val = self.avg if self.sum > 0 else self.elapsed_time
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val


class MetersDict(OrderedDict):
    """A sorted dictionary of :class:`Meters`.

    Meters are sorted according to a priority that is given when the
    meter is first added to the dictionary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.priorities = []

    def __setitem__(self, key, value):
        assert key not in self, "MetersDict doesn't support reassignment"
        priority, value = value
        bisect.insort(self.priorities, (priority, len(self.priorities), key))
        super().__setitem__(key, value)
        for _, _, key in self.priorities:  # reorder dict to match priorities
            self.move_to_end(key)

    def add_meter(self, key, meter, priority):
        self.__setitem__(key, (priority, meter))

    def state_dict(self):
        return [
            (pri, key, self[key].__class__.__name__, self[key].state_dict())
            for pri, _, key in self.priorities
            # can't serialize DerivedMeter instances
            if not isinstance(self[key], MetersDict._DerivedMeter)
        ]

    def load_state_dict(self, state_dict):
        self.clear()
        self.priorities.clear()
        for pri, key, meter_cls, meter_state in state_dict:
            meter = globals()[meter_cls]()
            meter.load_state_dict(meter_state)
            self.add_meter(key, meter, pri)

    def get_smoothed_value(self, key: str) -> float:
        """Get a single smoothed value."""
        meter = self[key]
        if isinstance(meter, MetersDict._DerivedMeter):
            return meter.fn(self)
        else:
            return meter.smoothed_value

    def get_smoothed_values(self) -> Dict[str, float]:
        """Get all smoothed values."""
        return OrderedDict([
            (key, self.get_smoothed_value(key))
            for key in self.keys()
            if not key.startswith("_")
        ])

    def reset(self):
        """Reset Meter instances."""
        for meter in self.values():
            if isinstance(meter, MetersDict._DerivedMeter):
                continue
            meter.reset()

    class _DerivedMeter(Meter):
        """A Meter whose values are derived from other Meters."""

        def __init__(self, fn):
            self.fn = fn

        def reset(self):
            pass
