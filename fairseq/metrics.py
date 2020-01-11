# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
A standalone module for aggregating metrics.

Metrics can be logged from anywhere using the `log_*` functions defined
in this module. The logged values will be aggregated dynamically based
on the aggregation context in which the logging occurs. See the
:func:`aggregate` context manager for more details.
"""

from collections import OrderedDict
import contextlib
import time
from typing import Callable, Dict, List, Optional
import uuid

from .meters import *


# Aggregation contexts are considered "active" when inside the scope
# created by the :func:`aggregate` context manager.
_aggregators = OrderedDict()
_active_aggregators = OrderedDict()


# The "default" aggregator observes all logged values.
_aggregators["default"] = MetersDict()
_active_aggregators["default"] = _aggregators["default"]


@contextlib.contextmanager
def aggregate(name: Optional[str] = None, exclusive: bool = False):
    """Context manager to aggregate metrics under a given name.

    Aggregations can be nested. If *exclusive* is ``False``, then logged
    metrics will be recorded along the entire stack of nested
    aggregators, including a global "default" aggregator. If *exclusive*
    is ``True``, then only the most recent aggregator will be used.

    Note that aggregation contexts are uniquely identified by their
    *name* (e.g., train, valid). Creating a context with an existing
    name will reuse the corresponding :class:`MetersDict` instance.
    If no name is given then a temporary aggregator will be created
    and reset when the context manager exits.

    Usage::

        with metrics.aggregate("train"):
            for step, batch in enumerate(epoch):
                with metrics.aggregate("train_inner") as agg:
                    metrics.log_scalar("loss", get_loss(batch))
                    if step % log_interval == 0:
                        print(agg.get_smoothed_value("loss"))
                        agg.reset()
        print(metrics.get_smoothed_values("train")["loss"])

    Args:
        name (str): name of the aggregation. Defaults to a
            random/temporary name if not given explicitly.
        exclusive (bool): only log to the most recent aggregation
            context, instead of all nested aggregations.
    """
    if name is None:
        # generate a temporary name
        name = str(uuid.uuid4())
        assert name not in _aggregators
        agg = MetersDict()
    else:
        assert name != "default"
        agg = _aggregators.setdefault(name, MetersDict())

    if exclusive:
        backup_aggregators = _active_aggregators.copy()
        _active_aggregators.clear()

    _active_aggregators[name] = agg
    yield agg
    del _active_aggregators[name]

    if exclusive:
        _active_aggregators.clear()
        _active_aggregators.update(backup_aggregators)


def get_active_aggregators() -> List[MetersDict]:
    return list(_active_aggregators.values())


def log_scalar(
    key: str,
    value: float,
    weight: float = 1,
    priority: int = 10,
    round: Optional[int] = None,
):
    """Log a scalar value.

    Args:
        key (str): name of the field to log
        value (float): value to log
        weight (float): weight that this value contributes to the average.
            A weight of 0 will always log the latest value.
        priority (int): smaller values are logged earlier in the output
        round (Optional[int]): number of digits to round to when displaying
    """
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, AverageMeter(round=round), priority)
        agg[key].update(value, weight)


def log_derived(key: str, fn: Callable[[MetersDict], float], priority: int = 20):
    """Log a scalar value derived from other meters.

    Args:
        key (str): name of the field to log
        fn (Callable[[MetersDict], float]): function that takes a single
            argument *meters* and returns the derived value
        priority (int): smaller values are logged earlier in the output
    """
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, MetersDict._DerivedMeter(fn), priority)


def log_speed(key: str, value: float, priority: int = 30, round: Optional[int] = None):
    """Log the rate of some quantity per second.

    Args:
        key (str): name of the field to log
        value (float): value to log
        priority (int): smaller values are logged earlier in the output
        round (Optional[int]): number of digits to round to when displaying
    """
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, TimeMeter(round=round), priority)
            agg[key].reset()  # reset meter on the first call
        else:
            agg[key].update(value)


def log_start_time(key: str, priority: int = 40, round: Optional[int] = None):
    """Log the duration of some event in seconds.

    The duration will be computed once :func:`log_stop_time` is called.

    Args:
        key (str): name of the field to log
        priority (int): smaller values are logged earlier in the output
        round (Optional[int]): number of digits to round to when displaying
    """
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, StopwatchMeter(round=round), priority)
        agg[key].start()


def log_stop_time(key: str, weight: float = 0.):
    """Log the duration of some event in seconds.

    The duration will be computed since :func:`log_start_time` was called.
    Set weight > 0 to report the average time instead of the sum.

    Args:
        key (str): name of the field to log
        weight (float): weight that this time contributes to the average
    """
    for agg in get_active_aggregators():
        agg[key].stop(weight)


def reset_meters(name: str):
    """Reset Meter instances aggregated under a given *name*."""
    meters = get_meters(name)
    if meters is not None:
        meters.reset()


def get_meter(name: str, key: str) -> Meter:
    """Get a single Meter instance aggregated under *name* and *key*.

    Returns:
        Meter or None if no metrics have been logged under *name* and *key*.
    """
    if name not in _aggregators:
        return None
    return _aggregators[name].get(key, None)


def get_meters(name: str) -> MetersDict:
    """Get Meter instances aggregated under a given *name*.

    Returns:
        MetersDict or None if no metrics have been logged under *name*.
    """
    return _aggregators.get(name, None)


def get_smoothed_value(name: str, key: str) -> float:
    """Get a single smoothed value.

    Raises:
        KeyError: if no metrics have been logged under *name* and *key*.
    """
    return _aggregators[name].get_smoothed_value(key)


def get_smoothed_values(name: str) -> Dict[str, float]:
    """Get smoothed values aggregated under a given *name*.

    Raises:
        KeyError: if no metrics have been logged under *name*.
    """
    return _aggregators[name].get_smoothed_values()


def state_dict():
    return OrderedDict([
        (name, agg.state_dict())
        for name, agg in _aggregators.items()
    ])


def load_state_dict(state_dict):
    for name, agg_state in state_dict.items():
        _aggregators[name] = MetersDict()
        _aggregators[name].load_state_dict(agg_state)
