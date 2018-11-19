# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import time
import numpy as np

from scipy.stats import pearsonr, spearmanr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n


class ClassificationMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, val_prefix=''):
        self.val_prefix = val_prefix
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.acc = 0
        self.mcc = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def update(self, tp, tn, fp, fn):
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn
        self.acc = (self.tp + self.tn) / ((self.tp + self.tn + self.fp + self.fn) or 1.0)
        self.mcc = (self.tp * self.tn - self.fp * self.fn) / (math.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)) or 1.0)
        self.precision = self.tp / ((self.tp + self.fp) or 1.0)
        self.recall = self.tp / ((self.tp + self.fn) or 1.0)
        self.f1 = 2 * self.precision * self.recall / ((self.precision + self.recall) or 1.0)

    def vals(self):
        def attach_prefix(s):
            return '{}_{}'.format(self.val_prefix, s) if len(self.val_prefix) > 0 else s
        return [
            (attach_prefix('tp'), self.tp),
            (attach_prefix('tn'), self.tn),
            (attach_prefix('fp'), self.fp),
            (attach_prefix('fn'), self.fn),
            (attach_prefix('acc'), self.acc),
            (attach_prefix('mcc'), self.mcc),
            (attach_prefix('f1'), self.f1),
        ]

class RegressionMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.xs = []
        self.ys = []

    def update(self, xs, ys):
        self.xs += xs
        self.ys += ys

    def vals(self):
        pearsons, _ = pearsonr(self.xs, self.ys)
        spearmans, _ = spearmanr(self.xs, self.ys)

        return [
            ('pearsons', pearsons),
            ('spearmans', spearmans),
        ]
