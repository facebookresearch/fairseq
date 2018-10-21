# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import ctypes
import math
import torch

try:
    from fairseq import libbleu
except ImportError as e:
    import sys
    sys.stderr.write('ERROR: missing libbleu.so. run `python setup.py build develop`\n')
    raise e


C = ctypes.cdll.LoadLibrary(libbleu.__file__)


class BleuStat(ctypes.Structure):
    _fields_ = [
        ('reflen', ctypes.c_size_t),
        ('predlen', ctypes.c_size_t),
        ('match1', ctypes.c_size_t),
        ('count1', ctypes.c_size_t),
        ('match2', ctypes.c_size_t),
        ('count2', ctypes.c_size_t),
        ('match3', ctypes.c_size_t),
        ('count3', ctypes.c_size_t),
        ('match4', ctypes.c_size_t),
        ('count4', ctypes.c_size_t),
    ]


class Scorer(object):
    def __init__(self, pad, eos, unk):
        self.stat = BleuStat()
        self.pad = pad
        self.eos = eos
        self.unk = unk
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            C.bleu_one_init(ctypes.byref(self.stat))
        else:
            C.bleu_zero_init(ctypes.byref(self.stat))

    def add(self, ref, pred):
        if not isinstance(ref, torch.IntTensor):
            raise TypeError('ref must be a torch.IntTensor (got {})'
                            .format(type(ref)))
        if not isinstance(pred, torch.IntTensor):
            raise TypeError('pred must be a torch.IntTensor(got {})'
                            .format(type(pred)))

        # don't match unknown words
        rref = ref.clone()
        assert not rref.lt(0).any()
        rref[rref.eq(self.unk)] = -999

        rref = rref.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        C.bleu_add(
            ctypes.byref(self.stat),
            ctypes.c_size_t(rref.size(0)),
            ctypes.c_void_p(rref.data_ptr()),
            ctypes.c_size_t(pred.size(0)),
            ctypes.c_void_p(pred.data_ptr()),
            ctypes.c_int(self.pad),
            ctypes.c_int(self.eos))

    def score(self, order=4):
        psum = sum(math.log(p) if p > 0 else float('-Inf')
                   for p in self.precision()[:order])
        return self.brevity() * math.exp(psum / order) * 100

    def precision(self):
        def ratio(a, b):
            return a / b if b > 0 else 0

        return [
            ratio(self.stat.match1, self.stat.count1),
            ratio(self.stat.match2, self.stat.count2),
            ratio(self.stat.match3, self.stat.count3),
            ratio(self.stat.match4, self.stat.count4),
        ]

    def brevity(self):
        r = self.stat.reflen / self.stat.predlen
        return min(1, math.exp(1 - r))

    def result_string(self, order=4):
        assert order <= 4, "BLEU scores for order > 4 aren't supported"
        fmt = 'BLEU{} = {:2.2f}, {:2.1f}'
        for _ in range(1, order):
            fmt += '/{:2.1f}'
        fmt += ' (BP={:.3f}, ratio={:.3f}, syslen={}, reflen={})'
        bleup = [p * 100 for p in self.precision()[:order]]
        return fmt.format(order, self.score(order=order), *bleup,
                          self.brevity(), self.stat.predlen/self.stat.reflen,
                          self.stat.predlen, self.stat.reflen)
