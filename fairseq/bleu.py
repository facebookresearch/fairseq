import ctypes
import math
import torch

try:
    from fairseq import libbleu
except ImportError as e:
    import sys
    sys.stderr.write('ERROR: missing libbleu.so. run `python setup.py install`\n')
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
    def __init__(self, pad, eos):
        self.stat = BleuStat()
        self.pad = pad
        self.eos = eos
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

        ref = ref.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        C.bleu_add(
            ctypes.byref(self.stat),
            ctypes.c_size_t(ref.size(0)),
            ctypes.c_void_p(ref.data_ptr()),
            ctypes.c_size_t(pred.size(0)),
            ctypes.c_void_p(pred.data_ptr()),
            ctypes.c_int(self.pad),
            ctypes.c_int(self.eos))

    def score(self):
        psum = sum(math.log(p) if p > 0 else float('-Inf') for p in self.precision())
        return self.brevity() * math.exp(psum / 4) * 100

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

    def result_string(self):
        fmt = 'BLEU4 = {:2.2f}, {:2.1f}/{:2.1f}/{:2.1f}/{:2.1f} '
        fmt += '(BP={:.3f}, syslen={}, reflen={})'
        bleup = [p * 100 for p in self.precision()]
        return fmt.format(self.score(), bleup[0], bleup[1], bleup[2], bleup[3],
                          self.brevity(), self.stat.predlen, self.stat.reflen)
