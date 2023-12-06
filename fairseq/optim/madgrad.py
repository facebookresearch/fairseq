# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from collections.abc import Collection
from typing import List

from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.optim.madgrad_base import MADGRAD
from omegaconf import II


@dataclass
class FairseqMADGRADConfig(FairseqDataclass):
    """Typically, the same learning rate schedule that is used for SGD or Adam may
    be used. The overall learning rate is not comparable to either method and
    should be determined by a hyper-parameter sweep.

    MADGRAD requires less weight decay than other methods, often as little as
    zero. Momentum values used for SGD or Adam's beta1 should work here also.
    On sparse problems both weight_decay and momentum should be set to 0.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate (default: 1e-2).
        momentum (float):
            Momentum value in the range [0,1) (default: 0.9).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        madgrad_eps (float):
            Term added to the denominator outside of the root operation to improve
            numerical stability. (default: 1e-6).
            This parameter is less important in MADGRAD than in Adam.
            On problems with very small gradients, setting this to 0 will improve convergence.
        decouple_decay (bool):
            Apply AdamW style decoupled weight decay (EXPERIMENTAL).
    """

    lr: List[float] = II("optimization.lr")
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    momentum: float = field(
        default=0.9, metadata={"help": "momentum factor in the range [0,1)"}
    )
    madgrad_eps: float = field(default=1e-6, metadata={"help": "denominator epsilon"})
    decouple_decay: bool = field(
        default=False, metadata={"help": "decouple weight decay"}
    )


@register_optimizer("madgrad", dataclass=FairseqMADGRADConfig)
class FairseqMADGRAD(FairseqOptimizer):
    """
    MADGRAD_: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
    Optimization.

    .. _MADGRAD: https://arxiv.org/abs/2101.11075

    MADGRAD is a general purpose optimizer that can be used in place of SGD or
    Adam may converge faster and generalize better. Currently GPU-only.
    """

    def __init__(self, cfg: FairseqMADGRADConfig, params):
        super().__init__(cfg)
        self._optimizer = MADGRAD(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "momentum": self.cfg.momentum,
            "weight_decay": self.cfg.weight_decay,
            "eps": self.cfg.madgrad_eps,
            "decouple_decay": self.cfg.decouple_decay,
        }
