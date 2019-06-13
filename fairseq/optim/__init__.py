# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os

from fairseq import registry
from fairseq.optim.fairseq_optimizer import FairseqOptimizer
from fairseq.optim.fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer
from fairseq.optim.bmuf import FairseqBMUF


__all__ = [
    'FairseqOptimizer',
    'FP16Optimizer',
    'MemoryEfficientFP16Optimizer',
]


_build_optimizer, register_optimizer, OPTIMIZER_REGISTRY = registry.setup_registry(
    '--optimizer',
    base_class=FairseqOptimizer,
    default='nag',
)


def build_optimizer(args, params, *extra_args, **extra_kwargs):
    params = list(filter(lambda p: p.requires_grad, params))
    return _build_optimizer(args, params, *extra_args, **extra_kwargs)


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.optim.' + module)
