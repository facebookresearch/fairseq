# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from fairseq import registry
from fairseq.optim.fairseq_optimizer import FairseqOptimizer, LegacyFairseqOptimizer # noqa
from fairseq.optim.fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer
from fairseq.optim.bmuf import FairseqBMUF  # noqa
from fairseq.optim.shard import shard_


OPTIMIZER_DATACLASS_REGISTRY = {}

__all__ = [
    'FairseqOptimizer',
    'FP16Optimizer',
    'MemoryEfficientFP16Optimizer',
    'shard_',
]


_build_optimizer, register_optimizer, OPTIMIZER_REGISTRY = registry.setup_registry(
    '--optimizer',
    base_class=FairseqOptimizer,
    required=True,
)


def build_optimizer(args, params, *extra_args, **extra_kwargs):
    if all(isinstance(p, dict) for p in params):
        params = [t for p in params for t in p.values()]
    params = list(filter(lambda p: p.requires_grad, params))
    return _build_optimizer(args, params, *extra_args, **extra_kwargs)


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.optim.' + module)
