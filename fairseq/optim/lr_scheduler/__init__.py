# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from fairseq import registry
from fairseq.optim.lr_scheduler.fairseq_lr_scheduler import FairseqLRScheduler, LegacyFairseqLRScheduler # noqa


LR_SCHEDULER_DATACLASS_REGISTRY = {}

build_lr_scheduler, register_lr_scheduler, LR_SCHEDULER_REGISTRY = registry.setup_registry(
    '--lr-scheduler',
    base_class=FairseqLRScheduler,
    default='fixed',
)

# automatically import any Python files in the optim/lr_scheduler/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.optim.lr_scheduler.' + module)
