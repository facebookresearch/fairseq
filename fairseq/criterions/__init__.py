# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os

from fairseq import registry
from fairseq.criterions.fairseq_criterion import FairseqCriterion


build_criterion, register_criterion, CRITERION_REGISTRY = registry.setup_registry(
    '--criterion',
    base_class=FairseqCriterion,
    default='cross_entropy',
)


# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.criterions.' + module)
