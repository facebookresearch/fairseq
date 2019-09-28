# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os

from .fairseq_criterion import FairseqCriterion


CRITERION_REGISTRY = {}
CRITERION_CLASS_NAMES = set()


def build_criterion(args, task):
    return CRITERION_REGISTRY[args.criterion](args, task)


def register_criterion(name):
    """Decorator to register a new criterion."""

    def register_criterion_cls(cls):
        if name in CRITERION_REGISTRY:
            raise ValueError('Cannot register duplicate criterion ({})'.format(name))
        if not issubclass(cls, FairseqCriterion):
            raise ValueError('Criterion ({}: {}) must extend FairseqCriterion'.format(name, cls.__name__))
        if cls.__name__ in CRITERION_CLASS_NAMES:
            # We use the criterion class name as a unique identifier in
            # checkpoints, so all criterions must have unique class names.
            raise ValueError('Cannot register criterion with duplicate class name ({})'.format(cls.__name__))
        CRITERION_REGISTRY[name] = cls
        CRITERION_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_criterion_cls


# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.criterions.' + module)
