# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os

from .fairseq_task import FairseqTask


TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()


def setup_task(args):
    return TASK_REGISTRY[args.task].setup_task(args)


def register_task(name):
    """Decorator to register a new task."""

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError('Cannot register duplicate task ({})'.format(name))
        if not issubclass(cls, FairseqTask):
            raise ValueError('Task ({}: {}) must extend FairseqTask'.format(name, cls.__name__))
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError('Cannot register task with duplicate class name ({})'.format(cls.__name__))
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_task_cls


# automatically import any Python files in the tasks/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.tasks.' + module)
