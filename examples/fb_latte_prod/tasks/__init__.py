#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This file imports all the relevant fb_* tasks for fairseq, so that all
# relevant tasks are imported at once.

import importlib
import os

# automatically import any Python files in the fb_latte_prod/tasks/ directory
tasks_dir = os.path.dirname(__file__)
for file in os.listdir(tasks_dir):
    path = os.path.join(tasks_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        task_name = file[:file.find('.py')] if file.endswith('.py') else file
        importlib.import_module('fairseq_latte_prod.tasks.' + task_name)
