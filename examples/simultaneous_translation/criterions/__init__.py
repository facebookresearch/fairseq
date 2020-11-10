# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        criterion_name = file[: file.find(".py")]
        importlib.import_module(
            "examples.simultaneous_translation.criterions." + criterion_name
        )
