# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

# automatically import any Python files in the current directory
cur_dir = os.path.dirname(__file__)
for file in os.listdir(cur_dir):
    path = os.path.join(cur_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        mod_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(__name__ + "." + mod_name)
