# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
from fairseq import registry
(
    build_scorer,
    register_scorer,
    SCORER_REGISTRIES,
    _
) = registry.setup_registry('--scorer-type')

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('scorers.' + module)
