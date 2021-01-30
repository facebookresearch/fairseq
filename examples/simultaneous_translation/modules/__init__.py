# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from fairseq import registry


(
    build_monotonic_attention,
    register_monotonic_attention,
    MONOTONIC_ATTENTION_REGISTRY,
    _,
) = registry.setup_registry("--simul-type")

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        model_name = file[: file.find(".py")]
        importlib.import_module(
            "examples.simultaneous_translation.modules." + model_name
        )
