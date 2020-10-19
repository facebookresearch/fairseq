# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from fairseq import registry


build_agent, register_agent, MONOTONIC_AGENT, _ = registry.setup_registry(
    "--agent-type"
)


DEFAULT_EOS = "</s>"
GET = 0
SEND = 1

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("agents." + module)
