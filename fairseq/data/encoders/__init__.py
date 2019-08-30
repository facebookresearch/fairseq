# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import os

from fairseq import registry


build_tokenizer, register_tokenizer, TOKENIZER_REGISTRY = registry.setup_registry(
    '--tokenizer',
    default=None,
)


build_bpe, register_bpe, BPE_REGISTRY = registry.setup_registry(
    '--bpe',
    default=None,
)


# automatically import any Python files in the encoders/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.data.encoders.' + module)
