# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools

from fairseq.hub_utils import BPEHubInterface as bpe  # noqa
from fairseq.hub_utils import TokenizerHubInterface as tokenizer  # noqa
from fairseq.models import MODEL_REGISTRY


dependencies = [
    'dataclasses',
    'hydra-core',
    'numpy',
    'regex',
    'requests',
    'torch',
]


# torch.hub doesn't build Cython components, so if they are not found then try
# to build them here
try:
    import fairseq.data.token_block_utils_fast
except (ImportError, ModuleNotFoundError):
    try:
        import cython
        import os
        from setuptools import sandbox
        sandbox.run_setup(
            os.path.join(os.path.dirname(__file__), 'setup.py'),
            ['build_ext', '--inplace'],
        )
    except (ImportError, ModuleNotFoundError):
        print(
            'Unable to build Cython components. Please make sure Cython is '
            'installed if the torch.hub model you are loading depends on it.'
        )


for _model_type, _cls in MODEL_REGISTRY.items():
    for model_name in _cls.hub_models().keys():
        globals()[model_name] = functools.partial(
            _cls.from_pretrained,
            model_name,
        )
    # to simplify the interface we only expose named models
    # globals()[_model_type] = _cls.from_pretrained
