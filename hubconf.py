# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import importlib


dependencies = [
    'dataclasses',
    'hydra',
    'numpy',
    'regex',
    'requests',
    'torch',
]


# Check for required dependencies and raise a RuntimeError if any are missing.
missing_deps = []
for dep in dependencies:
    try:
        importlib.import_module(dep)
    except ImportError:
        # Hack: the hydra package is provided under the "hydra-core" name in
        # pypi. We don't want the user mistakenly calling `pip install hydra`
        # since that will install an unrelated package.
        if dep == 'hydra':
            dep = 'hydra-core'
        missing_deps.append(dep)
if len(missing_deps) > 0:
    raise RuntimeError('Missing dependencies: {}'.format(', '.join(missing_deps)))


# torch.hub doesn't build Cython components, so if they are not found then try
# to build them here
try:
    import fairseq.data.token_block_utils_fast  # noqa
except ImportError:
    try:
        import cython  # noqa
        import os
        from setuptools import sandbox
        sandbox.run_setup(
            os.path.join(os.path.dirname(__file__), 'setup.py'),
            ['build_ext', '--inplace'],
        )
    except ImportError:
        print(
            'Unable to build Cython components. Please make sure Cython is '
            'installed if the torch.hub model you are loading depends on it.'
        )


from fairseq.hub_utils import BPEHubInterface as bpe  # noqa
from fairseq.hub_utils import TokenizerHubInterface as tokenizer  # noqa
from fairseq.models import MODEL_REGISTRY  # noqa


# automatically expose models defined in FairseqModel::hub_models
for _model_type, _cls in MODEL_REGISTRY.items():
    for model_name in _cls.hub_models().keys():
        globals()[model_name] = functools.partial(
            _cls.from_pretrained,
            model_name,
        )
