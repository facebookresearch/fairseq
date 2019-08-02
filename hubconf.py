# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools

from fairseq.hub_utils import BPEHubInterface as bpe  # noqa
from fairseq.hub_utils import TokenizerHubInterface as tokenizer  # noqa
from fairseq.models import MODEL_REGISTRY


dependencies = [
    'regex',
    'requests',
    'torch',
]


for _model_type, _cls in MODEL_REGISTRY.items():
    for model_name in _cls.hub_models().keys():
        globals()[model_name] = functools.partial(
            _cls.from_pretrained,
            model_name_or_path=model_name,
        )
    # to simplify the interface we only expose named models
    # globals()[_model_type] = _cls.from_pretrained
