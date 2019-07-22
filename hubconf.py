# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import functools

from fairseq.models import MODEL_REGISTRY


dependencies = [
    'regex',
    'requests',
    'sacremoses',
    'sentencepiece',
    'subword_nmt',
    'torch',
]


for model_type, _cls in MODEL_REGISTRY.items():
    for model_name in _cls.hub_models().keys():
        globals()[model_name] = functools.partial(
            _cls.from_pretrained,
            model_name_or_path=model_name,
        )
    # to simplify the interface we only expose named models
    #globals()[model_type] = _cls.from_pretrained
