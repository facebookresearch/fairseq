# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os

from .fairseq_decoder import FairseqDecoder  # noqa: F401
from .fairseq_encoder import FairseqEncoder  # noqa: F401
from .fairseq_incremental_decoder import FairseqIncrementalDecoder  # noqa: F401
from .fairseq_model import FairseqModel  # noqa: F401


MODEL_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}


def build_model(args, src_dict, dst_dict):
    return ARCH_MODEL_REGISTRY[args.arch].build_model(args, src_dict, dst_dict)


def register_model(name):
    """Decorator to register a new model (e.g., LSTM)."""

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        if not issubclass(cls, FairseqModel):
            raise ValueError('Model ({}: {}) must extend FairseqModel'.format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def register_model_architecture(model_name, arch_name):
    """Decorator to register a new model architecture (e.g., lstm_luong_wmt_en_de)."""

    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError('Cannot register model architecture for unknown model type ({})'.format(model_name))
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model architecture ({})'.format(arch_name))
        if not callable(fn):
            raise ValueError('Model architecture must be callable ({})'.format(arch_name))
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn

    return register_model_arch_fn


# automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.models.' + module)
