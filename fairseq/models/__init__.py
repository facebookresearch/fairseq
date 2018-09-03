# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import argparse
import importlib
import os

from .fairseq_decoder import FairseqDecoder  # noqa: F401
from .fairseq_encoder import FairseqEncoder  # noqa: F401
from .fairseq_incremental_decoder import FairseqIncrementalDecoder  # noqa: F401
from .fairseq_model import BaseFairseqModel, FairseqModel, FairseqLanguageModel  # noqa: F401

from .composite_encoder import CompositeEncoder  # noqa: F401


MODEL_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_MODEL_INV_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}


def build_model(args, task):
    return ARCH_MODEL_REGISTRY[args.arch].build_model(args, task)


def register_model(name):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqModel` for sequence-to-sequence
        tasks or :class:`FairseqLanguageModel` for language modeling tasks.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        if not issubclass(cls, BaseFairseqModel):
            raise ValueError('Model ({}: {}) must extend BaseFairseqModel'.format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def register_model_architecture(model_name, arch_name):
    """
    New model architectures can be added to fairseq with the
    :func:`register_model_architecture` function decorator. After registration,
    model architectures can be selected with the ``--arch`` command-line
    argument.

    For example::

        @register_model_architecture('lstm', 'lstm_luong_wmt_en_de')
        def lstm_luong_wmt_en_de(args):
            args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1000)
            (...)

    The decorated function should take a single argument *args*, which is a
    :class:`argparse.Namespace` of arguments parsed from the command-line. The
    decorated function should modify these arguments in-place to match the
    desired architecture.

    Args:
        model_name (str): the name of the Model (Model must already be
            registered)
        arch_name (str): the name of the model architecture (``--arch``)
    """

    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError('Cannot register model architecture for unknown model type ({})'.format(model_name))
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model architecture ({})'.format(arch_name))
        if not callable(fn):
            raise ValueError('Model architecture must be callable ({})'.format(arch_name))
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name, []).append(arch_name)
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn

    return register_model_arch_fn


# automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        model_name = file[:file.find('.py')]
        module = importlib.import_module('fairseq.models.' + model_name)

        # extra `model_parser` for sphinx
        if model_name in MODEL_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_archs = parser.add_argument_group('Named architectures')
            group_archs.add_argument('--arch', choices=ARCH_MODEL_INV_REGISTRY[model_name])
            group_args = parser.add_argument_group('Additional command-line arguments')
            MODEL_REGISTRY[model_name].add_args(group_args)
            globals()[model_name + '_parser'] = parser
