# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import argparse
import importlib
import os

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import merge_with_parent, populate_dataclass
from hydra.core.config_store import ConfigStore

from .composite_encoder import CompositeEncoder
from .distributed_fairseq_model import DistributedFairseqModel
from .fairseq_decoder import FairseqDecoder
from .fairseq_encoder import FairseqEncoder
from .fairseq_incremental_decoder import FairseqIncrementalDecoder
from .fairseq_model import (
    BaseFairseqModel,
    FairseqEncoderDecoderModel,
    FairseqEncoderModel,
    FairseqLanguageModel,
    FairseqModel,
    FairseqMultiModel,
)


MODEL_REGISTRY = {}
MODEL_DATACLASS_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_MODEL_NAME_REGISTRY = {}
ARCH_MODEL_INV_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}


__all__ = [
    "BaseFairseqModel",
    "CompositeEncoder",
    "DistributedFairseqModel",
    "FairseqDecoder",
    "FairseqEncoder",
    "FairseqEncoderDecoderModel",
    "FairseqEncoderModel",
    "FairseqIncrementalDecoder",
    "FairseqLanguageModel",
    "FairseqModel",
    "FairseqMultiModel",
]


def build_model(cfg: FairseqDataclass, task):

    model = None
    model_type = getattr(cfg, "_name", None) or getattr(cfg, "arch", None)

    if not model_type and len(cfg) == 1:
        # this is hit if config object is nested in directory that is named after model type

        model_type = next(iter(cfg))
        if model_type in MODEL_DATACLASS_REGISTRY:
            cfg = cfg[model_type]
        else:
            raise Exception(
                "Could not infer model type from directory. Please add _name field to indicate model type. "
                "Available models: "
                + str(MODEL_DATACLASS_REGISTRY.keys())
                + " Requested model type: "
                + model_type
            )

    if model_type in ARCH_MODEL_REGISTRY:
        # case 1: legacy models
        model = ARCH_MODEL_REGISTRY[model_type]
    elif model_type in MODEL_DATACLASS_REGISTRY:
        # case 2: config-driven models
        model = MODEL_REGISTRY[model_type]

    if model_type in MODEL_DATACLASS_REGISTRY:
        # set defaults from dataclass. note that arch name and model name can be the same
        dc = MODEL_DATACLASS_REGISTRY[model_type]
        if isinstance(cfg, argparse.Namespace):
            cfg = populate_dataclass(dc(), cfg)
        else:
            cfg = merge_with_parent(dc(), cfg)

    assert model is not None, (
        f"Could not infer model type from {cfg}. "
        f"Available models: "
        + str(MODEL_DATACLASS_REGISTRY.keys())
        + " Requested model type: "
        + model_type
    )

    return model.build_model(cfg, task)


def register_model(name, dataclass=None):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        if not issubclass(cls, BaseFairseqModel):
            raise ValueError(
                "Model ({}: {}) must extend BaseFairseqModel".format(name, cls.__name__)
            )
        MODEL_REGISTRY[name] = cls
        if dataclass is not None and not issubclass(dataclass, FairseqDataclass):
            raise ValueError(
                "Dataclass {} must extend FairseqDataclass".format(dataclass)
            )

        cls.__dataclass = dataclass
        if dataclass is not None:
            MODEL_DATACLASS_REGISTRY[name] = dataclass

            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="model", node=node, provider="fairseq")

            @register_model_architecture(name, name)
            def noop(_):
                pass

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
        def lstm_luong_wmt_en_de(cfg):
            args.encoder_embed_dim = getattr(cfg.model, 'encoder_embed_dim', 1000)
            (...)

    The decorated function should take a single argument *cfg*, which is a
    :class:`omegaconf.DictConfig`. The decorated function should modify these
    arguments in-place to match the desired architecture.

    Args:
        model_name (str): the name of the Model (Model must already be
            registered)
        arch_name (str): the name of the model architecture (``--arch``)
    """

    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                "Cannot register model architecture for unknown model type ({})".format(
                    model_name
                )
            )
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError(
                "Cannot register duplicate model architecture ({})".format(arch_name)
            )
        if not callable(fn):
            raise ValueError(
                "Model architecture must be callable ({})".format(arch_name)
            )
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_MODEL_NAME_REGISTRY[arch_name] = model_name
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name, []).append(arch_name)
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn

    return register_model_arch_fn


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("fairseq.models." + model_name)

        # extra `model_parser` for sphinx
        if model_name in MODEL_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_archs = parser.add_argument_group("Named architectures")
            group_archs.add_argument(
                "--arch", choices=ARCH_MODEL_INV_REGISTRY[model_name]
            )
            group_args = parser.add_argument_group("Additional command-line arguments")
            MODEL_REGISTRY[model_name].add_args(group_args)
            globals()[model_name + "_parser"] = parser
