# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.models.transformer import TransformerModel
from fairseq.models.fconv import FConvModel
from fairseq.models.fconv_self_att import FConvModelSelfAtt
from generator import Generator
from fairseq import options


dependencies = [
    'torch',
    'sacremoses',
    'subword_nmt',
]


def transformer(*args, **kwargs):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.
    """
    parser = options.get_interactive_generation_parser()
    model = TransformerModel.from_pretrained(parser, *args, **kwargs)
    return model


def fconv(*args, **kwargs):
    """
    A fully convolutional model, i.e. a convolutional encoder and a
    convolutional decoder, as described in `"Convolutional Sequence to Sequence
    Learning" (Gehring et al., 2017) <https://arxiv.org/abs/1705.03122>`_.
    """
    parser = options.get_interactive_generation_parser()
    model = FConvModel.from_pretrained(parser, *args, **kwargs)
    return model


def fconv_self_att(*args, **kwargs):
    parser = options.get_interactive_generation_parser()
    model = FConvModelSelfAtt.from_pretrained(parser, *args, **kwargs)
    return model


def generator(*args, **kwargs):
    parser = options.get_generation_parser(interactive=True)
    generator = Generator.from_pretrained(parser, *args, **kwargs)
    return generator
