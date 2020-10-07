# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from torch import nn

from fairseq.models import (
    register_model,
    register_model_architecture,
)

from .convtransformer import (
    ConvTransformerModel,
    ConvTransformerEncoder,
    convtransformer_espnet
)

from examples.simultaneous_translation.models import (
    TransformerMonotonicModel,
)

from examples.simultaneous_translation.modules import (
    TransformerMonotonicEncoderLayer
)

from fairseq import checkpoint_utils


@register_model('convtransformer_simul')
class SimulConvTransformerModel(ConvTransformerModel):

    @staticmethod
    def add_args(parser):
        super(SimulConvTransformerModel, SimulConvTransformerModel).add_args(parser)
        parser.add_argument(
            "--train-monotonic-only",
            action="store_true",
            default=False,
            help="Only train monotonic attention",
        )

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        tgt_dict = task.tgt_dict

        decoder = TransformerMonotonicModel.build_decoder(
            args, tgt_dict, embed_tokens)

        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder


@register_model('convtransformer_simul_uni')
class SimulConvTransformerModelUnidirectional(SimulConvTransformerModel):
    @classmethod
    def build_encoder(cls, args):
        encoder = ConvTransformerEncoderUnidirectional(args)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
        return encoder


class ConvTransformerEncoderUnidirectional(ConvTransformerEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.transformer_layers = nn.ModuleList([])
        self.transformer_layers.extend(
            [
                TransformerMonotonicEncoderLayer(args)
                for i in range(args.encoder_layers)
            ]
        )


@register_model_architecture(
    'convtransformer_simul',
    'convtransformer_simul'
)
def convtransformer_simul_trans_espnet(args):
    convtransformer_espnet(args)


@register_model_architecture(
    'convtransformer_simul_uni',
    'convtransformer_simul_uni'
)
def convtransformer_simul_trans_espnet_uni(args):
    convtransformer_espnet(args)
