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

from .vggtransformer import (
    lengths_to_encoder_padding_mask
)

from fairseq.models.fairseq_encoder import EncoderOut

import torch.nn.functional as F

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
        #if getattr(args, "load_pretrained_encoder_from", None):
            #encoder = checkpoint_utils.load_pretrained_component_from_model(
            #    component=encoder, checkpoint=args.load_pretrained_encoder_from
            #)
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

    def incremental_forward(self, src_tokens, incremental_states):
        bsz, max_seq_len, _ = src_tokens.size()
        src_lengths = (
            src_tokens.new_ones([1, 1]).long()
            * (incremental_states["steps"]['src'] * 4 + max_seq_len) # TODO: change hard coded 40
        )
        x = (
            src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
            .transpose(1, 2)
            .contiguous()
        )
        x = self.conv(x)
        bsz, _, output_seq_len, _ = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
        # Remove the results for lookahead
        x = x[1:-1]
        x = self.out(x)
        x = self.embed_scale * x

        subsampling_factor = 1.0 * max_seq_len / output_seq_len
        input_lengths = (src_lengths.float() / subsampling_factor).round().long()
        # Minus because of the lookahead
        input_lengths -= 1
        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            input_lengths, batch_first=True
        )

        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)

        x += positions[-x.size(0): ]
        #print(positions[-x.size(0):,0,0])
        #print(x[:,:,0])
        encoder_padding_mask = encoder_padding_mask[:, -x.size(0):]

        x = F.dropout(x, p=self.dropout, training=self.training)

        #print(x.size(), encoder_padding_mask.size())
        #if x.size(0) != encoder_padding_mask.size(1):

        for layer in self.transformer_layers:
            #if i == 1:
            #    print(x[:,:,0])
            x = layer.incremental_forward(
                x, encoder_padding_mask, incremental_states
            )

        if not encoder_padding_mask.any():
            maybe_encoder_padding_mask = None
        else:
            maybe_encoder_padding_mask = encoder_padding_mask

        return EncoderOut(
            encoder_out=x,
            encoder_padding_mask=maybe_encoder_padding_mask,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
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
