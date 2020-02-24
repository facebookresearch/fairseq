# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    Embedding,
    Linear,
    base_architecture,
    transformer_iwslt_de_en,
    transformer_vaswani_wmt_en_de_big,
    transformer_vaswani_wmt_en_fr_big
)

from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)


from examples.simultaneous_translation.modules.monotonic_transformer_layer import (
    TransformerMonotonicDecoderLayer,
    TransformerMonotonicEncoderLayer
)


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformer_unidirectional')
class TransformerUnidirectionalModel(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerMonotonicEncoder(args, src_dict, embed_tokens)

@register_model('transformer_monotonic')
class TransformerMonotonicModel(TransformerModel):

    #@classmethod
    #def build_model(cls, args, task):
    #    super_model = super().build_model(args, task)
    #    return cls(super_model.encoder, super_model.decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerMonotonicEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)

    def predict_from_states(self, states):
        src_indices = torch.LongTensor(
            [states["indices"]["src"][: 1 + states["steps"]["src"]]]
            )
        src_lengths = torch.LongTensor([src_indices.size(1)])

        tgt_indices = torch.LongTensor(
            [
                [self.decoder.dictionary.eos()]
                + states["indices"]["tgt"]
            ]
        )

        self.eval()

        # Update encoder state
        encoder_outs = self.encoder(src_indices, src_lengths)

        # Generate decoder state
        decoder_states, _ = self.decoder(
            tgt_indices, encoder_outs#, states
        )

        lprobs = self.get_normalized_probs(
            [decoder_states[:, -1:]], 
            log_probs=True
        )

        index = lprobs.argmax(dim=-1)

        token = self.decoder.dictionary.string(index) 

        return token, index[0, 0].item()

    def decision_from_states(self, states):
        if len(states["indices"]["src"]) == 0:
            return 0

        src_indices = torch.LongTensor(
            [states["indices"]["src"][: 1 + states["steps"]["src"]]]
            )
        src_lengths = torch.LongTensor([src_indices.size(1)])

        encoder_out_dict = self.encoder(src_indices, src_lengths)

        tgt_indices = torch.LongTensor(
            [
                [self.decoder.dictionary.eos()]
                + states["indices"]["tgt"]
            ]
        )

        # Only use the last token since we buffered 
        # the decoder states 
        # tgt_tokens = states["tgt_indices"][:, -1:]
        (
            x,
            encoder_out,
            encoder_padding_mask
        ) = self.decoder.pre_attention(
            tgt_indices,
            encoder_out_dict,
        )
        action = 1
        #print(states['src_indices'])
        for i, layer in enumerate(self.decoder.layers):
            #buffered_input = layer.encoder_attn._get_input_buffer(states)
            #if len(buffered_input) == 0:
            #    prev_attn_state = None
            #else:
            #    prev_attn_state = (
            #        buffered_input["prev_key"],
            #        buffered_input["prev_value"]
            #    )
            # Get action
            x, attn = layer(
                x=x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=None,
                #prev_attn_state=prev_attn_state,
            )
            # Update pointer
            #print(encoder_out.size())
            if attn["p_choose"].size(2) == 1:
                p_choose = attn["p_choose"][:, -1:, -1] 
            else:
                #import pdb; pdb.set_trace()
                pointer = layer.encoder_attn.get_pointer(states)["step"].clamp(0.0, attn["p_choose"].size(2) - 1)
                #print(attn["p_choose"].size(), pointer.max())
                p_choose = torch.gather(
                    attn["p_choose"][:,-1:,:], 2, pointer.long().unsqueeze(2))[:, :, -1]

                #print(layer.encoder_attn.get_pointer(states)["step"].t())
            
            layer.encoder_attn.set_pointer(states, p_choose)
            
            pointer = layer.encoder_attn.get_fastest_pointer(states)
            #print(layer.encoder_attn.get_pointer(states)["step"].t())
            if pointer.item() >= len(states["indices"]["src"]):
                action = 0
        #import pdb; pdb.set_trace()
        #print(action)
        return action


class TransformerMonotonicEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerMonotonicEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])


class TransformerMonotonicDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerMonotonicDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

    def pre_attention(self, prev_output_tokens, encoder_out_dict, incremental_state=None):
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_out = encoder_out_dict.encoder_out
        encoder_padding_mask = encoder_out_dict.encoder_padding_mask

        return x, encoder_out, encoder_padding_mask

    def extract_features(self, prev_output_tokens, encoder_out, incremental_state=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        (
            x,
            encoder_out,
            encoder_padding_mask
        ) = self.pre_attention(
            prev_output_tokens,
            encoder_out,
            incremental_state
        )
        # embed positions
        attn = None
        inner_states = [x]
        attn_list = []
        # decoder layers

        for i, layer in enumerate(self.layers):
             
            x, attn, _ = layer(
                x=x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self.buffered_future_mask(x)
                if incremental_state is None else None,
            )
            inner_states.append(x)
            attn_list.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn_list": attn_list, "encoder_out": encoder_out, "encoder_padding_mask": encoder_padding_mask}

    @staticmethod
    def increase_monotonic_step_buffer(encoder_out, num_layers, num_heads):
        if "monotonic_step_buffer" not in encoder_out:
            # initial monotonic step buffer
            encoder_out["monotonic_step_buffer"] = encoder_out['encoder_out'].new_zeros(
                encoder_out['encoder_out'].size(1),
                num_layers,
                num_heads,
                1
            ).long()
        else:
            encoder_out["monotonic_step_buffer"] = torch.cat(
                [
                    encoder_out["monotonic_step_buffer"],
                    encoder_out["monotonic_step_buffer"].new_zeros(
                        encoder_out["monotonic_step_buffer"].size(0),
                        num_layers,
                        num_heads,
                        1
                    )
                ],
                dim=3
            )

    @staticmethod
    def update_fastest_monotonic_step(layer, encoder_out, incremental_state):
        # current step
        # monotonic_step: bsz, num_heads
        monotonic_step = layer.encoder_attn._get_monotonic_buffer(incremental_state)["step"]

        encoder_out["monotonic_step_buffer"][:, layer.encoder_attn._fairseq_instance_id - 1, :, -1] = monotonic_step

    def calculate_latency(self, encoder_out):
        # src_lens: bsz, num_heads
        # src_lens is also considered here because fastest_monotonic_step could be
        # larger than max_src_len (which means it reach the eos).
        if encoder_out['encoder_padding_mask'] is not None:
            src_lens = (1 - encoder_out['encoder_padding_mask'].long()).sum(dim=1, keepdim=True)
        else:
            src_lens = (
                encoder_out['encoder_out'].size(0)
                * encoder_out['encoder_out']
                .new_ones([encoder_out['encoder_out'].size(1), 1]).long()
            )

        # delays: bsz, tgt_len
        # src_lens: bsz, 1

        for key, value in self.latency_inference(
            encoder_out["monotonic_step_buffer"], src_lens
        ).items():
            encoder_out[key] = value


@register_model_architecture(
    'transformer_monotonic',
    'transformer_monotonic'
)
def base_monotonic_rchitecture(args):
    base_architecture(args)
    args.encoder_unidirectional = getattr(args, 'encoder_unidirectional', False)


@register_model_architecture(
    'transformer_monotonic',
    'transformer_monotonic_iwslt_de_en'
)
def transformer_monotonic_iwslt_de_en(args):
    transformer_iwslt_de_en(args)
    base_monotonic_rchitecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture(
    'transformer_monotonic',
    'transformer_monotonic_vaswani_wmt_en_de_big'
)
def transformer_monotonic_vaswani_wmt_en_de_big(args):
    transformer_vaswani_wmt_en_de_big(args)
    base_architecture(args)


@register_model_architecture(
    'transformer_monotonic',
    'transformer_monotonic_vaswani_wmt_en_fr_big'
)
def transformer_monotonic_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_monotonic_vaswani_wmt_en_de_big(args)
            
@register_model_architecture(
    'transformer_unidirectional',
    'transformer_unidirectional_iwslt_de_en'
)
def transformer_unidirectional_iwslt_de_en(args):
    transformer_iwslt_de_en(args)

