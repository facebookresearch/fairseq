# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture,
)


from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    base_architecture,
    transformer_iwslt_de_en,
    transformer_vaswani_wmt_en_de_big,
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

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerMonotonicEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)

    def _indices_from_states(self, states):
        if type(states["indices"]["src"]) == list:
            if next(self.parameters()).is_cuda:
                tensor = torch.cuda.LongTensor
            else:
                tensor = torch.LongTensor

            src_indices = tensor(
                [states["indices"]["src"][: 1 + states["steps"]["src"]]]
            )

            tgt_indices = tensor(
                [
                    [self.decoder.dictionary.eos()]
                    + states["indices"]["tgt"]
                ]
            )
        else:
            src_indices = states["indices"]["src"][: 1 +
                                                   states["steps"]["src"]]
            tgt_indices = states["indices"]["tgt"]

        return src_indices, None, tgt_indices

    def predict_from_states(self, states):
        decoder_states = self.decoder.output_layer(
            states["decoder_features"]
        )
        lprobs = self.get_normalized_probs(
            [decoder_states[:, -1:]],
            log_probs=True
        )

        index = lprobs.argmax(dim=-1)

        token = self.decoder.dictionary.string(index)

        return token, index[0, 0].item()

    def decision_from_states(self, states):
        '''
        This funcion take states dictionary as input, and gives the agent
        a decision of whether read a token from server. Moreover, the decoder
        states are also calculated here so we can directly generate a target
        token without recompute every thing
        '''

        self.eval()

        if len(states["tokens"]["src"]) == 0:
            return 0

        src_indices, src_lengths, tgt_indices = self._indices_from_states(
            states)

        # Update encoder states if needed
        if (
            "encoder_states" not in states or
            states["encoder_states"][0].size(1) <= states["steps"]["src"]
        ):
            encoder_out_dict = self.encoder(src_indices, src_lengths)
            states["encoder_states"] = encoder_out_dict
        else:
            encoder_out_dict = states["encoder_states"]

        # online means we still need tokens to feed the model
        states["model_states"]["online"] = not (
            states["finish_read"]
            and len(states["tokens"]["src"]) == states["steps"]["src"]
        )

        states["model_states"]["steps"] = states["steps"]

        x, outputs = self.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=encoder_out_dict,
            incremental_state=states["model_states"],
            features_only=True,
        )

        states["decoder_features"] = x

        return outputs["action"]


class TransformerMonotonicEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.dictionary = dictionary
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

        self.dictionary = dictionary
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerMonotonicDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

    def pre_attention(
        self, prev_output_tokens, encoder_out_dict,
        incremental_state=None
    ):
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

    def post_attention(self, x):
        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x

    def extract_features(
        self, prev_output_tokens, encoder_out,
        incremental_state=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # incremental_state = None
        (
            x,
            encoder_outs,
            encoder_padding_mask
        ) = self.pre_attention(
            prev_output_tokens,
            encoder_out,
            incremental_state
        )
        attn = None
        inner_states = [x]
        attn_list = []
        step_list = []

        for i, layer in enumerate(self.layers):

            x, attn, _ = layer(
                x=x,
                encoder_out=encoder_outs,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self.buffered_future_mask(x)
                if incremental_state is None else None,
            )

            inner_states.append(x)
            attn_list.append(attn)

            if incremental_state is not None:
                curr_steps = layer.get_steps(incremental_state)
                step_list.append(curr_steps)

                if incremental_state.get("online", False):
                    p_choose = attn["p_choose"].squeeze(0).squeeze(1).gather(1, curr_steps.t())

                    new_steps = (
                        curr_steps
                        + (p_choose < 0.5).t().type_as(curr_steps)
                    )

                    if (new_steps >= incremental_state["steps"]["src"]).any():
                        # We need to prune the last self_attn saved_state
                        # if model decide not to read
                        # otherwise there will be duplicated saved_state
                        for j in range(i + 1):
                            self.layers[j].prune_incremental_state(
                                incremental_state)

                        return x, {"action": 0}

        if (
            incremental_state is not None
            and not incremental_state.get("online", False)
        ):
            # Here is for fast evaluation
            fastest_step = torch.max(
                torch.cat(step_list, dim=1),
                dim=1,
                keepdim=True
            )[0] + 1

            if "fastest_step" in incremental_state:
                incremental_state["fastest_step"] = torch.cat(
                    [incremental_state["fastest_step"], fastest_step],
                    dim=1
                )
            else:
                incremental_state["fastest_step"] = fastest_step

        x = self.post_attention(x)

        return x, {
            "action": 1,
            "attn_list": attn_list,
            "step_list": step_list,
            "encoder_out": encoder_out,
            "encoder_padding_mask": encoder_padding_mask,
        }

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        if "fastest_step" in incremental_state:
            incremental_state["fastest_step"] = (
                incremental_state["fastest_step"]
                .index_select(0, new_order)
            )


@register_model_architecture(
    'transformer_monotonic',
    'transformer_monotonic'
)
def base_monotonic_rchitecture(args):
    base_architecture(args)
    args.encoder_unidirectional = getattr(
        args, 'encoder_unidirectional', False)


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


@register_model_architecture(
    'transformer_monotonic',
    'transformer_monotonic_vaswani_wmt_en_fr_big'
)
def transformer_monotonic_vaswani_wmt_en_fr_big(args):
    transformer_monotonic_vaswani_wmt_en_fr_big(args)


@register_model_architecture(
    'transformer_unidirectional',
    'transformer_unidirectional_iwslt_de_en'
)
def transformer_unidirectional_iwslt_de_en(args):
    transformer_iwslt_de_en(args)
