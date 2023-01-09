from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn

from fairseq.models.streaming.modules.monotonic_transformer_layer import (
    TransformerMonotonicDecoderLayer,
)

from fairseq.models.transformer import (
    TransformerDecoder,
)
from torch import Tensor
import re
from argparse import Namespace
from fairseq.file_io import PathManager
from collections import OrderedDict
from fairseq.checkpoint_utils import load_checkpoint_to_cpu

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
READ_ACTION = 0
WRITE_ACTION = 1

TransformerMonotonicDecoderOut = NamedTuple(
    "TransformerMonotonicDecoderOut",
    [
        ("action", int),
        ("p_choose", Optional[Tensor]),
        ("attn_list", Optional[List[Optional[Dict[str, Tensor]]]]),
        ("inner_states", Optional[List[Optional[Dict[str, Tensor]]]]),
        ("encoder_out", Optional[Dict[str, List[Tensor]]]),
        ("encoder_padding_mask", Optional[Tensor]),
    ],
)


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
        self.layers.extend(
            [TransformerMonotonicDecoderLayer(args) for _ in range(args.decoder_layers)]
        )
        self.policy_criterion = getattr(args, "policy_criterion", "any")
        self.num_updates = None

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def pre_attention(
        self,
        prev_output_tokens,
        encoder_out_dict: Dict[str, List[Tensor]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        positions = (
            self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            )
            if self.embed_positions is not None
            else None
        )
        if len(prev_output_tokens.size()) == 3:
            assert prev_output_tokens.size(2) == 1
            prev_output_tokens = prev_output_tokens.squeeze(2)

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

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_out = encoder_out_dict["encoder_out"][-1]

        if "encoder_padding_mask" in encoder_out_dict:
            encoder_padding_mask = (
                encoder_out_dict["encoder_padding_mask"][0]
                if encoder_out_dict["encoder_padding_mask"]
                and len(encoder_out_dict["encoder_padding_mask"]) > 0
                else None
            )
        else:
            encoder_padding_mask = None

        return x, encoder_out, encoder_padding_mask

    def post_attention(self, x):
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x

    def clean_cache(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        end_id: Optional[int] = None,
    ):
        """
        Clean cache in the monotonic layers.
        The cache is generated because of a forward pass of decoder has run but no prediction,
        so that the self attention key value in decoder is written in the incremental state.
        end_id is the last idx of the layers
        """
        if end_id is None:
            end_id = len(self.layers)

        for index, layer in enumerate(self.layers):
            if index < end_id:
                layer.prune_incremental_state(incremental_state)

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,  # unused
        alignment_layer: Optional[int] = None,  # unused
        alignment_heads: Optional[int] = None,  # unsed
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # incremental_state = None
        assert encoder_out is not None
        (x, encoder_outs, encoder_padding_mask) = self.pre_attention(
            prev_output_tokens, encoder_out, incremental_state
        )
        attn = None
        inner_states = [x]
        attn_list: List[Optional[Dict[str, Tensor]]] = []

        p_choose = torch.tensor([1.0])

        for i, layer in enumerate(self.layers):

            x, attn, _ = layer(
                x=x,
                encoder_out=encoder_outs,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self.buffered_future_mask(x)
                if incremental_state is None
                else None,
            )

            inner_states.append(x)
            attn_list.append(attn)

            if incremental_state is not None and "online" in incremental_state:
                if_online = incremental_state["online"]["only"]
                assert if_online is not None
                if if_online.to(torch.bool):
                    # Online indicates that the encoder states are still changing
                    assert attn is not None
                    if self.policy_criterion == "any":
                        # Any head decide to read than read
                        head_read = layer.encoder_attn._get_monotonic_buffer(
                            incremental_state
                        )["head_read"]
                        assert head_read is not None
                        if head_read.any():
                            # We need to prune the last self_attn saved_state
                            # if model decide not to read
                            # otherwise there will be duplicated saved_state
                            self.clean_cache(incremental_state, i + 1)

                            return x, TransformerMonotonicDecoderOut(
                                action=0,
                                p_choose=p_choose,
                                attn_list=None,
                                encoder_out=None,
                                encoder_padding_mask=None,
                                inner_states=None,
                            )

        x = self.post_attention(x)

        return x, TransformerMonotonicDecoderOut(
            action=1,
            p_choose=p_choose,
            attn_list=attn_list,
            inner_states=inner_states,
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask,
        )

    @staticmethod
    def load_pretrained_monotonic_decoder_from_model(component, checkpoint: str):
        if not PathManager.exists(checkpoint):
            raise IOError("Model file not found: {}".format(checkpoint))
        state = load_checkpoint_to_cpu(checkpoint)
        component_type = "decoder"
        component_state_dict = OrderedDict()
        for key in state["model"].keys():
            if key.startswith(component_type):
                # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
                component_subkey = key[len(component_type) + 1 :]
                if "encoder_attn" in component_subkey:
                    component_subkey = component_subkey.replace("k_proj", "k_proj_soft")
                    component_subkey = component_subkey.replace("q_proj", "q_proj_soft")
                component_state_dict[component_subkey] = state["model"][key]

        keys = component.load_state_dict(component_state_dict, strict=False)
        for key in keys.missing_keys:
            if not (
                re.match(r"layers\.\d+\.encoder_attn\.[kq]_proj\.[weight|bias]", key)
                or re.match(r"layers\.\d+\.encoder_attn\.energy_bias", key)
            ):
                return key
        return component

    @classmethod
    def from_offline_decoder(
        cls, offline_decoder: TransformerDecoder, simul_args: Namespace
    ):
        new_args = offline_decoder.args
        for name, value in vars(simul_args).items():
            setattr(new_args, name, value)

        online_decoder = cls(
            new_args, offline_decoder.dictionary, offline_decoder.embed_tokens
        )

        state = offline_decoder.state_dict()
        component_state_dict = OrderedDict()

        for key in state.keys():
            if re.match(r"layers\.\d+\.encoder_attn", key):
                new_key = key.replace("k_proj", "k_proj_soft").replace(
                    "q_proj", "q_proj_soft"
                )
                component_state_dict[new_key] = state[key]
                if "waitk" not in simul_args.simul_type:
                    component_state_dict[key] = state[key]
            else:
                component_state_dict[key] = state[key]

        online_decoder.load_state_dict(component_state_dict)
        online_decoder.to(next(offline_decoder.parameters()).device)

        return online_decoder
