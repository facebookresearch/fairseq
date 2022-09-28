# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

from torch import Tensor

from fairseq.models.transformer import Linear
from fairseq.models.transformer.transformer_decoder_aug import AugTransformerDecoder


class AugTransformerUnitDecoder(AugTransformerDecoder):
    """Based on Transformer decoder, with support to decoding stacked units"""

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn, output_projection
        )
        self.n_frames_per_step = args.n_frames_per_step

        self.out_proj_n_frames = (
            Linear(
                self.output_embed_dim,
                self.output_embed_dim * self.n_frames_per_step,
                bias=False,
            )
            if self.n_frames_per_step > 1
            else None
        )

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        encoder_out_aug: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            encoder_out_aug=encoder_out_aug,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            bsz, seq_len, d = x.size()
            if self.out_proj_n_frames:
                x = self.out_proj_n_frames(x)
            x = self.output_layer(x.view(bsz, seq_len, self.n_frames_per_step, d))
            x = x.view(bsz, seq_len * self.n_frames_per_step, -1)
            if (
                incremental_state is None and self.n_frames_per_step > 1
            ):  # teacher-forcing mode in training
                x = x[
                    :, : -(self.n_frames_per_step - 1), :
                ]  # remove extra frames after <eos>

        return x, extra

    def upgrade_state_dict_named(self, state_dict, name):
        if self.n_frames_per_step > 1:
            move_keys = [
                (
                    f"{name}.project_in_dim.weight",
                    f"{name}.embed_tokens.project_in_dim.weight",
                )
            ]
            for from_k, to_k in move_keys:
                if from_k in state_dict and to_k not in state_dict:
                    state_dict[to_k] = state_dict[from_k]
                    del state_dict[from_k]
