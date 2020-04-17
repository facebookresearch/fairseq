# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Dict

import torch
from torch import Tensor

from fairseq.models import FairseqEncoder
from fairseq.models.fairseq_encoder import EncoderOut
from .t5_modules import T5Stack


class T5Encoder(FairseqEncoder):
    """
    T5 encoder for T5 model.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary=dictionary)

        # self.padding_idx = embed_tokens.padding_idx
        self.padding_idx = 0

        encoder_config = copy.deepcopy(args)
        encoder_config.is_decoder = False

        self.max_source_positions = 512

        self.t5_stack = T5Stack(config=encoder_config, embed_tokens=embed_tokens)

    def forward(self, src_tokens, src_lengths, cls_input=None, return_all_hiddens=False, **unused):
        encoder_padding_mask = ~ src_tokens.eq(self.padding_idx)
        hidden_states = self.t5_stack.forward(src_tokens, encoder_padding_mask)[0].transpose(0, 1)
        encoder_out = EncoderOut(
            encoder_out=hidden_states,
            encoder_padding_mask=encoder_padding_mask,
            encoder_embedding=None,
            encoder_states=None)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out: Dict[str, Tensor] = {}

        new_encoder_out["encoder_out"] = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_out["encoder_padding_mask"] = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_out["encoder_embedding"] = (
            encoder_out.encoder_embedding
            if encoder_out.encoder_embedding is None
            else encoder_out.encoder_embedding.index_select(0, new_order)
        )

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out["encoder_out"],  # T x B x C
            encoder_padding_mask=new_encoder_out["encoder_padding_mask"],  # B x T
            encoder_embedding=new_encoder_out["encoder_embedding"],  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )
