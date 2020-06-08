# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = torch.zeros(
                    (1, 1), device=input.device, dtype=input.dtype
                ).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = utils.make_positions(
                    input, self.padding_idx, onnx_trace=self.onnx_trace
                )
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class LearnedRelativePositionalEmbedding(nn.Module):
    """
    This module learns relative positional embeddings up to a fixed
    maximum size. These are masked for decoder and unmasked for encoder
    self attention.
    By default the embeddings are added to keys, but could be added to
    values as well.

    Args:
        max_relative_pos (int): the maximum relative positions to compute embeddings for
        num_heads (int): number of attention heads
        embedding_dim (int): depth of embeddings
        unmasked (bool): if the attention is unmasked (for transformer encoder)
        heads_share_embeddings (bool): if heads share the same relative positional embeddings
        add_to_values (bool): compute embeddings to be added to values as well
    """

    def __init__(
            self,
            max_relative_pos: int,
            num_heads: int,
            embedding_dim: int,
            unmasked: bool = False,
            heads_share_embeddings: bool = False,
            add_to_values: bool = False):
        super().__init__()
        self.max_relative_pos = max_relative_pos
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.unmasked = unmasked
        self.heads_share_embeddings = heads_share_embeddings
        self.add_to_values = add_to_values
        num_embeddings = (
            2 * max_relative_pos - 1
            if unmasked
            else max_relative_pos
        )
        embedding_size = (
            [num_embeddings, embedding_dim, 1]
            if heads_share_embeddings
            else [num_heads, num_embeddings, embedding_dim, 1]
        )
        if add_to_values:
            embedding_size[-1] = 2
        initial_stddev = embedding_dim**(-0.5)
        self.embeddings = nn.Parameter(torch.zeros(*embedding_size))
        nn.init.normal_(self.embeddings, mean=0.0, std=initial_stddev)

    def forward(self, query, saved_state=None):
        """
        Computes relative positional embeddings to be added to keys (and optionally values),
        multiplies the embeddings for keys with queries to create positional logits,
        returns the positional logits, along with embeddings for values (optionally)
        which could be added to values outside this module.

        Args:
            query (torch.Tensor): query tensor
            saved_state (dict): saved state from previous time step

        Shapes:
            query: `(length, batch_size*num_heads, embed_dim)`

        Returns:
            tuple(torch.Tensor):
                - positional logits
                - relative positional embeddings to be added to values
        """
        # During inference when previous states are cached
        if saved_state is not None and "prev_key" in saved_state:
            assert not self.unmasked, "This should only be for decoder attention"
            length = saved_state["prev_key"].shape[-2] + 1  # `length - 1` keys are cached,
                                                            # `+ 1` for the current time step
            decoder_step = True
        else:
            length = query.shape[0]
            decoder_step = False

        used_embeddings = self.get_embeddings_for_query(length)

        values_embeddings = (
            used_embeddings[..., 1]
            if self.add_to_values
            else None
        )
        positional_logits = self.calculate_positional_logits(query, used_embeddings[..., 0])
        positional_logits = self.relative_to_absolute_indexing(positional_logits, decoder_step)
        return (positional_logits, values_embeddings)

    def get_embeddings_for_query(self, length):
        """
        Extract the required embeddings. The maximum relative position between two time steps is
        `length` for masked case or `2*length - 1` for the unmasked case. If `length` is greater than
        `max_relative_pos`, we first pad the embeddings tensor with zero-embeddings, which represent
        embeddings when relative position is greater than `max_relative_pos`. In case `length` is
        less than `max_relative_pos`, we don't use the first `max_relative_pos - length embeddings`.

        Args:
            length (int): length of the query

        Returns:
            torch.Tensor: embeddings used by the query
        """
        pad_length = max(length - self.max_relative_pos, 0)
        start_pos = max(self.max_relative_pos - length, 0)
        if self.unmasked:
            with torch.no_grad():
                padded_embeddings = nn.functional.pad(
                    self.embeddings,
                    (0, 0, 0, 0, pad_length, pad_length)
                )
            used_embeddings = padded_embeddings.narrow(-3, start_pos, 2*length - 1)
        else:
            with torch.no_grad():
                padded_embeddings = nn.functional.pad(
                    self.embeddings,
                    (0, 0, 0, 0, pad_length, 0)
                )
            used_embeddings = padded_embeddings.narrow(-3, start_pos, length)
        return used_embeddings

    def calculate_positional_logits(self, query, relative_embeddings):
        """
        Multiplies query with the relative positional embeddings to create relative
        positional logits

        Args:
            query (torch.Tensor): Input tensor representing queries
            relative_embeddings (torch.Tensor): relative embeddings compatible with query

        Shapes:
            query: `(length, batch_size*num_heads, embed_dim)` if heads share embeddings
                   else `(length, batch_size, num_heads, embed_dim)`
            relative_embeddings: `(max_allowed_relative_positions, embed_dim)` if heads share embeddings
                                 else `(num_heads, max_allowed_relative_positions, embed_dim)`
                                 where `max_allowed_relative_positions` is `length` if masked
                                 else `2*length - 1`

        Returns:
            torch.Tensor: relative positional logits
        """
        if self.heads_share_embeddings:
            positional_logits = torch.einsum("lbd,md->lbm", query, relative_embeddings)
        else:
            query = query.view(query.shape[0], -1, self.num_heads, self.embedding_dim)
            positional_logits = torch.einsum("lbhd,hmd->lbhm", query, relative_embeddings)
            positional_logits = positional_logits.contiguous().view(
                positional_logits.shape[0], -1, positional_logits.shape[-1]
            )
        return positional_logits

    def relative_to_absolute_indexing(self, x, decoder_step):
        """
        Index tensor x (relative positional logits) in terms of absolute positions
        rather than relative positions. Last dimension of x represents relative position
        with respect to the first dimension, whereas returned tensor has both the first
        and last dimension indexed with absolute positions.

        Args:
            x (torch.Tensor): positional logits indexed by relative positions
            decoder_step (bool): is this is a single decoder step (during inference)

        Shapes:
            x: `(length, batch_size*num_heads, length)` for masked case or
               `(length, batch_size*num_heads, 2*length - 1)` for unmasked

        Returns:
            torch.Tensor: positional logits represented using absolute positions
        """
        length, bsz_heads, _ = x.shape

        if decoder_step:
            return x.contiguous().view(bsz_heads, 1, -1)

        if self.unmasked:
            x = nn.functional.pad(
                x,
                (0, 1)
            )
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length * 2 * length)
            x = nn.functional.pad(
                x,
                (0, length - 1)
            )
            # Reshape and slice out the padded elements.
            x = x.view(bsz_heads, length + 1, 2*length - 1)
            return x[:, :length, length-1:]
        else:
            x = nn.functional.pad(
                x,
                (1, 0)
            )
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length+1, length)
            return x[:, 1:, :]