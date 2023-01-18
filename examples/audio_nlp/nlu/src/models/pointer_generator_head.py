#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import FairseqDecoder
from fairseq.models.fairseq_encoder import EncoderOut

from fairseq.modules.multihead_attention import MultiheadAttention


class EncoderDecoderMultiheadAttention(nn.Module):
    """
    Multiheaded Scaled Dot Product Attention
    Implements equation:
    MultiHead(Q, K, V) = Concat(head_1,...,head_h)W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    Similarly to the above, d_k = d_v = d_model / h
    In this implementation, keys and values are both set to encoder output
    Inputs
      init:
        decoder_hidden_state_dim : dimensionality of decoder hidden state
        context_dim : dimensionality of encoder output
        kwargs :
          nheads : integer # of attention heads
          unseen_mask: if True, only attend to previous sequence positions
          src_lengths_mask: if True, mask padding based on src_lengths
      forward:
        decoder_state : [batch size, d_model]
        source_hids : [sequence length, batch size, d_model]
        src_lengths : [batch size]
      forward:
        query : [sequence length, batch size, d_model]
        key: [sequence length, batch size, d_model]
        value: [sequence length, batch size, d_model]
    Output
      result : [batch_size,  d_model]
    """

    def __init__(
        self,
        decoder_hidden_state_dim,
        context_dim,
        *,
        nheads=1,
        unseen_mask=False,
        src_length_mask=True,
    ):
        super().__init__()
        self.decoder_hidden_state_dim = decoder_hidden_state_dim
        self.context_dim = context_dim
        assert decoder_hidden_state_dim == context_dim
        d_model = decoder_hidden_state_dim  # for brevity
        assert d_model % nheads == 0

        if unseen_mask:
            raise NotImplementedError(
                "Unseen mask not supported with sequential decoding"
            )
        self._fair_attn = MultiheadAttention(d_model, nheads)
        self.use_src_length_mask = src_length_mask

    def forward(self, decoder_state, source_hids, encoder_padding_mask, squeeze=True):
        """
        Computes MultiheadAttention with respect to either a vector
        or a tensor
        Inputs:
            decoder_state: (bsz x decoder_hidden_state_dim) or
                (bsz x T x decoder_hidden_state_dim)
            source_hids: srclen x bsz x context_dim
            src_lengths: bsz x 1, actual sequence lengths
            squeeze: Whether or not to squeeze on the time dimension.
                Even if decoder_state.dim() is 2 dimensional an
                explicit time step dimension will be unsqueezed.
        Outputs:
          [batch_size, max_src_len] if decoder_state.dim() == 2 & squeeze
            or
          [batch_size, 1, max_src_len] if decoder_state.dim() == 2 & !squeeze
            or
          [batch_size, T, max_src_len] if decoder_state.dim() == 3 & !squeeze
            or
          [batch_size, T, max_src_len] if decoder_state.dim() == 3 & squeeze & T != 1
            or
          [batch_size, max_src_len] if decoder_state.dim() == 3 & squeeze & T == 1
        """
        if decoder_state.dim() == 3:
            query = decoder_state
        elif decoder_state.dim() == 2:
            query = decoder_state.unsqueeze(1)
        else:
            raise ValueError("decoder state must be either 2 or 3 dimensional")
        query = query.transpose(0, 1)
        value = key = source_hids

        attn, attn_weights = self._fair_attn.forward(
            query, key, value, key_padding_mask=encoder_padding_mask, need_weights=True
        )
        # attn.shape = T X bsz X embed_dim
        # attn_weights.shape = bsz X T X src_len

        attn_weights = attn_weights.transpose(0, 2)
        # attn_weights.shape = src_len X T X bsz

        if squeeze:
            attn = attn.squeeze(0)
            # attn.shape = squeeze(T) X bsz X embed_dim
            attn_weights = attn_weights.squeeze(1)
            # attn_weights.shape = src_len X squeeze(T) X bsz
            return attn, attn_weights
        return attn, attn_weights


class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input):
        """
        input: inputs: seq_len x batch_size x hidden_size,
               padding_mask: seq_len x batch_size
        output: batch_size x hidden_size
        """
        inputs, padding_mask = input
        max_len, batch_size, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(max_len, batch_size)
        # apply mask
        if padding_mask is not None:
            logits = logits.masked_fill_(padding_mask, float("-inf"))
        alphas = F.softmax(logits, dim=-1)

        # transpose inputs
        alphas = alphas.transpose(0, 1)
        inputs = inputs.transpose(0, 1)
        return torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)

	


class PointerGeneratorHead(FairseqDecoder):
    """
    The "DecoupledCopyHead"
    Given encoder states e_t and decoder states d_t,
    the final output distribution o_t is computed as the combination of
    the copy probability and the generation probability:
    o_t = P_{copy} * c_t + (1-P_{copy}) * g_t
    where g_t is calculated using d_t.
    c_t and P_{copy} is computed using an encoder-decoder attention.
    """

    def __init__(
        self,
        dictionary,
        embed_tokens,
        pointer_attention_heads: int,
        encoder_embed_dim: int,
        decoder_out_embed_dim: int,
        fixed_generation_vocab=None,
    ):
        super().__init__(dictionary)
        self.num_embeddings = len(dictionary)
        self.output_embed_dim = decoder_out_embed_dim
        self.share_input_output_embed = False  # args.share_decoder_input_output_embed
        self.embed_tokens = embed_tokens

        self.pointer_attention_heads = pointer_attention_heads
        self.pointer_projection = nn.Linear(encoder_embed_dim, self.output_embed_dim)
        self.pointer_prob_map = nn.Linear(self.output_embed_dim * 2, 1)
        self.pointer_attention = EncoderDecoderMultiheadAttention(
            self.output_embed_dim,
            self.output_embed_dim,
            nheads=self.pointer_attention_heads,
            src_length_mask=True,
        )
        self.fixed_vocab = not (fixed_generation_vocab is None)
        if self.fixed_vocab:
            assert isinstance(
                fixed_generation_vocab, list
            ), "List of indices is what is expected for fixed_generation_vocab"
            self.register_buffer(
                "fixed_generation_vocab_expanded",
                torch.tensor(fixed_generation_vocab, dtype=torch.long)
                .unsqueeze(0)
                .unsqueeze(0),
            )
            if not self.share_input_output_embed:
                # update output layer
                self.embed_out = nn.Parameter(
                    torch.Tensor(len(fixed_generation_vocab), self.output_embed_dim)
                )
                nn.init.normal_(
                    self.embed_out, mean=0, std=self.output_embed_dim ** -0.5
                )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            if self.fixed_vocab:
                return F.linear(
                    features,
                    self.embed_tokens.weight.index_select(
                        dim=0, index=self.fixed_generation_vocab_expanded[0][0]
                    ),
                )
            else:
                return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def forward(self, encoder_out, features):
        src_tokens = encoder_out["src_tokens"][0]
        encoder_outs = encoder_out["encoder_out"][0]
        encoder_padding_mask = encoder_out["encoder_padding_mask"][0]
        x, extra = features

        logits = self.output_layer(x)
        if self.fixed_vocab:
            # we now have to project to the full vocab size in order
            # to get the mixture of probability distributions right.
            optional_fixed_logits = torch.zeros(
                (logits.size(0), logits.size(1), self.num_embeddings),
                device=logits.device,
            )
            fixed_expanded = self.fixed_generation_vocab_expanded.repeat(
                logits.size(0), logits.size(1), 1
            )
            optional_fixed_logits.scatter_add_(
                2, fixed_expanded, F.softmax(logits, dim=2)
            )
        else:
            optional_fixed_logits = F.softmax(logits, dim=2)

        # pointer mechanism for copying
        encoder_outs = self.pointer_projection(encoder_outs)
        cur_src_attn, calc_src_attn_scores = self.pointer_attention(
            x,
            encoder_outs,
            encoder_padding_mask if encoder_padding_mask is not None else None,
            squeeze=False,
        )
        cur_src_attn = cur_src_attn.transpose(0, 1)
        calc_src_attn_scores = calc_src_attn_scores.transpose(0, 2)
        prob = torch.sigmoid(self.pointer_prob_map(torch.cat([cur_src_attn, x], dim=2)))
        vocab_attn_scores = torch.zeros(
            *optional_fixed_logits.size(), device=optional_fixed_logits.device
        ).to(optional_fixed_logits)
        src_tokens_expanded = src_tokens.unsqueeze(1).repeat(1, logits.size(1), 1)
        # calc_src_attn_scores are already probabilities
        vocab_attn_scores.scatter_add_(2, src_tokens_expanded, calc_src_attn_scores)
        explicit_copy_probs = (
            prob * optional_fixed_logits + (1 - prob) * vocab_attn_scores
        )
        explicit_copy_log_probs = (explicit_copy_probs + 1e-7).log()
        # full support occurs if not self.fixed_vocab otherwise not.
        # Taking log(0) = -inf which should not be an issue as long as the loss
        # does not touch it, which it shoudln't. If loss is nan then it's not a
        # straightforward copy task
        return explicit_copy_log_probs, extra

    def get_normalized_probs(self, net_output, log_probs, sample):
        # the output is already a softmax probability
        if log_probs:
            return net_output[0]
        else:
            raise ValueError("PointerGeneratorDecoder cannot return logits.")