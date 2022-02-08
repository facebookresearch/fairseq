#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import math
import re
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch import device as Device

from fairseq.models import FairseqEncoder
from fairseq.models.speech_to_text.utils import (
    NoOp,
    attention_suppression,
    layer_norm_backward_hook,
    lengths_to_padding_mask,
    segments_to_sequence,
)

try:
    import torch.ao.quantization as quantization
    from torch.ao.quantization.qconfig import (
        default_dynamic_qconfig,
        per_channel_dynamic_qconfig,
    )
except ImportError:
    import torch.quantization as quantization
    from torch.quantization.qconfig import (
        default_dynamic_qconfig,
        per_channel_dynamic_qconfig,
    )


class RelativePositionEmbedding(nn.Module):
    """
    Implementation according to https://arxiv.org/abs/1803.02155
    """

    def __init__(self, head_dim, max_position, norm_init=True):
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.embeddings = nn.Parameter(torch.Tensor(max_position * 2 + 1, head_dim))
        if norm_init:
            nn.init.xavier_normal_(self.embeddings)
        else:
            nn.init.xavier_uniform_(self.embeddings)

    def forward(self, input: Tensor):
        output = nn.functional.embedding(input.long(), self.embeddings)
        return output


class Fp32LayerNorm(nn.Module):
    def __init__(
        self,
        input_dim,
        clamp_grad=True,
        max_grad_value=256,
        eps=1e-5,
        elementwise_affine=True,
    ):
        super().__init__()
        self.torch_module = torch.nn.LayerNorm(
            input_dim, eps=eps, elementwise_affine=elementwise_affine
        )
        if clamp_grad:
            hook = partial(layer_norm_backward_hook, clamp_value=max_grad_value)
            self.torch_module.register_backward_hook(hook)

    def forward(self, input):
        output = torch.nn.functional.layer_norm(
            input.float(),
            self.torch_module.normalized_shape,
            self.torch_module.weight.float()
            if self.torch_module.weight is not None
            else None,
            self.torch_module.bias.float()
            if self.torch_module.bias is not None
            else None,
            self.torch_module.eps,
        ).type_as(input)
        return output


# ------------------------------------------------------------------------------
#   PositionwiseFF
# ------------------------------------------------------------------------------


class PositionwiseFF(nn.Module):
    """
    FFN layer in transformer.

    Args:
        input_dim: input embedding dimension
        ffn_dim: FFN layer inner dimension
        dropout_on_fc1: dropout for first linear layer
        dropout_on_fc2: dropout fr second linear layer
        activation_fn: activation function used after first linear layer. \
                Only relu or gelu is supported.

    """

    def __init__(
        self, input_dim, ffn_dim, dropout_on_fc1, dropout_on_fc2, activation_fn
    ):
        super(PositionwiseFF, self).__init__()

        self.input_dim = input_dim
        self.ffn_dim = ffn_dim
        if activation_fn == "relu":
            ac = nn.ReLU()
        elif activation_fn == "gelu":
            ac = nn.GELU()
        else:
            raise ValueError("Unsupported activation_fn = ({})".format(activation_fn))

        # fc1 -> ac -> dropout -> fc2 -> dropout
        self.module = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            ac,
            nn.Dropout(dropout_on_fc1),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout_on_fc2),
        )

        self.layer_norm = Fp32LayerNorm(input_dim)

    def forward(self, input):
        module_out = self.module(self.layer_norm(input))
        output = module_out + input

        return output

    def quantize_(self, params=None):
        if params and "per_channel" in params and params["per_channel"]:
            qconfig = per_channel_dynamic_qconfig
        else:
            qconfig = default_dynamic_qconfig
        quantization.quantize_dynamic(
            self, {torch.nn.Linear: qconfig}, dtype=torch.qint8, inplace=True
        )
        return self


# ------------------------------------------------------------------------------
#   SummarizationLayer
# ------------------------------------------------------------------------------


class SummarizationLayer(nn.Module):
    def __init__(self, method, segment_size, embedding_dim):
        super(SummarizationLayer, self).__init__()
        self.segment_size = segment_size
        self.embedding_dim = embedding_dim
        nonlin_match = re.match(r"nonlinear\((?P<act>[a-z]+),(?P<dim>[0-9]+)\)", method)
        self.method = method
        if method == "mean":
            self.module = nn.AvgPool1d(
                kernel_size=segment_size,
                stride=segment_size,
                ceil_mode=True,
            )
        elif method == "max":
            self.module = nn.MaxPool1d(
                kernel_size=segment_size,
                stride=segment_size,
                ceil_mode=True,
            )
        elif method == "linear":
            self.module = nn.Linear(segment_size, 1)
        elif nonlin_match:
            nonlin_args = nonlin_match.groupdict()
            act_type = nonlin_args["act"]
            hid_dim = int(nonlin_args["dim"])
            if act_type == "relu":
                act = nn.ReLU()
            elif act_type == "gelu":
                act = nn.GELU()
            else:
                raise ValueError("Unsupported activation_fn = ({})".format(act_type))
            self.module = nn.Sequential(
                nn.Linear(segment_size, hid_dim),
                act,
                nn.Linear(hid_dim, 1),
            )
        else:
            raise ValueError("Unsupported summarization method = ({})".format(method))

    def forward(self, input):
        # T, B, D -> B, D, T
        input = input.permute(1, 2, 0)

        if self.method == "mean" or self.method == "max":
            output = self.module(input)
            output = output.permute(2, 0, 1)
            return output

        full_seg_length = input.size(2) // self.segment_size * self.segment_size
        if full_seg_length > 0:
            # at least one seg is full
            B = input.size(0)
            D = input.size(1)
            input_todo = (
                input[:, :, :full_seg_length]
                .contiguous()
                .view(B, -1, self.segment_size)
            )
            output = self.module(input_todo)
            output = output.view(B, D, -1)
        else:
            output = input.new_zeros(input.size(0), input.size(1), 0)
        left = input.size(2) - full_seg_length
        if left > 0:
            # when last seg is not full, use zeros as last memory placeholder
            zeros = input.new_zeros(input.size(0), input.size(1), 1)
            output = torch.cat([output, zeros], dim=2)
        output = output.permute(2, 0, 1)
        return output


# ------------------------------------------------------------------------------
#   NoSegAugmentedMemoryMultiheadAttentionBmm
# ------------------------------------------------------------------------------


class NoSegAugmentedMemoryMultiheadAttentionBmm(nn.Module):
    """
    Whole utterance augmented memory multihead attention using BMM.

    Different with previous augmented memory multihead attention where
    the utterance is chunked into segments. Here we use attention mask
    achieve so. The input embedding [right_context, utterance, summary]
    is a concatenation of right context, utterance and summary.

    Right context block is the concatenation of all the right context for
    each segments. [right_context_0, right_context_1, ..., right_context_n]
    For example, if we have utterance = [v0, v1, v2, ...., v20]. segment
    size 8, right_context size 4. Then the right context blocks =
    [v8, v9, v10, v11, v16, v17, v18, v19, 0, 0, 0, 0], where v8, v9, v10,
    and v11 are the right context for first segment. v16, v17, v18 and v19
    are the right context for second segment. 0, 0, 0 and 0 are right context
    for the last segment.

    utterance is corresponding to input embedding sequence

    summary is concatenation of average of each segments. [summary_0,
    summary_1, ..., ].

    In augmented memory multihead attention, the query is [right_context,
    utterance, summary], key is [memory, right_context, utterance]. Different
    with AugmentedMemoryMultiheadAttentionBmm, memory here is passed from
    previous attention layer. For the first attention layer, memory is average
    of each segment.

    Memory is a concatenation of memory from each segments in previous attention
    layer. For example, current layer is i, then memory is [m_0, m_1, ..., m_n].
    Each m_k is the output from seg_k in layer i-1.

    args:
        input_dim: input embedding dimension
        num_heads: number of heads in multihead self-attention
        dropout: attention dropout
        std_scale: if std_scale is not None. The weak attention suppression is
            turned on. For std_scale = 0.5, all the attention smaller than
            mean + 0.5 * std will be suppressed.
        scaled_init: whether to use scaled init for linear weight
        tanh_on_mem: whether to use tanh on memory output
        use_mem: whether to use memory or not. When max_memory_size is 0, then
            we don't have memory anymore.
        layer_index: current self-attention layer index that is used in depth
            initialization
        max_relative_position: max relative position used in relative position
            embedding
        rpe_old_option: To be compatible with previous model. The previous model
            was trained with attention += attention + rpe. The correct equation
            should be attention = attention + rpe

    """

    def __init__(
        self,
        input_dim,
        num_heads,
        dropout=0.0,
        std_scale=None,
        scaled_init=False,
        tanh_on_mem=False,
        use_mem=True,
        mini_batches=False,
        negative_inf="-inf",
        layer_index=-1,
        max_relative_position=0,
        rpe_old_option=True,
    ):
        if input_dim % num_heads:
            raise ValueError(
                "input_dim ({}) must be divisible by num_heads ({})".format(
                    input_dim, num_heads
                )
            )

        super().__init__()

        embed_dim = input_dim
        self.e2h_kv = torch.nn.Linear(input_dim, 2 * input_dim, bias=True)
        self.e2h_q = torch.nn.Linear(input_dim, input_dim, bias=True)
        self.rpe_old_option = rpe_old_option
        if max_relative_position > 0:
            self.use_rpe = True
            self.rpe_k = RelativePositionEmbedding(
                head_dim=input_dim // num_heads,
                max_position=max_relative_position,
            )
            self.rpe_v = RelativePositionEmbedding(
                head_dim=input_dim // num_heads,
                max_position=max_relative_position,
            )
        else:
            self.use_rpe = False
            self.rpe_k = None
            self.rpe_v = None
        if scaled_init:
            if layer_index == -1:
                gain = 1.0 / math.sqrt(2)
            else:
                # https://arxiv.org/abs/2005.09684 depthwise initialization
                # stablize the training greatly. Use depthwise initialization to
                # replace incremental loss.
                gain = 1.0 / math.sqrt(layer_index + 1)
            torch.nn.init.xavier_uniform_(self.e2h_kv.weight, gain=gain)
            torch.nn.init.xavier_uniform_(self.e2h_q.weight, gain=gain)

        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.std_scale = std_scale
        self.use_mem = use_mem
        self.mini_batches = mini_batches
        self.negative_inf = negative_inf

        if tanh_on_mem:
            self.squash_mem = torch.tanh
            self.nonlinear_squash_mem = True
        else:
            self.squash_mem = NoOp()
            self.nonlinear_squash_mem = False

    def prepare_qkv(
        self,
        input: Tensor,
        mems: Tensor,
        lengths: Tensor,
        summary_length: int,
        lc_length: int,
    ):
        # T: right_context length + utterance_length  + summary_length
        T, B, D = input.shape
        mem_length = mems.size(0)
        utterance_length = torch.max(lengths)

        right_context_blocks_length = T - utterance_length - summary_length
        rc_block = input[:right_context_blocks_length, :, :]
        utterance_block = input[right_context_blocks_length : T - summary_length, :, :]

        if B == 1:
            padding_mask = None
        else:
            klengths = lengths + mem_length + right_context_blocks_length + lc_length
            padding_mask = lengths_to_padding_mask(lengths=klengths)

        mem_rc_input = torch.cat([mems, rc_block, utterance_block], dim=0)

        # In training lc_length = 0
        key_length = mem_rc_input.size(0) + lc_length
        rc_input_sum = input
        q = self.e2h_q(rc_input_sum)
        kv = self.e2h_kv(mem_rc_input)
        k, v = kv.chunk(chunks=2, dim=2)
        result_qkv = (q, k, v)
        input_shape = (T, B, D)
        result_lengths_info = (
            mem_length,
            utterance_length,
            right_context_blocks_length,
            key_length,
        )
        if padding_mask is not None:
            assert padding_mask.size(0) == B
            assert padding_mask.size(1) == key_length

        return result_qkv, input_shape, result_lengths_info, padding_mask

    def prepare_attention_weights(
        self,
        q: Tensor,
        new_k: Tensor,
        new_v: Tensor,
        input_shape: Tuple[int, int, int],
        rpe: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        T, B, D = input_shape
        q = (
            q.contiguous().view(-1, B * self.num_heads, self.head_dim).transpose(0, 1)
            * self.scaling
        )

        k = (
            new_k.contiguous()
            .view(-1, B * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        v = (
            new_v.contiguous()
            .view(-1, B * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attention_weights = torch.bmm(q, k.transpose(1, 2))
        if self.use_rpe and rpe is not None and self.rpe_v is not None:
            r_k = self.rpe_k(rpe)
            # [q, B*h, d] * [q, k, d] -> [B*h, q, k]
            attention_weights_rpe = torch.matmul(
                q.transpose(0, 1), r_k.transpose(1, 2)
            ).transpose(0, 1)
            attention_weights = attention_weights + attention_weights_rpe
        attention_weights_float = attention_weights.float()

        return attention_weights, attention_weights_float, v

    def prepare_attention_output(
        self,
        attention_weights: Tensor,
        attention_weights_float: Tensor,
        v: Tensor,
        input_shape: Tuple[int, int, int],
        key_length: int,
        padding_mask: Optional[Tensor],
        rpe: Optional[Tensor],
    ) -> Tensor:
        T, B, D = input_shape
        if padding_mask is not None:
            attention_weights_float = attention_weights_float.view(
                B, self.num_heads, T, key_length
            )
            attention_weights_float = attention_weights_float.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attention_weights_float = attention_weights_float.view(
                B * self.num_heads, T, key_length
            )

        if self.std_scale is not None:
            attention_weights_float = attention_suppression(
                attention_weights_float, self.std_scale
            )

        attention_weights_float = torch.nn.functional.softmax(
            attention_weights_float, dim=-1
        )
        attention_weights = attention_weights_float.type_as(attention_weights)

        attention_probs = torch.nn.functional.dropout(
            attention_weights, p=self.dropout, training=self.training
        )

        # [T, key_length, B, n_head]+ [key_length, B, n_head, d_head]
        # -> [T, B, n_head, d_head]
        attention = torch.bmm(attention_probs, v)
        if self.use_rpe and rpe is not None and self.rpe_v is not None:
            r_v = self.rpe_v(rpe)
            attention_rpe = torch.matmul(
                attention_probs.transpose(0, 1), r_v
            ).transpose(0, 1)

            if self.rpe_old_option:
                attention += attention + attention_rpe
            else:
                attention = attention + attention_rpe

        assert list(attention.shape) == [B * self.num_heads, T, self.head_dim]

        attention = attention.transpose(0, 1).contiguous().view(T, B, self.embed_dim)

        rc_output_memory = self.out_proj(attention)
        return rc_output_memory

    @torch.jit.unused
    def forward(
        self,
        input: Tensor,
        lengths: Tensor,
        mems: Tensor,
        attention_mask: Tensor,
        pre_mems: Optional[Tensor] = None,
        left_context_key: Optional[Tensor] = None,
        left_context_val: Optional[Tensor] = None,
        rpe: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        forward function for NoSegAugmentedMemoryMultiheadAttentionBmm in training.

        args:
            input: formed in the following way
                [right_context_0, right_contex_1, ..., seg_0, seg_1,
                ..., summary_0, summary_1,..]
            lengths: the length of query which is [seg_0, seg_1, ....]
            mems: [mem_0, mem_1, ...].
            attention_mask: attention mask for query = [right_context, query, summary]
                key = [mem, right_context, query]. This is only used for traing.

        """
        if self.use_mem:
            mem_length = mems.size(0)
            summary_length = mem_length + 1
            if pre_mems is not None:
                mems = torch.cat([pre_mems, mems], dim=0)
        else:
            mem_length = 0
            summary_length = 0

        # In training, lc_length = 0
        if left_context_key is not None:
            lc_length = left_context_key.size(0)
        else:
            lc_length = 0
        results = self.prepare_qkv(
            input=input,
            mems=mems,
            lengths=lengths,
            summary_length=summary_length,
            lc_length=lc_length,
        )
        result_qkv, input_shape, result_lengths_info, padding_mask = results
        q, k, v = result_qkv
        (
            mem_length,
            utterance_length,
            right_context_blocks_length,
            key_length,
        ) = result_lengths_info

        if left_context_key is not None:
            # add the cache key and value
            new_k = torch.cat(
                [
                    k[: mem_length + right_context_blocks_length, :, :],
                    left_context_key,
                    k[-utterance_length:, :, :],
                ],
                dim=0,
            )
            new_v = torch.cat(
                [
                    v[: mem_length + right_context_blocks_length, :, :],
                    left_context_val,
                    v[-utterance_length:, :, :],
                ],
                dim=0,
            )
            next_k = new_k[mem_length + right_context_blocks_length :, :, :]
            next_v = new_v[mem_length + right_context_blocks_length :, :, :]
        else:
            new_k = k
            new_v = v
            next_k = None
            next_v = None

        attention_weights, attention_weights_float, v = self.prepare_attention_weights(
            q=q,
            new_k=new_k,
            new_v=new_v,
            input_shape=input_shape,
            rpe=rpe,
        )

        # mask attention
        attention_mask = attention_mask.unsqueeze(0)
        attention_weights_float = attention_weights_float.masked_fill(
            attention_mask, float(self.negative_inf)
        )

        rc_output_memory = self.prepare_attention_output(
            attention_weights=attention_weights,
            attention_weights_float=attention_weights_float,
            v=v,
            input_shape=input_shape,
            key_length=key_length,
            padding_mask=padding_mask,
            rpe=rpe,
        )

        if self.use_mem:
            # next_m length equals to summary length - 1
            # last memory is ignored
            if self.mini_batches:
                next_m = rc_output_memory[-summary_length:]
            else:
                next_m = rc_output_memory[-summary_length:-1]

            next_m = self.squash_mem(next_m)
            # rc and output
            rc_output = rc_output_memory[:-summary_length]
            if not self.nonlinear_squash_mem:
                next_m = torch.clamp(next_m, min=-10, max=10)
        else:
            next_m = mems
            rc_output = rc_output_memory

        return rc_output, next_m, next_k, next_v

    @torch.jit.export
    def forward_jit(
        self,
        input: Tensor,
        lengths: Tensor,
        mems: Tensor,
        left_context_key: Tensor,
        left_context_val: Tensor,
        rpe: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        forward function for NoSegAugmentedMemoryMultiheadAttentionBmm in decoding.

        args:
            input: formed in the following way
                [right_context_0, right_contex_1, ..., seg_0, seg_1,
                ..., summary_0, summary_1,..]
            lengths: the length of query which is [seg_0, seg_1, ....]
            mems: [mem_0, mem_1, ...].
            left_context_key: left_context for key part. This is only used for online
                decoding. In training, this is empty tensor
            left_context_val: left_context for value part. This is only used for online
                decoding. In training, this is empty tensor

        """
        lc_length = left_context_key.size(0)

        # In decoding, summary_length = 1 or 0
        if self.use_mem:
            summary_length = 1
        else:
            summary_length = 0

        results = self.prepare_qkv(
            input=input,
            mems=mems,
            lengths=lengths,
            summary_length=summary_length,
            lc_length=lc_length,
        )
        result_qkv, input_shape, result_lengths_info, padding_mask = results
        q, k, v = result_qkv
        (
            mem_length,
            utterance_length,
            right_context_blocks_length,
            key_length,
        ) = result_lengths_info

        # add the cache key and value
        new_k = torch.cat(
            [
                k[: mem_length + right_context_blocks_length, :, :],
                left_context_key,
                k[-utterance_length:, :, :],
            ],
            dim=0,
        )
        new_v = torch.cat(
            [
                v[: mem_length + right_context_blocks_length, :, :],
                left_context_val,
                v[-utterance_length:, :, :],
            ],
            dim=0,
        )
        next_k = new_k[mem_length + right_context_blocks_length :, :, :]
        next_v = new_v[mem_length + right_context_blocks_length :, :, :]

        attention_weights, attention_weights_float, v = self.prepare_attention_weights(
            q=q,
            new_k=new_k,
            new_v=new_v,
            input_shape=input_shape,
            rpe=rpe,
        )
        # In online decoding, we don't have attention mask. But we still need
        # to disable the attention from summary query to memory
        attention_weights_float[:, -1, :mem_length] = float(self.negative_inf)
        rc_output_memory = self.prepare_attention_output(
            attention_weights=attention_weights,
            attention_weights_float=attention_weights_float,
            v=v,
            input_shape=input_shape,
            key_length=key_length,
            padding_mask=padding_mask,
            rpe=rpe,
        )

        # In decoding, summary length is 1
        if self.use_mem:
            next_m = rc_output_memory[-1:]
            next_m = self.squash_mem(next_m)
            # rc and output
            rc_output = rc_output_memory[:-1]
            if not self.nonlinear_squash_mem:
                next_m = torch.clamp(next_m, min=-10, max=10)
        else:
            rc_output = rc_output_memory
            # empty tensor as input mems
            next_m = mems

        return rc_output, next_m, next_k, next_v

    def quantize_(self, params=None):
        if params and "per_channel" in params and params["per_channel"]:
            qconfig = per_channel_dynamic_qconfig
        else:
            qconfig = default_dynamic_qconfig
        quantization.quantize_dynamic(
            self, {torch.nn.Linear: qconfig}, dtype=torch.qint8, inplace=True
        )
        return self


class NoSegAugmentedMemoryTransformer(nn.Module):
    """
    Whole utterance augmented memory transformer.

    This is not pyspeech nn layer. It is used as a module in a master layer where
    multiple transformers is used.
    """

    def __init__(
        self,
        input_dim,
        num_heads,
        ffn_dim,
        dropout_in_attn=0.0,
        dropout_on_attn=None,
        dropout_on_fc1=None,
        dropout_on_fc2=None,
        activation_fn="relu",
        tanh_on_mem=False,
        std_scale=None,
        scaled_init=False,
        segment_size=128,
        use_mem=True,
        mini_batches=False,
        negative_inf="-inf",
        layer_index=-1,
        summarization_method="mean",
        max_relative_position=0,
        rpe_old_option=True,
    ):
        super(NoSegAugmentedMemoryTransformer, self).__init__()

        self.attention = NoSegAugmentedMemoryMultiheadAttentionBmm(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout_in_attn,
            scaled_init=scaled_init,
            tanh_on_mem=tanh_on_mem,
            std_scale=std_scale,
            use_mem=use_mem,
            mini_batches=mini_batches,
            negative_inf=negative_inf,
            layer_index=layer_index,
            max_relative_position=max_relative_position,
        )
        self.dropout = nn.Dropout(dropout_on_attn)
        self.pos_ff = PositionwiseFF(
            input_dim=input_dim,
            ffn_dim=ffn_dim,
            dropout_on_fc1=dropout_on_fc1,
            dropout_on_fc2=dropout_on_fc2,
            activation_fn=activation_fn,
        )
        self.layer_norm_pre = Fp32LayerNorm(input_dim)
        self.layer_norm = Fp32LayerNorm(input_dim)
        self.segment_size = segment_size
        self.use_mem = use_mem

        self.memory_op = SummarizationLayer(
            summarization_method, segment_size, input_dim
        )

    def set_mini_batches(self, mini_batches):
        self.attention.mini_batches = mini_batches

    def gen_summary_queries(self, input):
        sum_input = self.memory_op(input)
        return sum_input

    def pre_attention_ops(self, input, right_context_blocks):
        rc_length = right_context_blocks.size(0)
        input_length = input.size(0)

        rc_and_input = torch.cat([right_context_blocks, input], dim=0)
        residual_input = rc_and_input
        rc_and_input = self.layer_norm_pre(rc_and_input)

        query_input = rc_and_input[-input_length:, :, :]
        return rc_length, input_length, residual_input, query_input, rc_and_input

    def after_attention_ops(self, attention_output, residual_input):
        output = self.dropout(attention_output)
        output = output + residual_input
        output = self.pos_ff(output)
        output = self.layer_norm(output)
        return output

    @torch.jit.export
    def forward_jit(
        self,
        input: Tensor,
        lengths: Tensor,
        mems: Tensor,
        left_context_key: Tensor,
        left_context_val: Tensor,
        right_context_blocks: Tensor,
        rpe: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        results = self.pre_attention_ops(input, right_context_blocks)
        rc_length, input_length, residual_input, query_input, rc_and_input = results

        # In online decoding, the summary query size is always 1 or 0
        if self.use_mem:
            summary_query = self.gen_summary_queries(query_input)
            summary_query = summary_query[0:1, :, :]
            rc_qu_su = torch.cat([rc_and_input, summary_query], dim=0)
        else:
            rc_qu_su = rc_and_input

        rc_output, next_m, next_k, next_v = self.attention.forward_jit(
            input=rc_qu_su,
            lengths=lengths,
            mems=mems,
            left_context_key=left_context_key,
            left_context_val=left_context_val,
            rpe=rpe,
        )
        rc_output = self.after_attention_ops(rc_output, residual_input)
        results = (
            rc_output[-input_length:, :, :],
            next_m,
            rc_output[0:rc_length, :, :],
            next_k,
            next_v,
        )
        return results

    @torch.jit.unused
    def forward(
        self,
        input,
        lengths,
        mems,
        right_context_blocks,
        attention_mask,
        pre_mems,
        left_context_key,
        left_context_val,
        rpe,
    ):

        results = self.pre_attention_ops(input, right_context_blocks)
        rc_length, input_length, residual_input, query_input, rc_and_input = results
        if self.use_mem:
            summary_query = self.gen_summary_queries(query_input)
            rc_qu_su = torch.cat([rc_and_input, summary_query], dim=0)
        else:
            rc_qu_su = rc_and_input

        rc_output, next_m, next_k, next_v = self.attention(
            input=rc_qu_su,
            lengths=lengths,
            mems=mems,
            attention_mask=attention_mask,
            pre_mems=pre_mems,
            left_context_key=left_context_key,
            left_context_val=left_context_val,
            rpe=rpe,
        )

        # [TODO] Note memory did not go through pos_ff. What happen if we pass
        # memory through the pos_ff as well?
        rc_output = self.after_attention_ops(rc_output, residual_input)
        results = (
            rc_output[-input_length:, :, :],
            next_m,
            rc_output[0:rc_length, :, :],
            next_k,
            next_v,
        )

        return results


class NoSegAugmentedMemoryTransformerEncoderLayer(FairseqEncoder):
    """
    Whole utterance augmented memory transformer encoder layer. This is a master layer
    where we can define multiple augmented memory transformers. There are two reasons
    to setup the master layer.
    1. We only need to define once about the attention mask. All the layers in the master
       layer share the same mask.
    2. pyspeech nn layer has special input and output format. Defining one master layer is
       easier to passing memory between different layes inside the master layer

    args:
        input_dim: input embedding dimension
        num_heads: number of heads in multihead self-attention
        ffn_dim: ffn dimension in FFN layer
        num_layers: number of augmented memory transformer layers
        dropout_in_attn: dropout used in multi-head self-attention
        dropout_on_attn: dropout used for output from te multihead self-attention
        dropout_on_fc1: dropout used in FFN layer for the first linear layer
        dropout_on_fc2: dropout used in FFN layer for the second linear layer
        segment_size: segment size for each segment
        context_config: (left_context_size, right_context_size) defines the surround context size
            for each segment
        max_memory_size: maximum memory size used for each segment
        scaled_init: whether use scaled init for weight initialization in attention layer
        std_scale: if std_scale is not None. The weak attention suppression is
            turned on. For std_scale = 0.5, all the attention smaller than
            mean + 0.5 * std will be suppressed.
        activation_fn: activation function used in FFN layer. [ReLU, GELU] supported
        tanh_on_mem: whether use tanh on memory
        mini_batches: use mini-btach training
        negative_inf: the negative infinity value used in attention masking. default is "-inf".
            For some situation, e.g. LM. it is better to use "-1e8" to avoid nan issue.
        summarization_method: method to generate segment summrization embedding
        max_relative_position: max relatie position for relative position embedding
        rpe_old_option: To be compatible with previous model. The previous model
            was trained with attention += attention + rpe. The correct equation
            should be attention = attention + rpe
        [TODO]: remove the rpe_old_option by the end of 2021 Q1.

    """

    def __init__(
        self,
        input_dim,
        num_heads,
        ffn_dim,
        num_layers=1,
        dropout_in_attn=0.0,
        dropout_on_attn=0.0,
        dropout_on_fc1=0.0,
        dropout_on_fc2=0.0,
        segment_size=128,
        context_config=(0, 0),
        max_memory_size=0,
        scaled_init=True,
        std_scale=None,
        activation_fn="relu",
        tanh_on_mem=False,
        mini_batches=False,
        negative_inf="-inf",
        deep_init=True,
        summarization_method="mean",
        max_relative_position=0,
        rpe_old_option=True,
    ):
        super().__init__(None)
        if input_dim % num_heads:
            raise ValueError(
                "input_dim ({}) must be divisible by num_heads ({})".format(
                    input_dim, num_heads
                )
            )

        # we used to support growing memory size. However, it will cause
        # cross stream batching failure. Now we need to have exact max memory size
        if max_memory_size < 0:
            raise ValueError("max_memory_size must be >= 0")

        # Only assign right_context. In decoding, left context will be cached.
        # No need to let the online decoder to re-assign the left context
        self.left_context, self.right_context = context_config
        self.segment_size = segment_size
        self.memory_dim = input_dim
        self.max_memory_size = max_memory_size
        self.mini_batches = mini_batches
        if self.max_memory_size != 0:
            self.use_mem = True
        else:
            self.use_mem = False

        self.memory_op = SummarizationLayer(
            summarization_method, segment_size, input_dim
        )

        self.layers = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.max_relative_position = max_relative_position
        if self.max_relative_position > 0:
            self.use_rpe = True
        else:
            self.use_rpe = False
        for i in range(self.num_layers):
            if deep_init:
                layer_index = i
            else:
                layer_index = -1

            self.layers.append(
                NoSegAugmentedMemoryTransformer(
                    num_heads=num_heads,
                    input_dim=input_dim,
                    ffn_dim=ffn_dim,
                    dropout_in_attn=dropout_in_attn,
                    dropout_on_attn=dropout_on_attn,
                    dropout_on_fc1=dropout_on_fc1,
                    dropout_on_fc2=dropout_on_fc2,
                    segment_size=segment_size,
                    std_scale=std_scale,
                    activation_fn=activation_fn,
                    tanh_on_mem=tanh_on_mem,
                    scaled_init=scaled_init,
                    use_mem=self.use_mem,
                    mini_batches=mini_batches,
                    negative_inf=negative_inf,
                    layer_index=layer_index,
                    summarization_method=summarization_method,
                    max_relative_position=max_relative_position,
                    rpe_old_option=rpe_old_option,
                )
            )

    def set_mini_batches(self, mini_batches):
        # handy function only used for unit test
        self.mini_batches = mini_batches
        for layer in self.layers:
            layer.set_mini_batches(mini_batches)

    def _get_relative_position(
        self,
        input: Tensor,
        max_relative_position: int,
        left_context_length: int,
        past_length: int,
        is_decoding: bool,
    ):
        # For training, we copy the right context to the start of the utterance
        # First dimension in distance is corresponding to query.
        # [right context, utterance, summary vector]
        # Second dimension in distance is corresponding to key.
        # [Memory bank, right context, utterance]
        # For summary vector in query part, the distance with
        # all other position is 2*max_position. For memory bank in key,
        # the distance with all other positions is 0.

        T, B, D = input.shape
        num_segs = math.ceil((T - self.right_context) / self.segment_size)

        # utterance
        u_st = past_length * self.segment_size
        u_ed = u_st + T
        utterance_ranges = torch.arange(u_st, u_ed - self.right_context)

        # left context. Only in minibatch or decoding
        left_context_ranges = torch.arange(u_st - left_context_length, u_st)

        # Right context block
        # right context + utterance
        right_context_blocks = []
        for i in range(0, num_segs - 1):
            st = (i + 1) * self.segment_size + u_st
            ed = st + self.right_context
            assert ed < u_ed
            temp = torch.arange(st, ed)
            right_context_blocks.append(temp)
        right_context_blocks.append(torch.arange(u_ed - self.right_context, u_ed))
        right_context_ranges = torch.cat(right_context_blocks)

        if self.use_mem:
            # Memory bank
            # The position for memory -n, .., -1
            if is_decoding:
                memory_size = min(past_length, self.max_memory_size)
            else:
                memory_size = num_segs + past_length - 1
            memory_bank_ranges = torch.arange(
                -max_relative_position - 1, -max_relative_position - 1 - memory_size, -1
            )

            # summary vector
            # The position for summary vector as the T+max_relative_position+1.
            # After the clamping, the relative position is max_relative_position
            summary_pos_st = u_ed + max_relative_position + 1
            summary_vector_ranges = torch.arange(
                summary_pos_st, summary_pos_st + num_segs
            )

            key_ranges = torch.cat(
                [
                    memory_bank_ranges,
                    right_context_ranges,
                    left_context_ranges,
                    utterance_ranges,
                ]
            )

            query_ranges = torch.cat(
                [right_context_ranges, utterance_ranges, summary_vector_ranges]
            )
        else:
            key_ranges = torch.cat(
                [right_context_ranges, left_context_ranges, utterance_ranges]
            )

            query_ranges = torch.cat([right_context_ranges, utterance_ranges])

        distance = key_ranges[None, :] - query_ranges[:, None]
        distance_clamp = (
            torch.clamp(distance, -max_relative_position, max_relative_position)
            + max_relative_position
        )
        distance_clamp = distance_clamp.to(input.device).long().detach()
        return distance_clamp

    def _get_attention_mask(self, input, past_length=0, left_context_cache=0):
        # attention mask for each query contains three parts:
        # 1. memory part
        # 2. left_context + segment
        # 3. right_context_block
        # so for each segment and its correspoinding right context block,
        # the attention matrix is formed by 9 parts:
        # [0, m, 0, 0, right_context, 0, 0, seg, 0]
        # [before memory, memory, after memory, before right context, right_context,
        #  after right context, before seg, seg, after seg]
        #
        # Query is formed in the way as [right_context_blocks, utterance, summary]
        #
        # Note: put m and right_context before segment is convenient
        # for padding_mask operation.
        # Key lengths = m_length + right_context_block_length + lengths
        utterance_length, batch_size, _ = input.shape
        summary_length = math.ceil(utterance_length / self.segment_size)
        num_segs = summary_length
        rc_length = self.right_context * num_segs
        rc = self.right_context
        lc = self.left_context

        # using mini-batches, there is left context cache available for current
        # sequence.
        lcc = left_context_cache

        # max_memory_size is 0 then we don't have memory and summary
        # past_length is the memory carry from previous sequence
        if self.use_mem:
            mem_length = num_segs - 1 + past_length
        else:
            mem_length = 0
        rc_mask = []
        query_mask = []
        summary_mask = []
        for j in range(0, num_segs):
            ssize = min(self.segment_size, utterance_length - j * self.segment_size)

            rc_size = rc
            rc_mat = []
            q_mat = []
            s_mat = []
            m_start = max(j + past_length - self.max_memory_size, 0)

            # max_memory_size is 0, then we don't use memory
            if self.use_mem:
                # part 0: before memory
                rc_mat.append(input.new_zeros(rc_size, m_start))
                q_mat.append(input.new_zeros(ssize, m_start))
                s_mat.append(input.new_zeros(1, m_start))

                # part 1: memory
                col_1 = j + past_length - m_start
                rc_mat.append(torch.ones(rc_size, col_1, device=input.device))
                q_mat.append(torch.ones(ssize, col_1, device=input.device))
                # based on D22875746, disable summary query attention
                # on memeory is better for long form utterance
                s_mat.append(input.new_zeros(1, col_1))

                # part 2: after memory
                col_2 = mem_length - (j + past_length)
                rc_mat.append(input.new_zeros(rc_size, col_2))
                q_mat.append(input.new_zeros(ssize, col_2))
                s_mat.append(input.new_zeros(1, col_2))

            # part 3: before right context
            rc_start = j * rc
            rc_mat.append(input.new_zeros(rc_size, rc_start))
            q_mat.append(input.new_zeros(ssize, rc_start))
            s_mat.append(input.new_zeros(1, rc_start))

            # part 4: right context
            rc_end = rc_start + rc
            col_4 = rc
            rc_mat.append(torch.ones(rc_size, col_4, device=input.device))
            q_mat.append(torch.ones(ssize, col_4, device=input.device))
            s_mat.append(torch.ones(1, col_4, device=input.device))

            # part 5: after right context
            col_5 = rc_length - rc_end
            rc_mat.append(input.new_zeros(rc_size, col_5))
            q_mat.append(input.new_zeros(ssize, col_5))
            s_mat.append(input.new_zeros(1, col_5))

            # part 6: before query segment
            seg_start = max(j * self.segment_size + lcc - lc, 0)
            rc_mat.append(input.new_zeros(rc_size, seg_start))
            q_mat.append(input.new_zeros(ssize, seg_start))
            s_mat.append(input.new_zeros(1, seg_start))

            # part 7: query segment
            # note: right context is put in right context block
            # here we only need to consider about left context
            seg_end = min((j + 1) * self.segment_size + lcc, utterance_length + lcc)
            col_7 = seg_end - seg_start
            rc_mat.append(torch.ones(rc_size, col_7, device=input.device))
            q_mat.append(torch.ones(ssize, col_7, device=input.device))
            s_mat.append(torch.ones(1, col_7, device=input.device))

            # part 8: after query segment
            col_8 = utterance_length + lcc - seg_end
            rc_mat.append(input.new_zeros(rc_size, col_8))
            q_mat.append(input.new_zeros(ssize, col_8))
            s_mat.append(input.new_zeros(1, col_8))

            rc_mask.append(torch.cat(rc_mat, dim=1))
            query_mask.append(torch.cat(q_mat, dim=1))
            summary_mask.append(torch.cat(s_mat, dim=1))

        # no memory, then we don't need summary either
        if self.use_mem:
            attention_mask = (
                1
                - torch.cat(
                    [
                        torch.cat(rc_mask, dim=0),
                        torch.cat(query_mask, dim=0),
                        torch.cat(summary_mask, dim=0),
                    ],
                    dim=0,
                )
            ).to(torch.bool)
        else:
            attention_mask = (
                1
                - torch.cat(
                    [torch.cat(rc_mask, dim=0), torch.cat(query_mask, dim=0)], dim=0
                )
            ).to(torch.bool)

        return attention_mask

    @torch.jit.export
    def init_state(
        self, batch_size: int, device: Optional[Device] = None
    ) -> List[Tensor]:
        empty_memory = torch.zeros(
            self.num_layers,
            self.max_memory_size,
            batch_size,
            self.memory_dim,
            device=device,
        )
        left_context_key = torch.zeros(
            self.num_layers,
            self.left_context,
            batch_size,
            self.memory_dim,
            device=device,
        )
        left_context_val = torch.zeros(
            self.num_layers,
            self.left_context,
            batch_size,
            self.memory_dim,
            device=device,
        )
        past_length = torch.zeros(1, batch_size, dtype=torch.int32, device=device)

        return [empty_memory, left_context_key, left_context_val, past_length]

    @torch.jit.export
    def batch_state(self, states: List[List[Tensor]]) -> List[Tensor]:
        if len(states) == 0:
            return []
        batched_m = []
        batched_lc_key = []
        batched_lc_val = []
        batched_past_length = []
        for state in states:
            if len(state) == 0:
                continue
            m, lc_key, lc_val, past_length = state
            batched_m.append(m)
            batched_lc_key.append(lc_key)
            batched_lc_val.append(lc_val)
            batched_past_length.append(past_length)

        if (
            (len(batched_m) == 0)
            or (len(batched_lc_key) == 0)
            or (len(batched_lc_val) == 0)
            or (len(batched_past_length) == 0)
        ):
            return [
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
            ]

        batched_m = torch.cat(batched_m, dim=2)
        batched_lc_key = torch.cat(batched_lc_key, dim=2)
        batched_lc_val = torch.cat(batched_lc_val, dim=2)
        batched_past_length = torch.cat(batched_past_length, dim=1)
        return [batched_m, batched_lc_key, batched_lc_val, batched_past_length]

    @torch.jit.export
    def reorder_state(self, state: List[Tensor], indices: Tensor) -> List[Tensor]:
        if len(state) == 0:
            return []
        m, lc_key, lc_val, past_length = state
        indices = indices.to(device=m.device)
        reord_m = torch.index_select(m, 2, indices)
        reord_lc_key = torch.index_select(lc_key, 2, indices)
        reord_lc_val = torch.index_select(lc_val, 2, indices)
        reord_past_length = torch.index_select(past_length, 1, indices)
        return [reord_m, reord_lc_key, reord_lc_val, reord_past_length]

    @torch.jit.export
    def reset_state(self, state: List[Tensor], indices: Tensor) -> List[Tensor]:
        m, lc_key, lc_val, past_length = state
        m = m.index_fill(dim=2, index=indices, value=0.0)
        lc_key = lc_key.index_fill(dim=2, index=indices, value=0.0)
        lc_val = lc_val.index_fill(dim=2, index=indices, value=0.0)
        past_length = past_length.index_fill(dim=1, index=indices, value=0)

        return [m, lc_key, lc_val, past_length]

    @torch.jit.export
    def state_size(self) -> int:
        return 4

    @torch.jit.export
    def batch_size_in_state(
        self, state: Optional[List[Tensor]], sloppy: bool = True
    ) -> Optional[int]:
        if state is None:
            return None
        return state[0].size(2)

    def gen_summary_queries(self, input):
        sum_input = self.memory_op(input)
        return sum_input

    def _gen_right_context_padded_input(self, input):
        # This function deals with input that is already
        # padded with right context (e.g. minibatch training)
        right_context_blocks = []
        T, B, D = input.shape
        num_segs = math.ceil((T - self.right_context) / self.segment_size)
        for i in range(0, num_segs - 1):
            st = (i + 1) * self.segment_size
            ed = st + self.right_context
            assert ed < T
            temp = input[st:ed, :, :]
            right_context_blocks.append(temp)

        # last segment right context is already available
        right_context_blocks.append(input[T - self.right_context :, :, :])
        return torch.cat(right_context_blocks, dim=0)

    def _gen_segs_right_context(self, input, lengths):
        segments = []
        T, B, D = input.size()
        nT = T - self.right_context

        # assume input is right context padded
        num_segs = math.ceil(nT / self.segment_size)
        # pad zeros to the utterance to make sure each
        # segment has the same right context. For the
        for i in range(0, num_segs - 1):
            st = i * self.segment_size
            ed = min(T, st + self.segment_size + self.right_context)
            temp = input[st:ed, :, :]
            rest_lengths = torch.clamp(
                lengths - self.segment_size, min=0, max=nT - (i + 1) * self.segment_size
            )
            segments.append((temp, lengths - rest_lengths + self.right_context))
            lengths = rest_lengths

        last_seg = input[st + self.segment_size :, :, :]
        segments.append((last_seg, rest_lengths + self.right_context))

        return segments

    @torch.jit.unused
    def forward(
        self, input: Tensor, padding_masks: Tensor, state: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        # Xutai: originally the second argument is lengths.
        lengths = (~padding_masks).sum(dim=1).long()
        # mini batch training.
        if self.mini_batches:
            return self.forward_mini_batches(input, lengths, state)

        # regular full sequence training. Note, assume the right context in provided
        # in the input.
        T, B, D = input.size()
        right_context_blocks = self._gen_right_context_padded_input(input)

        # generate the relative positional embedding
        if self.use_rpe:
            rpe = self._get_relative_position(
                input=input,
                max_relative_position=self.max_relative_position,
                left_context_length=0,
                past_length=0,
                is_decoding=False,
            )
        else:
            rpe = None
        input = input[: T - self.right_context, :, :]

        attention_mask = self._get_attention_mask(input)

        # firt layer use each segment mean as memory
        # ignore the last one seg average
        if self.use_mem:
            mems = self.gen_summary_queries(input)[:-1, :, :]
        else:
            mems = torch.zeros(0, input.size(1), input.size(2), device=input.device)
            mems = mems.type_as(input)

        output = input
        all_outputs = []

        for layer in self.layers:
            output, mems, right_context_blocks, _, _ = layer(
                input=output,
                lengths=lengths,
                attention_mask=attention_mask,
                mems=mems,
                right_context_blocks=right_context_blocks,
                pre_mems=None,
                left_context_key=None,
                left_context_val=None,
                rpe=rpe,
            )
            all_outputs.append(output)
        return output, padding_masks, [], all_outputs

    def forward_jit_mini_batch_init(
        self,
        seg: Tensor,
        state: Optional[List[Tensor]] = None,
        is_decoding: bool = False,
    ):
        # Prepare state. In whole sequence training, state is ignored.
        # For minibatch training, we need to prepare state
        if state is None:
            state = self.init_state(batch_size=seg.size(1), device=seg.device)
            if seg.dtype == torch.half:
                state = [state[0].half(), state[1].half(), state[2].half(), state[3]]

        if self.use_mem:
            # note input average only on seg, not on right context
            # first layer use each segmetn mean as memory. the last
            # one segment average is used in state
            full_mems = self.gen_summary_queries(seg)
            if is_decoding:
                mems = full_mems[0:1, :, :]
                state_mems = torch.cat([state[0][0], mems], dim=0)
            else:
                mems = full_mems[:-1, :, :]
                state_mems = torch.cat([state[0][0], full_mems], dim=0)
        else:
            mems = state[0][0]
            state_mems = mems

        # track processed segment number or memory number
        # the same batch as the same bumber of past length
        past_length = state[3][0][0].item()
        past_left_context = min(past_length * self.segment_size, self.left_context)
        past_length = min(self.max_memory_size, past_length)

        return state, mems, state_mems, past_length, past_left_context

    def state_update_before(
        self, layer: int, state: List[Tensor], past_length: int, past_left_context: int
    ):
        pre_mems = state[0][layer][self.max_memory_size - past_length :, :, :]
        lc_key = state[1][layer][self.left_context - past_left_context :, :, :]
        lc_val = state[2][layer][self.left_context - past_left_context :, :, :]
        return pre_mems, lc_key, lc_val

    def state_update_after(
        self,
        layer: int,
        state: List[Tensor],
        mems: Tensor,
        next_key: Tensor,
        next_val: Tensor,
        mems_list: List[Tensor],
        lc_key_list: List[Tensor],
        lc_val_list: List[Tensor],
    ):
        # mems is used for next layer
        if layer < self.num_layers - 1:
            state_mems = torch.cat([state[0][layer + 1], mems], dim=0)
            mems_list.append(state_mems[-self.max_memory_size :, :, :])

        # when mems pass to next sequence, we need the last memory. when mems
        # use for the next layer, we can ignore the last memory
        mems = mems[:-1, :, :]

        # note state[1][i] and state[2][i] original length equals to self.left_context
        new_k = torch.cat([state[1][layer], next_key], dim=0)
        new_v = torch.cat([state[2][layer], next_val], dim=0)
        lc_key_list.append(new_k[-self.left_context :, :, :])
        lc_val_list.append(new_v[-self.left_context :, :, :])
        return mems_list, lc_key_list, lc_val_list, mems

    def state_update_after_loop(
        self,
        state: List[Tensor],
        mems_list: List[Tensor],
        lc_key_list: List[Tensor],
        lc_val_list: List[Tensor],
        update_length: int,
    ):
        state[0] = torch.stack(mems_list, dim=0)
        state[1] = torch.stack(lc_key_list, dim=0)
        state[2] = torch.stack(lc_val_list, dim=0)
        state[3] = state[3] + update_length
        return state

    @torch.jit.unused
    def forward_mini_batches(
        self, input: Tensor, lengths: Tensor, state: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        T, B, D = input.size()

        # input without right context
        seg = input[: T - self.right_context, :, :]

        # get right context blocks
        right_context_blocks = self._gen_right_context_padded_input(input)

        mems_list = []
        lc_key_list = []
        lc_val_list = []
        results = self.forward_jit_mini_batch_init(seg, state, False)
        state, mems, state_mems, past_length, past_left_context = results

        # relative position embedding
        if self.use_rpe:
            rpe = self._get_relative_position(
                input=input,
                max_relative_position=self.max_relative_position,
                left_context_length=past_left_context,
                past_length=past_length,
                is_decoding=False,
            )
        else:
            rpe = None

        # get attention mask based on seg (not include right context) and available
        # left context
        attention_mask = self._get_attention_mask(seg, past_length, past_left_context)
        mems_list.append(state_mems[-self.max_memory_size :, :, :])
        output = seg
        i = 0
        all_outputs = []
        for layer in self.layers:
            # In order to make cross stream batching work, mem, left context key
            # and left context value in the state should always be the same shape.
            # We use the past length to track the processed segment number. In this
            # way, we take out the essential memory, left context key and left
            # context val from the state. After finish the forward for current segment
            # we add the new memory, left context key and left context value into the
            # staate and trim out the oldest part to keep the shape consistent.
            pre_mems, lc_key, lc_val = self.state_update_before(
                i, state, past_length, past_left_context
            )

            output, mems, right_context_blocks, next_key, next_val = layer.forward(
                input=output,
                lengths=lengths,
                attention_mask=attention_mask,
                mems=mems,
                right_context_blocks=right_context_blocks,
                pre_mems=pre_mems,
                left_context_key=lc_key,
                left_context_val=lc_val,
                rpe=rpe,
            )
            all_outputs.append(output)
            mems_list, lc_key_list, lc_val_list, mems = self.state_update_after(
                layer=i,
                state=state,
                mems=mems,
                next_key=next_key,
                next_val=next_val,
                mems_list=mems_list,
                lc_key_list=lc_key_list,
                lc_val_list=lc_val_list,
            )

            i += 1

        # update state
        update_length = math.ceil((T - self.right_context) / self.segment_size)
        state = self.state_update_after_loop(
            state=state,
            mems_list=mems_list,
            lc_key_list=lc_key_list,
            lc_val_list=lc_val_list,
            update_length=update_length,
        )

        return output, lengths, state, all_outputs

    def forward_jit_test(
        self, input: Tensor, lengths: Tensor, state: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        This one simulate sequence encoder forward jit. This is for unit test purpose.
        It is not used in training or decoding. Note, extra_right_context is set in
        the model. In unit test, input = [utterance, right_context], lengths =
        [utterance_length].
        args:
            input: input utterance
            lengths: utterance input length
            state: None here. input is whole utterance
        """
        # [TODO] sequence_to_segment has bug in lengths.
        seg_src_tokens_lengths = self._gen_segs_right_context(input, lengths)

        seg_enc_tokens_lengths: List[Tuple[Tensor, Tensor]] = []
        state: Optional[List[Tensor]] = None
        for seg_src_tokens, seg_src_lengths in seg_src_tokens_lengths:
            seg_enc_tokens, seg_enc_lengths, state = self.forward_jit(
                input=seg_src_tokens, lengths=seg_src_lengths, state=state
            )
            seg_enc_tokens_lengths.append((seg_enc_tokens, seg_enc_lengths))

        enc_tokens, enc_lengths = segments_to_sequence(
            segments=seg_enc_tokens_lengths, time_axis=0
        )

        state = []  # returns trivial state

        return enc_tokens, enc_lengths, state

    @torch.jit.export
    def forward_jit(
        self, input: Tensor, lengths: Tensor, state: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Forward helper for online decoding.

        args:
            input: [seg, right_context]. We assume in online we
                always padding the right context to the preset right context size.
                For the last segment, we may have short segment size, but right
                context size is the same as other segments
            lengths: utterance input length is the utterance segment length and
                     right context size
            state: [memory, left_context_key, left_context_val]. To improve throughput,
                in addition to memory, we also cache key and value for left_context in
                multihead self-attention
        """
        # In online decoding, input = [segment, right_context]
        # Lengths = [segment_length, right_context_length]
        # so we need strip right context in output
        T, B, D = input.size()
        rc_str = T - self.right_context
        rc_end = T
        right_context_blocks = input[rc_str:rc_end, :, :]
        seg = input[:rc_str, :, :]
        lengths = torch.clamp(lengths - self.right_context, min=0)
        mems_list = []
        lc_key_list = []
        lc_val_list = []

        results = self.forward_jit_mini_batch_init(seg, state, True)
        state, mems, state_mems, past_length, past_left_context = results

        # relative position embedding
        if self.use_rpe:
            rpe = self._get_relative_position(
                input=input,
                max_relative_position=self.max_relative_position,
                left_context_length=past_left_context,
                past_length=past_length,
                is_decoding=True,
            )
        else:
            rpe = None

        # memory for first layer.
        mems_list.append(state_mems[-self.max_memory_size :, :, :])
        output = seg
        i = 0
        for layer in self.layers:
            # In order to make cross stream batching work, mem, left context key
            # and left context value in the state should always be the same shape.
            # We use the past length to track the processed segment number. In this
            # way, we take out the essential memory, left context key and left
            # context val from the state. After finish the forward for current segment
            # we add the new memory, left context key and left context value into the
            # staate and trim out the oldest part to keep the shape consistent.
            true_mems, lc_key, lc_val = self.state_update_before(
                layer=i,
                state=state,
                past_length=past_length,
                past_left_context=past_left_context,
            )

            output, mems, right_context_blocks, next_key, next_val = layer.forward_jit(
                input=output,
                lengths=lengths,
                mems=true_mems,
                right_context_blocks=right_context_blocks,
                left_context_key=lc_key,
                left_context_val=lc_val,
                rpe=rpe,
            )
            # mems is used for next layer
            mems_list, lc_key_list, lc_val_list, _ = self.state_update_after(
                layer=i,
                state=state,
                mems_list=mems_list,
                mems=mems,
                next_key=next_key,
                next_val=next_val,
                lc_key_list=lc_key_list,
                lc_val_list=lc_val_list,
            )
            i += 1

        # update state
        state = self.state_update_after_loop(
            state=state,
            mems_list=mems_list,
            lc_key_list=lc_key_list,
            lc_val_list=lc_val_list,
            update_length=1,
        )

        return output, lengths, state

    def quantize_(self, params=None):
        if params and "per_channel" in params and params["per_channel"]:
            qconfig = per_channel_dynamic_qconfig
        else:
            qconfig = default_dynamic_qconfig
        quantization.quantize_dynamic(
            self, {torch.nn.Linear: qconfig}, dtype=torch.qint8, inplace=True
        )
        return self


# ------------------------------------------------------------------------------
#   Emformer encoder for seq2seq model
#   This is a wrapper over the original emformer
# ------------------------------------------------------------------------------
def emformer_encoder(klass):
    class SpeechEncoder(klass):
        def __init__(self, args):
            super().__init__(args)
            stride = SpeechEncoder.conv_layer_stride(args)
            trf_left_context = args.segment_left_context // stride
            trf_right_context = args.segment_right_context // stride
            context_config = [trf_left_context, trf_right_context]
            self.transformer_layers = nn.ModuleList(
                [
                    NoSegAugmentedMemoryTransformerEncoderLayer(
                        input_dim=args.encoder_embed_dim,
                        num_heads=args.encoder_attention_heads,
                        ffn_dim=args.encoder_ffn_embed_dim,
                        num_layers=args.encoder_layers,
                        dropout_in_attn=args.dropout,
                        dropout_on_attn=args.dropout,
                        dropout_on_fc1=args.dropout,
                        dropout_on_fc2=args.dropout,
                        activation_fn=args.activation_fn,
                        context_config=context_config,
                        segment_size=args.segment_length,
                        max_memory_size=args.max_memory_size,
                        scaled_init=True,  # TODO: use constant for now.
                        tanh_on_mem=args.amtrf_tanh_on_mem,
                    )
                ]
            )

        def forward(self, src_tokens, src_lengths):
            encoder_out = super().forward(src_tokens, src_lengths)
            output = encoder_out["encoder_out"][0]
            encoder_padding_masks = encoder_out["encoder_padding_mask"][0]

            # This is because that in the original implementation
            # the output didn't consider the last segment as right context.
            encoder_padding_masks = encoder_padding_masks[:, : output.size(0)]

            return {
                "encoder_out": [output],
                "encoder_padding_mask": [encoder_padding_masks],
                "encoder_embedding": [],
                "encoder_states": [],
                "src_tokens": [],
                "src_lengths": [],
            }

        @staticmethod
        def conv_layer_stride(args):
            # TODO: make it configurable from the args
            return 4

    SpeechEncoder.__name__ = klass.__name__
    return SpeechEncoder
