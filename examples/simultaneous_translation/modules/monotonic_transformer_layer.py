# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.modules import (
    LayerNorm,
    TransformerEncoderLayer,
    TransformerDecoderLayer
)

from . import build_monotonic_attention


class TransformerMonotonicEncoderLayer(TransformerEncoderLayer):

    def forward(self, x, encoder_padding_mask):
        seq_len, _, _ = x.size()
        attn_mask = x.new_ones([seq_len, seq_len]).triu(1)
        attn_mask = attn_mask.masked_fill(attn_mask.bool(), float('-inf'))
        return super().forward(x, encoder_padding_mask, attn_mask)


class TransformerMonotonicDecoderLayer(TransformerDecoderLayer):

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__(
            args,
            no_encoder_attn=True,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn
        )
        self.encoder_attn = build_monotonic_attention(args)
        self.encoder_attn_layer_norm = LayerNorm(
            self.embed_dim,
            export=getattr(args, 'char_inputs', False)
        )

    def prune_incremental_state(self, incremental_state):
        def prune(module):
            input_buffer = module._get_input_buffer(incremental_state)
            for key in ["prev_key", "prev_value"]:
                if input_buffer[key].size(2) > 1:
                    input_buffer[key] = input_buffer[key][:, :, :-1, :]
                else:
                    input_buffer = {}
                    break
            module._set_input_buffer(incremental_state, input_buffer)
        prune(self.self_attn)

    def get_steps(self, incremental_state):
        return (
            self.encoder_attn
            ._get_monotonic_buffer(
                incremental_state
            ).get("step", 0)
        )
