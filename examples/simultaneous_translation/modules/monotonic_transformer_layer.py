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

    def incremental_forward(self, x, encoder_padding_mask, incremental_states):
        if incremental_states["steps"]["src"] == 0:
            prev_key_len = 0
        else:
            prev_key, prev_value = self.get_prev_key_value(incremental_states)
            prev_key_len = prev_key.size(2)

        seq_len = prev_key_len + x.size(0)

        attn_mask = x.new_ones([x.size(0), seq_len]).triu(1)
        attn_mask = attn_mask.masked_fill(attn_mask.bool(), -1e8)

        residual = x


        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            incremental_state=incremental_states
        )

        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x

    def get_prev_key_value(self, incremental_states):
        input_buffer = self.self_attn._get_input_buffer(incremental_states)
        return input_buffer['prev_key'], input_buffer['prev_value']

    def set_prev_key_value(self, incremental_states, key, value):
        self.self_attn._set_input_buffer(
            incremental_states,
            {'prev_key': key, 'prev_value': value}
        )


class TransformerMonotonicDecoderLayer(TransformerDecoderLayer):

    def __init__(
        self, args, no_encoder_attn=False,
        add_bias_kv=False, add_zero_attn=False
    ):
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

    def get_head_step(self, incremental_state):
        return (
            self.encoder_attn
            ._get_monotonic_buffer(incremental_state).get("head_step")
        )

    def get_head_read(self, incremental_state):
        return (
            self.encoder_attn
            ._get_monotonic_buffer(incremental_state).get("head_read")
        )

    def prune_incremental_state(self, incremental_state):
        def prune(module):
            input_buffer = module._get_input_buffer(incremental_state)
            for key in ["prev_key", "prev_value"]:
                if input_buffer[key].size(2) > 1:
                    # Remove the last key or value since there is a read action
                    input_buffer[key] = input_buffer[key][:, :, :-1, :]
                else:
                    # the first step on the target side
                    input_buffer = {}
                    break
            module._set_input_buffer(incremental_state, input_buffer)
        prune(self.self_attn)
