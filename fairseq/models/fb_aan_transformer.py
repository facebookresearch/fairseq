# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerDecoder, TransformerModel
from fairseq.modules import FairseqDropout, LayerNorm, MultiheadAttention
from torch import Tensor


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("aan_transformer")
class AANTransformerModel(TransformerModel):
    """
    Implements a variant of the model described in "Accelerating Neural
    Transformer via an Average Attention Network" (Zhang et al., 2018)
    <https://arxiv.org/abs/1805.00631`_.

    Different from paper, we use a single gate for AAN gating function (mixing
    AAN and residual via sigmoid(z) and 1-sigmoid(z) rather than sigmoid(z_1)
    and sigmoid (z_2).

    Fixed configuration for FB production: No additional FFN for AAN block.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder
    """

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return AANTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


@with_incremental_state
class AverageAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

    def forward(
        self,
        value,
        mask_trick: bool = False,
        mask_future_timesteps: bool = False,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """Input shape: Time x Batch x Channel
        ` mask_trick` is to use matrix multiplication instead of cumulative sum
         to average the inputs.
         Future timesteps can be masked with the
         `mask_future_timesteps` argument. Padding elements can be excluded from
         the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
         batch x src_len, where padding elements are indicated by 1s.
        """

        assert mask_future_timesteps or incremental_state is None
        if incremental_state is None:
            return self._forward(value, mask_trick, mask_future_timesteps)
        else:
            return self._forward_incremental(
                value, mask_trick, mask_future_timesteps, incremental_state
            )

    def _forward(self, value, mask_trick: bool, mask_future_timesteps: bool):
        length, batch_size = value.size()[:2]
        if not mask_future_timesteps:
            attn = value.mean(dim=0, keepdim=True).repeat(length, 1, 1)
            attn_weights = None
        elif mask_trick:
            v = value.transpose(0, 1)
            # no TorchScript support for specifying start in arange()
            attn_weights = torch.arange(length, out=torch.zeros([0]).to(v)) + 1
            attn_weights = (
                attn_weights.reciprocal_().unsqueeze_(1).repeat(1, length).tril(0)
            )
            attn_weights = attn_weights.unsqueeze_(0).repeat(batch_size, 1, 1)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )
            attn = torch.bmm(attn_weights, v)
            attn = attn.transpose(0, 1).contiguous()
        else:
            # no TorchScript support for specifying start in arange()
            attn_weights = (
                torch.arange(length, out=torch.zeros([0]).to(value)) + 1
            ).view(length, 1, 1)
            attn = value.cumsum(0) / attn_weights
            attn_weights = None
        return attn, attn_weights

    def _forward_incremental(
        self,
        value,
        mask_trick: bool,
        mask_future_timesteps: bool,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    ):
        if mask_trick:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_vec" in saved_state:
                prev_vec = saved_state["prev_vec"]
                assert prev_vec is not None
                value = torch.cat([prev_vec, value], dim=0)
            saved_state["prev_vec"] = value
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
            attn_weights = None
            attn = value.mean(0, keepdim=True)
        else:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_sum" in saved_state:
                prev_sum = saved_state["prev_sum"]
                assert prev_sum is not None
                curr_sum = prev_sum + value

                prev_pos = saved_state["prev_pos"]
                assert prev_pos is not None
                pos = prev_pos + 1
                attn = curr_sum / pos
            else:
                curr_sum = value
                attn = value
                pos = torch.ones([1]).int()
            saved_state["prev_sum"] = curr_sum
            saved_state["prev_pos"] = pos
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
            attn_weights = None
        return attn, attn_weights

    def extra_repr(self):
        return "embed_dim={}, dropout={}".format(self.embed_dim, self.dropout)

    def reorder_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in ("prev_vec", "prev_sum"):
                if k in input_buffer:
                    input_buffer_k = input_buffer[k]
                    if input_buffer_k is not None and input_buffer_k.size(1) > 1:
                        input_buffer[k] = input_buffer_k.index_select(1, new_order)
        if incremental_state is not None:
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)


class AANTransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.avg_attn = AverageAttention(self.embed_dim, dropout=args.attention_dropout)

        # differently than original paper, we use a single gate
        self.aan_gating_fc = Linear(self.embed_dim * 2, self.embed_dim)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
            self.activation_dropout_module = FairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )

        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = self.build_fc2(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, _ = self.avg_attn(
            value=x,
            mask_trick=self.training,
            mask_future_timesteps=True,
            incremental_state=incremental_state,
        )

        # differently than original paper, we use a single gate
        gate = torch.sigmoid(self.aan_gating_fc(torch.cat([residual, x], dim=-1)))
        x = gate * x + (1 - gate) * residual

        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


class AANTransformerDecoder(TransformerDecoder):
    def build_decoder_layer(self, args, no_encoder_attn=False):
        return AANTransformerDecoderLayer(args, no_encoder_attn)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("aan_transformer", "aan_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
