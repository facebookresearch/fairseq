# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
)

EncoderOut = namedtuple(
    "TransformerEncoderOut",
    [
        "encoder_out",  # T x B x C
        "encoder_padding_mask",  # B x T
        "encoder_embedding",  # B x T x C
        "encoder_states",  # List[T x B x C]
    ],
)


class TransformerEncoderEmbedding(nn.Module):
    """Encoder Embedding + Positional Embedding"""

    def __init__(self, args, embed_tokens):
        super().__init__()
        self.dropout = args.dropout
        self.max_source_positions = args.max_source_positions
        self.embed_tokens = embed_tokens
        if isinstance(embed_tokens, nn.ModuleList):
            self.padding_idx = embed_tokens[0].padding_idx
            embed_dim = sum(e.embedding_dim for e in embed_tokens)
        else:
            self.padding_idx = embed_tokens.padding_idx
            embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward(self, input):
        # embed tokens and positions
        src_tokens = input[0]
        prev_output_tokens = input[2]
        if isinstance(self.embed_tokens, nn.ModuleList):
            x_embed_list = []
            for embed_tokens_part in self.embed_tokens:
                x_embed_list.append(embed_tokens_part(src_tokens))

            embedded = torch.cat(x_embed_list, dim=-1)
        else:
            embedded = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * embedded
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        return (x, encoder_padding_mask, prev_output_tokens)


class TransformerEncoderLayerNorm(nn.Module):
    """
    Layer norm at the the end of all encoder layers if
    args.encoder_enormalize_before = True
    """

    def __init__(self, args, embed_dim):
        super().__init__()
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, input):
        x = input[0]
        encoder_padding_mask = input[1]
        prev_output_tokens = input[2]
        if self.layer_norm:
            x = self.layer_norm(x)
        # keeping track of the incremental_state is not supported yet
        return (x, encoder_padding_mask, prev_output_tokens)


class TransformerDecoderEmbedding(nn.Module):
    """Decoder Embedding + Positional Embedding"""

    def __init__(self, args, embed_tokens):
        super().__init__()
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed
        input_embed_dim = (
            sum(e.embedding_dim for e in embed_tokens)
            if isinstance(embed_tokens, nn.ModuleList)
            else embed_tokens.embedding_dim
        )
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        padding_idx = (
            embed_tokens[0].padding_idx
            if isinstance(embed_tokens, nn.ModuleList)
            else embed_tokens.padding_idx
        )
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

    def forward(self, input):
        mt_task = False
        if isinstance(input, tuple):
            if len(input) == 3:
                encoder_out = input[0]
                encoder_padding_mask = input[1]
                prev_output_tokens = input[2]
                incremental_state = None  # Hardcoding to avoid passing of None objects
                mt_task = True
            else:
                # HACK for now, need to fix (TODO sidgoyal)
                prev_output_tokens = input[0]
                # discard "src_lengths"
                encoder_out = None
                encoder_padding_mask = None
                incremental_state = None

        else:
            prev_output_tokens = input
            encoder_out = None
            encoder_padding_mask = None
            incremental_state = None

        positions = (
            self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions

        if isinstance(self.embed_tokens, nn.ModuleList):
            x_embed_list = []
            for embed_tokens_part in self.embed_tokens:
                x_embed_list.append(embed_tokens_part(prev_output_tokens))

            x = self.embed_scale * torch.cat(x_embed_list, dim=-1)
        else:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        if mt_task:
            return (x, encoder_out, encoder_padding_mask)
        return x


class TransformerDecoderOutputLayer(nn.Module):
    def __init__(self, args, embed_tokens, dictionary):
        super().__init__()
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.embed_tokens = embed_tokens
        self.output_embed_dim = args.decoder_output_dim
        embed_dim = args.decoder_embed_dim

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )
        self.adaptive_softmax = None
        if args.adaptive_softmax_cutoff is not None:
            assert not isinstance(embed_tokens, nn.ModuleList)
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_tokens = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(
                self.embed_tokens, mean=0, std=self.output_embed_dim ** -0.5
            )

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, input, apply_final_proj=True):
        if isinstance(input, tuple):
            x = input[0]
        else:
            x = input

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        if apply_final_proj:
            x = self.output_layer(x)
        return x

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                if isinstance(self.embed_tokens, nn.ModuleList):
                    output = None
                    for i, emb in enumerate(self.embed_tokens):
                        sidx = i * emb.embedding_dim
                        eidx = (i + 1) * emb.embedding_dim
                        if output is None:
                            output = F.linear(features[:, :, sidx:eidx], emb.weight)
                        else:
                            output += F.linear(features[:, :, sidx:eidx], emb.weight)

                    return output
                else:
                    return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_tokens)
        else:
            return features


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, input):
        """
        Args:
            input (Tuple):
                input[0] (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
                input[1] (ByteTensor/FloatTensor): encoder padding mask -
                    binary ByteTensor of shape `(batch, src_len)` where padding elements
                    are indicated by ``1``.
                input[2] (LongTensor): previous decoder outputs of shape
                    `(batch, tgt_len)`, for teacher forcing)
        Returns:
            output (Tuple):
                output[0] (Tensor): encoded output of shape `(batch, src_len, embed_dim)`
                output[1] (ByteTensor/FloatTensor): encoder padding mask
                output[2] (LongTensor): previous decoder outputs
        """
        x = input[0]
        encoder_padding_mask = input[1]
        prev_output_tokens = input[2]
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=encoder_padding_mask
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return (x, encoder_padding_mask, prev_output_tokens)

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
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
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
        )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
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
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, input):
        """
        Args:
            input (Tuple):
                input[0] (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
                input[1] (Tensor): encoder output of shape `(batch, src_len, embed_dim)`
                input[2] (ByteTensor/FloatTensor): encoder padding mask -
                    binary ByteTensor of shape `(batch, src_len)` where padding elements
                    are indicated by ``1``.
        Returns:
            output (Tuple):
                output[0] (Tensor): encoded output of shape `(batch, src_len, embed_dim)`
                output[1] (ByteTensor/FloatTensor): encoder padding mask
                output[2] (LongTensor): previous decoder outputs
        """
        # Note: incremental state is not yet supported
        mt_task = False
        if isinstance(input, tuple):
            x = input[0]
            encoder_out = input[1]
            encoder_padding_mask = input[2]
            incremental_state = None
            mt_task = True
        else:
            x = input
            encoder_out = None
            encoder_padding_mask = None
            incremental_state = None

        if incremental_state is None:
            self_attn_mask = self.buffered_future_mask(x)
        else:
            self_attn_mask = None

        # TODO: add back prev_self_attn_state, prev_attn_state,
        # self_attn_padding_mask
        prev_self_attn_state = None
        prev_attn_state = None
        self_attn_padding_mask = None

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        if mt_task:
            return (x, encoder_out, encoder_padding_mask)
        return x

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
