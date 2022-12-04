# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, Linear
from fairseq.modules import (
    AdaptiveSoftmax,
    CharacterTokenEmbedder,
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
)
from fairseq.modules.character_token_embedder import CHAR_PAD_IDX
from fairseq.modules.fb_bidirectional_multihead_attention import (
    BidirectionalMultiheadSelfAttention,
)

logger = logging.getLogger(__name__)


@register_model("bi_transformer_lm")
class BiTransformerLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout",
            default=0.1,
            type=float,
            metavar="D",
            help="dropout probability",
        )
        parser.add_argument(
            "--attention-dropout",
            default=0.0,
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--adaptive-softmax-cutoff",
            metavar="EXPR",
            help="comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion",
        )
        parser.add_argument(
            "--adaptive-softmax-dropout",
            type=float,
            metavar="D",
            help="sets adaptive softmax dropout for the tail projections",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--no-token-positional-embeddings",
            action="store_true",
            help="if set, disables positional embeddings (outside self attention)",
        )
        parser.add_argument(
            "--character-embeddings",
            action="store_true",
            help="if set, uses character embedding convolutions to produce token embeddings",
        )
        parser.add_argument(
            "--character-filters",
            type=str,
            metavar="LIST",
            default="[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]",
            help="size of character embeddings",
        )
        parser.add_argument(
            "--character-embedding-dim",
            type=int,
            metavar="N",
            default=4,
            help="size of character embeddings",
        )
        parser.add_argument(
            "--char-embedder-highway-layers",
            type=int,
            metavar="N",
            default=2,
            help="number of highway layers for character token embeddder",
        )
        parser.add_argument(
            "--linear-final-layer",
            action="store_true",
            help="if set, uses a simple linear layer for the final prediction that combines the "
            "forward and backward tower instead of an attentional layer",
        )
        parser.add_argument(
            "--linear-final-layer-bias",
            action="store_true",
            help="if set, has a bias on the final linear layer",
        )
        parser.add_argument(
            "--no-bias-kv",
            action="store_true",
            help="if set, pads attn with zero instead of adding a learnable bias kv",
        )
        parser.add_argument(
            "--max-char-len",
            type=int,
            metavar="N",
            default=50,
            help="if set and char_inputs, max characters to use per token",
        )
        # below two arguments are only used during inference / finetuning
        parser.add_argument(
            "--char-inputs",
            action="store_true",
            help="if set, model takes character ids as input",
        )
        parser.add_argument(
            "--unmask-curr-state",
            action="store_true",
            help="if set, there will be no mask for current state",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_bi_lm_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = args.tokens_per_sample
        if not getattr(args, "max_target_positions", None):
            args.max_target_positions = args.tokens_per_sample

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
                max_char_len=args.max_char_len,
                char_inputs=args.char_inputs,
            )
        else:
            embed_tokens = Embedding(
                len(task.dictionary), args.decoder_embed_dim, task.dictionary.pad()
            )

        logger.info(args)

        decoder = BiTransformerDecoder(args, task.output_dictionary, embed_tokens)
        return BiTransformerLanguageModel(decoder)

    @property
    def supported_targets(self):
        return {"self", "past", "future"}

    def get_layers_by_depth_for_fine_tuning(self):
        decoder_layers = self.decoder.get_layers_by_depth_for_fine_tuning()
        return [
            {"decoder.%s" % name: layer for name, layer in layers.items()}
            for layers in decoder_layers
        ]


class BiTransformerClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.proj = Linear(2 * embed_dim, num_classes)

    def forward(self, features, padding_mask=None, **kwargs):
        assert features.size(1) >= 2  # B x T x C

        # extract endpoints for classification
        x = features
        if x.size(1) == 2:
            x = x.view(x.size(0), -1)
        else:
            left = x[:, 0, :]
            if padding_mask is None:
                right = x[:, -1, :]
            else:
                eos_idx = (~padding_mask).int().sum(dim=1) - 1
                eos_idx += (torch.arange(eos_idx.size(0)) * x.size(1)).type_as(eos_idx)
                right = x.contiguous().view(-1, x.size(-1))[eos_idx]
            x = torch.cat([left, right], dim=1)

        return self.proj(x)


class BiTransformerDecoder(FairseqDecoder):
    """Transformer decoder."""

    def __init__(self, args, dictionary, embed_tokens, classification_head=None):
        super().__init__(dictionary)
        self.onnx_trace = False
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.share_input_output_embed = args.share_decoder_input_output_embed

        self.embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.self_target = args.self_target
        self.future_target = args.future_target
        self.past_target = args.past_target
        self.char_inputs = args.char_inputs

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(self.embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                self.embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.forward_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    args,
                    no_encoder_attn=True,
                    add_bias_kv=not args.no_bias_kv,
                    add_zero_attn=args.no_bias_kv,
                )
                for _ in range(args.decoder_layers)
            ]
        )
        self.backward_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    args,
                    no_encoder_attn=True,
                    add_bias_kv=not args.no_bias_kv,
                    add_zero_attn=args.no_bias_kv,
                )
                for _ in range(args.decoder_layers)
            ]
        )

        self.full_attn_layer = None
        self.full_linear_layer = None

        if self.self_target:
            if args.linear_final_layer:
                self.full_linear_layer = Linear(
                    self.embed_dim * 2, self.embed_dim, args.linear_final_layer_bias
                )
            else:
                self.full_attn_layer = BidirectionalTransformerDecoderLayer(args)

        self.load_softmax = not getattr(args, "remove_head", False)
        self.embed_out = None
        self.adaptive_softmax = None
        self.classification_head = classification_head

        if self.load_softmax:
            if args.adaptive_softmax_cutoff is not None:
                self.adaptive_softmax = AdaptiveSoftmax(
                    len(dictionary),
                    args.decoder_embed_dim,
                    utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                    dropout=args.adaptive_softmax_dropout,
                )
            elif not self.share_input_output_embed:
                self.embed_out = nn.Parameter(
                    torch.Tensor(len(dictionary), self.embed_dim)
                )
                nn.init.normal_(self.embed_out, mean=0, std=self.embed_dim**-0.5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, src_tokens, **kwargs):
        x, extra = self.extract_features(src_tokens, **kwargs)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, src_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, seq_len, embed_dim)`
                - a dictionary of additional data, where 'attn' contains the attention over the final
                  states (concatenated from forward and backward towers) and 'inner_states' is a list
                  of internal model states used to compute the predictions (for example to use in ELMO).
                  The first element is the token embeddings (with the positional embeddings added).
                  The next n elements are tuples of the hidden states for the forward and backward towers.
                  The last element is the output of the final full layer on top of the towers and would be
                  equivalent to the logits if adaptive softmax is used.
                  NOTE: unlike the logits, the format for all hidden states is T x B x C
        """
        # compute padding mask
        if self.char_inputs:
            # casting to byte for onnx
            padding_mask = src_tokens[:, :, 0].eq(CHAR_PAD_IDX).bool()
        else:
            padding_mask = src_tokens.eq(self.padding_idx).bool()

        # embed positions
        positional_input = self.padding_idx * padding_mask.long()
        positions = (
            self.embed_positions(positional_input)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if positions is not None:
            x += positions
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        fwd_x = bwd_x = x.transpose(0, 1)

        inner_states = [fwd_x]

        future_mask = self.buffered_future_mask(fwd_x)
        past_mask = self.buffered_past_mask(bwd_x)
        if not padding_mask.any():
            padding_mask = None

        # decoder layers
        for fwd, back in zip(self.forward_layers, self.backward_layers):
            fwd_x, _, _ = fwd(
                fwd_x,
                self_attn_mask=future_mask,
                self_attn_padding_mask=padding_mask,
            )
            bwd_x, _, _ = back(
                bwd_x,
                self_attn_mask=past_mask,
                self_attn_padding_mask=padding_mask,
            )
            inner_states.extend((fwd_x, bwd_x))

        if self.self_target:
            if self.full_attn_layer is not None:
                x, attn = self.full_attn_layer(
                    fwd_x,
                    bwd_x,
                    padding_mask,
                )
                inner_states.append(x)
            elif self.full_linear_layer is not None:
                zeros = x.new_zeros(1, fwd_x.size(1), fwd_x.size(2))
                fwd_x = torch.cat([zeros, fwd_x[:-1]], dim=0)
                bwd_x = torch.cat([bwd_x[1:], zeros], dim=0)
                x = torch.cat([fwd_x, bwd_x], dim=-1)
                x = self.full_linear_layer(x)
                attn = None
                inner_states.append(x)
            x = [x]
        else:
            x = []
            attn = None

        if self.future_target:
            x.append(fwd_x)
        if self.past_target:
            x.append(bwd_x)

        # T x B x C -> B x T x C
        x = [z.transpose(0, 1) for z in x]

        if len(x) == 1:
            x = x[0]

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.classification_head:
            return self.classification_head(features, **kwargs)

        x = features

        if not isinstance(x, list):
            x = [x]

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed and hasattr(self.embed_tokens, "weight"):
                x = [F.linear(x, self.embed_tokens.weight) for x in x]
            elif self.embed_out is not None:
                x = [F.linear(x, self.embed_out) for x in x]

        if len(x) == 1:
            x = x[0]

        return x

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if self.onnx_trace:
            a = torch._dim_arange(tensor, 0).unsqueeze(0).repeat(dim, 1)
            b = torch._dim_arange(tensor, 0).unsqueeze(1).repeat(1, dim)
            future_mask = a > b
            future_mask_neg_inf = torch.where(
                future_mask, torch.Tensor([float("-Inf")]), torch.Tensor([0])
            ).type_as(tensor)
            return future_mask_neg_inf

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

    def buffered_past_mask(self, tensor):
        dim = tensor.size(0)
        if self.onnx_trace:
            a = torch._dim_arange(tensor, 0).unsqueeze(0).repeat(dim, 1)
            b = torch._dim_arange(tensor, 0).unsqueeze(1).repeat(1, dim)
            past_mask = a < b
            past_mask_neg_inf = torch.where(
                past_mask, torch.Tensor([float("-Inf")]), torch.Tensor([0])
            ).type_as(tensor)
            return past_mask_neg_inf
        if (
            not hasattr(self, "_past_mask")
            or self._past_mask is None
            or self._past_mask.device != tensor.device
        ):
            self._past_mask = torch.tril(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), -1
            )
        if self._past_mask.size(0) < dim:
            self._past_mask = torch.tril(
                utils.fill_with_neg_inf(self._past_mask.resize_(dim, dim)), -1
            )
        return self._past_mask[:dim, :dim]

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            state_dict[name + ".embed_positions._float_tensor"] = torch.FloatTensor(1)
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if k.startswith(name + ".adaptive_softmax.") or k.startswith(
                    name + ".embed_out"
                ):
                    del state_dict[k]
        return state_dict

    def get_layers_by_depth_for_fine_tuning(self):
        """
        Returns a list of module dictionaries, where each module dictionary
        (name -> module) contains modules at the same "depth" in the model.
        The first module dictionary corresponds to the lowest level layer (embeddings)
        and the last corresponds to the highest level layer.
        """
        emb_layers = self._module_dict(("embed_tokens", "embed_positions"))
        fwd_bwd_layers = [
            {"forward_layers.%d" % i: fwd, "backward_layers.%d" % i: bwd}
            for i, (fwd, bwd) in enumerate(
                zip(self.forward_layers, self.backward_layers)
            )
        ]
        top_layers = self._module_dict(("full_attn_layer", "full_linear_layer"))
        return [emb_layers] + fwd_bwd_layers + [top_layers]

    def _module_dict(self, attributes):
        return {
            attr: getattr(self, attr)
            for attr in attributes
            if getattr(self, attr, None) is not None
        }


class BidirectionalTransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = BidirectionalMultiheadSelfAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            mask_curr_state=not args.unmask_curr_state,
        )
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

        self.fwd_layer_norm = LayerNorm(self.embed_dim, export=args.char_inputs)
        self.bwd_layer_norm = LayerNorm(self.embed_dim, export=args.char_inputs)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=args.char_inputs)

    def forward(self, fwd_x, bwd_x, key_padding_mask):
        fwd_x = self.maybe_layer_norm(self.fwd_layer_norm, fwd_x, before=True)
        bwd_x = self.maybe_layer_norm(self.bwd_layer_norm, bwd_x, before=True)
        x, attn = self.self_attn(
            fwd_x=fwd_x,
            bwd_x=bwd_x,
            key_padding_mask=key_padding_mask,
        )
        x = self.dropout_module(x)
        x = self.maybe_layer_norm(self.fwd_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


@register_model_architecture("bi_transformer_lm", "bi_transformer_lm")
def base_bi_lm_architecture(args):
    # by default bi-directional language models predict the current token (self)
    args.self_target = getattr(
        args, "self_target", not getattr(args, "exclude_self_target", False)
    )

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(
        args, "adaptive_softmax_dropout", args.dropout
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)
    args.character_filters = getattr(
        args,
        "character_filters",
        "[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]",
    )
    args.character_embedding_dim = getattr(args, "character_embedding_dim", 128)
    args.char_embedder_highway_layers = getattr(args, "char_embedder_highway_layers", 2)
    args.linear_final_layer = getattr(args, "linear_final_layer", False)
    args.linear_final_layer_bias = getattr(args, "linear_final_layer_bias", False)

    args.future_target = getattr(args, "future_target", False)
    args.past_target = getattr(args, "past_target", False)

    args.no_bias_kv = getattr(args, "no_bias_kv", False)
    args.char_inputs = getattr(args, "char_inputs", False)
    args.unmask_curr_state = getattr(args, "unmask_curr_state", False)
    args.max_char_len = getattr(args, "max_char_len", 50)

    # otherwise model training is unstable
    args.decoder_normalize_before = True


@register_model_architecture("bi_transformer_lm", "bi_transformer_lm_big")
def bi_transformer_lm_big(args):
    args.self_target = True
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_bi_lm_architecture(args)


@register_model_architecture("bi_transformer_lm", "bi_transformer_lm_bpe_large")
def bi_transformer_lm_bpe_large(args):
    args.self_target = True
    # TODO support query formulation
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    base_bi_lm_architecture(args)


@register_model_architecture("bi_transformer_lm", "bi_transformer_lm_big_non_cloze")
def bi_transformer_lm_big_non_cloze(args):
    bi_transformer_lm_big(args)
    args.self_target = False
    args.future_target = True
    args.past_target = True


@register_model_architecture("bi_transformer_lm", "bi_transformer_lm_huge")
def bi_transformer_lm_huge(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)  # 2.6B params
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8192)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.activation_fn = getattr(args, "activation_fn", "gelu_fast")
    base_bi_lm_architecture(args)


@register_model_architecture("bi_transformer_lm", "bi_transformer_lm_huge_relu")
def bi_transformer_lm_huge_relu(args):
    args.activation_fn = getattr(args, "activation_fn", "relu")
    bi_transformer_lm_huge(args)
