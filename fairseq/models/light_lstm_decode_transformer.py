# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from fairseq import options as fairseq_options
from fairseq import utils as fairseq_utils
from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.adaptive_softmax import AdaptiveSoftmax
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_model import FairseqEncoderDecoderModel
from fairseq.models.fairseq_incremental_decoder import \
    FairseqIncrementalDecoder
from fairseq.models.transformer import EncoderOut, TransformerEncoder

DEFAULT_MAX_SOURCE_POSITIONS = 1024


@register_model('light_lstm_decode_transformer')
class LightLSTMDecodeTransformerModel(FairseqEncoderDecoderModel):
    """Transformer with lightweight LSTM based decoder.

    This model takes advantage of the performance of the transformer
    encoder while scaling linearly in decoding speed and memory against
    generated sequence length.

    Rather than using multi-head attention in every layer in the
    decoder, only 1 single-headed attention block is used in the final
    layer of the decoder.

    This model follows closely in implementation to fairseq.models.lstm

    The model provides the following command-line arguments

    .. argparse::
        :ref: fairseq.models.lstm_decode_transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            '--activation-fn',
            choices=fairseq_utils.get_available_activation_fns(),
            help='activation function to use')

        parser.add_argument(
            '--dropout', type=float, metavar='D',
            help='dropout probability')
        parser.add_argument(
            '--attention-dropout', type=float, metavar='D',
            help='dropout probability for attention weights')
        parser.add_argument(
            '--activation-dropout', '--relu-dropout', type=float, metavar='D',
            help='dropout probability after activation in FFN.')

        parser.add_argument(
            '--encoder-embed-path', type=str, metavar='STR',
            help='path to pre-trained encoder embedding')
        parser.add_argument(
            '--encoder-embed-dim', type=int, metavar='N',
            help='encoder embedding dimension')
        parser.add_argument(
            '--encoder-ffn-embed-dim', type=int, metavar='N',
            help='encoder embedding dimension for FFN')
        parser.add_argument(
            '--encoder-layers', type=int, metavar='N',
            help='num encoder layers')
        parser.add_argument(
            '--encoder-attention-heads', type=int, metavar='N',
            help='num encoder attention heads')
        parser.add_argument(
            '--encoder-normalize-before', action='store_true',
            help='apply layernorm before each encoder block')
        parser.add_argument(
            '--encoder-learned-pos', action='store_true',
            help='use learned positional embeddings in the encoder')

        parser.add_argument(
            '--decoder-embed-path', type=str, metavar='STR',
            help='path to pre-trained decoder embedding')
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='decoder embedding dimension')
        parser.add_argument(
            '--decoder-ffn-embed-dim', type=int, metavar='N',
            help='decoder embedding dimension for FFN')
        parser.add_argument(
            '--decoder-layers', type=int, metavar='N',
            help='number of decoder layers')
        parser.add_argument(
            '--decoder-out-embed-dim', type=int, metavar='N',
            help='decoder output embedding dimension')

        parser.add_argument(
            '--adaptive-softmax-cutoff', metavar='EXPR',
            help='comma separated list of adaptive softmax cutoff points. '
            'Must be used with adaptive_loss criterion'),
        parser.add_argument(
            '--adaptive-softmax-dropout', type=float, metavar='D',
            help='sets adaptive softmax dropout for the tail projections')

        parser.add_argument(
            '--share-decoder-input-output-embed', action='store_true',
            help='share decoder input and output embeddings')
        parser.add_argument(
            '--share-all-embeddings', action='store_true',
            help='share encoder, decoder and output embeddings'
            ' (requires shared dictionary and embed dim)')

        parser.add_argument(
            '--layernorm-embedding', action='store_true',
            help='add layernorm to embedding')
        parser.add_argument(
            '--no-scale-embedding', action='store_true',
            help='if True, dont scale embeddings')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)

        if getattr(args, 'max_source_positions', None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = fairseq_utils.parse_embedding(path)
                fairseq_utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError(
                    '--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            if (
                args.decoder_embed_path and
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    '--share-all-embeddings not compatible with '
                    '--decoder-embed-path'
                )
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return LightLSTMDecodeTransformerDecoder(
            dictionary=tgt_dict,
            embed_tokens=embed_tokens,
            embed_dim=args.decoder_embed_dim,
            ffn_embed_dim=args.decoder_ffn_embed_dim,
            encoder_embed_dim=args.encoder_embed_dim,
            num_layers=args.decoder_layers,
            dropout=args.dropout,
            activation_dropout=args.activation_dropout,
            share_input_output_embed=args.share_decoder_input_output_embed,
            no_encoder_attn=False,
            adaptive_softmax_cutoff=args.adaptive_softmax_cutoff,
            adaptive_softmax_dropout=args.adaptive_softmax_dropout,
            layernorm_embedding=args.layernorm_embedding,
            no_scale_embedding=args.no_scale_embedding,
        )


class LightLSTMDecodeTransformerDecoder(FairseqIncrementalDecoder):
    """Multihead attention decoder with lighter LSTM self-attention."""

    def __init__(
        self,
        dictionary,
        embed_tokens: nn.Embedding,
        embed_dim: int = 512,
        ffn_embed_dim: int = 512,
        encoder_embed_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        activation_dropout: float = 0.0,
        share_input_output_embed: bool = False,
        no_encoder_attn: bool = False,
        adaptive_softmax_cutoff: Optional[str] = None,
        adaptive_softmax_dropout: float = 0.0,
        layernorm_embedding: bool = False,
        no_scale_embedding: bool = False,
    ):
        super().__init__(dictionary)

        self.dropout = dropout
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        input_embed_dim = embed_tokens.embedding_dim
        if share_input_output_embed:
            output_embed_dim = input_embed_dim
        else:
            output_embed_dim = embed_dim

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = None
        if embed_dim != input_embed_dim:
            self.project_in_dim = Linear(
                input_embed_dim,
                embed_dim,
                bias=False
            )

        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=embed_dim * 2 if layer == 0 else embed_dim,
                hidden_size=embed_dim,
                bias=True,
            )
            for layer in range(num_layers)
        ])

        self.attention = None
        if not no_encoder_attn:
            self.attention = AttentionLayer(embed_dim, encoder_embed_dim)

        self.fc1 = Linear(self.hidden_size, self.hidden_size * 4)
        self.fc2 = Linear(self.hidden_size * 4, self.hidden_size)
        self.final_layer_norm = LayerNorm(self.hidden_size)

        self.project_out_dim = None
        if embed_dim != output_embed_dim:
            self.project_out_dim = Linear(embed_dim, output_embed_dim)

        self.adaptive_softmax = None
        self.embed_out = None
        if adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                fairseq_options.eval_str_list(
                    adaptive_softmax_cutoff, type=int),
                dropout=adaptive_softmax_dropout,
            )
        elif not share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(
                len(dictionary), output_embed_dim))
            nn.init.normal_(
                self.embed_out, mean=0, std=output_embed_dim ** -0.5)

        if layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)

    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: Optional[EncoderOut],
        incremental_state: Optional[dict] = None,
        features_only: bool = False,
        **extra_args
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            prev_output_tokens (torch.Tensor): Previous decoder outputs of
                shape `(batch, tgt_len)`.
            encoder_out (Optional[EncoderOut]): Encoder outputs.
            incremental_state (Optional[dict], optional): Dictionary used to
                    cache states for efficient incremental decoding.
                    Defaults to None.
            features_only (bool, optional): Only return features without
                applying output layer. Defaults to False.

        Returns:
            Tuple[torch.Tensor, dict]:
                - Decoder output of shape `(batch, tgt_len, vocab)`.
                - A dictionary with model specific outputs.
        """  # noqa
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            **extra_args
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.embed_out is not None:
                return F.linear(features, self.embed_out)
            else:
                return F.linear(features, self.embed_tokens.weight)
        else:
            return features

    def extract_features(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: Optional[EncoderOut],
        incremental_state: Optional[dict] = None,
        **extra_args
    ) -> Tuple[torch.Tensor, dict]:
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        prev_states, input_feed = fairseq_utils.get_incremental_state(
            self,
            incremental_state,
            self.__class__.__name__
        )
        if prev_states is None:
            prev_states = [None] * self.num_layers
            input_feed = x.new_zeros(bsz, self.embed_dim)

        attn_scores = []
        outs = []
        for j in range(seqlen):
            # Input combines current tensor with context from previous timestep
            input = torch.cat((x[j, :, :], input_feed), dim=1)
            for i, rnn in enumerate(self.layers):
                hidden, cell = rnn(input, prev_states[i])
                input = F.dropout(
                    hidden,
                    p=self.dropout,
                    training=self.training
                )
                prev_states[i] = hidden, cell

            if self.attention is not None:
                out, attn_score = self.attention(hidden, encoder_out)
                attn_scores.append(attn_score)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout, training=self.training)

            # Save input current state as input feed for next timestep
            input_feed = out

            outs.append(out)

        fairseq_utils.set_incremental_state(
            self,
            incremental_state,
            self.__class__.__name__,
            (prev_states, input_feed)
        )

        # Collect outputs across time steps
        x = torch.stack(outs, dim=0)

        # ffn
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if not self.training and len(attn_scores) > 0:
            attn_scores = torch.stack(attn_scores, dim=1)
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        return x, {'attn': attn_scores}

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = fairseq_utils.get_incremental_state(
            self,
            incremental_state,
            self.__class__.__name__
        )
        if cached_state is None:
            return

        prev_states, input_feed = cached_state
        input_feed = input_feed.index_select(0, new_order)
        for i, (hidden, cell) in enumerate(prev_states):
            prev_states[i] = (
                hidden.index_select(0, new_order),
                cell.index_select(0, new_order)
            )

        fairseq_utils.set_incremental_state(
            self,
            incremental_state,
            self.__class__.__name__,
            (prev_states, input_feed)
        )

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class AttentionLayer(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        encoder_embed_dim: int,
        bias: bool = False
    ):
        super().__init__()

        self.input_proj = Linear(embed_dim, encoder_embed_dim, bias=bias)
        self.output_proj = Linear(
            embed_dim + encoder_embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor, encoder_out: EncoderOut):
        x_input = x
        x = self.input_proj(x)

        attn_weights = (encoder_out.encoder_out * x.unsqueeze(0)).sum(dim=2)
        if encoder_out.encoder_padding_mask is not None:
            attn_weights = attn_weights.masked_fill_(
                encoder_out.encoder_padding_mask.t(),
                float('-inf')
            )
        attn_probs_float = F.softmax(attn_weights.float(), dim=0)
        attn_probs = attn_probs_float.type_as(x_input)

        x = (attn_probs.unsqueeze(2) * encoder_out.encoder_out).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, x_input), dim=1)))
        return x, attn_probs


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture(
    'light_lstm_decode_transformer', 'light_lstm_decode_transformer')
def base_architecture(args):
    args.activation_fn = getattr(args, 'activation_fn', 'relu')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)

    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = \
        getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 1)

    args.adaptive_softmax_cutoff = \
        getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = \
        getattr(args, 'adaptive_softmax_dropout', 0)

    args.share_decoder_input_output_embed = \
        getattr(args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)

    # To account for dummy args
    args.encoder_layerdrop = 0
    args.no_token_positional_embeddings = False
