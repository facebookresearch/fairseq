# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fairseq import options as fairseq_options
from fairseq import utils as fairseq_utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.data.dictionary import Dictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_incremental_decoder import \
    FairseqIncrementalDecoder
from fairseq.models.fairseq_model import FairseqEncoderDecoderModel
from fairseq.models.transformer import TransformerEncoder, EncoderOut, Linear
from fairseq.modules.adaptive_softmax import AdaptiveSoftmax
from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.multihead_attention import MultiheadAttention

BufferType = Tuple[torch.Tensor, torch.Tensor]
DEFAULT_MAX_SOURCE_POSITIONS = 1024


@register_model('lstm_decode_transformer')
class LSTMDecodeTransformerModel(FairseqEncoderDecoderModel):
    """Transformer with LSTM based decoder.

    This model takes advantage of the performance of the transformer
    encoder while scaling linearly in decoding speed and memory against
    generated sequence length.

    Similar to the standard transformer, this model takes advantage
    of multi-head attention to attend on encoder outputs.
    However, instead of using multi-head self-attention, this model uses
    a LSTM blocks instead.

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
            '--decoder-attention-heads', type=int, metavar='N',
            help='num decoder attention heads')
        parser.add_argument(
            '--decoder-out-embed-dim', type=int, metavar='N',
            help='decoder output embedding dimension')
        parser.add_argument(
            '--decoder-normalize-before', action='store_true',
            help='apply layernorm before each decoder block')

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
        return LSTMDecodeTransformerDecoder(
            dictionary=tgt_dict,
            embed_tokens=embed_tokens,
            embed_dim=args.decoder_embed_dim,
            ffn_embed_dim=args.decoder_ffn_embed_dim,
            encoder_embed_dim=args.encoder_embed_dim,
            num_layers=args.decoder_layers,
            num_heads=args.decoder_attention_heads,
            activation_fn=args.activation_fn,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            share_input_output_embed=args.share_decoder_input_output_embed,
            normalize_before=args.decoder_normalize_before,
            no_encoder_attn=False,
            adaptive_softmax_cutoff=args.adaptive_softmax_cutoff,
            layernorm_embedding=args.layernorm_embedding,
            no_scale_embedding=args.no_scale_embedding,
        )


class LSTMDecodeTransformerDecoder(FairseqIncrementalDecoder):
    """Multihead attention decoder with LSTM instead of self-attn."""

    def __init__(
        self,
        dictionary: Dictionary,
        embed_tokens: nn.Embedding,
        embed_dim: int = 512,
        ffn_embed_dim: int = 512,
        encoder_embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        activation_fn: str = 'relu',
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        share_input_output_embed: bool = False,
        normalize_before: bool = False,
        no_encoder_attn: bool = False,
        adaptive_softmax_cutoff: Optional[str] = None,
        layernorm_embedding: bool = False,
        no_scale_embedding: bool = False,
    ):
        super().__init__(dictionary)

        self.dropout = dropout
        self.share_input_output_embed = share_input_output_embed

        output_embed_dim = input_embed_dim = embed_tokens.embedding_dim

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if no_scale_embedding else math.sqrt(embed_dim)

        self.in_proj = None
        if embed_dim != input_embed_dim:
            self.in_proj = Linear(input_embed_dim, embed_dim, bias=False)

        self.out_proj = None
        if embed_dim != output_embed_dim:
            self.out_proj = Linear(embed_dim, output_embed_dim, bias=False)

        self.layers = nn.ModuleList([LSTMTransformerDecoderLayer(
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            encoder_embed_dim=encoder_embed_dim,
            num_heads=num_heads,
            activation_fn=activation_fn,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            normalize_before=normalize_before,
            no_encoder_attn=no_encoder_attn
        ) for _ in range(num_layers)])

        self.adaptive_softmax = None
        if adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                fairseq_options.eval_str_list(adaptive_softmax_cutoff),
                dropout=dropout,
            )

        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(
                self.embed_out,
                mean=0,
                std=output_embed_dim ** -0.5
            )

        self.layer_norm = None
        if normalize_before:
            self.layer_norm = LayerNorm(embed_dim)

        self.layernorm_embedding = None
        if layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)

    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: EncoderOut = None,
        incremental_state: Optional[dict] = None,
        features_only: bool = False,
        **extra_args,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            prev_output_tokens (torch.Tensor):
                Previous decoder outputs of shape `(batch, tgt_len)`.
            encoder_out (EncoderOut, optional):
                Output from encoder. Defaults to None.
            incremental_state (Optional[dict], optional):
                Dictionary caching tensors for efficient sequence generation.
                Defaults to None.
            features_only (bool, optional):
                Only return features without applying output layer.
                Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                Tensor of shape `(seq_len, batch, embed_dim)` and
                meta data like atten weights.
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
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """This should be strictly infinite."""
        return int(1e6)

    def extract_features(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: EncoderOut = None,
        incremental_state: Optional[dict] = None,
        features_only: bool = False,
        **unused,
    ):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.in_proj is not None:
            x = self.in_proj(x)
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # BTC -> TBC
        x = x.transpose(0, 1)

        attn = None
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out=encoder_out,
                incremental_state=incremental_state
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # TBC -> BTC
        x = x.transpose(0, 1)

        if self.out_proj is not None:
            x = self.out_proj(x)

        return x, {'attn': [attn]}


@with_incremental_state
class LSTMTransformerDecoderLayer(nn.Module):
    """Multihead attention decoder layer with LSTM instead of self-attn."""

    def __init__(
        self,
        embed_dim: int = 512,
        ffn_embed_dim: int = 2048,
        encoder_embed_dim: int = 512,
        num_heads: int = 8,
        activation_fn: str = 'relu',
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        normalize_before: bool = False,
        no_encoder_attn: bool = False,
    ):
        super().__init__()

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.normalize_before = normalize_before

        self.activation_fn = fairseq_utils.get_activation_fn(activation_fn)

        # To be determined if applying LSTM directly is helpful
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0.0,
            bidirectional=False
        )
        self.layer_norm = LayerNorm(embed_dim)

        if no_encoder_attn:
            self.attn = None
            self.attn_layer_norm = None
        else:
            self.attn = MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                kdim=encoder_embed_dim,
                vdim=encoder_embed_dim,
                dropout=attention_dropout,
                bias=True,
                encoder_decoder_attention=True,
            )
            self.attn_layer_norm = LayerNorm(embed_dim)

        self.fc1 = Linear(embed_dim, ffn_embed_dim)
        self.fc2 = Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim)

        self.need_attn = True

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def _get_input_buffer(self, incremental_state: dict) -> BufferType:
        return fairseq_utils.get_incremental_state(
            self,
            incremental_state=incremental_state,
            key='rnn_hidden_state'
        )

    def _set_input_buffer(self, incremental_state: dict, buffer: BufferType):
        return fairseq_utils.set_incremental_state(
            self,
            incremental_state=incremental_state,
            key='rnn_hidden_state',
            value=buffer
        )

    def reorder_incremental_state(
        self,
        incremental_state: dict,
        new_order: torch.Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        buffer = self._get_input_buffer(incremental_state)
        if buffer is None:
            return

        h, c = buffer
        new_buffer = (
            h.index_select(1, new_order),
            c.index_select(1, new_order)
        )
        self._set_input_buffer(incremental_state, new_buffer)
        return incremental_state

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x (torch.Tensor):
                Tensor of shape `(seq_len, batch, embed_dim)`
            encoder_out (Optional[EncoderOut], optional):
                Encoder output. Defaults to None.
            incremental_state (Optional[dict], optional):
                Dictionary caching tensors for efficient sequence generation.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                Tensor of shape `(seq_len, batch, embed_dim)` and
                attention weights.
        """  # noqa
        prev_states = None
        if incremental_state is not None:
            prev_states = self._get_input_buffer(incremental_state)

        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        seqlen, bsz, _ = x.size()
        packed_x = pack_padded_sequence(x, [seqlen] * bsz)
        packed_x, new_states = self.rnn(packed_x, prev_states)
        x, _ = pad_packed_sequence(packed_x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x += residual
        if not self.normalize_before:
            x = self.layer_norm(x)

        if incremental_state is not None:
            self._set_input_buffer(incremental_state, new_states)

        attn = None
        if self.attn is not None:
            residual = x
            if self.normalize_before:
                x = self.attn_layer_norm(x)
            x, attn = self.attn(
                query=x,
                key=encoder_out.encoder_out,
                value=encoder_out.encoder_out,
                key_padding_mask=encoder_out.encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x += residual
            if not self.normalize_before:
                x = self.attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x += residual
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


@register_model_architecture(
    'lstm_decode_transformer', 'lstm_decode_transformer')
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
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = \
        getattr(args, "decoder_normalize_before", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_softmax_cutoff = \
        getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = \
        getattr(args, 'adaptive_softmax_dropout', 0)

    args.share_decoder_input_output_embed = \
        getattr(args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)

    args.quant_noise_pq = 0
    args.encoder_layerdrop = 0
    args.no_token_positional_embeddings = False
