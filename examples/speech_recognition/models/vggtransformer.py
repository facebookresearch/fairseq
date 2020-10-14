# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import LinearizedConvolution
from examples.speech_recognition.data.data_utils import lengths_to_encoder_padding_mask
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer, VGGBlock


@register_model("asr_vggtransformer")
class VGGTransformerModel(FairseqEncoderDecoderModel):
    """
    Transformers with convolutional context for ASR
    https://arxiv.org/abs/1904.11660
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="encoder input dimension per input channel",
        )
        parser.add_argument(
            "--vggblock-enc-config",
            type=str,
            metavar="EXPR",
            help="""
    an array of tuples each containing the configuration of one vggblock:
    [(out_channels,
      conv_kernel_size,
      pooling_kernel_size,
      num_conv_layers,
      use_layer_norm), ...])
            """,
        )
        parser.add_argument(
            "--transformer-enc-config",
            type=str,
            metavar="EXPR",
            help=""""
    a tuple containing the configuration of the encoder transformer layers
    configurations:
    [(input_dim,
      num_heads,
      ffn_dim,
      normalize_before,
      dropout,
      attention_dropout,
      relu_dropout), ...]')
            """,
        )
        parser.add_argument(
            "--enc-output-dim",
            type=int,
            metavar="N",
            help="""
    encoder output dimension, can be None. If specified, projecting the
    transformer output to the specified dimension""",
        )
        parser.add_argument(
            "--in-channels",
            type=int,
            metavar="N",
            help="number of encoder input channels",
        )
        parser.add_argument(
            "--tgt-embed-dim",
            type=int,
            metavar="N",
            help="embedding dimension of the decoder target tokens",
        )
        parser.add_argument(
            "--transformer-dec-config",
            type=str,
            metavar="EXPR",
            help="""
    a tuple containing the configuration of the decoder transformer layers
    configurations:
    [(input_dim,
      num_heads,
      ffn_dim,
      normalize_before,
      dropout,
      attention_dropout,
      relu_dropout), ...]
            """,
        )
        parser.add_argument(
            "--conv-dec-config",
            type=str,
            metavar="EXPR",
            help="""
    an array of tuples for the decoder 1-D convolution config
        [(out_channels, conv_kernel_size, use_layer_norm), ...]""",
        )

    @classmethod
    def build_encoder(cls, args, task):
        return VGGTransformerEncoder(
            input_feat_per_channel=args.input_feat_per_channel,
            vggblock_config=eval(args.vggblock_enc_config),
            transformer_config=eval(args.transformer_enc_config),
            encoder_output_dim=args.enc_output_dim,
            in_channels=args.in_channels,
        )

    @classmethod
    def build_decoder(cls, args, task):
        return TransformerDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.tgt_embed_dim,
            transformer_config=eval(args.transformer_dec_config),
            conv_config=eval(args.conv_dec_config),
            encoder_output_dim=args.enc_output_dim,
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        base_architecture(args)

        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        return cls(encoder, decoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs


DEFAULT_ENC_VGGBLOCK_CONFIG = ((32, 3, 2, 2, False),) * 2
DEFAULT_ENC_TRANSFORMER_CONFIG = ((256, 4, 1024, True, 0.2, 0.2, 0.2),) * 2
# 256: embedding dimension
# 4: number of heads
# 1024: FFN
# True: apply layerNorm before (dropout + resiaul) instead of after
# 0.2 (dropout): dropout after MultiheadAttention and second FC
# 0.2 (attention_dropout): dropout in MultiheadAttention
# 0.2 (relu_dropout): dropout after ReLu
DEFAULT_DEC_TRANSFORMER_CONFIG = ((256, 2, 1024, True, 0.2, 0.2, 0.2),) * 2
DEFAULT_DEC_CONV_CONFIG = ((256, 3, True),) * 2


# TODO: repace transformer encoder config from one liner
# to explicit args to get rid of this transformation
def prepare_transformer_encoder_params(
    input_dim,
    num_heads,
    ffn_dim,
    normalize_before,
    dropout,
    attention_dropout,
    relu_dropout,
):
    args = argparse.Namespace()
    args.encoder_embed_dim = input_dim
    args.encoder_attention_heads = num_heads
    args.attention_dropout = attention_dropout
    args.dropout = dropout
    args.activation_dropout = relu_dropout
    args.encoder_normalize_before = normalize_before
    args.encoder_ffn_embed_dim = ffn_dim
    return args


def prepare_transformer_decoder_params(
    input_dim,
    num_heads,
    ffn_dim,
    normalize_before,
    dropout,
    attention_dropout,
    relu_dropout,
):
    args = argparse.Namespace()
    args.decoder_embed_dim = input_dim
    args.decoder_attention_heads = num_heads
    args.attention_dropout = attention_dropout
    args.dropout = dropout
    args.activation_dropout = relu_dropout
    args.decoder_normalize_before = normalize_before
    args.decoder_ffn_embed_dim = ffn_dim
    return args


class VGGTransformerEncoder(FairseqEncoder):
    """VGG + Transformer encoder"""

    def __init__(
        self,
        input_feat_per_channel,
        vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG,
        transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
        encoder_output_dim=512,
        in_channels=1,
        transformer_context=None,
        transformer_sampling=None,
    ):
        """constructor for VGGTransformerEncoder

        Args:
            - input_feat_per_channel: feature dim (not including stacked,
              just base feature)
            - in_channel: # input channels (e.g., if stack 8 feature vector
                together, this is 8)
            - vggblock_config: configuration of vggblock, see comments on
                DEFAULT_ENC_VGGBLOCK_CONFIG
            - transformer_config: configuration of transformer layer, see comments
                on DEFAULT_ENC_TRANSFORMER_CONFIG
            - encoder_output_dim: final transformer output embedding dimension
            - transformer_context: (left, right) if set, self-attention will be focused
              on (t-left, t+right)
            - transformer_sampling: an iterable of int, must match with
              len(transformer_config), transformer_sampling[i] indicates sampling
              factor for i-th transformer layer, after multihead att and feedfoward
              part
        """
        super().__init__(None)

        self.num_vggblocks = 0
        if vggblock_config is not None:
            if not isinstance(vggblock_config, Iterable):
                raise ValueError("vggblock_config is not iterable")
            self.num_vggblocks = len(vggblock_config)

        self.conv_layers = nn.ModuleList()
        self.in_channels = in_channels
        self.input_dim = input_feat_per_channel

        if vggblock_config is not None:
            for _, config in enumerate(vggblock_config):
                (
                    out_channels,
                    conv_kernel_size,
                    pooling_kernel_size,
                    num_conv_layers,
                    layer_norm,
                ) = config
                self.conv_layers.append(
                    VGGBlock(
                        in_channels,
                        out_channels,
                        conv_kernel_size,
                        pooling_kernel_size,
                        num_conv_layers,
                        input_dim=input_feat_per_channel,
                        layer_norm=layer_norm,
                    )
                )
                in_channels = out_channels
                input_feat_per_channel = self.conv_layers[-1].output_dim

        transformer_input_dim = self.infer_conv_output_dim(
            self.in_channels, self.input_dim
        )
        # transformer_input_dim is the output dimension of VGG part

        self.validate_transformer_config(transformer_config)
        self.transformer_context = self.parse_transformer_context(transformer_context)
        self.transformer_sampling = self.parse_transformer_sampling(
            transformer_sampling, len(transformer_config)
        )

        self.transformer_layers = nn.ModuleList()

        if transformer_input_dim != transformer_config[0][0]:
            self.transformer_layers.append(
                Linear(transformer_input_dim, transformer_config[0][0])
            )
        self.transformer_layers.append(
            TransformerEncoderLayer(
                prepare_transformer_encoder_params(*transformer_config[0])
            )
        )

        for i in range(1, len(transformer_config)):
            if transformer_config[i - 1][0] != transformer_config[i][0]:
                self.transformer_layers.append(
                    Linear(transformer_config[i - 1][0], transformer_config[i][0])
                )
            self.transformer_layers.append(
                TransformerEncoderLayer(
                    prepare_transformer_encoder_params(*transformer_config[i])
                )
            )

        self.encoder_output_dim = encoder_output_dim
        self.transformer_layers.extend(
            [
                Linear(transformer_config[-1][0], encoder_output_dim),
                LayerNorm(encoder_output_dim),
            ]
        )

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        src_tokens: padded tensor (B, T, C * feat)
        src_lengths: tensor of original lengths of input utterances (B,)
        """
        bsz, max_seq_len, _ = src_tokens.size()
        x = src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
        x = x.transpose(1, 2).contiguous()
        # (B, C, T, feat)

        for layer_idx in range(len(self.conv_layers)):
            x = self.conv_layers[layer_idx](x)

        bsz, _, output_seq_len, _ = x.size()

        # (B, C, T, feat) -> (B, T, C, feat) -> (T, B, C, feat) -> (T, B, C * feat)
        x = x.transpose(1, 2).transpose(0, 1)
        x = x.contiguous().view(output_seq_len, bsz, -1)

        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
        # TODO: shouldn't subsampling_factor determined in advance ?
        input_lengths = (src_lengths.float() / subsampling_factor).ceil().long()

        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            input_lengths, batch_first=True
        )
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        attn_mask = self.lengths_to_attn_mask(input_lengths, subsampling_factor)

        transformer_layer_idx = 0

        for layer_idx in range(len(self.transformer_layers)):

            if isinstance(self.transformer_layers[layer_idx], TransformerEncoderLayer):
                x = self.transformer_layers[layer_idx](
                    x, encoder_padding_mask, attn_mask
                )

                if self.transformer_sampling[transformer_layer_idx] != 1:
                    sampling_factor = self.transformer_sampling[transformer_layer_idx]
                    x, encoder_padding_mask, attn_mask = self.slice(
                        x, encoder_padding_mask, attn_mask, sampling_factor
                    )

                transformer_layer_idx += 1

            else:
                x = self.transformer_layers[layer_idx](x)

        # encoder_padding_maks is a (T x B) tensor, its [t, b] elements indicate
        # whether encoder_output[t, b] is valid or not (valid=0, invalid=1)

        return {
            "encoder_out": x,  # (T, B, C)
            "encoder_padding_mask": encoder_padding_mask.t()
            if encoder_padding_mask is not None
            else None,
            # (B, T) --> (T, B)
        }

    def infer_conv_output_dim(self, in_channels, input_dim):
        sample_seq_len = 200
        sample_bsz = 10
        x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
        for i, _ in enumerate(self.conv_layers):
            x = self.conv_layers[i](x)
        x = x.transpose(1, 2)
        mb, seq = x.size()[:2]
        return x.contiguous().view(mb, seq, -1).size(-1)

    def validate_transformer_config(self, transformer_config):
        for config in transformer_config:
            input_dim, num_heads = config[:2]
            if input_dim % num_heads != 0:
                msg = (
                    "ERROR in transformer config {}: ".format(config)
                    + "input dimension {} ".format(input_dim)
                    + "not dividable by number of heads {}".format(num_heads)
                )
                raise ValueError(msg)

    def parse_transformer_context(self, transformer_context):
        """
        transformer_context can be the following:
        -   None; indicates no context is used, i.e.,
            transformer can access full context
        -   a tuple/list of two int; indicates left and right context,
            any number <0 indicates infinite context
                * e.g., (5, 6) indicates that for query at x_t, transformer can
                access [t-5, t+6] (inclusive)
                * e.g., (-1, 6) indicates that for query at x_t, transformer can
                access [0, t+6] (inclusive)
        """
        if transformer_context is None:
            return None

        if not isinstance(transformer_context, Iterable):
            raise ValueError("transformer context must be Iterable if it is not None")

        if len(transformer_context) != 2:
            raise ValueError("transformer context must have length 2")

        left_context = transformer_context[0]
        if left_context < 0:
            left_context = None

        right_context = transformer_context[1]
        if right_context < 0:
            right_context = None

        if left_context is None and right_context is None:
            return None

        return (left_context, right_context)

    def parse_transformer_sampling(self, transformer_sampling, num_layers):
        """
        parsing transformer sampling configuration

        Args:
            - transformer_sampling, accepted input:
                * None, indicating no sampling
                * an Iterable with int (>0) as element
            - num_layers, expected number of transformer layers, must match with
              the length of transformer_sampling if it is not None

        Returns:
            - A tuple with length num_layers
        """
        if transformer_sampling is None:
            return (1,) * num_layers

        if not isinstance(transformer_sampling, Iterable):
            raise ValueError(
                "transformer_sampling must be an iterable if it is not None"
            )

        if len(transformer_sampling) != num_layers:
            raise ValueError(
                "transformer_sampling {} does not match with the number "
                "of layers {}".format(transformer_sampling, num_layers)
            )

        for layer, value in enumerate(transformer_sampling):
            if not isinstance(value, int):
                raise ValueError("Invalid value in transformer_sampling: ")
            if value < 1:
                raise ValueError(
                    "{} layer's subsampling is {}.".format(layer, value)
                    + " This is not allowed! "
                )
        return transformer_sampling

    def slice(self, embedding, padding_mask, attn_mask, sampling_factor):
        """
        embedding is a (T, B, D) tensor
        padding_mask is a (B, T) tensor or None
        attn_mask is a (T, T) tensor or None
        """
        embedding = embedding[::sampling_factor, :, :]
        if padding_mask is not None:
            padding_mask = padding_mask[:, ::sampling_factor]
        if attn_mask is not None:
            attn_mask = attn_mask[::sampling_factor, ::sampling_factor]

        return embedding, padding_mask, attn_mask

    def lengths_to_attn_mask(self, input_lengths, subsampling_factor=1):
        """
        create attention mask according to sequence lengths and transformer
        context

        Args:
            - input_lengths: (B, )-shape Int/Long tensor; input_lengths[b] is
              the length of b-th sequence
            - subsampling_factor: int
                * Note that the left_context and right_context is specified in
                  the input frame-level while input to transformer may already
                  go through subsampling (e.g., the use of striding in vggblock)
                  we use subsampling_factor to scale the left/right context

        Return:
            - a (T, T) binary tensor or None, where T is max(input_lengths)
                * if self.transformer_context is None, None
                * if left_context is None,
                    * attn_mask[t, t + right_context + 1:] = 1
                    * others = 0
                * if right_context is None,
                    * attn_mask[t, 0:t - left_context] = 1
                    * others = 0
                * elsif
                    * attn_mask[t, t - left_context: t + right_context + 1] = 0
                    * others = 1
        """
        if self.transformer_context is None:
            return None

        maxT = torch.max(input_lengths).item()
        attn_mask = torch.zeros(maxT, maxT)

        left_context = self.transformer_context[0]
        right_context = self.transformer_context[1]
        if left_context is not None:
            left_context = math.ceil(self.transformer_context[0] / subsampling_factor)
        if right_context is not None:
            right_context = math.ceil(self.transformer_context[1] / subsampling_factor)

        for t in range(maxT):
            if left_context is not None:
                st = 0
                en = max(st, t - left_context)
                attn_mask[t, st:en] = 1
            if right_context is not None:
                st = t + right_context + 1
                st = min(st, maxT - 1)
                attn_mask[t, st:] = 1

        return attn_mask.to(input_lengths.device)

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
            1, new_order
        )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(1, new_order)
        return encoder_out


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
        conv_config=DEFAULT_DEC_CONV_CONFIG,
        encoder_output_dim=512,
    ):

        super().__init__(dictionary)
        vocab_size = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(vocab_size, embed_dim, self.padding_idx)

        self.conv_layers = nn.ModuleList()
        for i in range(len(conv_config)):
            out_channels, kernel_size, layer_norm = conv_config[i]
            if i == 0:
                conv_layer = LinearizedConv1d(
                    embed_dim, out_channels, kernel_size, padding=kernel_size - 1
                )
            else:
                conv_layer = LinearizedConv1d(
                    conv_config[i - 1][0],
                    out_channels,
                    kernel_size,
                    padding=kernel_size - 1,
                )
            self.conv_layers.append(conv_layer)
            if layer_norm:
                self.conv_layers.append(nn.LayerNorm(out_channels))
            self.conv_layers.append(nn.ReLU())

        self.layers = nn.ModuleList()
        if conv_config[-1][0] != transformer_config[0][0]:
            self.layers.append(Linear(conv_config[-1][0], transformer_config[0][0]))
        self.layers.append(TransformerDecoderLayer(
            prepare_transformer_decoder_params(*transformer_config[0])
        ))

        for i in range(1, len(transformer_config)):
            if transformer_config[i - 1][0] != transformer_config[i][0]:
                self.layers.append(
                    Linear(transformer_config[i - 1][0], transformer_config[i][0])
                )
            self.layers.append(TransformerDecoderLayer(
                prepare_transformer_decoder_params(*transformer_config[i])
            ))
        self.fc_out = Linear(transformer_config[-1][0], vocab_size)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        target_padding_mask = (
            (prev_output_tokens == self.padding_idx).to(prev_output_tokens.device)
            if incremental_state is None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)

        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)

        for layer in self.conv_layers:
            if isinstance(layer, LinearizedConvolution):
                x = layer(x, incremental_state)
            else:
                x = layer(x)

        # B x T x C -> T x B x C
        x = self._transpose_if_inference(x, incremental_state)

        # decoder layers
        for layer in self.layers:
            if isinstance(layer, TransformerDecoderLayer):
                x, *_ = layer(
                    x,
                    (encoder_out["encoder_out"] if encoder_out is not None else None),
                    (
                        encoder_out["encoder_padding_mask"].t()
                        if encoder_out["encoder_padding_mask"] is not None
                        else None
                    ),
                    incremental_state,
                    self_attn_mask=(
                        self.buffered_future_mask(x)
                        if incremental_state is None
                        else None
                    ),
                    self_attn_padding_mask=(
                        target_padding_mask if incremental_state is None else None
                    ),
                )
            else:
                x = layer(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        x = self.fc_out(x)

        return x, None

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

    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x

    def _transpose_if_inference(self, x, incremental_state):
        if incremental_state:
            x = x.transpose(0, 1)
        return x

@register_model("asr_vggtransformer_encoder")
class VGGTransformerEncoderModel(FairseqEncoderModel):
    def __init__(self, encoder):
        super().__init__(encoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="encoder input dimension per input channel",
        )
        parser.add_argument(
            "--vggblock-enc-config",
            type=str,
            metavar="EXPR",
            help="""
    an array of tuples each containing the configuration of one vggblock
    [(out_channels, conv_kernel_size, pooling_kernel_size,num_conv_layers), ...]
    """,
        )
        parser.add_argument(
            "--transformer-enc-config",
            type=str,
            metavar="EXPR",
            help="""
    a tuple containing the configuration of the Transformer layers
    configurations:
    [(input_dim,
      num_heads,
      ffn_dim,
      normalize_before,
      dropout,
      attention_dropout,
      relu_dropout), ]""",
        )
        parser.add_argument(
            "--enc-output-dim",
            type=int,
            metavar="N",
            help="encoder output dimension, projecting the LSTM output",
        )
        parser.add_argument(
            "--in-channels",
            type=int,
            metavar="N",
            help="number of encoder input channels",
        )
        parser.add_argument(
            "--transformer-context",
            type=str,
            metavar="EXPR",
            help="""
    either None or a tuple of two ints, indicating left/right context a
    transformer can have access to""",
        )
        parser.add_argument(
            "--transformer-sampling",
            type=str,
            metavar="EXPR",
            help="""
    either None or a tuple of ints, indicating sampling factor in each layer""",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture_enconly(args)
        encoder = VGGTransformerEncoderOnly(
            vocab_size=len(task.target_dictionary),
            input_feat_per_channel=args.input_feat_per_channel,
            vggblock_config=eval(args.vggblock_enc_config),
            transformer_config=eval(args.transformer_enc_config),
            encoder_output_dim=args.enc_output_dim,
            in_channels=args.in_channels,
            transformer_context=eval(args.transformer_context),
            transformer_sampling=eval(args.transformer_sampling),
        )
        return cls(encoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (T, B, D) tensor
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        # lprobs is a (T, B, D) tensor
        # we need to transoose to get (B, T, D) tensor
        lprobs = lprobs.transpose(0, 1).contiguous()
        lprobs.batch_first = True
        return lprobs


class VGGTransformerEncoderOnly(VGGTransformerEncoder):
    def __init__(
        self,
        vocab_size,
        input_feat_per_channel,
        vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG,
        transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
        encoder_output_dim=512,
        in_channels=1,
        transformer_context=None,
        transformer_sampling=None,
    ):
        super().__init__(
            input_feat_per_channel=input_feat_per_channel,
            vggblock_config=vggblock_config,
            transformer_config=transformer_config,
            encoder_output_dim=encoder_output_dim,
            in_channels=in_channels,
            transformer_context=transformer_context,
            transformer_sampling=transformer_sampling,
        )
        self.fc_out = Linear(self.encoder_output_dim, vocab_size)

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        src_tokens: padded tensor (B, T, C * feat)
        src_lengths: tensor of original lengths of input utterances (B,)
        """

        enc_out = super().forward(src_tokens, src_lengths)
        x = self.fc_out(enc_out["encoder_out"])
        # x = F.log_softmax(x, dim=-1)
        # Note: no need this line, because model.get_normalized_prob will call
        # log_softmax
        return {
            "encoder_out": x,  # (T, B, C)
            "encoder_padding_mask": enc_out["encoder_padding_mask"],  # (T, B)
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (1e6, 1e6)  # an arbitrary large number


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    # nn.init.uniform_(m.weight, -0.1, 0.1)
    # nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    # m.weight.data.uniform_(-0.1, 0.1)
    # if bias:
    #     m.bias.data.uniform_(-0.1, 0.1)
    return m


def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


# seq2seq models
def base_architecture(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 40)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", DEFAULT_ENC_VGGBLOCK_CONFIG
    )
    args.transformer_enc_config = getattr(
        args, "transformer_enc_config", DEFAULT_ENC_TRANSFORMER_CONFIG
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.in_channels = getattr(args, "in_channels", 1)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 128)
    args.transformer_dec_config = getattr(
        args, "transformer_dec_config", DEFAULT_ENC_TRANSFORMER_CONFIG
    )
    args.conv_dec_config = getattr(args, "conv_dec_config", DEFAULT_DEC_CONV_CONFIG)
    args.transformer_context = getattr(args, "transformer_context", "None")


@register_model_architecture("asr_vggtransformer", "vggtransformer_1")
def vggtransformer_1(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.transformer_enc_config = getattr(
        args,
        "transformer_enc_config",
        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 14",
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 1024)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 128)
    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
    args.transformer_dec_config = getattr(
        args,
        "transformer_dec_config",
        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 4",
    )


@register_model_architecture("asr_vggtransformer", "vggtransformer_2")
def vggtransformer_2(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.transformer_enc_config = getattr(
        args,
        "transformer_enc_config",
        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 16",
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 1024)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
    args.transformer_dec_config = getattr(
        args,
        "transformer_dec_config",
        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 6",
    )


@register_model_architecture("asr_vggtransformer", "vggtransformer_base")
def vggtransformer_base(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.transformer_enc_config = getattr(
        args, "transformer_enc_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 12"
    )

    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
    args.transformer_dec_config = getattr(
        args, "transformer_dec_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 6"
    )
    # Size estimations:
    # Encoder:
    #   - vggblock param: 64*1*3*3 + 64*64*3*3 + 128*64*3*3  + 128*128*3 = 258K
    #   Transformer:
    #   - input dimension adapter: 2560 x 512 -> 1.31M
    #   - transformer_layers (x12) --> 37.74M
    #       * MultiheadAttention: 512*512*3 (in_proj) + 512*512 (out_proj) = 1.048M
    #       * FFN weight: 512*2048*2 = 2.097M
    #   - output dimension adapter: 512 x 512 -> 0.26 M
    # Decoder:
    #   - LinearizedConv1d: 512 * 256 * 3 + 256 * 256 * 3 * 3
    #   - transformer_layer: (x6) --> 25.16M
    #        * MultiheadAttention (self-attention): 512*512*3 + 512*512 = 1.048M
    #        * MultiheadAttention (encoder-attention): 512*512*3 + 512*512 = 1.048M
    #        * FFN: 512*2048*2 = 2.097M
    # Final FC:
    #   - FC: 512*5000 = 256K (assuming vocab size 5K)
    # In total:
    #       ~65 M


# CTC models
def base_architecture_enconly(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 40)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(32, 3, 2, 2, True)] * 2"
    )
    args.transformer_enc_config = getattr(
        args, "transformer_enc_config", "((256, 4, 1024, True, 0.2, 0.2, 0.2),) * 2"
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.in_channels = getattr(args, "in_channels", 1)
    args.transformer_context = getattr(args, "transformer_context", "None")
    args.transformer_sampling = getattr(args, "transformer_sampling", "None")


@register_model_architecture("asr_vggtransformer_encoder", "vggtransformer_enc_1")
def vggtransformer_enc_1(args):
    # vggtransformer_1 is the same as vggtransformer_enc_big, except the number
    # of layers is increased to 16
    # keep it here for backward compatiablity purpose
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.transformer_enc_config = getattr(
        args,
        "transformer_enc_config",
        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 16",
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 1024)
