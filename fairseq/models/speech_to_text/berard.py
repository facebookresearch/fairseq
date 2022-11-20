#!/usr/bin/env python3

from ast import literal_eval
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)


@register_model("s2t_berard")
class BerardModel(FairseqEncoderDecoderModel):
    """Implementation of a model similar to https://arxiv.org/abs/1802.04200

    Paper title: End-to-End Automatic Speech Translation of Audiobooks
    An implementation is available in tensorflow at
    https://github.com/eske/seq2seq
    Relevant files in this implementation are the config
    (https://github.com/eske/seq2seq/blob/master/config/LibriSpeech/AST.yaml)
    and the model code
    (https://github.com/eske/seq2seq/blob/master/translate/models.py).
    The encoder and decoder try to be close to the original implementation.
    The attention is an MLP as in Bahdanau et al.
    (https://arxiv.org/abs/1409.0473).
    There is no state initialization by averaging the encoder outputs.
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--input-layers",
            type=str,
            metavar="EXPR",
            help="List of linear layer dimensions. These "
            "layers are applied to the input features and "
            "are followed by tanh and possibly dropout.",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            metavar="D",
            help="Dropout probability to use in the encoder/decoder. "
            "Note that this parameters control dropout in various places, "
            "there is no fine-grained control for dropout for embeddings "
            "vs LSTM layers for example.",
        )
        parser.add_argument(
            "--in-channels",
            type=int,
            metavar="N",
            help="Number of encoder input channels. " "Typically value is 1.",
        )
        parser.add_argument(
            "--conv-layers",
            type=str,
            metavar="EXPR",
            help="List of conv layers " "(format: (channels, kernel, stride)).",
        )
        parser.add_argument(
            "--num-blstm-layers",
            type=int,
            metavar="N",
            help="Number of encoder bi-LSTM layers.",
        )
        parser.add_argument(
            "--lstm-size", type=int, metavar="N", help="LSTM hidden size."
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="Embedding dimension of the decoder target tokens.",
        )
        parser.add_argument(
            "--decoder-hidden-dim",
            type=int,
            metavar="N",
            help="Decoder LSTM hidden dimension.",
        )
        parser.add_argument(
            "--decoder-num-layers",
            type=int,
            metavar="N",
            help="Number of decoder LSTM layers.",
        )
        parser.add_argument(
            "--attention-dim",
            type=int,
            metavar="N",
            help="Hidden layer dimension in MLP attention.",
        )
        parser.add_argument(
            "--output-layer-dim",
            type=int,
            metavar="N",
            help="Hidden layer dim for linear layer prior to output projection.",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )

    @classmethod
    def build_encoder(cls, args, task):
        encoder = BerardEncoder(
            input_layers=literal_eval(args.input_layers),
            conv_layers=literal_eval(args.conv_layers),
            in_channels=args.input_channels,
            input_feat_per_channel=args.input_feat_per_channel,
            num_blstm_layers=args.num_blstm_layers,
            lstm_size=args.lstm_size,
            dropout=args.dropout,
        )
        if getattr(args, "load_pretrained_encoder_from", None) is not None:
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task):
        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            num_layers=args.decoder_num_layers,
            hidden_size=args.decoder_hidden_dim,
            dropout=args.dropout,
            encoder_output_dim=2 * args.lstm_size,  # bidirectional
            attention_dim=args.attention_dim,
            output_layer_dim=args.output_layer_dim,
        )
        if getattr(args, "load_pretrained_decoder_from", None) is not None:
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task)

        return cls(encoder, decoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        # lprobs is a (B, T, D) tensor
        lprobs.batch_first = True
        return lprobs


class BerardEncoder(FairseqEncoder):
    def __init__(
        self,
        input_layers: List[int],
        conv_layers: List[Tuple[int]],
        in_channels: int,
        input_feat_per_channel: int,
        num_blstm_layers: int,
        lstm_size: int,
        dropout: float,
    ):
        """
        Args:
            input_layers: list of linear layer dimensions. These layers are
                applied to the input features and are followed by tanh and
                possibly dropout.
            conv_layers: list of conv2d layer configurations. A configuration is
                a tuple (out_channels, conv_kernel_size, stride).
            in_channels: number of input channels.
            input_feat_per_channel: number of input features per channel. These
                are speech features, typically 40 or 80.
            num_blstm_layers: number of bidirectional LSTM layers.
            lstm_size: size of the LSTM hidden (and cell) size.
            dropout: dropout probability. Dropout can be applied after the
                linear layers and LSTM layers but not to the convolutional
                layers.
        """
        super().__init__(None)

        self.input_layers = nn.ModuleList()
        in_features = input_feat_per_channel
        for out_features in input_layers:
            if dropout > 0:
                self.input_layers.append(
                    nn.Sequential(
                        nn.Linear(in_features, out_features), nn.Dropout(p=dropout)
                    )
                )
            else:
                self.input_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

        self.in_channels = in_channels
        self.input_dim = input_feat_per_channel
        self.conv_kernel_sizes_and_strides = []
        self.conv_layers = nn.ModuleList()
        lstm_input_dim = input_layers[-1]
        for conv_layer in conv_layers:
            out_channels, conv_kernel_size, conv_stride = conv_layer
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    stride=conv_stride,
                    padding=conv_kernel_size // 2,
                )
            )
            self.conv_kernel_sizes_and_strides.append((conv_kernel_size, conv_stride))
            in_channels = out_channels
            lstm_input_dim //= conv_stride

        lstm_input_dim *= conv_layers[-1][0]
        self.lstm_size = lstm_size
        self.num_blstm_layers = num_blstm_layers
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_size,
            num_layers=num_blstm_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.output_dim = 2 * lstm_size  # bidirectional
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        """
        Args
            src_tokens: padded tensor (B, T, C * feat)
            src_lengths: tensor of original lengths of input utterances (B,)
        """
        bsz, max_seq_len, _ = src_tokens.size()
        # (B, C, T, feat)
        x = (
            src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
            .transpose(1, 2)
            .contiguous()
        )

        for input_layer in self.input_layers:
            x = input_layer(x)
            x = torch.tanh(x)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        bsz, _, output_seq_len, _ = x.size()

        # (B, C, T, feat) -> (B, T, C, feat) -> (T, B, C, feat) ->
        # (T, B, C * feat)
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)

        input_lengths = src_lengths.clone()
        for k, s in self.conv_kernel_sizes_and_strides:
            p = k // 2
            input_lengths = (input_lengths.float() + 2 * p - k) / s + 1
            input_lengths = input_lengths.floor().long()

        packed_x = nn.utils.rnn.pack_padded_sequence(x, input_lengths)

        h0 = x.new(2 * self.num_blstm_layers, bsz, self.lstm_size).zero_()
        c0 = x.new(2 * self.num_blstm_layers, bsz, self.lstm_size).zero_()
        packed_outs, _ = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_outs)
        if self.dropout is not None:
            x = self.dropout(x)

        encoder_padding_mask = (
            lengths_to_padding_mask(output_lengths).to(src_tokens.device).t()
        )

        return {
            "encoder_out": x,  # (T, B, C)
            "encoder_padding_mask": encoder_padding_mask,  # (T, B)
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
            1, new_order
        )
        encoder_out["encoder_padding_mask"] = encoder_out[
            "encoder_padding_mask"
        ].index_select(1, new_order)
        return encoder_out


class MLPAttention(nn.Module):
    """The original attention from Badhanau et al. (2014)

    https://arxiv.org/abs/1409.0473, based on a Multi-Layer Perceptron.
    The attention score between position i in the encoder and position j in the
    decoder is: alpha_ij = V_a * tanh(W_ae * enc_i + W_ad * dec_j + b_a)
    """

    def __init__(self, decoder_hidden_state_dim, context_dim, attention_dim):
        super().__init__()

        self.context_dim = context_dim
        self.attention_dim = attention_dim
        # W_ae and b_a
        self.encoder_proj = nn.Linear(context_dim, self.attention_dim, bias=True)
        # W_ad
        self.decoder_proj = nn.Linear(
            decoder_hidden_state_dim, self.attention_dim, bias=False
        )
        # V_a
        self.to_scores = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, decoder_state, source_hids, encoder_padding_mask):
        """The expected input dimensions are:
        decoder_state: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        """
        src_len, bsz, _ = source_hids.size()
        # (src_len*bsz) x context_dim (to feed through linear)
        flat_source_hids = source_hids.view(-1, self.context_dim)
        # (src_len*bsz) x attention_dim
        encoder_component = self.encoder_proj(flat_source_hids)
        # src_len x bsz x attention_dim
        encoder_component = encoder_component.view(src_len, bsz, self.attention_dim)
        # 1 x bsz x attention_dim
        decoder_component = self.decoder_proj(decoder_state).unsqueeze(0)
        # Sum with broadcasting and apply the non linearity
        # src_len x bsz x attention_dim
        hidden_att = torch.tanh(
            (decoder_component + encoder_component).view(-1, self.attention_dim)
        )
        # Project onto the reals to get attentions scores (src_len x bsz)
        attn_scores = self.to_scores(hidden_att).view(src_len, bsz)

        # Mask + softmax (src_len x bsz)
        if encoder_padding_mask is not None:
            attn_scores = (
                attn_scores.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attn_scores)
            )  # FP16 support: cast to float and back
        # srclen x bsz
        normalized_masked_attn_scores = F.softmax(attn_scores, dim=0)

        # Sum weighted sources (bsz x context_dim)
        attn_weighted_context = (
            source_hids * normalized_masked_attn_scores.unsqueeze(2)
        ).sum(dim=0)

        return attn_weighted_context, normalized_masked_attn_scores


class LSTMDecoder(FairseqIncrementalDecoder):
    def __init__(
        self,
        dictionary,
        embed_dim,
        num_layers,
        hidden_size,
        dropout,
        encoder_output_dim,
        attention_dim,
        output_layer_dim,
    ):
        """
        Args:
            dictionary: target text dictionary.
            embed_dim: embedding dimension for target tokens.
            num_layers: number of LSTM layers.
            hidden_size: hidden size for LSTM layers.
            dropout: dropout probability. Dropout can be applied to the
                embeddings, the LSTM layers, and the context vector.
            encoder_output_dim: encoder output dimension (hidden size of
                encoder LSTM).
            attention_dim: attention dimension for MLP attention.
            output_layer_dim: size of the linear layer prior to output
                projection.
        """
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for layer_id in range(num_layers):
            input_size = embed_dim if layer_id == 0 else encoder_output_dim
            self.layers.append(
                nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
            )

        self.context_dim = encoder_output_dim
        self.attention = MLPAttention(
            decoder_hidden_state_dim=hidden_size,
            context_dim=encoder_output_dim,
            attention_dim=attention_dim,
        )

        self.deep_output_layer = nn.Linear(
            hidden_size + encoder_output_dim + embed_dim, output_layer_dim
        )
        self.output_projection = nn.Linear(output_layer_dim, num_embeddings)

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs
    ):
        encoder_padding_mask = encoder_out["encoder_padding_mask"]
        encoder_outs = encoder_out["encoder_out"]

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        srclen = encoder_outs.size(0)

        # embed tokens
        embeddings = self.embed_tokens(prev_output_tokens)
        x = embeddings
        if self.dropout is not None:
            x = self.dropout(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental
        # generation)
        cached_state = utils.get_incremental_state(
            self, incremental_state, "cached_state"
        )
        if cached_state is not None:
            prev_hiddens, prev_cells = cached_state
        else:
            prev_hiddens = [encoder_out["encoder_out"].mean(dim=0)] * self.num_layers
            prev_cells = [x.new_zeros(bsz, self.hidden_size)] * self.num_layers

        attn_scores = x.new_zeros(bsz, srclen)
        attention_outs = []
        outs = []
        for j in range(seqlen):
            input = x[j, :, :]
            attention_out = None
            for i, layer in enumerate(self.layers):
                # the previous state is one layer below except for the bottom
                # layer where the previous state is the state emitted by the
                # top layer
                hidden, cell = layer(
                    input,
                    (
                        prev_hiddens[(i - 1) % self.num_layers],
                        prev_cells[(i - 1) % self.num_layers],
                    ),
                )
                if self.dropout is not None:
                    hidden = self.dropout(hidden)
                prev_hiddens[i] = hidden
                prev_cells[i] = cell
                if attention_out is None:
                    attention_out, attn_scores = self.attention(
                        hidden, encoder_outs, encoder_padding_mask
                    )
                    if self.dropout is not None:
                        attention_out = self.dropout(attention_out)
                    attention_outs.append(attention_out)
                input = attention_out

            # collect the output of the top layer
            outs.append(hidden)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, "cached_state", (prev_hiddens, prev_cells)
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)
        attention_outs_concat = torch.cat(attention_outs, dim=0).view(
            seqlen, bsz, self.context_dim
        )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        attention_outs_concat = attention_outs_concat.transpose(0, 1)

        # concat LSTM output, attention output and embedding
        # before output projection
        x = torch.cat((x, attention_outs_concat, embeddings), dim=2)
        x = self.deep_output_layer(x)
        x = torch.tanh(x)
        if self.dropout is not None:
            x = self.dropout(x)
        # project back to size of vocabulary
        x = self.output_projection(x)

        # to return the full attn_scores tensor, we need to fix the decoder
        # to account for subsampling input frames
        # return x, attn_scores
        return x, None

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(
            self, incremental_state, "cached_state"
        )
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, "cached_state", new_state)


@register_model_architecture(model_name="s2t_berard", arch_name="s2t_berard")
def berard(args):
    """The original version: "End-to-End Automatic Speech Translation of
    Audiobooks" (https://arxiv.org/abs/1802.04200)
    """
    args.input_layers = getattr(args, "input_layers", "[256, 128]")
    args.conv_layers = getattr(args, "conv_layers", "[(16, 3, 2), (16, 3, 2)]")
    args.num_blstm_layers = getattr(args, "num_blstm_layers", 3)
    args.lstm_size = getattr(args, "lstm_size", 256)
    args.dropout = getattr(args, "dropout", 0.2)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 128)
    args.decoder_num_layers = getattr(args, "decoder_num_layers", 2)
    args.decoder_hidden_dim = getattr(args, "decoder_hidden_dim", 512)
    args.attention_dim = getattr(args, "attention_dim", 512)
    args.output_layer_dim = getattr(args, "output_layer_dim", 128)
    args.load_pretrained_encoder_from = getattr(
        args, "load_pretrained_encoder_from", None
    )
    args.load_pretrained_decoder_from = getattr(
        args, "load_pretrained_decoder_from", None
    )


@register_model_architecture(model_name="s2t_berard", arch_name="s2t_berard_256_3_3")
def berard_256_3_3(args):
    """Used in
    * "Harnessing Indirect Training Data for End-to-End Automatic Speech
    Translation: Tricks of the Trade" (https://arxiv.org/abs/1909.06515)
    * "CoVoST: A Diverse Multilingual Speech-To-Text Translation Corpus"
    (https://arxiv.org/pdf/2002.01320.pdf)
    * "Self-Supervised Representations Improve End-to-End Speech Translation"
    (https://arxiv.org/abs/2006.12124)
    """
    args.decoder_num_layers = getattr(args, "decoder_num_layers", 3)
    berard(args)


@register_model_architecture(model_name="s2t_berard", arch_name="s2t_berard_512_3_2")
def berard_512_3_2(args):
    args.num_blstm_layers = getattr(args, "num_blstm_layers", 3)
    args.lstm_size = getattr(args, "lstm_size", 512)
    args.dropout = getattr(args, "dropout", 0.3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_num_layers = getattr(args, "decoder_num_layers", 2)
    args.decoder_hidden_dim = getattr(args, "decoder_hidden_dim", 1024)
    args.attention_dim = getattr(args, "attention_dim", 512)
    args.output_layer_dim = getattr(args, "output_layer_dim", 256)
    berard(args)


@register_model_architecture(model_name="s2t_berard", arch_name="s2t_berard_512_5_3")
def berard_512_5_3(args):
    args.num_blstm_layers = getattr(args, "num_blstm_layers", 5)
    args.lstm_size = getattr(args, "lstm_size", 512)
    args.dropout = getattr(args, "dropout", 0.3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_num_layers = getattr(args, "decoder_num_layers", 3)
    args.decoder_hidden_dim = getattr(args, "decoder_hidden_dim", 1024)
    args.attention_dim = getattr(args, "attention_dim", 512)
    args.output_layer_dim = getattr(args, "output_layer_dim", 256)
    berard(args)
