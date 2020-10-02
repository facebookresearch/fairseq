#!/usr/bin/env python3

from ast import literal_eval
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqModel,
    register_model,
    register_model_architecture,
)
from .berard import BerardModel

from examples.simultaneous_translation.modules import (
    build_monotonic_attention
)


@register_model("berard_simul")
class BerardSimulASTModel(BerardModel):
    @staticmethod
    def add_args(parser):
        super(BerardSimulASTModel, BerardSimulASTModel).add_args(parser)


    @classmethod
    def build_encoder(cls, args, task):
        if getattr(args, 'encoder_hidden_size', None) is None:
            args.encoder_hidden_size = args.lstm_size
        encoder = BerardSimulEncoder(
            input_layers=literal_eval(args.input_layers),
            conv_layers=literal_eval(args.conv_layers),
            in_channels=args.in_channels,
            input_feat_per_channel=args.input_feat_per_channel,
            num_lstm_layers=args.num_lstm_layers,
            lstm_size=args.encoder_hidden_size,
            dropout=args.dropout,
            use_energy=getattr(args, "use_energy", False),
        )
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task):
        decoder = LSTMSimulDecoder(args, task.target_dictionary)
        if getattr(args, "load_pretrained_decoder_from", None):
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

@register_model("berard_simul_text")
class BerardSimulTextModel(BerardSimulASTModel):
    @staticmethod
    def add_args(parser):
        super(BerardSimulTextModel, BerardSimulTextModel).add_args(parser)
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
    @classmethod
    def build_encoder(cls, args, task):
        from fairseq.models.lstm import LSTMEncoder
        args.encoder_hidden_size = args.lstm_size
        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.lstm_size,
            num_layers=args.num_lstm_layers,
            dropout_in=args.dropout,
            dropout_out=args.dropout,
            bidirectional=False,
            pretrained_embed=None,
        )
        return encoder


class BerardSimulEncoder(FairseqEncoder):

    def __init__(
        self,
        input_layers: List[int],
        conv_layers: List[Tuple[int]],
        in_channels: int,
        input_feat_per_channel: int,
        num_lstm_layers: int,
        lstm_size: int,
        dropout: float,
        use_energy: float,
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
            in_channels = out_channels
            lstm_input_dim //= conv_stride

        lstm_input_dim *= conv_layers[-1][0]
        self.lstm_size = lstm_size
        self.num_lstm_layers = num_lstm_layers
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_size,
            num_layers=num_lstm_layers,
            dropout=dropout,
            bidirectional=False,
        )
        self.output_dim = lstm_size  # bidirectional
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.use_energy = use_energy

    def forward(self, src_tokens, src_lengths, **kwargs):
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

        segmentation = kwargs.get("segmentation", None)

        for input_layer in self.input_layers:
            x = input_layer(x)
            x = torch.tanh(x)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            if segmentation is not None:
                kernal_size = conv_layer.kernel_size[0]
                stride = conv_layer.stride[0]
                padding = conv_layer.padding[0]
                segmentation = nn.functional.max_pool1d(
                    segmentation.float().unsqueeze(1),
                    kernel_size=kernal_size,
                    stride=stride,
                    padding=padding
                ).squeeze(1)

        if segmentation is not None:
            segmentation = segmentation.t()


        bsz, _, output_seq_len, _ = x.size()

        # (B, C, T, feat) -> (B, T, C, feat) -> (T, B, C, feat) ->
        # (T, B, C * feat)
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)

        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
        input_lengths = (src_lengths.float() / subsampling_factor).ceil().long()

        packed_x = nn.utils.rnn.pack_padded_sequence(x, input_lengths.data.tolist())

        h0 = x.new(self.num_lstm_layers, bsz, self.lstm_size).zero_()
        c0 = x.new(self.num_lstm_layers, bsz, self.lstm_size).zero_()
        packed_outs, _ = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_outs)
        if self.dropout is not None:
            x = self.dropout(x)

        # need to debug this -- find a simpler/elegant way in pytorch APIs
        encoder_padding_mask = (
            (
                torch.arange(output_seq_len).view(1, output_seq_len).expand(bsz, -1)
                >= output_lengths.view(bsz, 1).expand(-1, output_seq_len)
            )
            .t()
            .to(x.device)
        )  # (B x T) -> (T x B)

        return {
            "encoder_out": x,
            "encoder_padding_mask": encoder_padding_mask,  # (T, B)
            "encoder_segmentation": segmentation
        }  # (T, B, C)  # (B, )

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
            1, new_order
        )

        encoder_out["encoder_padding_mask"] = encoder_out[
            "encoder_padding_mask"
        ].index_select(1, new_order)

        return encoder_out


class LSTMSimulDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary):
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
        self.num_layers = args.decoder_num_layers
        self.hidden_size = args.decoder_hidden_dim
        self.embed_dim = args.decoder_embed_dim
        encoder_output_dim = args.encoder_hidden_size
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(num_embeddings, self.embed_dim, self.padding_idx)
        if getattr(args, 'dropout', 0.0) > 0.0:
            self.dropout = nn.Dropout(p=args.dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for layer_id in range(self.num_layers):
            input_size = self.embed_dim if layer_id == 0 else encoder_output_dim
            self.layers.append(
                nn.LSTMCell(input_size=input_size, hidden_size=self.hidden_size)
            )

        self.context_dim = encoder_output_dim
        self.simul_type = args.simul_type
        self.attention = build_monotonic_attention(args)

        self.deep_output_layer = nn.Linear(
            self.hidden_size + encoder_output_dim + self.embed_dim,
            args.output_layer_dim
        )
        self.output_projection = nn.Linear(args.output_layer_dim, num_embeddings)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, **kwargs):
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
            prev_hiddens = [x.new_zeros(bsz, self.hidden_size)] * self.num_layers
            prev_cells = [x.new_zeros(bsz, self.hidden_size)] * self.num_layers

        #attn_scores = x.new_zeros(bsz, srclen)
        prev_alpha = None
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
                    if incremental_state is None:
                        self.attention.set_target_step(j)
                    attention_out = self.attention(
                        hidden,
                        encoder_outs,
                        prev_alpha,
                        incremental_state
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
        # tgt_len, bsz, src_len


        return x, {'encoder_padding_mask' : encoder_padding_mask}

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

        step_list = utils.get_incremental_state(self.attention, incremental_state, "step")
        for i, step in enumerate(step_list):
            step_list[i] = step.index_select(1, new_order)
        utils.set_incremental_state(self.attention, incremental_state, "step", step_list)


@register_model_architecture(model_name="berard_simul", arch_name="berard_simul_ast")
def berard_simul_ast(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.in_channels = getattr(args, "in_channels", 1)
    args.input_layers = getattr(args, "input_layers", "[256, 128]")
    args.conv_layers = getattr(args, "conv_layers", "[(16, 3, 2), (16, 3, 2)]")
    args.num_lstm_layers = getattr(args, "num_lstm_layers", 3)
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
    args.use_energy = getattr(args, "use_energy", False)

@register_model_architecture(model_name="berard_simul_text", arch_name="berard_simul_text_iwslt")
def berard_simul_mt(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.num_lstm_layers = getattr(args, "num_lstm_layers", 2)
    args.lstm_size = getattr(args, "lstm_size", 512)
    args.dropout = getattr(args, "dropout", 0.3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
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
