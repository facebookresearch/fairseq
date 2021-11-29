# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torch import nn
from torch.nn import functional as F

from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import LSTMCellWithZoneOut, LocationAttention


logger = logging.getLogger(__name__)


def encoder_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


class Tacotron2Encoder(FairseqEncoder):
    def __init__(self, args, src_dict, embed_speaker):
        super().__init__(src_dict)
        self.padding_idx = src_dict.pad()
        self.embed_speaker = embed_speaker
        self.spk_emb_proj = None
        if embed_speaker is not None:
            self.spk_emb_proj = nn.Linear(
                args.encoder_embed_dim + args.speaker_embed_dim, args.encoder_embed_dim
            )

        self.embed_tokens = nn.Embedding(
            len(src_dict), args.encoder_embed_dim, padding_idx=self.padding_idx
        )

        assert args.encoder_conv_kernel_size % 2 == 1
        self.convolutions = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    args.encoder_embed_dim,
                    args.encoder_embed_dim,
                    kernel_size=args.encoder_conv_kernel_size,
                    padding=((args.encoder_conv_kernel_size - 1) // 2),
                ),
                nn.BatchNorm1d(args.encoder_embed_dim),
                nn.ReLU(),
                nn.Dropout(args.encoder_dropout),
            )
            for _ in range(args.encoder_conv_layers)
        )

        self.lstm = nn.LSTM(
            args.encoder_embed_dim,
            args.encoder_embed_dim // 2,
            num_layers=args.encoder_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.apply(encoder_init)

    def forward(self, src_tokens, src_lengths=None, speaker=None, **kwargs):
        x = self.embed_tokens(src_tokens)
        x = x.transpose(1, 2).contiguous()  # B x T x C -> B x C x T
        for conv in self.convolutions:
            x = conv(x)
        x = x.transpose(1, 2).contiguous()  # B x C x T -> B x T x C

        src_lengths = src_lengths.cpu().long()
        x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, batch_first=True)
        x = self.lstm(x)[0]
        x = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)[0]

        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        if self.embed_speaker is not None:
            seq_len, bsz, _ = x.size()
            emb = self.embed_speaker(speaker).expand(seq_len, bsz, -1)
            x = self.spk_emb_proj(torch.cat([x, emb], dim=2))

        return {
            "encoder_out": [x],  # B x T x C
            "encoder_padding_mask": encoder_padding_mask,  # B x T
        }


class Prenet(nn.Module):
    def __init__(self, in_dim, n_layers, n_units, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(in_dim if i == 0 else n_units, n_units), nn.ReLU())
            for i in range(n_layers)
        )
        self.dropout = dropout

    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(layer(x), p=self.dropout)  # always applies dropout
        return x


class Postnet(nn.Module):
    def __init__(self, in_dim, n_channels, kernel_size, n_layers, dropout):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        assert kernel_size % 2 == 1
        for i in range(n_layers):
            cur_layers = (
                [
                    nn.Conv1d(
                        in_dim if i == 0 else n_channels,
                        n_channels if i < n_layers - 1 else in_dim,
                        kernel_size=kernel_size,
                        padding=((kernel_size - 1) // 2),
                    ),
                    nn.BatchNorm1d(n_channels if i < n_layers - 1 else in_dim),
                ]
                + ([nn.Tanh()] if i < n_layers - 1 else [])
                + [nn.Dropout(dropout)]
            )
            nn.init.xavier_uniform_(
                cur_layers[0].weight,
                torch.nn.init.calculate_gain("tanh" if i < n_layers - 1 else "linear"),
            )
            self.convolutions.append(nn.Sequential(*cur_layers))

    def forward(self, x):
        x = x.transpose(1, 2)  # B x T x C -> B x C x T
        for conv in self.convolutions:
            x = conv(x)
        return x.transpose(1, 2)


def decoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("tanh"))


class Tacotron2Decoder(FairseqIncrementalDecoder):
    def __init__(self, args, src_dict):
        super().__init__(None)
        self.args = args
        self.n_frames_per_step = args.n_frames_per_step
        self.out_dim = args.output_frame_dim * args.n_frames_per_step

        self.prenet = Prenet(
            self.out_dim, args.prenet_layers, args.prenet_dim, args.prenet_dropout
        )

        # take prev_context, prev_frame, (speaker embedding) as input
        self.attention_lstm = LSTMCellWithZoneOut(
            args.zoneout,
            args.prenet_dim + args.encoder_embed_dim,
            args.decoder_lstm_dim,
        )

        # take attention_lstm output, attention_state, encoder_out as input
        self.attention = LocationAttention(
            args.attention_dim,
            args.encoder_embed_dim,
            args.decoder_lstm_dim,
            (1 + int(args.attention_use_cumprob)),
            args.attention_conv_dim,
            args.attention_conv_kernel_size,
        )

        # take attention_lstm output, context, (gated_latent) as input
        self.lstm = nn.ModuleList(
            LSTMCellWithZoneOut(
                args.zoneout,
                args.encoder_embed_dim + args.decoder_lstm_dim,
                args.decoder_lstm_dim,
            )
            for i in range(args.decoder_lstm_layers)
        )

        proj_in_dim = args.encoder_embed_dim + args.decoder_lstm_dim
        self.feat_proj = nn.Linear(proj_in_dim, self.out_dim)
        self.eos_proj = nn.Linear(proj_in_dim, 1)

        self.postnet = Postnet(
            self.out_dim,
            args.postnet_conv_dim,
            args.postnet_conv_kernel_size,
            args.postnet_layers,
            args.postnet_dropout,
        )

        self.ctc_proj = None
        if getattr(args, "ctc_weight", 0.0) > 0.0:
            self.ctc_proj = nn.Linear(self.out_dim, len(src_dict))

        self.apply(decoder_init)

    def _get_states(self, incremental_state, enc_out):
        bsz, in_len, _ = enc_out.size()
        alstm_h = self.get_incremental_state(incremental_state, "alstm_h")
        if alstm_h is None:
            alstm_h = enc_out.new_zeros(bsz, self.args.decoder_lstm_dim)
        alstm_c = self.get_incremental_state(incremental_state, "alstm_c")
        if alstm_c is None:
            alstm_c = enc_out.new_zeros(bsz, self.args.decoder_lstm_dim)

        lstm_h = self.get_incremental_state(incremental_state, "lstm_h")
        if lstm_h is None:
            lstm_h = [
                enc_out.new_zeros(bsz, self.args.decoder_lstm_dim)
                for _ in range(self.args.decoder_lstm_layers)
            ]
        lstm_c = self.get_incremental_state(incremental_state, "lstm_c")
        if lstm_c is None:
            lstm_c = [
                enc_out.new_zeros(bsz, self.args.decoder_lstm_dim)
                for _ in range(self.args.decoder_lstm_layers)
            ]

        attn_w = self.get_incremental_state(incremental_state, "attn_w")
        if attn_w is None:
            attn_w = enc_out.new_zeros(bsz, in_len)
        attn_w_cum = self.get_incremental_state(incremental_state, "attn_w_cum")
        if attn_w_cum is None:
            attn_w_cum = enc_out.new_zeros(bsz, in_len)
        return alstm_h, alstm_c, lstm_h, lstm_c, attn_w, attn_w_cum

    def _get_init_attn_c(self, enc_out, enc_mask):
        bsz = enc_out.size(0)
        if self.args.init_attn_c == "zero":
            return enc_out.new_zeros(bsz, self.args.encoder_embed_dim)
        elif self.args.init_attn_c == "avg":
            enc_w = (~enc_mask).type(enc_out.type())
            enc_w = enc_w / enc_w.sum(dim=1, keepdim=True)
            return torch.sum(enc_out * enc_w.unsqueeze(2), dim=1)
        else:
            raise ValueError(f"{self.args.init_attn_c} not supported")

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        target_lengths=None,
        **kwargs,
    ):
        enc_mask = encoder_out["encoder_padding_mask"]
        enc_out = encoder_out["encoder_out"][0]
        in_len = enc_out.size(1)

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:, :]
        bsz, out_len, _ = prev_output_tokens.size()

        prenet_out = self.prenet(prev_output_tokens)
        (alstm_h, alstm_c, lstm_h, lstm_c, attn_w, attn_w_cum) = self._get_states(
            incremental_state, enc_out
        )
        attn_ctx = self._get_init_attn_c(enc_out, enc_mask)

        attn_out = enc_out.new_zeros(bsz, in_len, out_len)
        feat_out = enc_out.new_zeros(bsz, out_len, self.out_dim)
        eos_out = enc_out.new_zeros(bsz, out_len)
        for t in range(out_len):
            alstm_in = torch.cat((attn_ctx, prenet_out[:, t, :]), dim=1)
            alstm_h, alstm_c = self.attention_lstm(alstm_in, (alstm_h, alstm_c))

            attn_state = attn_w.unsqueeze(1)
            if self.args.attention_use_cumprob:
                attn_state = torch.stack((attn_w, attn_w_cum), dim=1)
            attn_ctx, attn_w = self.attention(enc_out, enc_mask, alstm_h, attn_state)
            attn_w_cum = attn_w_cum + attn_w
            attn_out[:, :, t] = attn_w

            for i, cur_lstm in enumerate(self.lstm):
                if i == 0:
                    lstm_in = torch.cat((attn_ctx, alstm_h), dim=1)
                else:
                    lstm_in = torch.cat((attn_ctx, lstm_h[i - 1]), dim=1)
                lstm_h[i], lstm_c[i] = cur_lstm(lstm_in, (lstm_h[i], lstm_c[i]))

            proj_in = torch.cat((attn_ctx, lstm_h[-1]), dim=1)
            feat_out[:, t, :] = self.feat_proj(proj_in)
            eos_out[:, t] = self.eos_proj(proj_in).squeeze(1)
        self.attention.clear_cache()

        self.set_incremental_state(incremental_state, "alstm_h", alstm_h)
        self.set_incremental_state(incremental_state, "alstm_c", alstm_c)
        self.set_incremental_state(incremental_state, "lstm_h", lstm_h)
        self.set_incremental_state(incremental_state, "lstm_c", lstm_c)
        self.set_incremental_state(incremental_state, "attn_w", attn_w)
        self.set_incremental_state(incremental_state, "attn_w_cum", attn_w_cum)

        post_feat_out = feat_out + self.postnet(feat_out)
        eos_out = eos_out.view(bsz, out_len, 1)
        return post_feat_out, eos_out, {"attn": attn_out, "feature_out": feat_out}


@register_model("tacotron_2")
class Tacotron2Model(FairseqEncoderDecoderModel):
    """
    Implementation for https://arxiv.org/pdf/1712.05884.pdf
    """

    @staticmethod
    def add_args(parser):
        # encoder
        parser.add_argument("--encoder-dropout", type=float)
        parser.add_argument("--encoder-embed-dim", type=int)
        parser.add_argument("--encoder-conv-layers", type=int)
        parser.add_argument("--encoder-conv-kernel-size", type=int)
        parser.add_argument("--encoder-lstm-layers", type=int)
        # decoder
        parser.add_argument("--attention-dim", type=int)
        parser.add_argument("--attention-conv-dim", type=int)
        parser.add_argument("--attention-conv-kernel-size", type=int)
        parser.add_argument("--prenet-dropout", type=float)
        parser.add_argument("--prenet-layers", type=int)
        parser.add_argument("--prenet-dim", type=int)
        parser.add_argument("--postnet-dropout", type=float)
        parser.add_argument("--postnet-layers", type=int)
        parser.add_argument("--postnet-conv-dim", type=int)
        parser.add_argument("--postnet-conv-kernel-size", type=int)
        parser.add_argument("--init-attn-c", type=str)
        parser.add_argument("--attention-use-cumprob", action="store_true")
        parser.add_argument("--zoneout", type=float)
        parser.add_argument("--decoder-lstm-layers", type=int)
        parser.add_argument("--decoder-lstm-dim", type=int)
        parser.add_argument("--output-frame-dim", type=int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_updates = 0

    @classmethod
    def build_model(cls, args, task):
        embed_speaker = task.get_speaker_embeddings(args)
        encoder = Tacotron2Encoder(args, task.src_dict, embed_speaker)
        decoder = Tacotron2Decoder(args, task.src_dict)
        return cls(encoder, decoder)

    def forward_encoder(self, src_tokens, src_lengths, **kwargs):
        return self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self._num_updates = num_updates


@register_model_architecture("tacotron_2", "tacotron_2")
def base_architecture(args):
    # encoder
    args.encoder_dropout = getattr(args, "encoder_dropout", 0.5)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_conv_layers = getattr(args, "encoder_conv_layers", 3)
    args.encoder_conv_kernel_size = getattr(args, "encoder_conv_kernel_size", 5)
    args.encoder_lstm_layers = getattr(args, "encoder_lstm_layers", 1)
    # decoder
    args.attention_dim = getattr(args, "attention_dim", 128)
    args.attention_conv_dim = getattr(args, "attention_conv_dim", 32)
    args.attention_conv_kernel_size = getattr(args, "attention_conv_kernel_size", 15)
    args.prenet_dropout = getattr(args, "prenet_dropout", 0.5)
    args.prenet_layers = getattr(args, "prenet_layers", 2)
    args.prenet_dim = getattr(args, "prenet_dim", 256)
    args.postnet_dropout = getattr(args, "postnet_dropout", 0.5)
    args.postnet_layers = getattr(args, "postnet_layers", 5)
    args.postnet_conv_dim = getattr(args, "postnet_conv_dim", 512)
    args.postnet_conv_kernel_size = getattr(args, "postnet_conv_kernel_size", 5)
    args.init_attn_c = getattr(args, "init_attn_c", "zero")
    args.attention_use_cumprob = getattr(args, "attention_use_cumprob", True)
    args.zoneout = getattr(args, "zoneout", 0.1)
    args.decoder_lstm_layers = getattr(args, "decoder_lstm_layers", 2)
    args.decoder_lstm_dim = getattr(args, "decoder_lstm_dim", 1024)
    args.output_frame_dim = getattr(args, "output_frame_dim", 80)
