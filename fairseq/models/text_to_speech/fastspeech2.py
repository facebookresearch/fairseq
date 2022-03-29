# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torch import nn

from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from fairseq.models.text_to_speech.tacotron2 import Postnet
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
)

logger = logging.getLogger(__name__)


def model_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    return m


class PositionwiseFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel_size, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv1d(
                in_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                hidden_dim,
                in_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
        )
        self.layer_norm = LayerNorm(in_dim)
        self.dropout = self.dropout_module = FairseqDropout(
            p=dropout, module_name=self.__class__.__name__
        )

    def forward(self, x):
        # B x T x C
        residual = x
        x = self.ffn(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class FFTLayer(torch.nn.Module):
    def __init__(
        self, embed_dim, n_heads, hidden_dim, kernel_size, dropout, attention_dropout
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dim, n_heads, dropout=attention_dropout, self_attention=True
        )
        self.layer_norm = LayerNorm(embed_dim)
        self.ffn = PositionwiseFeedForward(
            embed_dim, hidden_dim, kernel_size, dropout=dropout
        )

    def forward(self, x, padding_mask=None):
        # B x T x C
        residual = x
        x = x.transpose(0, 1)
        x, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=padding_mask, need_weights=False
        )
        x = x.transpose(0, 1)
        x = self.layer_norm(x + residual)
        return self.ffn(x)


class LengthRegulator(nn.Module):
    def forward(self, x, durations):
        # x: B x T x C
        out_lens = durations.sum(dim=1)
        max_len = out_lens.max()
        bsz, seq_len, dim = x.size()
        out = x.new_zeros((bsz, max_len, dim))

        for b in range(bsz):
            indices = []
            for t in range(seq_len):
                indices.extend([t] * utils.item(durations[b, t]))
            indices = torch.tensor(indices, dtype=torch.long).to(x.device)
            out_len = utils.item(out_lens[b])
            out[b, :out_len] = x[b].index_select(0, indices)

        return out, out_lens


class VariancePredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                args.encoder_embed_dim,
                args.var_pred_hidden_dim,
                kernel_size=args.var_pred_kernel_size,
                padding=(args.var_pred_kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(args.var_pred_hidden_dim)
        self.dropout_module = FairseqDropout(
            p=args.var_pred_dropout, module_name=self.__class__.__name__
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                args.var_pred_hidden_dim,
                args.var_pred_hidden_dim,
                kernel_size=args.var_pred_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.ln2 = nn.LayerNorm(args.var_pred_hidden_dim)
        self.proj = nn.Linear(args.var_pred_hidden_dim, 1)

    def forward(self, x):
        # Input: B x T x C; Output: B x T
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln1(x))
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln2(x))
        return self.proj(x).squeeze(dim=2)


class VarianceAdaptor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.length_regulator = LengthRegulator()
        self.duration_predictor = VariancePredictor(args)
        self.pitch_predictor = VariancePredictor(args)
        self.energy_predictor = VariancePredictor(args)

        n_bins, steps = self.args.var_pred_n_bins, self.args.var_pred_n_bins - 1
        self.pitch_bins = torch.linspace(args.pitch_min, args.pitch_max, steps)
        self.embed_pitch = Embedding(n_bins, args.encoder_embed_dim)
        self.energy_bins = torch.linspace(args.energy_min, args.energy_max, steps)
        self.embed_energy = Embedding(n_bins, args.encoder_embed_dim)

    def get_pitch_emb(self, x, tgt=None, factor=1.0):
        out = self.pitch_predictor(x)
        bins = self.pitch_bins.to(x.device)
        if tgt is None:
            out = out * factor
            emb = self.embed_pitch(torch.bucketize(out, bins))
        else:
            emb = self.embed_pitch(torch.bucketize(tgt, bins))
        return out, emb

    def get_energy_emb(self, x, tgt=None, factor=1.0):
        out = self.energy_predictor(x)
        bins = self.energy_bins.to(x.device)
        if tgt is None:
            out = out * factor
            emb = self.embed_energy(torch.bucketize(out, bins))
        else:
            emb = self.embed_energy(torch.bucketize(tgt, bins))
        return out, emb

    def forward(
        self,
        x,
        padding_mask,
        durations=None,
        pitches=None,
        energies=None,
        d_factor=1.0,
        p_factor=1.0,
        e_factor=1.0,
    ):
        # x: B x T x C
        log_dur_out = self.duration_predictor(x)
        dur_out = torch.clamp(
            torch.round((torch.exp(log_dur_out) - 1) * d_factor).long(), min=0
        )
        dur_out.masked_fill_(padding_mask, 0)

        pitch_out, pitch_emb = self.get_pitch_emb(x, pitches, p_factor)
        x = x + pitch_emb
        energy_out, energy_emb = self.get_energy_emb(x, energies, e_factor)
        x = x + energy_emb

        x, out_lens = self.length_regulator(
            x, dur_out if durations is None else durations
        )

        return x, out_lens, log_dur_out, pitch_out, energy_out


class FastSpeech2Encoder(FairseqEncoder):
    def __init__(self, args, src_dict, embed_speaker):
        super().__init__(src_dict)
        self.args = args
        self.padding_idx = src_dict.pad()
        self.n_frames_per_step = args.n_frames_per_step
        self.out_dim = args.output_frame_dim * args.n_frames_per_step

        self.embed_speaker = embed_speaker
        self.spk_emb_proj = None
        if embed_speaker is not None:
            self.spk_emb_proj = nn.Linear(
                args.encoder_embed_dim + args.speaker_embed_dim, args.encoder_embed_dim
            )

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_tokens = Embedding(
            len(src_dict), args.encoder_embed_dim, padding_idx=self.padding_idx
        )

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))
        self.dec_pos_emb_alpha = nn.Parameter(torch.ones(1))

        self.encoder_fft_layers = nn.ModuleList(
            FFTLayer(
                args.encoder_embed_dim,
                args.encoder_attention_heads,
                args.fft_hidden_dim,
                args.fft_kernel_size,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
            )
            for _ in range(args.encoder_layers)
        )

        self.var_adaptor = VarianceAdaptor(args)

        self.decoder_fft_layers = nn.ModuleList(
            FFTLayer(
                args.decoder_embed_dim,
                args.decoder_attention_heads,
                args.fft_hidden_dim,
                args.fft_kernel_size,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
            )
            for _ in range(args.decoder_layers)
        )

        self.out_proj = nn.Linear(args.decoder_embed_dim, self.out_dim)

        self.postnet = None
        if args.add_postnet:
            self.postnet = Postnet(
                self.out_dim,
                args.postnet_conv_dim,
                args.postnet_conv_kernel_size,
                args.postnet_layers,
                args.postnet_dropout,
            )

        self.apply(model_init)

    def forward(
        self,
        src_tokens,
        src_lengths=None,
        speaker=None,
        durations=None,
        pitches=None,
        energies=None,
        **kwargs,
    ):
        x = self.embed_tokens(src_tokens)

        enc_padding_mask = src_tokens.eq(self.padding_idx)
        x += self.pos_emb_alpha * self.embed_positions(enc_padding_mask)
        x = self.dropout_module(x)

        for layer in self.encoder_fft_layers:
            x = layer(x, enc_padding_mask)

        if self.embed_speaker is not None:
            bsz, seq_len, _ = x.size()
            emb = self.embed_speaker(speaker).expand(bsz, seq_len, -1)
            x = self.spk_emb_proj(torch.cat([x, emb], dim=2))

        x, out_lens, log_dur_out, pitch_out, energy_out = self.var_adaptor(
            x, enc_padding_mask, durations, pitches, energies
        )

        dec_padding_mask = lengths_to_padding_mask(out_lens)
        x += self.dec_pos_emb_alpha * self.embed_positions(dec_padding_mask)
        for layer in self.decoder_fft_layers:
            x = layer(x, dec_padding_mask)

        x = self.out_proj(x)
        x_post = None
        if self.postnet is not None:
            x_post = x + self.postnet(x)
        return x, x_post, out_lens, log_dur_out, pitch_out, energy_out


@register_model("fastspeech2")
class FastSpeech2Model(FairseqEncoderModel):
    """
    Implementation for https://arxiv.org/abs/2006.04558
    """

    NON_AUTOREGRESSIVE = True

    @classmethod
    def hub_models(cls):
        base_url = "http://dl.fbaipublicfiles.com/fairseq/s2"
        model_ids = [
            "fastspeech2-en-ljspeech",
            "fastspeech2-en-200_speaker-cv4",
        ]
        return {i: f"{base_url}/{i}.tar.gz" for i in model_ids}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        config_yaml="config.yaml",
        vocoder: str = "griffin_lim",
        fp16: bool = False,
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            config_yaml=config_yaml,
            vocoder=vocoder,
            fp16=fp16,
            **kwargs,
        )
        return TTSHubInterface(x["args"], x["task"], x["models"][0])

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--output-frame-dim", type=int)
        parser.add_argument("--speaker-embed-dim", type=int)
        # FFT blocks
        parser.add_argument("--fft-hidden-dim", type=int)
        parser.add_argument("--fft-kernel-size", type=int)
        parser.add_argument("--attention-dropout", type=float)
        parser.add_argument("--encoder-layers", type=int)
        parser.add_argument("--encoder-embed-dim", type=int)
        parser.add_argument("--encoder-attention-heads", type=int)
        parser.add_argument("--decoder-layers", type=int)
        parser.add_argument("--decoder-embed-dim", type=int)
        parser.add_argument("--decoder-attention-heads", type=int)
        # variance predictor
        parser.add_argument("--var-pred-n-bins", type=int)
        parser.add_argument("--var-pred-hidden-dim", type=int)
        parser.add_argument("--var-pred-kernel-size", type=int)
        parser.add_argument("--var-pred-dropout", type=float)
        # postnet
        parser.add_argument("--add-postnet", action="store_true")
        parser.add_argument("--postnet-dropout", type=float)
        parser.add_argument("--postnet-layers", type=int)
        parser.add_argument("--postnet-conv-dim", type=int)
        parser.add_argument("--postnet-conv-kernel-size", type=int)

    def __init__(self, encoder, args, src_dict):
        super().__init__(encoder)
        self._num_updates = 0

        out_dim = args.output_frame_dim * args.n_frames_per_step
        self.ctc_proj = None
        if getattr(args, "ctc_weight", 0.0) > 0.0:
            self.ctc_proj = nn.Linear(out_dim, len(src_dict))

    @classmethod
    def build_model(cls, args, task):
        embed_speaker = task.get_speaker_embeddings(args)
        encoder = FastSpeech2Encoder(args, task.src_dict, embed_speaker)
        return cls(encoder, args, task.src_dict)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self._num_updates = num_updates

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        logits = self.ctc_proj(net_output[0])
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)


@register_model_architecture("fastspeech2", "fastspeech2")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.2)
    args.output_frame_dim = getattr(args, "output_frame_dim", 80)
    args.speaker_embed_dim = getattr(args, "speaker_embed_dim", 64)
    # FFT blocks
    args.fft_hidden_dim = getattr(args, "fft_hidden_dim", 1024)
    args.fft_kernel_size = getattr(args, "fft_kernel_size", 9)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    # variance predictor
    args.var_pred_n_bins = getattr(args, "var_pred_n_bins", 256)
    args.var_pred_hidden_dim = getattr(args, "var_pred_hidden_dim", 256)
    args.var_pred_kernel_size = getattr(args, "var_pred_kernel_size", 3)
    args.var_pred_dropout = getattr(args, "var_pred_dropout", 0.5)
    # postnet
    args.add_postnet = getattr(args, "add_postnet", False)
    args.postnet_dropout = getattr(args, "postnet_dropout", 0.5)
    args.postnet_layers = getattr(args, "postnet_layers", 5)
    args.postnet_conv_dim = getattr(args, "postnet_conv_dim", 512)
    args.postnet_conv_kernel_size = getattr(args, "postnet_conv_kernel_size", 5)
