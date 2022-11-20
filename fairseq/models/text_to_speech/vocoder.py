# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fairseq.data.audio.audio_utils import (
    TTSSpectrogram,
    get_fourier_basis,
    get_mel_filters,
    get_window,
)
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.text_to_speech.codehifigan import CodeGenerator as CodeHiFiGANModel
from fairseq.models.text_to_speech.hifigan import Generator as HiFiGANModel
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface

logger = logging.getLogger(__name__)


class PseudoInverseMelScale(torch.nn.Module):
    def __init__(self, n_stft, n_mels, sample_rate, f_min, f_max) -> None:
        super(PseudoInverseMelScale, self).__init__()
        self.n_mels = n_mels
        basis = get_mel_filters(sample_rate, (n_stft - 1) * 2, n_mels, f_min, f_max)
        basis = torch.pinverse(basis)  # F x F_mel
        self.register_buffer("basis", basis)

    def forward(self, melspec: torch.Tensor) -> torch.Tensor:
        # pack batch
        shape = melspec.shape  # B_1 x ... x B_K x F_mel x T
        n_mels, time = shape[-2], shape[-1]
        melspec = melspec.view(-1, n_mels, time)

        freq, _ = self.basis.size()  # F x F_mel
        assert self.n_mels == n_mels, (self.n_mels, n_mels)
        specgram = self.basis.matmul(melspec).clamp(min=0)

        # unpack batch
        specgram = specgram.view(shape[:-2] + (freq, time))
        return specgram


class GriffinLim(torch.nn.Module):
    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_iter: int,
        window_fn=torch.hann_window,
    ):
        super(GriffinLim, self).__init__()
        self.transform = TTSSpectrogram(
            n_fft, win_length, hop_length, return_phase=True
        )

        basis = get_fourier_basis(n_fft)
        basis = torch.pinverse(n_fft / hop_length * basis).T[:, None, :]
        basis *= get_window(window_fn, n_fft, win_length)
        self.register_buffer("basis", basis)

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_iter = n_iter

        self.tiny = 1.1754944e-38

    @classmethod
    def get_window_sum_square(
        cls, n_frames, hop_length, win_length, n_fft, window_fn=torch.hann_window
    ) -> torch.Tensor:
        w_sq = get_window(window_fn, n_fft, win_length) ** 2
        n = n_fft + hop_length * (n_frames - 1)
        x = torch.zeros(n, dtype=torch.float32)
        for i in range(n_frames):
            ofst = i * hop_length
            x[ofst : min(n, ofst + n_fft)] += w_sq[: max(0, min(n_fft, n - ofst))]
        return x

    def inverse(self, magnitude: torch.Tensor, phase) -> torch.Tensor:
        x = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        x = F.conv_transpose1d(x, self.basis, stride=self.hop_length)
        win_sum_sq = self.get_window_sum_square(
            magnitude.shape[-1],
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
        ).to(magnitude.device)
        # remove modulation effects
        approx_nonzero_indices = win_sum_sq > self.tiny
        x[:, :, approx_nonzero_indices] /= win_sum_sq[approx_nonzero_indices]
        x *= self.n_fft / self.hop_length
        x = x[:, :, self.n_fft // 2 :]
        x = x[:, :, : -self.n_fft // 2 :]
        return x

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        angles = np.angle(np.exp(2j * np.pi * np.random.rand(*specgram.shape)))
        angles = torch.from_numpy(angles).to(specgram)
        _specgram = specgram.view(-1, specgram.shape[-2], specgram.shape[-1])
        waveform = self.inverse(_specgram, angles).squeeze(1)
        for _ in range(self.n_iter):
            _, angles = self.transform(waveform)
            waveform = self.inverse(_specgram, angles).squeeze(1)
        return waveform.squeeze(0)


class GriffinLimVocoder(nn.Module):
    def __init__(
        self,
        sample_rate,
        win_size,
        hop_size,
        n_fft,
        n_mels,
        f_min,
        f_max,
        window_fn,
        spec_bwd_max_iter=32,
        fp16=False,
    ):
        super().__init__()
        self.inv_mel_transform = PseudoInverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
        )
        self.gl_transform = GriffinLim(
            n_fft=n_fft,
            win_length=win_size,
            hop_length=hop_size,
            window_fn=window_fn,
            n_iter=spec_bwd_max_iter,
        )
        if fp16:
            self.half()
            self.inv_mel_transform.half()
            self.gl_transform.half()
        else:
            self.float()
            self.inv_mel_transform.float()
            self.gl_transform.float()

    def forward(self, x):
        # x: (B x) T x D -> (B x) 1 x T
        # NOTE: batched forward produces noisier waveform. recommend running
        # one utterance at a time
        self.eval()
        x = x.exp().transpose(-1, -2)
        x = self.inv_mel_transform(x)
        x = self.gl_transform(x)
        return x

    @classmethod
    def from_data_cfg(cls, args, data_cfg: S2TDataConfig):
        feat_cfg = data_cfg.config["features"]
        window_fn = getattr(torch, feat_cfg["window_fn"] + "_window")
        return cls(
            sample_rate=feat_cfg["sample_rate"],
            win_size=int(feat_cfg["win_len_t"] * feat_cfg["sample_rate"]),
            hop_size=int(feat_cfg["hop_len_t"] * feat_cfg["sample_rate"]),
            n_fft=feat_cfg["n_fft"],
            n_mels=feat_cfg["n_mels"],
            f_min=feat_cfg["f_min"],
            f_max=feat_cfg["f_max"],
            window_fn=window_fn,
            spec_bwd_max_iter=args.spec_bwd_max_iter,
            fp16=args.fp16,
        )


class HiFiGANVocoder(nn.Module):
    def __init__(
        self, checkpoint_path: str, model_cfg: Dict[str, str], fp16: bool = False
    ) -> None:
        super().__init__()
        self.model = HiFiGANModel(model_cfg)
        state_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(state_dict["generator"])
        if fp16:
            self.model.half()
        logger.info(f"loaded HiFiGAN checkpoint from {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B x) T x D -> (B x) 1 x T
        model = self.model.eval()
        if len(x.shape) == 2:
            return model(x.unsqueeze(0).transpose(1, 2)).detach().squeeze(0)
        else:
            return model(x.transpose(-1, -2)).detach()

    @classmethod
    def from_data_cfg(cls, args, data_cfg: S2TDataConfig):
        vocoder_cfg = data_cfg.vocoder
        assert vocoder_cfg.get("type", "griffin_lim") == "hifigan"
        with open(vocoder_cfg["config"]) as f:
            model_cfg = json.load(f)
        return cls(vocoder_cfg["checkpoint"], model_cfg, fp16=args.fp16)


@register_model("CodeHiFiGANVocoder")
class CodeHiFiGANVocoder(BaseFairseqModel):
    def __init__(
        self, checkpoint_path: str, model_cfg: Dict[str, str], fp16: bool = False
    ) -> None:
        super().__init__()
        self.model = CodeHiFiGANModel(model_cfg)
        if torch.cuda.is_available():
            state_dict = torch.load(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict["generator"])
        self.model.eval()
        if fp16:
            self.model.half()
        self.model.remove_weight_norm()
        logger.info(f"loaded CodeHiFiGAN checkpoint from {checkpoint_path}")

    def forward(self, x: Dict[str, torch.Tensor], dur_prediction=False) -> torch.Tensor:
        assert "code" in x
        x["dur_prediction"] = dur_prediction

        # remove invalid code
        mask = x["code"] >= 0
        x["code"] = x["code"][mask].unsqueeze(dim=0)
        if "f0" in x:
            f0_up_ratio = x["f0"].size(1) // x["code"].size(1)
            mask = mask.unsqueeze(2).repeat(1, 1, f0_up_ratio).view(-1, x["f0"].size(1))
            x["f0"] = x["f0"][mask].unsqueeze(dim=0)

        return self.model(**x).detach().squeeze()

    @classmethod
    def from_data_cfg(cls, args, data_cfg):
        vocoder_cfg = data_cfg.vocoder
        assert vocoder_cfg is not None, "vocoder not specified in the data config"
        with open(vocoder_cfg["config"]) as f:
            model_cfg = json.load(f)
        return cls(vocoder_cfg["checkpoint"], model_cfg, fp16=args.fp16)

    @classmethod
    def hub_models(cls):
        base_url = "http://dl.fbaipublicfiles.com/fairseq/vocoder"
        model_ids = [
            "unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur",
            "unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10_dur",
            "unit_hifigan_HK_layer12.km2500_frame_TAT-TTS",
        ]
        return {i: f"{base_url}/{i}.tar.gz" for i in model_ids}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        config="config.json",
        fp16: bool = False,
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            config_yaml=config,
            fp16=fp16,
            is_vocoder=True,
            **kwargs,
        )

        with open(f"{x['args']['data']}/{config}") as f:
            vocoder_cfg = json.load(f)
        assert len(x["args"]["model_path"]) == 1, "Too many vocoder models in the input"

        vocoder = CodeHiFiGANVocoder(x["args"]["model_path"][0], vocoder_cfg)
        return VocoderHubInterface(vocoder_cfg, vocoder)


def get_vocoder(args, data_cfg: S2TDataConfig):
    if args.vocoder == "griffin_lim":
        return GriffinLimVocoder.from_data_cfg(args, data_cfg)
    elif args.vocoder == "hifigan":
        return HiFiGANVocoder.from_data_cfg(args, data_cfg)
    elif args.vocoder == "code_hifigan":
        return CodeHiFiGANVocoder.from_data_cfg(args, data_cfg)
    else:
        raise ValueError("Unknown vocoder")
