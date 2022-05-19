# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchaudio


EMBEDDER_PARAMS = {
    'num_mels': 40,
    'n_fft': 512,
    'emb_dim': 256,
    'lstm_hidden': 768,
    'lstm_layers': 3,
    'window': 80,
    'stride': 40,
}


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary
    computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class LinearNorm(nn.Module):
    def __init__(self, hp):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(hp["lstm_hidden"], hp["emb_dim"])

    def forward(self, x):
        return self.linear_layer(x)


class SpeechEmbedder(nn.Module):
    def __init__(self, hp):
        super(SpeechEmbedder, self).__init__()
        self.lstm = nn.LSTM(hp["num_mels"],
                            hp["lstm_hidden"],
                            num_layers=hp["lstm_layers"],
                            batch_first=True)
        self.proj = LinearNorm(hp)
        self.hp = hp

    def forward(self, mel):
        # (num_mels, T) -> (num_mels, T', window)
        mels = mel.unfold(1, self.hp["window"], self.hp["stride"])
        mels = mels.permute(1, 2, 0)  # (T', window, num_mels)
        x, _ = self.lstm(mels)  # (T', window, lstm_hidden)
        x = x[:, -1, :]  # (T', lstm_hidden), use last frame only
        x = self.proj(x)  # (T', emb_dim)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)  # (T', emb_dim)

        x = x.mean(dim=0)
        if x.norm(p=2) != 0:
            x = x / x.norm(p=2)
        return x


class SpkrEmbedder(nn.Module):
    RATE = 16000

    def __init__(
        self,
        embedder_path,
        embedder_params=EMBEDDER_PARAMS,
        rate=16000,
        hop_length=160,
        win_length=400,
        pad=False,
    ):
        super(SpkrEmbedder, self).__init__()
        embedder_pt = torch.load(embedder_path, map_location="cpu")
        self.embedder = SpeechEmbedder(embedder_params)
        self.embedder.load_state_dict(embedder_pt)
        self.embedder.eval()
        set_requires_grad(self.embedder, requires_grad=False)
        self.embedder_params = embedder_params

        self.register_buffer('mel_basis', torch.from_numpy(
            librosa.filters.mel(
                sr=self.RATE,
                n_fft=self.embedder_params["n_fft"],
                n_mels=self.embedder_params["num_mels"])
        )
                             )

        self.resample = None
        if rate != self.RATE:
            self.resample = torchaudio.transforms.Resample(rate, self.RATE)
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad = pad

    def get_mel(self, y):
        if self.pad and y.shape[-1] < 14000:
            y = F.pad(y, (0, 14000 - y.shape[-1]))

        window = torch.hann_window(self.win_length).to(y)
        y = torch.stft(y, n_fft=self.embedder_params["n_fft"],
                       hop_length=self.hop_length,
                       win_length=self.win_length,
                       window=window)
        magnitudes = torch.norm(y, dim=-1, p=2) ** 2
        mel = torch.log10(self.mel_basis @ magnitudes + 1e-6)
        return mel

    def forward(self, inputs):
        dvecs = []
        for wav in inputs:
            mel = self.get_mel(wav)
            if mel.dim() == 3:
                mel = mel.squeeze(0)
            dvecs += [self.embedder(mel)]
        dvecs = torch.stack(dvecs)

        dvec = torch.mean(dvecs, dim=0)
        dvec = dvec / torch.norm(dvec)

        return dvec
