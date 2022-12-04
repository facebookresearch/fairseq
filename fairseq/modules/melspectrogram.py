import torch
from torch import nn


class MelSpectrogram(nn.Module):
    """
    Waveform to MelSpectrogram extractor
    """

    def __init__(
        self,
        sr,
        n_fft,
        win_length,
        hop_length,
        pre_emph=True,
        normalize=True,
        log=True,
        power=2.0,
        n_mels=80,
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.preemphasis_coeff = 0.97
        self.log = log

        self.pre_emph = pre_emph

        import torchaudio

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=power,
            n_mels=n_mels,
            normalized=normalize,
        )

    def _amp_to_db(self, x, minimum=1e-5):
        return 20 * torch.log10(torch.clamp(x, min=minimum))

    def _normalize(self, feat):
        return torch.clamp(
            (feat - self.MIN_LEVEL_DB) / -self.MIN_LEVEL_DB, min=0, max=1
        )

    def _preemphasis(self, waveform):
        waveform = torch.cat(
            [
                waveform[:, :1],
                waveform[:, 1:] - self.preemphasis_coeff * waveform[:, :-1],
            ],
            dim=-1,
        )
        return waveform

    def forward(self, waveform):
        if self.pre_emph:
            waveform = self._preemphasis(waveform.float())
        self.mel_spec = self.mel_spec.float()
        melspecgram = self.mel_spec(waveform.float())
        if self.log:
            melspecgram = melspecgram.log10()
        return melspecgram.transpose(1, 2)
