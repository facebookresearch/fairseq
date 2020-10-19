import os.path as op
from typing import BinaryIO, Optional, Tuple, Union

import numpy as np


def get_waveform(
    path_or_fp: Union[str, BinaryIO], normalization=True
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit mono-channel WAV or FLAC.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): Normalize values to [-1, 1] (Default: True)
    """
    if isinstance(path_or_fp, str):
        ext = op.splitext(op.basename(path_or_fp))[1]
        if ext not in {".flac", ".wav"}:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLAC file")

    waveform, sample_rate = sf.read(path_or_fp, dtype="float32")
    if not normalization:
        waveform *= 2 ** 15  # denormalized to 16-bit signed integers
    return waveform, sample_rate


def _get_kaldi_fbank(waveform, sample_rate, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via PyKaldi."""
    try:
        from kaldi.feat.mel import MelBanksOptions
        from kaldi.feat.fbank import FbankOptions, Fbank
        from kaldi.feat.window import FrameExtractionOptions
        from kaldi.matrix import Vector

        mel_opts = MelBanksOptions()
        mel_opts.num_bins = n_bins
        frame_opts = FrameExtractionOptions()
        frame_opts.samp_freq = sample_rate
        opts = FbankOptions()
        opts.mel_opts = mel_opts
        opts.frame_opts = frame_opts
        fbank = Fbank(opts=opts)
        features = fbank.compute(Vector(waveform), 1.0).numpy()
        return features
    except ImportError:
        return None


def _get_torchaudio_fbank(waveform, sample_rate, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via TorchAudio."""
    try:
        import torch
        import torchaudio.compliance.kaldi as ta_kaldi

        waveform = torch.from_numpy(waveform).unsqueeze(0)
        features = ta_kaldi.fbank(
            waveform, num_mel_bins=n_bins, sample_frequency=sample_rate
        )
        return features.numpy()
    except ImportError:
        return None


def get_fbank(path_or_fp: Union[str, BinaryIO], n_bins=80) -> np.ndarray:
    """Get mel-filter bank features via PyKaldi or TorchAudio. Prefer PyKaldi
    (faster CPP implementation) to TorchAudio (Python implementation). Note that
    Kaldi/TorchAudio requires 16-bit signed integers as inputs and hence the
    waveform should not be normalized."""
    sound, sample_rate = get_waveform(path_or_fp, normalization=False)

    features = _get_kaldi_fbank(sound, sample_rate, n_bins)
    if features is None:
        features = _get_torchaudio_fbank(sound, sample_rate, n_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable "
            "online filterbank feature extraction"
        )

    return features
