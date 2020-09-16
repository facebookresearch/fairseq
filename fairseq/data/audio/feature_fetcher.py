import io
import os.path as op
from typing import Union, BinaryIO

import numpy as np


def _get_fbank_features_kaldi(sound, sr, num_mel_bins=80):
    try:
        from kaldi.feat.mel import MelBanksOptions
        from kaldi.feat.fbank import FbankOptions, Fbank
        from kaldi.feat.window import FrameExtractionOptions
        from kaldi.matrix import Vector

        mel_opts = MelBanksOptions()
        mel_opts.num_bins = num_mel_bins
        frame_opts = FrameExtractionOptions()
        frame_opts.samp_freq = sr
        opts = FbankOptions()
        opts.mel_opts = mel_opts
        opts.frame_opts = frame_opts
        fbank = Fbank(opts=opts)
        features = fbank.compute(Vector(sound), 1.0).numpy()
        return features
    except ImportError:
        return None


def _get_fbank_features_torchaudio(sound, sr, num_mel_bins=80):
    try:
        import torch
        import torchaudio.compliance.kaldi as ta_kaldi
        sound = torch.tensor([sound.tolist()])
        torch.manual_seed(1)
        features = ta_kaldi.fbank(sound, num_mel_bins=num_mel_bins,
                                  sample_frequency=sr)
        return features.numpy()
    except ImportError:
        return None


def get_waveform(path_or_fp: Union[str, BinaryIO]):
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError('Please install soundfile to load audio file')

    sound, sr = sf.read(path_or_fp)
    return sound, sr


def get_fbank_features(path_or_fp: Union[str, BinaryIO], num_mel_bins=80,
                       utt_cmvn=True):
    sound, sr = get_waveform(path_or_fp)
    sound *= 32768

    features = _get_fbank_features_kaldi(sound, sr, num_mel_bins)
    if features is None:
        features = _get_fbank_features_torchaudio(sound, sr, num_mel_bins)
    if features is None:
        raise ImportError('Please install pyKaldi or torchaudio to enable '
                          'online filterbank feature extraction')

    if utt_cmvn:
        square_sums = (features ** 2).sum(axis=0)
        mean = features.mean(axis=0)

        features = np.subtract(features, mean)
        var = square_sums / features.shape[0] - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-8))
        features = np.divide(features, std)

    return features


def is_npy(data_bytes):
    return data_bytes[0] == 147 and data_bytes[1] == 78


def is_flac(data_bytes):
    return data_bytes[0] == 102 and data_bytes[1] == 76


def is_wav(data_bytes):
    return data_bytes[0] == 82 and data_bytes[1] == 73


def load_from_uncompressed_zip(file_path, offset, file_size):
    with open(file_path, 'rb') as f:
        f.seek(offset)
        data = f.read(file_size)
    return data


def fetch_features_from_audio_or_npy_file(path, use_audio=False):
    file_ext = op.splitext(op.basename(path))[1]
    if file_ext == '.npy':
        features = np.load(path)
    elif file_ext in {'.flac', '.wav'}:
        features = get_waveform(path)[0] if use_audio else get_fbank_features(path)
    else:
        raise ValueError(f'Invalid file format for "{path}"')
    return features


def fetch_features_from_zip_file(path, byte_offset, byte_size, use_audio=False):
    assert path.endswith('.zip')
    data = load_from_uncompressed_zip(path, byte_offset, byte_size)
    if is_npy(data):
        features = np.load(io.BytesIO(data))
    elif is_flac(data) or is_wav(data):
        f = io.BytesIO(data)
        features = get_waveform(f)[0] if use_audio else get_fbank_features(f)
    else:
        raise ValueError(f'Invalid file format for "{path}"')
    return features


def fetch_features(audio_path: str, use_audio=False):
    _audio_path, *ptr = audio_path.split(':')
    if not op.exists(_audio_path):
        raise FileNotFoundError(f'File not found: {_audio_path}')

    if len(ptr) == 0:
        features = fetch_features_from_audio_or_npy_file(_audio_path,
                                                         use_audio=use_audio)
    elif len(ptr) == 2:
        ptr = [int(i) for i in ptr]
        features = fetch_features_from_zip_file(_audio_path, ptr[0], ptr[1],
                                                use_audio=use_audio)
    else:
        raise ValueError(f'Invalid audio path: {audio_path}')
    return features
