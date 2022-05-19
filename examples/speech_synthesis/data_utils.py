# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
from pathlib import Path
from typing import Optional, List, Dict
import zipfile
import tempfile
from dataclasses import dataclass
from itertools import groupby

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from examples.speech_to_text.data_utils import load_tsv_to_dicts
from fairseq.data.audio.audio_utils import (
    TTSSpectrogram, TTSMelScale, parse_path, read_from_stored_zip, is_npy_data
)


def trim_or_pad_to_target_length(
        data_1d_or_2d: np.ndarray, target_length: int
) -> np.ndarray:
    assert len(data_1d_or_2d.shape) in {1, 2}
    delta = data_1d_or_2d.shape[0] - target_length
    if delta >= 0:  # trim if being longer
        data_1d_or_2d = data_1d_or_2d[: target_length]
    else:  # pad if being shorter
        if len(data_1d_or_2d.shape) == 1:
            data_1d_or_2d = np.concatenate(
                [data_1d_or_2d, np.zeros(-delta)], axis=0
            )
        else:
            data_1d_or_2d = np.concatenate(
                [data_1d_or_2d, np.zeros((-delta, data_1d_or_2d.shape[1]))],
                axis=0
            )
    return data_1d_or_2d


def extract_logmel_spectrogram(
        waveform: torch.Tensor, sample_rate: int,
        output_path: Optional[Path] = None, win_length: int = 1024,
        hop_length: int = 256, n_fft: int = 1024,
        win_fn: callable = torch.hann_window, n_mels: int = 80,
        f_min: float = 0., f_max: float = 8000, eps: float = 1e-5,
        overwrite: bool = False, target_length: Optional[int] = None
):
    if output_path is not None and output_path.is_file() and not overwrite:
        return

    spectrogram_transform = TTSSpectrogram(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        window_fn=win_fn
    )
    mel_scale_transform = TTSMelScale(
        n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
        n_stft=n_fft // 2 + 1
    )
    spectrogram = spectrogram_transform(waveform)
    mel_spec = mel_scale_transform(spectrogram)
    logmel_spec = torch.clamp(mel_spec, min=eps).log()
    assert len(logmel_spec.shape) == 3 and logmel_spec.shape[0] == 1
    logmel_spec = logmel_spec.squeeze().t()  # D x T -> T x D
    if target_length is not None:
        logmel_spec = trim_or_pad_to_target_length(logmel_spec, target_length)

    if output_path is not None:
        np.save(output_path.as_posix(), logmel_spec)
    else:
        return logmel_spec


def extract_pitch(
        waveform: torch.Tensor, sample_rate: int,
        output_path: Optional[Path] = None, hop_length: int = 256,
        log_scale: bool = True, phoneme_durations: Optional[List[int]] = None
):
    if output_path is not None and output_path.is_file():
        return

    try:
        import pyworld
    except ImportError:
        raise ImportError("Please install PyWORLD: pip install pyworld")

    _waveform = waveform.squeeze(0).double().numpy()
    pitch, t = pyworld.dio(
        _waveform, sample_rate, frame_period=hop_length / sample_rate * 1000
    )
    pitch = pyworld.stonemask(_waveform, pitch, t, sample_rate)

    if phoneme_durations is not None:
        pitch = trim_or_pad_to_target_length(pitch, sum(phoneme_durations))
        try:
            from scipy.interpolate import interp1d
        except ImportError:
            raise ImportError("Please install SciPy: pip install scipy")
        nonzero_ids = np.where(pitch != 0)[0]
        if len(nonzero_ids) == 0:
            print((f"{output_path} has all empty values in the pitch contour"))
            return
        elif len(nonzero_ids) == 1:
            print((f"{output_path} has only one non-zero values in the pitch contour"))
            return
        else:
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))
        d_cumsum = np.cumsum(np.concatenate([np.array([0]), phoneme_durations]))
        pitch = np.array(
            [
                np.mean(pitch[d_cumsum[i-1]: d_cumsum[i]])
                for i in range(1, len(d_cumsum))
            ]
        )
        assert len(pitch) == len(phoneme_durations)

    if log_scale:
        pitch = np.log(pitch + 1)

    if output_path is not None:
        np.save(output_path.as_posix(), pitch)
    else:
        return pitch


def extract_energy(
        waveform: torch.Tensor, output_path: Optional[Path] = None,
        hop_length: int = 256, n_fft: int = 1024, log_scale: bool = True,
        phoneme_durations: Optional[List[int]] = None
):
    if output_path is not None and output_path.is_file():
        return

    assert len(waveform.shape) == 2 and waveform.shape[0] == 1
    waveform = waveform.view(1, 1, waveform.shape[1])
    waveform = F.pad(
        waveform.unsqueeze(1), [n_fft // 2, n_fft // 2, 0, 0],
        mode="reflect"
    )
    waveform = waveform.squeeze(1)

    fourier_basis = np.fft.fft(np.eye(n_fft))
    cutoff = int((n_fft / 2 + 1))
    fourier_basis = np.vstack(
        [np.real(fourier_basis[:cutoff, :]),
         np.imag(fourier_basis[:cutoff, :])]
    )

    forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
    forward_transform = F.conv1d(
        waveform, forward_basis, stride=hop_length, padding=0
    )

    real_part = forward_transform[:, :cutoff, :]
    imag_part = forward_transform[:, cutoff:, :]
    magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
    energy = torch.norm(magnitude, dim=1).squeeze(0).numpy()

    if phoneme_durations is not None:
        energy = trim_or_pad_to_target_length(energy, sum(phoneme_durations))
        d_cumsum = np.cumsum(np.concatenate([np.array([0]), phoneme_durations]))
        energy = np.array(
            [
                np.mean(energy[d_cumsum[i - 1]: d_cumsum[i]])
                for i in range(1, len(d_cumsum))
            ]
        )
        assert len(energy) == len(phoneme_durations)

    if log_scale:
        energy = np.log(energy + 1)

    if output_path is not None:
        np.save(output_path.as_posix(), energy)
    else:
        return energy


def get_global_cmvn(feature_root: Path, output_path: Optional[Path] = None):
    mean_x, mean_x2, n_frames = None, None, 0
    feature_paths = feature_root.glob("*.npy")
    for p in tqdm(feature_paths):
        with open(p, 'rb') as f:
            frames = np.load(f).squeeze()

        n_frames += frames.shape[0]

        cur_mean_x = frames.sum(axis=0)
        if mean_x is None:
            mean_x = cur_mean_x
        else:
            mean_x += cur_mean_x

        cur_mean_x2 = (frames ** 2).sum(axis=0)
        if mean_x2 is None:
            mean_x2 = cur_mean_x2
        else:
            mean_x2 += cur_mean_x2

    mean_x /= n_frames
    mean_x2 /= n_frames
    var_x = mean_x2 - mean_x ** 2
    std_x = np.sqrt(np.maximum(var_x, 1e-10))

    if output_path is not None:
        with open(output_path, 'wb') as f:
            np.savez(f, mean=mean_x, std=std_x)
    else:
        return {"mean": mean_x, "std": std_x}


def ipa_phonemize(text, lang="en-us", use_g2p=False):
    if use_g2p:
        assert lang == "en-us", "g2pE phonemizer only works for en-us"
        try:
            from g2p_en import G2p
            g2p = G2p()
            return " ".join("|" if p == " " else p for p in g2p(text))
        except ImportError:
            raise ImportError(
                "Please install phonemizer: pip install g2p_en"
            )
    else:
        try:
            from phonemizer import phonemize
            from phonemizer.separator import Separator
            return phonemize(
                text, backend='espeak', language=lang,
                separator=Separator(word="| ", phone=" ")
            )
        except ImportError:
            raise ImportError(
                "Please install phonemizer: pip install phonemizer"
            )


@dataclass
class ForceAlignmentInfo(object):
    tokens: List[str]
    frame_durations: List[int]
    start_sec: Optional[float]
    end_sec: Optional[float]


def get_mfa_alignment_by_sample_id(
        textgrid_zip_path: str, sample_id: str, sample_rate: int,
        hop_length: int, silence_phones: List[str] = ("sil", "sp", "spn")
) -> ForceAlignmentInfo:
    try:
        import tgt
    except ImportError:
        raise ImportError("Please install TextGridTools: pip install tgt")

    filename = f"{sample_id}.TextGrid"
    out_root = Path(tempfile.gettempdir())
    tgt_path = out_root / filename
    with zipfile.ZipFile(textgrid_zip_path) as f_zip:
        f_zip.extract(filename, path=out_root)
    textgrid = tgt.io.read_textgrid(tgt_path.as_posix())
    os.remove(tgt_path)

    phones, frame_durations = [], []
    start_sec, end_sec, end_idx = 0, 0, 0
    for t in textgrid.get_tier_by_name("phones")._objects:
        s, e, p = t.start_time, t.end_time, t.text
        # Trim leading silences
        if len(phones) == 0:
            if p in silence_phones:
                continue
            else:
                start_sec = s
        phones.append(p)
        if p not in silence_phones:
            end_sec = e
            end_idx = len(phones)
        r = sample_rate / hop_length
        frame_durations.append(int(np.round(e * r) - np.round(s * r)))
    # Trim tailing silences
    phones = phones[:end_idx]
    frame_durations = frame_durations[:end_idx]

    return ForceAlignmentInfo(
        tokens=phones, frame_durations=frame_durations, start_sec=start_sec,
        end_sec=end_sec
    )


def get_mfa_alignment(
        textgrid_zip_path: str, sample_ids: List[str], sample_rate: int,
        hop_length: int
) -> Dict[str, ForceAlignmentInfo]:
    return {
        i: get_mfa_alignment_by_sample_id(
            textgrid_zip_path, i, sample_rate, hop_length
        ) for i in tqdm(sample_ids)
    }


def get_unit_alignment(
        id_to_unit_tsv_path: str, sample_ids: List[str]
) -> Dict[str, ForceAlignmentInfo]:
    id_to_units = {
        e["id"]: e["units"] for e in load_tsv_to_dicts(id_to_unit_tsv_path)
    }
    id_to_units = {i: id_to_units[i].split() for i in sample_ids}
    id_to_units_collapsed = {
        i: [uu for uu, _ in groupby(u)] for i, u in id_to_units.items()
    }
    id_to_durations = {
        i: [len(list(g)) for _, g in groupby(u)] for i, u in id_to_units.items()
    }

    return {
        i: ForceAlignmentInfo(
            tokens=id_to_units_collapsed[i], frame_durations=id_to_durations[i],
            start_sec=None, end_sec=None
        )
        for i in sample_ids
    }


def get_feature_value_min_max(feature_paths: List[str]):
    v_min, v_max = 1e-8, -1e-8
    for p in tqdm(feature_paths):
        _path, slice_ptr = parse_path(p)
        assert len(slice_ptr) == 2
        byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
        assert is_npy_data(byte_data)
        path_or_fp = io.BytesIO(byte_data)
        features = np.load(path_or_fp).squeeze()
        v_min = min(v_min, features.min().item())
        v_max = max(v_max, features.max().item())
    return v_min, v_max
