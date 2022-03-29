# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from scipy.interpolate import interp1d
import torchaudio

from fairseq.tasks.text_to_speech import (
    batch_compute_distortion, compute_rms_dist
)


def batch_mel_spectral_distortion(
        y1, y2, sr, normalize_type="path", mel_fn=None
):
    """
    https://arxiv.org/pdf/2011.03568.pdf

    Same as Mel Cepstral Distortion, but computed on log-mel spectrograms.
    """
    if mel_fn is None or mel_fn.sample_rate != sr:
        mel_fn = torchaudio.transforms.MelSpectrogram(
            sr, n_fft=int(0.05 * sr), win_length=int(0.05 * sr),
            hop_length=int(0.0125 * sr), f_min=20, n_mels=80,
            window_fn=torch.hann_window
        ).to(y1[0].device)
    offset = 1e-6
    return batch_compute_distortion(
        y1, y2, sr, lambda y: torch.log(mel_fn(y) + offset).transpose(-1, -2),
        compute_rms_dist, normalize_type
    )


# This code is based on
# "https://github.com/bastibe/MAPS-Scripts/blob/master/helper.py"
def _same_t_in_true_and_est(func):
    def new_func(true_t, true_f, est_t, est_f):
        assert type(true_t) is np.ndarray
        assert type(true_f) is np.ndarray
        assert type(est_t) is np.ndarray
        assert type(est_f) is np.ndarray

        interpolated_f = interp1d(
            est_t, est_f, bounds_error=False, kind='nearest', fill_value=0
        )(true_t)
        return func(true_t, true_f, true_t, interpolated_f)

    return new_func


@_same_t_in_true_and_est
def gross_pitch_error(true_t, true_f, est_t, est_f):
    """The relative frequency in percent of pitch estimates that are
    outside a threshold around the true pitch. Only frames that are
    considered pitched by both the ground truth and the estimator (if
    applicable) are considered.
    """

    correct_frames = _true_voiced_frames(true_t, true_f, est_t, est_f)
    gross_pitch_error_frames = _gross_pitch_error_frames(
        true_t, true_f, est_t, est_f
    )
    return np.sum(gross_pitch_error_frames) / np.sum(correct_frames)


def _gross_pitch_error_frames(true_t, true_f, est_t, est_f, eps=1e-8):
    voiced_frames = _true_voiced_frames(true_t, true_f, est_t, est_f)
    true_f_p_eps = [x + eps for x in true_f]
    pitch_error_frames = np.abs(est_f / true_f_p_eps - 1) > 0.2
    return voiced_frames & pitch_error_frames


def _true_voiced_frames(true_t, true_f, est_t, est_f):
    return (est_f != 0) & (true_f != 0)


def _voicing_decision_error_frames(true_t, true_f, est_t, est_f):
    return (est_f != 0) != (true_f != 0)


@_same_t_in_true_and_est
def f0_frame_error(true_t, true_f, est_t, est_f):
    gross_pitch_error_frames = _gross_pitch_error_frames(
        true_t, true_f, est_t, est_f
    )
    voicing_decision_error_frames = _voicing_decision_error_frames(
        true_t, true_f, est_t, est_f
    )
    return (np.sum(gross_pitch_error_frames) +
            np.sum(voicing_decision_error_frames)) / (len(true_t))


@_same_t_in_true_and_est
def voicing_decision_error(true_t, true_f, est_t, est_f):
    voicing_decision_error_frames = _voicing_decision_error_frames(
        true_t, true_f, est_t, est_f
    )
    return np.sum(voicing_decision_error_frames) / (len(true_t))
