# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi


class LogMelFeatureReader:
    """
    Wrapper class to run inference on HuBERT model.
    Helps extract features for a given audio file.
    """

    def __init__(self, *args, **kwargs):
        self.num_mel_bins = kwargs.get("num_mel_bins", 80)
        self.frame_length = kwargs.get("frame_length", 25.0)

    def get_feats(self, file_path, channel_id=None):
        wav, sr = sf.read(file_path)
        if channel_id is not None:
            assert wav.ndim == 2, \
                f"Expected stereo input when channel_id is given ({file_path})"
            wav = wav[:, channel_id-1]
        feats = torch.from_numpy(wav).float()
        feats = kaldi.fbank(
            feats.unsqueeze(0),
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            sample_frequency=sr,
        )
        return feats
