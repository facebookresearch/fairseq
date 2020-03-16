# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from fairseq.data import FairseqDataset

from . import data_utils
from .collaters import Seq2SeqCollater
from examples.speech_recognition.data import AsrDataset
import torch


class AstDataset(AsrDataset):
    """
    A dataset representing speech and corresponding transcription.

    Args:
        aud_paths: (List[str]): A list of str with paths to audio files.
        aud_durations_ms (List[int]): A list of int containing the durations of
            audio files.
        tgt (List[torch.LongTensor]): A list of LongTensors containing the indices
            of target transcriptions.
        tgt_dict (~fairseq.data.Dictionary): target vocabulary.
        ids (List[str]): A list of utterance IDs.
        speakers (List[str]): A list of speakers corresponding to utterances.
        num_mel_bins (int): Number of triangular mel-frequency bins (default: 80)
        frame_length (float): Frame length in milliseconds (default: 25.0)
        frame_shift (float): Frame shift in milliseconds (default: 10.0)
    """

    def __init__(
        self, aud_paths, aud_durations_ms, tgt,
        tgt_dict, ids, speakers,
        num_mel_bins=80, frame_length=25.0, frame_shift=10.0, 
        online_features=True, use_energy=False, segmenter=None,
        mv_norm=False,
    ):
        super().__init__(
            aud_paths, aud_durations_ms, tgt,
            tgt_dict, ids, speakers,
            num_mel_bins, frame_length, frame_shift
        )
        self.online_features = online_features
        self.use_energy = use_energy
        self.segmenter = segmenter
        self.mv_norm = mv_norm

    def __getitem__(self, index):
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
        tgt_item = self.tgt[index] if self.tgt is not None else None

        path = self.aud_paths[index]
        if not os.path.exists(path):
            raise FileNotFoundError("Audio file not found: {}".format(path))

        if self.online_features:
            sound, sample_rate = torchaudio.load_wav(path)
            output = kaldi.fbank(
                sound,
                num_mel_bins=self.num_mel_bins,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                use_energy=self.use_energy,
            )
        else:
            feature_path = f'{path}.{self.num_mel_bins}-{self.frame_length}-{self.frame_shift}{"-use_energy"*self.use_energy}.fbank'
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"Can't find the file {feature_path}. \nHave you exatracted the offline features? Or do you want to use --online-features?")
            output = torch.load(
                f'{path}.{self.num_mel_bins}-{self.frame_length}-{self.frame_shift}{"-use_energy"*self.use_energy}.fbank'
            )
            
        if self.use_energy:
            energy, output = output[:, 0], output[:, 1:]
        else:
            energy = None

        import pdb; pdb.set_trace()
        if self.mv_norm:
            output_cmvn = data_utils.apply_mv_norm(output)
        else:
            output_cmvn = output
        
        return {"id": index, "data": [output_cmvn.detach(), tgt_item]}
