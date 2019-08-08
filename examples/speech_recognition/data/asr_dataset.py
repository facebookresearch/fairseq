# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from fairseq.data import FairseqDataset

from . import data_utils
from .collaters import Seq2SeqCollater


class AsrDataset(FairseqDataset):
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
        num_mel_bins=80, frame_length=25.0, frame_shift=10.0
    ):
        assert frame_length > 0
        assert frame_shift > 0
        assert all(x > frame_length for x in aud_durations_ms)
        self.frame_sizes = [
            int(1 + (d - frame_length) / frame_shift)
            for d in aud_durations_ms
        ]

        assert len(aud_paths) > 0
        assert len(aud_paths) == len(aud_durations_ms)
        assert len(aud_paths) == len(tgt)
        assert len(aud_paths) == len(ids)
        assert len(aud_paths) == len(speakers)
        self.aud_paths = aud_paths
        self.tgt_dict = tgt_dict
        self.tgt = tgt
        self.ids = ids
        self.speakers = speakers
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift

    def __getitem__(self, index):
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
        tgt_item = self.tgt[index] if self.tgt is not None else None

        path = self.aud_paths[index]
        if not os.path.exists(path):
            raise FileNotFoundError("Audio file not found: {}".format(path))
        sound, sample_rate = torchaudio.load_wav(path)
        output = kaldi.fbank(
            sound,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift
        )
        output_cmvn = data_utils.apply_mv_norm(output)
        self.collater = Seq2SeqCollater(
            0, 1, pad_index=self.tgt_dict.pad(),
            eos_index=self.tgt_dict.eos(), move_eos_to_beginning=True
        )

        return {"id": index, "data": [output_cmvn.detach(), tgt_item]}

    def __len__(self):
        return len(self.aud_paths)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[int]): sample indices to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        return self.collater.collate(samples)

    def num_tokens(self, index):
        return self.frame_sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.frame_sizes[index],
            len(self.tgt[index]) if self.tgt is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self))
