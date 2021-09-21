# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.abs

from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch

from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset, SpeechToTextDatasetCreator, S2TDataConfig,
    _collate_frames, get_features_or_waveform
)
from fairseq.data import Dictionary, data_utils as fairseq_data_utils


@dataclass
class TextToSpeechDatasetItem(object):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    speaker_id: Optional[int] = None
    duration: Optional[torch.Tensor] = None
    pitch: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None


class TextToSpeechDataset(SpeechToTextDataset):
    def __init__(
            self,
            split: str,
            is_train_split: bool,
            cfg: S2TDataConfig,
            audio_paths: List[str],
            n_frames: List[int],
            src_texts: Optional[List[str]] = None,
            tgt_texts: Optional[List[str]] = None,
            speakers: Optional[List[str]] = None,
            src_langs: Optional[List[str]] = None,
            tgt_langs: Optional[List[str]] = None,
            ids: Optional[List[str]] = None,
            tgt_dict: Optional[Dictionary] = None,
            pre_tokenizer=None,
            bpe_tokenizer=None,
            n_frames_per_step=1,
            speaker_to_id=None,
            durations: Optional[List[List[int]]] = None,
            pitches: Optional[List[str]] = None,
            energies: Optional[List[str]] = None
    ):
        super(TextToSpeechDataset, self).__init__(
            split, is_train_split, cfg, audio_paths, n_frames,
            src_texts=src_texts, tgt_texts=tgt_texts, speakers=speakers,
            src_langs=src_langs, tgt_langs=tgt_langs, ids=ids,
            tgt_dict=tgt_dict, pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer, n_frames_per_step=n_frames_per_step,
            speaker_to_id=speaker_to_id
        )
        self.durations = durations
        self.pitches = pitches
        self.energies = energies

    def __getitem__(self, index: int) -> TextToSpeechDatasetItem:
        s2t_item = super().__getitem__(index)

        duration, pitch, energy = None, None, None
        if self.durations is not None:
            duration = torch.tensor(
                self.durations[index] + [0], dtype=torch.long  # pad 0 for EOS
            )
        if self.pitches is not None:
            pitch = get_features_or_waveform(self.pitches[index])
            pitch = torch.from_numpy(
                np.concatenate((pitch, [0]))  # pad 0 for EOS
            ).float()
        if self.energies is not None:
            energy = get_features_or_waveform(self.energies[index])
            energy = torch.from_numpy(
                np.concatenate((energy, [0]))  # pad 0 for EOS
            ).float()
        return TextToSpeechDatasetItem(
            index=index, source=s2t_item.source, target=s2t_item.target,
            speaker_id=s2t_item.speaker_id, duration=duration, pitch=pitch,
            energy=energy
        )

    def collater(self, samples: List[TextToSpeechDatasetItem]) -> Dict[str, Any]:
        if len(samples) == 0:
            return {}

        src_lengths, order = torch.tensor(
            [s.target.shape[0] for s in samples], dtype=torch.long
        ).sort(descending=True)
        id_ = torch.tensor([s.index for s in samples],
                           dtype=torch.long).index_select(0, order)
        feat = _collate_frames(
            [s.source for s in samples], self.cfg.use_audio_input
        ).index_select(0, order)
        target_lengths = torch.tensor(
            [s.source.shape[0] for s in samples], dtype=torch.long
        ).index_select(0, order)

        src_tokens = fairseq_data_utils.collate_tokens(
            [s.target for s in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        ).index_select(0, order)

        speaker = None
        if self.speaker_to_id is not None:
            speaker = torch.tensor(
                [s.speaker_id for s in samples], dtype=torch.long
            ).index_select(0, order).view(-1, 1)

        bsz, _, d = feat.size()
        prev_output_tokens = torch.cat(
            (feat.new_zeros((bsz, 1, d)), feat[:, :-1, :]), dim=1
        )

        durations, pitches, energies = None, None, None
        if self.durations is not None:
            durations = fairseq_data_utils.collate_tokens(
                [s.duration for s in samples], 0
            ).index_select(0, order)
            assert src_tokens.shape[1] == durations.shape[1]
        if self.pitches is not None:
            pitches = _collate_frames([s.pitch for s in samples], True)
            pitches = pitches.index_select(0, order)
            assert src_tokens.shape[1] == pitches.shape[1]
        if self.energies is not None:
            energies = _collate_frames([s.energy for s in samples], True)
            energies = energies.index_select(0, order)
            assert src_tokens.shape[1] == energies.shape[1]
        src_texts = [self.tgt_dict.string(samples[i].target) for i in order]

        return {
            "id": id_,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "prev_output_tokens": prev_output_tokens,
            },
            "speaker": speaker,
            "target": feat,
            "durations": durations,
            "pitches": pitches,
            "energies": energies,
            "target_lengths": target_lengths,
            "ntokens": sum(target_lengths).item(),
            "nsentences": len(samples),
            "src_texts": src_texts,
        }


class TextToSpeechDatasetCreator(SpeechToTextDatasetCreator):
    KEY_DURATION = "duration"
    KEY_PITCH = "pitch"
    KEY_ENERGY = "energy"

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id
    ) -> TextToSpeechDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        durations = [s.get(cls.KEY_DURATION, None) for s in samples]
        durations = [
            None if dd is None else [int(d) for d in dd.split(" ")]
            for dd in durations
        ]
        durations = None if any(dd is None for dd in durations) else durations

        pitches = [s.get(cls.KEY_PITCH, None) for s in samples]
        pitches = [
            None if pp is None else (audio_root / pp).as_posix()
            for pp in pitches
        ]
        pitches = None if any(pp is None for pp in pitches) else pitches

        energies = [s.get(cls.KEY_ENERGY, None) for s in samples]
        energies = [
            None if ee is None else (audio_root / ee).as_posix()
            for ee in energies]
        energies = None if any(ee is None for ee in energies) else energies

        return TextToSpeechDataset(
            split_name, is_train_split, cfg, audio_paths, n_frames,
            src_texts, tgt_texts, speakers, src_langs, tgt_langs, ids, tgt_dict,
            pre_tokenizer, bpe_tokenizer, n_frames_per_step, speaker_to_id,
            durations, pitches, energies
        )
