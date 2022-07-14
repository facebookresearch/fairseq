# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
from fairseq.data.fairseq_dataset import FairseqDataset
import torch

from typing import Dict, List
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDatasetItem, _collate_frames
from fairseq.data import data_utils as fairseq_data_utils


CONFIGS = {
    "num_positive": 1,
    "num_negative": 5,
    "max_words": 5
}


def split_into_words(samples, tgt_dict):
    words = []
    for x in samples:
        sample_words = []
        start_word_idx = 0
        for ph_idx in range(x.target.size(0)):
            if tgt_dict[x.target[ph_idx]].startswith("\u2581"):
                sample_words.append(x.target[start_word_idx:ph_idx])
                start_word_idx = ph_idx
        sample_words.append(x.target[start_word_idx:ph_idx])
        words.append(sample_words)
    return words



class SpeechTextRetrievalDataset(BaseWrapperDataset):
    def collater(
        self, samples: List[SpeechToTextDatasetItem], return_order: bool = False
    ) -> Dict:
        assert len(samples) > 1

        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        frames = _collate_frames([x.source for x in samples], self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            # We are assuming sentencepiece-like segmentation
            # of target text (phonemes in our case)
            words = split_into_words(samples, self.tgt_dict)
            
            target = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [x.target.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                eos_idx=None,
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(x.target.size(0) for x in samples)

        speaker = None
        if self.speaker_to_id is not None:
            speaker = (
                torch.tensor([s.speaker_id for s in samples], dtype=torch.long)
                .index_select(0, order)
                .view(-1, 1)
            )

        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
            "prev_output_tokens": prev_output_tokens,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": speaker,
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out  