# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from fairseq.data import (FairseqDataset, Dictionary,
                          data_utils as fairseq_data_utils)
from fairseq.data.audio.feature_fetcher import fetch_features
from fairseq.data.audio.transforms import CompositeTransform

logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO
)
logger = logging.getLogger(__name__)


def _collate_frames(frames: List[torch.Tensor],
                    is_audio_input: bool = False) -> torch.Tensor:
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    if is_audio_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out


class SpeechToTextDataset(FairseqDataset):
    """
    A dataset representing speech and corresponding transcription.

    Args:
        audio_paths: (List[str]): A list of str with paths to audio
            feature files.
        n_frames (List[int]): A list of int containing the durations (ms) of
            audio files.
        tgt_dict (~fairseq.data.Dictionary): target vocabulary.
        ids (List[str]): A list of utterance IDs.
    """

    LANG_TAG_TEMPLATE = '<lang:{}>'

    def __init__(
            self,
            split: str,
            is_train_split: bool,
            config: Dict,
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
    ):
        self.split, self.is_train_split = split, is_train_split
        self.config = config
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or \
               (tgt_dict is not None and tgt_texts is not None)
        self.tgt_dict = tgt_dict
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.ids = ids
        self.shuffle = config.get('shuffle', False) if is_train_split else False

        self.transforms = self.get_transforms(split, is_train_split)

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer

        self.use_audio_input = config.get('use_audio_input', False)

        self.prepend_tgt_lang_tag = config.get('prepend_tgt_lang_tag', False)
        self.check_tgt_lang_tag()

        logger.info(self.__repr__())

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(split="{self.split}", n_samples={self.n_samples}, ' \
               f'prepend_tgt_lang_tag={self.prepend_tgt_lang_tag}, ' \
               f'shuffle={self.shuffle}, transforms={self.transforms})'

    def get_transforms(self, split, is_train):
        from copy import deepcopy
        cfg = deepcopy(self.config)
        _cur = cfg.get('transforms', {})
        cur = _cur.get(split)
        cur = _cur.get('_train') if cur is None and is_train else cur
        cur = _cur.get('_eval') if cur is None and not is_train else cur
        cur = _cur.get('_all') if cur is None else cur
        cfg['transforms'] = cur
        return CompositeTransform.from_config_dict(cfg)

    @classmethod
    def is_lang_tag(cls, token):
        pattern = cls.LANG_TAG_TEMPLATE.replace('{}', '(.*)')
        return re.match(pattern, token)

    def check_tgt_lang_tag(self):
        if self.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [self.LANG_TAG_TEMPLATE.format(t)
                             for t in set(self.tgt_langs)]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)

    def tokenize_text(self, text: str):
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.bpe_tokenizer is not None:
            text = self.bpe_tokenizer.encode(text)
        return text

    def __getitem__(
            self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        source = fetch_features(self.audio_paths[index],
                                use_audio=self.use_audio_input)

        if self.transforms is not None:
            source = self.transforms(source)
        source = torch.from_numpy(source).float()

        target = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            if self.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                tokenized = lang_tag + ' ' + tokenized
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
        return index, source, target

    def __len__(self):
        return self.n_samples

    def collater(
            self, samples: List[Tuple[int, torch.Tensor, torch.Tensor]]
    ) -> dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _ in samples], dtype=torch.long)
        frames = _collate_frames([s for _, s, _ in samples],
                                 self.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor(
            [s.size(0) for _, s, _ in samples], dtype=torch.long
        )
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t in samples], self.tgt_dict.pad(),
                self.tgt_dict.eos(), left_pad=False, move_eos_to_beginning=False
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, t in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t in samples], self.tgt_dict.pad(),
                self.tgt_dict.eos(), left_pad=False, move_eos_to_beginning=True
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t.size(0) for _, _, t in samples)

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if self.use_audio_input:
            padding_mask = None
            if len(n_frames) > 1:
                padding_mask = frames.new_full(frames.size(), True,
                                               dtype=torch.bool)
                for i, l in enumerate(n_frames):
                    padding_mask[i, :l] = False
            out['net_input']['padding_mask'] = padding_mask
        return out

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        t_len = 0
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            t_len = len(tokenized.split(' '))
        return self.n_frames[index], t_len

    @property
    def sizes(self):
        return np.array(self.n_frames)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of frame counts
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        raise False
