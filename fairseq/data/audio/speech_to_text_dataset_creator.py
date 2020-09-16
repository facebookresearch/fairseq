# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as op
import logging
from typing import Dict, List
import csv

import numpy as np

from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset
from fairseq.data import ConcatDataset, ResamplingDataset

logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
logger = logging.getLogger(__name__)


class SpeechToTextDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = 'id', 'audio', 'n_frames'
    KEY_TGT_TEXT = 'tgt_text'
    # optional columns
    KEY_SPEAKER, KEY_SRC_TEXT = 'speaker', 'src_text'
    KEY_SRC_LANG, KEY_TGT_LANG = 'src_lang', 'tgt_lang'
    # default values
    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ''

    @classmethod
    def _from_list(cls, split_name: str, is_train_split,
                   samples: List[List[Dict]], config: Dict, tgt_dict,
                   pre_tokenizer, bpe_tokenizer) -> SpeechToTextDataset:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []
        audio_root = config.get('audio_root', '')
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend([op.join(audio_root, ss[cls.KEY_AUDIO])
                                for ss in s])
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend([ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT)
                              for ss in s])
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER)
                             for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG)
                              for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG)
                              for ss in s])
        return SpeechToTextDataset(
            split_name, is_train_split, config, audio_paths, n_frames,
            src_texts, tgt_texts, speakers, src_langs, tgt_langs, ids, tgt_dict,
            pre_tokenizer, bpe_tokenizer
        )

    @classmethod
    def _get_size_ratios(cls, ids: List[str], sizes: List[int],
                         alpha: float = 1.):
        _sizes = np.array(sizes)
        prob = _sizes / _sizes.sum()
        smoothed_prob = prob ** alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        size_ratio = (smoothed_prob * _sizes.sum()) / _sizes

        o_str = str({_i: f"{prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"original sampling probability: {o_str}")
        p_str = str({_i: f"{smoothed_prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"balanced sampling probability: {p_str}")
        sr_str = str({_id: f"{size_ratio[i]:.3f}" for i, _id in enumerate(ids)})
        logger.info(f"balanced sampling size ratio: {sr_str}")
        return size_ratio.tolist()

    @classmethod
    def from_tsv(cls, root: str, config: Dict, splits: str, tgt_dict,
                 pre_tokenizer, bpe_tokenizer, is_train_split: bool, epoch: int,
                 seed: int) -> SpeechToTextDataset:
        samples = []
        _splits = splits.split(',')
        for split in _splits:
            tsv_path = op.join(root, f'{split}.tsv')
            if not op.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            with open(tsv_path) as f:
                reader = csv.DictReader(
                    f, delimiter='\t', quotechar=None, doublequote=False,
                    lineterminator='\n', quoting=csv.QUOTE_NONE
                )
                samples.append([dict(e) for e in reader])
                assert len(samples) > 0

        datasets = [cls._from_list(name, is_train_split, [s], config, tgt_dict,
                                   pre_tokenizer, bpe_tokenizer)
                    for name, s in zip(_splits, samples)]

        sampling_alpha = config.get('sampling_alpha', 1.)
        if is_train_split and len(_splits) > 1 and sampling_alpha != 1.:
            # balanced sampling
            size_ratios = cls._get_size_ratios(
                _splits, [len(s) for s in samples], alpha=sampling_alpha
            )
            datasets = [
                ResamplingDataset(d, size_ratio=r, seed=seed, epoch=epoch,
                                  replace=(r >= 1.))
                for d, r in zip(datasets, size_ratios)
            ]
        return ConcatDataset(datasets)
