# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset
)
from fairseq.data.audio.data_cfg import S2TDataConfig
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDatasetItem,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator
)

logger = logging.getLogger(__name__)


@dataclass
class SpeechToTextDatasetItemWithDomain(SpeechToTextDatasetItem):
    src_lang_id: Optional[torch.Tensor] = None
    tgt_lang_id: Optional[torch.Tensor] = None
    domain_id: Optional[torch.Tensor] = None


class SpeechToTextDatasetWithDomain(SpeechToTextDataset):

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
        src_lang_ids: Optional[List[int]] = None,
        tgt_lang_ids: Optional[List[int]] = None,
        domain_ids: Optional[List[int]] = None
    ):
        super().__init__(
            split, is_train_split, cfg, audio_paths, n_frames,
            src_texts, tgt_texts, speakers, src_langs, tgt_langs,
            ids, tgt_dict, pre_tokenizer, bpe_tokenizer,
            n_frames_per_step, speaker_to_id
        )
        assert src_lang_ids is None or len(src_lang_ids) == self.n_samples
        assert tgt_lang_ids is None or len(tgt_lang_ids) == self.n_samples
        assert domain_ids is None or len(domain_ids) == self.n_samples

        self.src_lang_ids = src_lang_ids
        self.tgt_lang_ids = tgt_lang_ids
        self.domain_ids = domain_ids

    def __getitem__(self, index: int) -> SpeechToTextDatasetItemWithDomain:
        item = super().__getitem__(index)
        src_lang_id = self.src_lang_ids[index]
        tgt_lang_id = self.tgt_lang_ids[index]
        domain_id = self.domain_ids[index]
        return SpeechToTextDatasetItemWithDomain(
            index=item.index, source=item.source,
            target=item.target, speaker_id=item.speaker_id,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id,
            domain_id=domain_id
        )

    def collater(
        self, samples: List[SpeechToTextDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        out = super().collater(samples, return_order=True)
        order = out["order"]
        src_lang_ids = torch.tensor([x.src_lang_id for x in samples], dtype=torch.long).index_select(0, order)
        tgt_lang_ids = torch.tensor([x.tgt_lang_id for x in samples], dtype=torch.long).index_select(0, order)
        domain_ids = torch.tensor([x.domain_id for x in samples], dtype=torch.long).index_select(0, order)

        out["src_lang_ids"] = src_lang_ids
        out["tgt_lang_ids"] = tgt_lang_ids
        out["domain_ids"] = domain_ids
        if not return_order:
            del out["order"]
        return out


class SpeechToTextDatasetCreatorWithDomain(SpeechToTextDatasetCreator):
    KEY_SRC_LANG_ID, KEY_TGT_LANG_ID = "src_lang_id", "tgt_lang_id"
    KEY_DOMAIN_ID = "domain_id"
    # default values
    DEFAULT_SRC_LANG_ID, DEFAULT_TGT_LANG_ID, DEFAULT_DOMAIN_ID = 0, 0, 0

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
    ) -> SpeechToTextDatasetWithDomain:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]
        src_lang_ids = [s.get(cls.KEY_SRC_LANG_ID, cls.DEFAULT_SRC_LANG_ID) for s in samples]
        tgt_lang_ids = [s.get(cls.KEY_TGT_LANG_ID, cls.DEFAULT_TGT_LANG_ID) for s in samples]
        domain_ids = [s.get(cls.KEY_DOMAIN_ID, cls.DEFAULT_DOMAIN_ID) for s in samples]
        return SpeechToTextDatasetWithDomain(
            split_name,
            is_train_split,
            cfg,
            audio_paths,
            n_frames,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            speakers=speakers,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            tgt_dict=tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            n_frames_per_step=n_frames_per_step,
            speaker_to_id=speaker_to_id,
            src_lang_ids=src_lang_ids,
            tgt_lang_ids=tgt_lang_ids,
            domain_ids=domain_ids
        )

    @classmethod
    def _load_samples_from_tsv(
        cls,
        root: str,
        split: str,
        src_lang_map,
        tgt_lang_map,
        domain_map
    ):
        # metadata from split
        _, src_lang, tgt_lang, domain = split.split("_")
        src_lang_id = src_lang_map[src_lang]
        tgt_lang_id = tgt_lang_map[tgt_lang]
        domain_id = domain_map[domain]

        samples = SpeechToTextDatasetCreator._load_samples_from_tsv(root, split)
        for s in samples:
            s.update({
                cls.KEY_SRC_LANG_ID: src_lang_id,
                cls.KEY_TGT_LANG_ID: tgt_lang_id,
                cls.KEY_DOMAIN_ID: domain_id
            })
        return samples

    @classmethod
    def _from_tsv(
        cls,
        root: str,
        cfg: S2TDataConfig,
        split: str,
        tgt_dict,
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id,
        src_lang_map: Dict[str, int],
        tgt_lang_map: Dict[str, int],
        domain_map: Dict[str, int]
    ) -> SpeechToTextDatasetItemWithDomain:
        samples = cls._load_samples_from_tsv(
            root, split, src_lang_map,
            tgt_lang_map, domain_map
        )
        return cls._from_list(
            split, is_train_split, samples, cfg, tgt_dict, pre_tokenizer,
            bpe_tokenizer, n_frames_per_step, speaker_to_id
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2TDataConfig,
        splits: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        src_lang_map: Dict[str, int],
        tgt_lang_map: Dict[str, int],
        domain_map: Dict[str, int],
        n_frames_per_step: int = 1,
        speaker_to_id=None
    ) -> SpeechToTextDatasetWithDomain:
        datasets = [
            cls._from_tsv(
                root, cfg, split, tgt_dict, is_train_split, pre_tokenizer, bpe_tokenizer, n_frames_per_step, speaker_to_id, src_lang_map, tgt_lang_map, domain_map
            )
            for split in splits.split(",")
        ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
