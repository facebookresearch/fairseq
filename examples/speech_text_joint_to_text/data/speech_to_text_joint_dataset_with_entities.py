# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import torch

from fairseq.data import Dictionary
from fairseq.data.audio.speech_to_text_joint_dataset import \
    S2TJointDataConfig, SpeechToTextJointDataset, SpeechToTextJointDatasetCreator

logger = logging.getLogger(__name__)


class SpeechToTextJointDatasetWithEntitiesItem(NamedTuple):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    src_txt_tokens: Optional[torch.Tensor] = None
    tgt_lang_tag: Optional[int] = None
    src_lang_tag: Optional[int] = None
    tgt_alignment: Optional[torch.Tensor] = None
    entities: Optional[torch.Tensor] = None


class SpeechToTextJointWithEntitiesDataset(SpeechToTextJointDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: S2TJointDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        entities: Optional[List[List[str]]] = None,
        tgt_dict: Optional[Dictionary] = None,
        src_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        src_pre_tokenizer=None,
        src_bpe_tokenizer=None,
        append_eos: Optional[bool] = True,
        alignment: Optional[List[str]] = None,
        use_src_lang_id: Optional[int] = 0,
    ):
        super().__init__(
            split,
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
            src_dict=src_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            append_eos=append_eos,
            alignment=alignment,
            use_src_lang_id=use_src_lang_id,
        )
        self.entities = entities
        self.source_entities = True

    def __getitem__(self, index: int) -> SpeechToTextJointDatasetWithEntitiesItem:
        st2t_dataset_item = super().__getitem__(index)
        tokenized_entities = None
        if self.source_entities:
            if self.entities is not None:
                tokenized_entities = [
                    self.src_dict.encode_line(
                        self.tokenize(self.src_bpe_tokenizer, self.tokenize(self.src_pre_tokenizer, e.strip())),
                        add_if_not_exist=False,
                        append_eos=False).long()
                    for e in self.entities[index]]
        else:
            if self.entities is not None:
                tokenized_entities = [
                    self.tgt_dict.encode_line(
                        self.tokenize(self.bpe_tokenizer, self.tokenize(self.pre_tokenizer, e.strip())).strip('▁ '),
                        add_if_not_exist=False,
                        append_eos=False).long()
                    for e in self.entities[index]]
        return SpeechToTextJointDatasetWithEntitiesItem(
            index=st2t_dataset_item.index,
            source=st2t_dataset_item.source,
            target=st2t_dataset_item.target,
            src_txt_tokens=st2t_dataset_item.src_txt_tokens,
            tgt_lang_tag=st2t_dataset_item.tgt_lang_tag,
            src_lang_tag=st2t_dataset_item.src_lang_tag,
            tgt_alignment=st2t_dataset_item.tgt_alignment,
            entities=tokenized_entities
        )


class SpeechToTextJointWithEntitiesDatasetCreator(SpeechToTextJointDatasetCreator):
    KEY_ENTITIES = "entities"

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TJointDataConfig,
        tgt_dict,
        src_dict,
        pre_tokenizer,
        bpe_tokenizer,
        src_pre_tokenizer,
        src_bpe_tokenizer,
        append_eos,
        use_src_lang_id,
    ) -> SpeechToTextJointWithEntitiesDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        def split_entities(entities_string):
            if entities_string == "":
                return []
            return entities_string.split(";")

        entities = [split_entities(s.get(cls.KEY_ENTITIES, "")) for s in samples]
        tgt_alignment = None
        if cls.KEY_ALIGN in samples[0].keys():
            tgt_alignment = [s[cls.KEY_ALIGN] for s in samples]
        return SpeechToTextJointWithEntitiesDataset(
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
            entities=entities,
            tgt_dict=tgt_dict,
            src_dict=src_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            append_eos=append_eos,
            alignment=tgt_alignment,
            use_src_lang_id=use_src_lang_id,
        )
