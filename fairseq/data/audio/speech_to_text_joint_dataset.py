# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple

import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    S2TDataConfig,
    SpeechToTextDatasetCreator,
)


logger = logging.getLogger(__name__)


class S2TJointDataConfig(S2TDataConfig):
    """Wrapper class for data config YAML"""

    @property
    def src_vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("src_vocab_filename", "src_dict.txt")

    @property
    def src_pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("src_pre_tokenizer", {"tokenizer": None})

    @property
    def src_bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply on source text after pre-tokenization.
        Returning a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("src_bpe_tokenizer", {"bpe": None})

    @property
    def prepend_tgt_lang_tag_no_change(self) -> bool:
        """Prepend target lang ID token as the prev_output_tokens BOS (e.g. for
        to-many multilingual setting). No change needed during inference.
        """
        return self.config.get("prepend_tgt_lang_tag_no_change", False)


class SpeechToTextJointDatasetItem(NamedTuple):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    src_txt_tokens: Optional[torch.Tensor] = None
    tgt_lang_tag: Optional[int] = None


class SpeechToTextJointDataset(SpeechToTextDataset):
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
        tgt_dict: Optional[Dictionary] = None,
        src_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        src_pre_tokenizer=None,
        src_bpe_tokenizer=None,
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
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
        )

        self.src_dict = src_dict
        self.src_pre_tokenizer = src_pre_tokenizer
        self.src_bpe_tokenizer = src_bpe_tokenizer

    def get_tokenized_src_text(self, index: int):
        text = self.tokenize(self.src_pre_tokenizer, self.src_texts[index])
        text = self.tokenize(self.src_bpe_tokenizer, text)
        return text

    def __getitem__(self, index: int) -> SpeechToTextJointDatasetItem:
        s2t_dataset_item = super().__getitem__(index)
        src_tokens = None
        if self.src_texts is not None and self.src_dict is not None:
            src_tokens = self.get_tokenized_src_text(index)
            src_tokens = self.src_dict.encode_line(
                src_tokens, add_if_not_exist=False, append_eos=True
            ).long()
        tgt_lang_tag = None
        if self.cfg.prepend_tgt_lang_tag_no_change:
            # prepend_tgt_lang_tag_no_change: modify prev_output_tokens instead
            tgt_lang_tag = self.get_lang_tag_idx(self.tgt_langs[index], self.tgt_dict)

        return SpeechToTextJointDatasetItem(
            index=index,
            source=s2t_dataset_item.source,
            target=s2t_dataset_item.target,
            src_txt_tokens=src_tokens,
            tgt_lang_tag=tgt_lang_tag,
        )

    def __len__(self):
        return self.n_samples

    def collater(self, samples: List[SpeechToTextJointDatasetItem]) -> Dict:
        s2t_out = super().collater(samples, return_order=True)
        if s2t_out == {}:
            return s2t_out
        net_input, order = s2t_out["net_input"], s2t_out["order"]

        if self.src_texts is not None and self.src_dict is not None:
            src_txt_tokens = fairseq_data_utils.collate_tokens(
                [x.src_txt_tokens for x in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            src_txt_tokens = src_txt_tokens.index_select(0, order)
            src_txt_lengths = torch.tensor(
                [x.src_txt_tokens.size()[0] for x in samples], dtype=torch.long
            ).index_select(0, order)
            net_input["src_txt_tokens"] = src_txt_tokens
            net_input["src_txt_lengths"] = src_txt_lengths

        if self.tgt_texts is not None and samples[0].tgt_lang_tag is not None:
            for i in range(len(samples)):
                net_input["prev_output_tokens"][i][0] = samples[order[i]].tgt_lang_tag

        out = {
            "id": s2t_out["id"],
            "net_input": net_input,
            "target": s2t_out["target"],
            "target_lengths": s2t_out["target_lengths"],
            "ntokens": s2t_out["ntokens"],
            "nsentences": len(samples),
        }
        return out


class SpeechToTextJointDatasetCreator(SpeechToTextDatasetCreator):
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
    ) -> SpeechToTextJointDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]
        return SpeechToTextJointDataset(
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
            src_dict=src_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
        )

    @classmethod
    def _from_tsv(
        cls,
        root: str,
        cfg: S2TJointDataConfig,
        split: str,
        tgt_dict,
        src_dict,
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
        src_pre_tokenizer,
        src_bpe_tokenizer,
    ) -> SpeechToTextJointDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            cfg,
            tgt_dict,
            src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            src_pre_tokenizer,
            src_bpe_tokenizer,
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2TJointDataConfig,
        splits: str,
        tgt_dict,
        src_dict,
        pre_tokenizer,
        bpe_tokenizer,
        src_pre_tokenizer,
        src_bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
    ) -> SpeechToTextJointDataset:
        datasets = [
            cls._from_tsv(
                root,
                cfg,
                split,
                tgt_dict,
                src_dict,
                is_train_split,
                pre_tokenizer,
                bpe_tokenizer,
                src_pre_tokenizer,
                src_bpe_tokenizer,
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
