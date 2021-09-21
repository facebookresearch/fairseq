# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.abs

import csv
import logging
import os.path as op
from typing import List, Optional

import numpy as np
import torch
from fairseq.data import Dictionary
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig
)
from fairseq.data.audio.text_to_speech_dataset import (
    TextToSpeechDataset, TextToSpeechDatasetCreator
)

logger = logging.getLogger(__name__)


class FrmTextToSpeechDataset(TextToSpeechDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2TDataConfig,
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
        do_chunk=False,
        chunk_bound=-1,
        chunk_init=50,
        chunk_incr=5,
        add_eos=True,
        dedup=True,
        ref_fpu=-1
    ):
        # It assumes texts are encoded at a fixed frame-rate
        super().__init__(
            split=split,
            is_train_split=is_train_split,
            data_cfg=data_cfg,
            audio_paths=audio_paths,
            n_frames=n_frames,
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
            speaker_to_id=speaker_to_id
        )

        self.do_chunk = do_chunk
        self.chunk_bound = chunk_bound
        self.chunk_init = chunk_init
        self.chunk_incr = chunk_incr
        self.add_eos = add_eos
        self.dedup = dedup
        self.ref_fpu = ref_fpu

        self.chunk_size = -1

        if do_chunk:
            assert self.chunk_incr >= 0
            assert self.pre_tokenizer is None

    def __getitem__(self, index):
        index, source, target, speaker_id, _, _, _ = super().__getitem__(index)
        if target[-1].item() == self.tgt_dict.eos_index:
            target = target[:-1]

        fpu = source.size(0) / target.size(0)  # frame-per-unit
        fps = self.n_frames_per_step
        assert (
            self.ref_fpu == -1 or
            abs((fpu * fps - self.ref_fpu) / self.ref_fpu) < 0.1
        ), f"{fpu*fps} != {self.ref_fpu}"

        # only chunk training split
        if self.is_train_split and self.do_chunk and self.chunk_size > 0:
            lang = target[:int(self.data_cfg.prepend_tgt_lang_tag)]
            text = target[int(self.data_cfg.prepend_tgt_lang_tag):]
            size = len(text)
            chunk_size = min(self.chunk_size, size)
            chunk_start = np.random.randint(size - chunk_size + 1)
            text = text[chunk_start:chunk_start+chunk_size]
            target = torch.cat((lang, text), 0)

            f_size = int(np.floor(chunk_size * fpu))
            f_start = int(np.floor(chunk_start * fpu))
            assert(f_size > 0)
            source = source[f_start:f_start+f_size, :]

        if self.dedup:
            target = torch.unique_consecutive(target)

        if self.add_eos:
            eos_idx = self.tgt_dict.eos_index
            target = torch.cat((target, torch.LongTensor([eos_idx])), 0)

        return index, source, target, speaker_id

    def set_epoch(self, epoch):
        if self.is_train_split and self.do_chunk:
            old = self.chunk_size
            self.chunk_size = self.chunk_init + epoch * self.chunk_incr
            if self.chunk_bound > 0:
                self.chunk_size = min(self.chunk_size, self.chunk_bound)
            logger.info((
                f"{self.split}: setting chunk size "
                f"from {old} to {self.chunk_size}"
            ))


class FrmTextToSpeechDatasetCreator(TextToSpeechDatasetCreator):
    # inherit for key names
    @classmethod
    def from_tsv(
        cls,
        root: str,
        data_cfg: S2TDataConfig,
        split: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        n_frames_per_step: int,
        speaker_to_id,
        do_chunk: bool = False,
        chunk_bound: int = -1,
        chunk_init: int = 50,
        chunk_incr: int = 5,
        add_eos: bool = True,
        dedup: bool = True,
        ref_fpu: float = -1
    ) -> FrmTextToSpeechDataset:
        tsv_path = op.join(root, f"{split}.tsv")
        if not op.isfile(tsv_path):
            raise FileNotFoundError(f"Dataset not found: {tsv_path}")
        with open(tsv_path) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            s = [dict(e) for e in reader]
            assert len(s) > 0

        ids = [ss[cls.KEY_ID] for ss in s]
        audio_paths = [
            op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s
        ]
        n_frames = [int(ss[cls.KEY_N_FRAMES]) for ss in s]
        tgt_texts = [ss[cls.KEY_TGT_TEXT] for ss in s]
        src_texts = [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
        speakers = [ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s]
        src_langs = [ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s]
        tgt_langs = [ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s]

        return FrmTextToSpeechDataset(
            split=split,
            is_train_split=is_train_split,
            data_cfg=data_cfg,
            audio_paths=audio_paths,
            n_frames=n_frames,
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
            do_chunk=do_chunk,
            chunk_bound=chunk_bound,
            chunk_init=chunk_init,
            chunk_incr=chunk_incr,
            add_eos=add_eos,
            dedup=dedup,
            ref_fpu=ref_fpu
        )
