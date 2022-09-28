# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import re
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from fairseq.data import ConcatDataset, Dictionary, FairseqDataset, ResamplingDataset
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data import encoders
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.data_cfg import S2TDataConfig
from fairseq.data.audio.dataset_transforms import CompositeAudioDatasetTransform
from fairseq.data.audio.dataset_transforms.concataugment import ConcatAugment
from fairseq.data.audio.dataset_transforms.noisyoverlapaugment import (
    NoisyOverlapAugment,
)
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.waveform_transforms import CompositeAudioWaveformTransform

logger = logging.getLogger(__name__)


def _collate_frames(
    frames: List[torch.Tensor], is_audio_input: bool = False
) -> torch.Tensor:
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


def _is_int_or_np_int(n):
    return isinstance(n, int) or (
        isinstance(n, np.generic) and isinstance(n.item(), int)
    )


@dataclass
class SpeechToTextDatasetItem(object):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    speaker_id: Optional[int] = None


class SpeechToTextDataset(FairseqDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

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
        append_eos=True,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.cfg = cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
            tgt_dict is not None and tgt_texts is not None
        )
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.speakers = speakers
        self.tgt_dict = tgt_dict
        self.check_tgt_lang_tag()
        self.ids = ids
        self.shuffle = cfg.shuffle if is_train_split else False

        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.cfg.get_feature_transforms(split, is_train_split)
        )
        self.waveform_transforms = CompositeAudioWaveformTransform.from_config_dict(
            self.cfg.get_waveform_transforms(split, is_train_split)
        )
        # TODO: add these to data_cfg.py
        self.dataset_transforms = CompositeAudioDatasetTransform.from_config_dict(
            self.cfg.get_dataset_transforms(split, is_train_split)
        )

        # check proper usage of transforms
        if self.feature_transforms and self.cfg.use_audio_input:
            logger.warning(
                "Feature transforms will not be applied. To use feature transforms, "
                "set use_audio_input as False in config."
            )

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.n_frames_per_step = n_frames_per_step
        self.speaker_to_id = speaker_to_id

        self.tgt_lens = self.get_tgt_lens_and_check_oov()
        self.append_eos = append_eos

        logger.info(self.__repr__())

    def get_tgt_lens_and_check_oov(self):
        if self.tgt_texts is None:
            return [0 for _ in range(self.n_samples)]
        tgt_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_tokenized_tgt_text(i).split(" ")
            oov_tokens = [
                t
                for t in tokenized
                if self.tgt_dict.index(t) == self.tgt_dict.unk_index
            ]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            tgt_lens.append(len(tokenized))
        logger.info(f"'{self.split}' has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return tgt_lens

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"prepend_tgt_lang_tag={self.cfg.prepend_tgt_lang_tag}, "
            f"n_frames_per_step={self.n_frames_per_step}, "
            f"shuffle={self.shuffle}, "
            f"feature_transforms={self.feature_transforms}, "
            f"waveform_transforms={self.waveform_transforms}, "
            f"dataset_transforms={self.dataset_transforms})"
        )

    @classmethod
    def is_lang_tag(cls, token):
        pattern = cls.LANG_TAG_TEMPLATE.replace("{}", "(.*)")
        return re.match(pattern, token)

    def check_tgt_lang_tag(self):
        if self.cfg.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.tgt_langs)
            ]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)

    @classmethod
    def tokenize(cls, tokenizer, text: str):
        return text if tokenizer is None else tokenizer.encode(text)

    def get_tokenized_tgt_text(self, index: Union[int, List[int]]):
        if _is_int_or_np_int(index):
            text = self.tgt_texts[index]
        else:
            text = " ".join([self.tgt_texts[i] for i in index])

        text = self.tokenize(self.pre_tokenizer, text)
        text = self.tokenize(self.bpe_tokenizer, text)
        return text

    def pack_frames(self, feature: torch.Tensor):
        if self.n_frames_per_step == 1:
            return feature
        n_packed_frames = feature.shape[0] // self.n_frames_per_step
        feature = feature[: self.n_frames_per_step * n_packed_frames]
        return feature.reshape(n_packed_frames, -1)

    @classmethod
    def get_lang_tag_idx(cls, lang: str, dictionary: Dictionary):
        lang_tag_idx = dictionary.index(cls.LANG_TAG_TEMPLATE.format(lang))
        assert lang_tag_idx != dictionary.unk()
        return lang_tag_idx

    def _get_source_audio(self, index: Union[int, List[int]]) -> torch.Tensor:
        """
        Gives source audio for given index with any relevant transforms
        applied. For ConcatAug, source audios for given indices are
        concatenated in given order.
        Args:
            index (int or List[int]): index—or in the case of ConcatAug,
            indices—to pull the source audio for
        Returns:
            source audios concatenated for given indices with
            relevant transforms appplied
        """
        if _is_int_or_np_int(index):
            source = get_features_or_waveform(
                self.audio_paths[index],
                need_waveform=self.cfg.use_audio_input,
                use_sample_rate=self.cfg.use_sample_rate,
                waveform_transforms=self.waveform_transforms,
            )
        else:
            source = np.concatenate(
                [
                    get_features_or_waveform(
                        self.audio_paths[i],
                        need_waveform=self.cfg.use_audio_input,
                        use_sample_rate=self.cfg.use_sample_rate,
                        waveform_transforms=self.waveform_transforms,
                    )
                    for i in index
                ]
            )
        if self.cfg.use_audio_input:
            source = torch.from_numpy(source).float()
            if self.cfg.standardize_audio:
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
        else:
            if self.feature_transforms is not None:
                source = self.feature_transforms(source)
            source = torch.from_numpy(source).float()
        return source

    def __getitem__(self, index: int) -> SpeechToTextDatasetItem:
        has_concat = self.dataset_transforms.has_transform(ConcatAugment)
        if has_concat:
            concat = self.dataset_transforms.get_transform(ConcatAugment)
            indices = concat.find_indices(index, self.n_frames, self.n_samples)

        source = self._get_source_audio(indices if has_concat else index)
        source = self.pack_frames(source)

        target = None
        if self.tgt_texts is not None:
            tokenized = self.get_tokenized_tgt_text(indices if has_concat else index)
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=self.append_eos
            ).long()
            if self.cfg.prepend_tgt_lang_tag:
                lang_tag_idx = self.get_lang_tag_idx(
                    self.tgt_langs[index], self.tgt_dict
                )
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)

        if self.cfg.prepend_bos_and_append_tgt_lang_tag:
            bos = torch.LongTensor([self.tgt_dict.bos()])
            lang_tag_idx = self.get_lang_tag_idx(self.tgt_langs[index], self.tgt_dict)
            assert lang_tag_idx != self.tgt_dict.unk()
            lang_tag_idx = torch.LongTensor([lang_tag_idx])
            target = torch.cat((bos, target, lang_tag_idx), 0)

        speaker_id = None
        if self.speaker_to_id is not None:
            speaker_id = self.speaker_to_id[self.speakers[index]]
        return SpeechToTextDatasetItem(
            index=index, source=source, target=target, speaker_id=speaker_id
        )

    def __len__(self):
        return self.n_samples

    def collater(
        self, samples: List[SpeechToTextDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)

        sources = [x.source for x in samples]
        has_NOAug = self.dataset_transforms.has_transform(NoisyOverlapAugment)
        if has_NOAug and self.cfg.use_audio_input:
            NOAug = self.dataset_transforms.get_transform(NoisyOverlapAugment)
            sources = NOAug(sources)

        frames = _collate_frames(sources, self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.size(0) for x in sources], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
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

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        return self.n_frames[index], self.tgt_lens[index]

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False


class TextTargetMultitaskData(object):
    # mandatory columns
    KEY_ID, KEY_TEXT = "id", "tgt_text"

    def __init__(self, args, split, tgt_dict):
        samples = SpeechToTextDatasetCreator._load_samples_from_tsv(args.data, split)
        self.data = {s[self.KEY_ID]: s[self.KEY_TEXT] for s in samples}
        self.dict = tgt_dict
        self.append_eos = args.decoder_type != "ctc"
        self.pre_tokenizer = self.build_tokenizer(args)
        self.bpe_tokenizer = self.build_bpe(args)

    @classmethod
    def tokenize(cls, tokenizer, text: str):
        return text if tokenizer is None else tokenizer.encode(text)

    def get_tokenized_tgt_text(self, index: int):
        text = self.tokenize(self.pre_tokenizer, self.data[index])
        text = self.tokenize(self.bpe_tokenizer, text)
        return text

    def build_tokenizer(self, args):
        pre_tokenizer = args.config.get("pre_tokenizer")
        if pre_tokenizer is not None:
            logger.info(f"pre-tokenizer: {pre_tokenizer}")
            return encoders.build_tokenizer(Namespace(**pre_tokenizer))
        else:
            return None

    def build_bpe(self, args):
        bpe_tokenizer = args.config.get("bpe_tokenizer")
        if bpe_tokenizer is not None:
            logger.info(f"tokenizer: {bpe_tokenizer}")
            return encoders.build_bpe(Namespace(**bpe_tokenizer))
        else:
            return None

    def get(self, sample_id):
        if sample_id in self.data:
            tokenized = self.get_tokenized_tgt_text(sample_id)
            return self.dict.encode_line(
                tokenized,
                add_if_not_exist=False,
                append_eos=self.append_eos,
            )
        else:
            logger.warning(f"no target for {sample_id}")
            return torch.IntTensor([])

    def collater(self, samples: List[torch.Tensor]) -> torch.Tensor:
        out = fairseq_data_utils.collate_tokens(
            samples,
            self.dict.pad(),
            eos_idx=self.dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        ).long()

        prev_out = fairseq_data_utils.collate_tokens(
            samples,
            self.dict.pad(),
            eos_idx=self.dict.eos(),
            left_pad=False,
            move_eos_to_beginning=True,
        ).long()

        target_lengths = torch.tensor([t.size(0) for t in samples], dtype=torch.long)
        ntokens = sum(t.size(0) for t in samples)

        output = {
            "prev_output_tokens": prev_out,
            "target": out,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
        }

        return output


class SpeechToTextMultitaskDataset(SpeechToTextDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multitask_data = {}

    def add_multitask_dataset(self, task_name, task_data):
        self.multitask_data[task_name] = task_data

    def __getitem__(
        self, index: int
    ) -> Tuple[SpeechToTextDatasetItem, Dict[str, torch.Tensor]]:
        s2t_data = super().__getitem__(index)

        multitask_target = {}
        sample_id = self.ids[index]
        tgt_lang = self.tgt_langs[index]
        for task_name, task_dataset in self.multitask_data.items():
            multitask_target[task_name] = task_dataset.get(sample_id, tgt_lang)

        return s2t_data, multitask_target

    def collater(
        self, samples: List[Tuple[SpeechToTextDatasetItem, Dict[str, torch.Tensor]]]
    ) -> Dict:
        if len(samples) == 0:
            return {}

        out = super().collater([s for s, _ in samples], return_order=True)
        order = out["order"]
        del out["order"]

        for task_name, task_dataset in self.multitask_data.items():
            if "multitask" not in out:
                out["multitask"] = {}
            d = [s[task_name] for _, s in samples]
            task_target = task_dataset.collater(d)
            out["multitask"][task_name] = {
                "target": task_target["target"].index_select(0, order),
                "target_lengths": task_target["target_lengths"].index_select(0, order),
                "ntokens": task_target["ntokens"],
            }
            out["multitask"][task_name]["net_input"] = {
                "prev_output_tokens": task_target["prev_output_tokens"].index_select(
                    0, order
                ),
            }

        return out


class SpeechToTextDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
    KEY_TGT_TEXT = "tgt_text"
    # optional columns
    KEY_SPEAKER, KEY_SRC_TEXT = "speaker", "src_text"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""

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
        speaker_to_id,
        multitask: Optional[Dict] = None,
    ) -> SpeechToTextDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        has_multitask = len(multitask) > 0
        dataset_cls = (
            SpeechToTextMultitaskDataset if has_multitask else SpeechToTextDataset
        )

        ds = dataset_cls(
            split=split_name,
            is_train_split=is_train_split,
            cfg=cfg,
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
        )

        if has_multitask:
            for task_name, task_obj in multitask.items():
                task_data = TextTargetMultitaskData(
                    task_obj.args, split_name, task_obj.target_dictionary
                )
                ds.add_multitask_dataset(task_name, task_data)
        return ds

    @classmethod
    def get_size_ratios(
        cls, datasets: List[SpeechToTextDataset], alpha: float = 1.0
    ) -> List[float]:
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""

        id_to_lp, lp_to_sz = {}, defaultdict(int)
        for ds in datasets:
            lang_pairs = {f"{s}->{t}" for s, t in zip(ds.src_langs, ds.tgt_langs)}
            assert len(lang_pairs) == 1
            lang_pair = list(lang_pairs)[0]
            id_to_lp[ds.split] = lang_pair
            lp_to_sz[lang_pair] += sum(ds.n_frames)

        sz_sum = sum(v for v in lp_to_sz.values())
        lp_to_prob = {k: v / sz_sum for k, v in lp_to_sz.items()}
        lp_to_tgt_prob = {k: v**alpha for k, v in lp_to_prob.items()}
        prob_sum = sum(v for v in lp_to_tgt_prob.values())
        lp_to_tgt_prob = {k: v / prob_sum for k, v in lp_to_tgt_prob.items()}
        lp_to_sz_ratio = {
            k: (lp_to_tgt_prob[k] * sz_sum) / v for k, v in lp_to_sz.items()
        }
        size_ratio = [lp_to_sz_ratio[id_to_lp[ds.split]] for ds in datasets]

        p_formatted = {
            k: f"{lp_to_prob[k]:.3f}->{lp_to_tgt_prob[k]:.3f}" for k in lp_to_sz
        }
        logger.info(f"sampling probability balancing: {p_formatted}")
        sr_formatted = {ds.split: f"{r:.3f}" for ds, r in zip(datasets, size_ratio)}
        logger.info(f"balanced sampling size ratio: {sr_formatted}")
        return size_ratio

    @classmethod
    def _load_samples_from_tsv(cls, root: str, split: str):
        tsv_path = Path(root) / f"{split}.tsv"
        if not tsv_path.is_file():
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
            samples = [dict(e) for e in reader]
        if len(samples) == 0:
            raise ValueError(f"Empty manifest: {tsv_path}")
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
        multitask: Optional[Dict] = None,
    ) -> SpeechToTextDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            cfg,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            n_frames_per_step,
            speaker_to_id,
            multitask,
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
        n_frames_per_step: int = 1,
        speaker_to_id=None,
        multitask: Optional[Dict] = None,
    ) -> SpeechToTextDataset:
        datasets = [
            cls._from_tsv(
                root=root,
                cfg=cfg,
                split=split,
                tgt_dict=tgt_dict,
                is_train_split=is_train_split,
                pre_tokenizer=pre_tokenizer,
                bpe_tokenizer=bpe_tokenizer,
                n_frames_per_step=n_frames_per_step,
                speaker_to_id=speaker_to_id,
                multitask=multitask,
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
