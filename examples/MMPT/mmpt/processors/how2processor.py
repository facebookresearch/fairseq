# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) Facebook, Inc. All Rights Reserved


import torch
import math
import pickle
import random
import os
import numpy as np

from collections import deque
from typing import Optional, Tuple, List
from .processor import (
    Processor,
    MetaProcessor,
    TextProcessor,
    Aligner,
    MMAttentionMask2DProcessor
)

from ..utils import ShardedTensor


class How2MetaProcessor(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        path = self._get_split_path(config)
        with open(path) as fd:
            self.data = [line.strip() for line in fd]

    def __getitem__(self, idx):
        video_id = self.data[idx]
        return video_id, video_id


class ShardedHow2MetaProcessor(How2MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.split = str(config.split)
        self.vfeat_dir = config.vfeat_dir
        self._init_shard()

    def _init_shard(self):
        if self.split == "train":
            meta_fn = os.path.join(self.vfeat_dir, "train" + "_meta.pkl")
            with open(meta_fn, "rb") as fr:
                meta = pickle.load(fr)
        elif self.split == "valid":
            meta_fn = os.path.join(self.vfeat_dir, "val" + "_meta.pkl")
            with open(meta_fn, "rb") as fr:
                meta = pickle.load(fr)
        elif self.split == "test":
            print("use how2 val as test.")
            meta_fn = os.path.join(self.vfeat_dir, "val" + "_meta.pkl")
            with open(meta_fn, "rb") as fr:
                meta = pickle.load(fr)
        else:
            raise ValueError("unsupported for MetaProcessor:", self.split)
        video_id_to_shard = {}
        for shard_id in meta:
            for video_idx, video_id in enumerate(meta[shard_id]):
                video_id_to_shard[video_id] = (shard_id, video_idx)
        self.video_id_to_shard = video_id_to_shard

    def __getitem__(self, idx):
        video_id, video_id = super().__getitem__(idx)
        shard_id, shard_idx = self.video_id_to_shard[video_id]
        meta = (video_id, idx, shard_id, shard_idx)
        return meta, meta


class ShardedVideoProcessor(Processor):
    """
    mmaped shards of numpy video features.
    """

    def __init__(self, config):
        self.split = str(config.split)
        self.vfeat_dir = config.vfeat_dir

    def __call__(self, video_id):
        _, _, shard_id, video_idx = video_id
        if self.split == "train":
            shard = ShardedTensor.load(
                os.path.join(self.vfeat_dir, "train" + "_" + str(shard_id)),
                "r"
            )
        elif self.split == "valid":
            shard = ShardedTensor.load(
                os.path.join(self.vfeat_dir, "val" + "_" + str(shard_id)),
                "r"
            )
        elif self.split == "test":
            shard = ShardedTensor.load(
                os.path.join(self.vfeat_dir, "val" + "_" + str(shard_id)),
                "r"
            )
        else:
            raise ValueError("unknown split", self.split)
        feat = shard[video_idx]
        return feat


class ShardedTextProcessor(Processor):
    def __init__(self, config):
        self.tfeat_dir = str(config.tfeat_dir)
        self.split = str(config.split)

    def __call__(self, video_id):
        _, _, shard_id, shard_idx = video_id
        if self.split == "train":
            target_path = self.tfeat_dir + "train" + "_" + str(shard_id)
        elif self.split == "valid":
            target_path = self.tfeat_dir + "val" + "_" + str(shard_id)
        elif self.split == "test":
            target_path = self.tfeat_dir + "val" + "_" + str(shard_id)
        else:
            raise ValueError("unknown split", self.split)

        startend = ShardedTensor.load(
            target_path + ".startends", "r")[shard_idx]
        cap_ids = ShardedTensor.load(
            target_path + ".caps_ids", "r")[shard_idx]
        cap = []
        for clip_idx in range(len(cap_ids)):
            clip = cap_ids[clip_idx]
            cap.append(clip[clip != -1].tolist())
        start, end = startend[:, 0].tolist(), startend[:, 1].tolist()
        return {"start": start, "end": end, "cap": cap}


class FixedLenAligner(Aligner):
    """
    In the model we assume text is on the left (closer to BERT formulation)
    and video is on the right.
    We fix the total length of text + video.
    max_video_len is in number of secs.
    max_text_len is in number of tokens.

    special tokens formats:
    we use the format [CLS] [SEP] text tokens [SEP] [PAD] ...
    [CLS] will be splitted out into:
    [CLS] video tokens [SEP] text tokens [SEP] [PAD] ...
    token_type_ids will be generated by the model (for now).
    0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
    | first sequence    | second sequence |
    so each sequence owns a [SEP] token for no-ops.
    """

    def __init__(self, config):
        super().__init__(config)
        self.text_clip_sampler = TextClipSamplingProcessor(
            self.max_len - self.max_video_len - 3
        )
        """
        decide subsampling:
        `config.subsampling` will change batch_size in trainer.
        `config.clip_per_video` (used by RetriTask) doesn't
            change batch_size in trainer.
        """
        subsampling = config.subsampling \
            if config.subsampling is not None else None
        if config.clip_per_video is not None:
            subsampling = config.clip_per_video
        self.subsampling = subsampling

    def _get_text_maxlen(self):
        # use max text len
        return self.text_clip_sampler.max_text_len

    def __call__(self, video_id, video_feature, text_feature):
        from transformers import default_data_collator
        video_idx = video_id[1]
        if self.subsampling is not None and self.subsampling >= 1:
            batch = []
            for _ in range(self.subsampling):
                centerclip_idx = random.randint(
                                    0, len(text_feature["start"]) - 1)
                batch.append(
                    self.sampling(
                        video_idx,
                        video_feature,
                        text_feature,
                        centerclip_idx,
                        self._get_text_maxlen()
                    ))
            batch = self.batch_post_processing(batch, video_feature)
            batch = default_data_collator(batch)
        else:
            raise ValueError(
                "dataset.subsampling must be >= 1 for efficient video loading.")
            batch = self.sampling(video_idx, video_feature, text_feature)
            batch = self.batch_post_processing(batch, video_feature)

        batch["video_id"] = video_id if isinstance(video_id, str) \
            else video_id[0]
        # e2e: make sure frame ids is into tensor.
        assert torch.is_tensor(batch["vfeats"])
        return batch

    def sampling(
        self,
        video_idx,
        video_feature,
        text_feature,
        centerclip_idx=None,
        sampled_max_text_len=None,
    ):
        text_clip_indexs = self.text_clip_sampler(
            text_feature, centerclip_idx,
            sampled_max_text_len
        )
        if isinstance(video_feature, np.ndarray):
            video_len = len(video_feature)
        else:
            video_len = math.ceil(text_feature["end"][-1])

        video_end = min(
            math.ceil(text_feature["end"][text_clip_indexs[-1]]),
            video_len
        )
        video_start = max(
            min(
                math.floor(text_feature["start"][text_clip_indexs[0]]),
                video_end),
            0
        )

        video_clips = {"start": [video_start], "end": [video_end]}

        # tensorize.
        vfeats, vmasks = self._build_video_seq(
            video_feature, video_clips
        )
        caps, cmasks = self._build_text_seq(
            text_feature, text_clip_indexs
        )

        text_start = text_clip_indexs[0]
        text_end = text_clip_indexs[-1] + 1

        return {
            "caps": caps,
            "cmasks": cmasks,
            "vfeats": vfeats,
            "vmasks": vmasks,
            "video_start": video_start,
            "video_end": video_end,
            "text_start": text_start,
            "text_end": text_end,
        }


class VariedLenAligner(FixedLenAligner):
    def __init__(self, config):
        super().__init__(config)
        self.sampled_min_len = config.sampled_min_len
        self.sampled_max_len = config.sampled_max_len

    def _get_text_maxlen(self):
        return random.randint(self.sampled_min_len, self.sampled_max_len)


class StartClipAligner(VariedLenAligner):
    def sampling(
        self,
        video_idx,
        video_feature,
        text_feature,
        centerclip_idx=None,
        sampled_max_text_len=None,
    ):
        return super().sampling(
            video_idx, video_feature, text_feature, 0)


class OverlappedAligner(VariedLenAligner):
    """video clip and text clip has overlappings
    but may not be the same start/end."""
    def __init__(self, config):
        super().__init__(config)
        self.sampled_video_min_len = config.sampled_video_min_len
        self.sampled_video_max_len = config.sampled_video_max_len

        self.video_clip_sampler = VideoClipSamplingProcessor()

    def _get_video_maxlen(self):
        return random.randint(
            self.sampled_video_min_len, self.sampled_video_max_len)

    def sampling(
        self,
        video_idx,
        video_feature,
        text_feature,
        centerclip_idx=None,
        sampled_max_text_len=None,
    ):
        text_clip_indexs = self.text_clip_sampler(
            text_feature, centerclip_idx,
            sampled_max_text_len
        )
        if isinstance(video_feature, np.ndarray):
            video_len = len(video_feature)
        else:
            video_len = math.ceil(text_feature["end"][-1])
        low = math.floor(text_feature["start"][text_clip_indexs[0]])
        high = math.ceil(text_feature["end"][text_clip_indexs[-1]])
        if low < high:
            center = random.randint(low, high)
        else:
            center = int((low + high) // 2)
        center = max(0, min(video_feature.shape[0] - 1, center))

        assert 0 <= center < video_feature.shape[0]

        video_clips = self.video_clip_sampler(
            video_len, self._get_video_maxlen(), center
        )
        video_start = video_clips["start"][0]
        video_end = video_clips["end"][0]

        # tensorize.
        vfeats, vmasks = self._build_video_seq(
            video_feature, video_clips
        )
        caps, cmasks = self._build_text_seq(
            text_feature, text_clip_indexs
        )

        text_start = text_clip_indexs[0]
        text_end = text_clip_indexs[-1] + 1

        return {
            "caps": caps,
            "cmasks": cmasks,
            "vfeats": vfeats,
            "vmasks": vmasks,
            "video_start": video_start,
            "video_end": video_end,
            "text_start": text_start,
            "text_end": text_end,
        }


class MFMMLMAligner(FixedLenAligner):
    """
    `FixedLenAligner` with Masked Language Model and Masked Frame Model.
    """

    def __init__(self, config):
        super().__init__(config)
        keep_prob = config.keep_prob if config.keep_prob is not None else 1.0
        self.text_clip_sampler = TextClipSamplingProcessor(
            self.max_len - self.max_video_len - 3, keep_prob
        )
        self.sampled_min_len = config.sampled_min_len
        self.sampled_max_len = config.sampled_max_len
        self.masked_token_sampler = TextMaskingProcessor(config)
        self.mm_type = config.mm_type \
            if config.mm_type is not None else "full"
        self.attnmasker = MMAttentionMask2DProcessor() \
            if self.mm_type == "textgen" else None
        self.masked_frame_sampler = FrameMaskingProcessor(config)
        self.lazy_vfeat_mask = (
            False if config.lazy_vfeat_mask is None else config.lazy_vfeat_mask
        )
        self.mm_prob = config.mm_prob if config.mm_prob is not None else 0.

    def __call__(self, video_id, video_feature, text_feature):
        from transformers import default_data_collator
        if self.subsampling is not None and self.subsampling > 1:
            batch = []
            for _ in range(self.subsampling):
                centerclip_idx = random.randint(
                                    0, len(text_feature["start"]) - 1)
                sampled_max_text_len = random.randint(
                    self.sampled_min_len, self.sampled_max_len
                )
                batch.append(
                    self.sampling(
                        video_id,
                        video_feature,
                        text_feature,
                        centerclip_idx,
                        sampled_max_text_len,
                    )
                )
            batch = self.batch_post_processing(batch, video_feature)
            batch = default_data_collator(batch)
        else:
            batch = self.sampling(video_id, video_feature, text_feature)
            batch = self.batch_post_processing(batch, video_feature)
        batch["video_id"] = video_id if isinstance(video_id, str) \
            else video_id[0]
        return batch

    def sampling(
        self,
        video_id,
        video_feature,
        text_feature,
        centerclip_idx=None,
        sampled_max_text_len=None,
    ):
        output = FixedLenAligner.sampling(self,
            video_id, video_feature, text_feature,
            centerclip_idx, sampled_max_text_len)

        masking_text, masking_video = None, None
        if random.random() < self.mm_prob:
            if random.random() > 0.5:
                masking_text, masking_video = self.mm_type, "no"
            else:
                masking_text, masking_video = "no", "full"
        video_feats = output["vfeats"] if not self.lazy_vfeat_mask else None
        video_label = self.masked_frame_sampler(
            output["vmasks"], masking_video, vfeats=video_feats)
        caps, text_label = self.masked_token_sampler(
            output["caps"], masking_text)

        output.update({
            "caps": caps,
            "video_label": video_label,
            "text_label": text_label,
        })

        if self.attnmasker is not None:
            attention_mask = self.attnmasker(
                output["vmasks"], output["cmasks"], masking_text)
            output.update({
                "attention_mask": attention_mask
            })
        return output


class FrameMaskingProcessor(Processor):
    def __init__(self, config):
        self.mfm_probability = 0.15
        if config.mfm_probability is not None:
            self.mfm_probability = config.mfm_probability

    def __call__(self, vmasks, modality_masking=None, vfeats=None):
        """
        We perform lazy masking to save data transfer time.
        It only generates video_labels by default and MFM model
        will do actualy masking.
        Return: `video_label` is a binary mask.
        """
        video_label = vmasks.clone()
        if modality_masking is not None:
            if modality_masking == "full":
                probability_matrix = torch.full(video_label.shape, 1.)
            elif modality_masking == "no":
                probability_matrix = torch.full(video_label.shape, 0.)
            elif modality_masking == "inverse":
                probability_matrix = torch.full(
                    video_label.shape, 1. - self.mfm_probability)
            else:
                raise ValueError("unknown modality masking.", modality_masking)
        else:
            probability_matrix = torch.full(
                video_label.shape, self.mfm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # We only compute loss on masked tokens
        video_label[~masked_indices] = 0
        if vfeats is not None:
            vfeats[video_label, :] = 0.0
        return video_label


class TextGenerationProcessor(Processor):
    def __init__(self, tokenizer):
        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, inputs):
        labels = inputs.clone()
        # [CLS] [SEP] for video
        labels[:2] = -100
        # keep [SEP] for text.
        pad_mask = labels == self.pad_token_id
        labels[pad_mask] = -100
        inputs[2:] = torch.cat([
            torch.LongTensor([self.bos_token_id]),
            inputs[2:-1]])
        inputs[pad_mask] = self.pad_token_id
        assert len(inputs) == len(labels)
        return inputs, labels


class TextMaskingProcessor(Processor):
    def __init__(self, config):
        """this function is borrowed from
        `transformers/data/data_collator.DataCollatorForLanguageModeling`"""
        self.mlm_probability = 0.15
        if config.mlm_probability is not None:
            self.mlm_probability = config.mlm_probability
        self.bert_name = config.bert_name
        # [CLS] is used as bos_token and [SEP] is used as eos_token.
        # https://huggingface.co/transformers/master/model_doc/bertgeneration.html
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.bert_name, bos_token="[CLS]", eos_token="[SEP]")
        self.textgen = TextGenerationProcessor(self.tokenizer)

    def __call__(
        self, inputs: torch.Tensor,
        modality_masking=None,
        special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        expand modality_masking into
            None: traditional bert masking.
            "no": no masking.
            "full": all [MASK] token for generation.
            "gen": autoregressive generation.
        """
        """
        Prepare masked tokens inputs/labels for masked language modeling:
        80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training
        # (with probability `self.mlm_probability`)
        if modality_masking is not None:
            if modality_masking == "full":
                probability_matrix = torch.full(labels.shape, 1.)
            elif modality_masking == "no":
                probability_matrix = torch.full(labels.shape, 0.)
            elif modality_masking.startswith("textgen"):
                # [CLS] [SEP] <s> ...
                inputs, labels = self.textgen(inputs)
                if "mask" not in modality_masking:
                    return inputs, labels
                inputs = self.mask_input(inputs, special_tokens_mask)
                return inputs, labels
            elif modality_masking == "mask":
                inputs = self.mask_input(inputs, special_tokens_mask)
                labels = torch.full(inputs.shape, -100)
                return inputs, labels
            elif modality_masking == "inverse":
                probability_matrix = torch.full(labels.shape, 1. - self.mlm_probability)
            else:
                raise ValueError("unknown modality masking.", modality_masking)
        else:
            probability_matrix = torch.full(labels.shape, self.mlm_probability)

        if special_tokens_mask is None:
            special_tokens_mask = self.get_special_tokens_mask(
                labels.tolist(), already_has_special_tokens=True
            )
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time,
        # we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(
                torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input
        # tokens unchanged
        return inputs, labels

    def mask_input(self, inputs, special_tokens_mask=None):
        # the following is new with masked autoregressive.
        probability_matrix = torch.full(
            inputs.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = self.get_special_tokens_mask(
                inputs.tolist(), already_has_special_tokens=True
            )
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        indices_replaced = (
            torch.bernoulli(
                torch.full(inputs.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(inputs.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), inputs.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]
        return inputs

    def get_special_tokens_mask(
        self, token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Note: the version from transformers do not consider pad
        as special tokens.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if"
                    "the provided sequence of "
                    "ids is already formated with special tokens "
                    "for the model."
                )
            return list(map(lambda x: 1 if x in [
                self.tokenizer.sep_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.pad_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]


class TextClipSamplingProcessor(Processor):
    def __init__(self, max_text_len, keep_prob=1.0):
        self.max_text_len = max_text_len
        self.max_video_len = 256  # always hold.
        self.keep_prob = keep_prob

    def __call__(
        self,
        text_feature,
        centerclip_idx=None,
        sampled_max_text_len=None,
        sampled_max_video_len=None,
    ):
        # Let's use all caps for now and see if 256 can cover all of them.
        if sampled_max_text_len is not None:
            max_text_len = sampled_max_text_len
        else:
            max_text_len = self.max_text_len
        if sampled_max_video_len is not None:
            max_video_len = sampled_max_video_len
        else:
            max_video_len = self.max_video_len

        t_num_clips = len(text_feature["start"])

        if centerclip_idx is None:
            centerclip_idx = random.randint(0, t_num_clips - 1)

        start_idx, end_idx = centerclip_idx, centerclip_idx + 1
        text_clip_indexs = deque()
        text_clip_indexs.append(start_idx)
        text_len = len(text_feature["cap"][start_idx])

        video_len = max(
            0,
            text_feature["end"][start_idx]
            - text_feature["start"][start_idx],
        )

        while (
            (start_idx > 0 or end_idx < t_num_clips)
            and text_len < max_text_len
            and video_len < max_video_len
        ):
            if random.random() > 0.5 and end_idx < t_num_clips:
                # skip the next one?
                if random.random() > self.keep_prob and (end_idx + 1) < t_num_clips:
                    end_idx = end_idx + 1
                text_clip_indexs.append(end_idx)
                text_len += len(text_feature["cap"][end_idx])
                end_idx += 1
            elif start_idx > 0:
                if random.random() > self.keep_prob and (start_idx - 1) > 0:
                    start_idx = start_idx - 1
                start_idx -= 1
                text_clip_indexs.insert(0, start_idx)
                text_len += len(text_feature["cap"][start_idx])
            else:
                if end_idx < t_num_clips:
                    if random.random() > self.keep_prob and (end_idx + 1) < t_num_clips:
                        end_idx = end_idx + 1
                    text_clip_indexs.append(end_idx)
                    text_len += len(text_feature["cap"][end_idx])
                    end_idx += 1
                else:
                    return text_clip_indexs
            video_len = max(
                0,
                text_feature["end"][text_clip_indexs[-1]]
                - text_feature["start"][text_clip_indexs[0]],
            )
        return text_clip_indexs


class VideoClipSamplingProcessor(Processor):
    def __call__(self, video_len, max_video_len, center):
        """
        `video_len`: length of the video.
        `max_video_len`: maximum video tokens allowd in a sequence.
        `center`: initial starting index.
        """
        assert center >= 0 and center < video_len
        t_clip_len = 0
        start, end = center, center
        while (start > 0 or end < video_len) and t_clip_len < max_video_len:
            # decide the direction to grow.
            if start <= 0:
                end += 1
            elif end >= video_len:
                start -= 1
            elif random.random() > 0.5:
                end += 1
            else:
                start -= 1
            t_clip_len += 1
        return {"start": [start], "end": [end]}


class How2MILNCEAligner(FixedLenAligner):
    """reference: `antoine77340/MIL-NCE_HowTo100M/video_loader.py`"""

    def __init__(self, config):
        super().__init__(config)
        self.num_candidates = 4
        self.min_time = 5.0
        self.num_sec = 3.2
        # self.num_sec = self.num_frames / float(self.fps)  num_frames=16 / fps = 5
        # self.num_frames = 16

    def sampling(
        self,
        video_id,
        video_feature,
        text_feature,
        centerclip_idx=None,  # will be ignored.
        sampled_max_text_len=None  # will be ignored.
    ):
        text, start, end = self._get_text(text_feature)
        video = self._get_video(video_feature, start, end)

        vfeats = torch.zeros((self.max_video_len, video_feature.shape[1]))
        vmasks = torch.zeros((self.max_video_len,), dtype=torch.bool)
        vfeats[: video.shape[0]] = torch.from_numpy(np.array(video))
        vmasks[: video.shape[0]] = 1

        caps, cmasks = [], []
        for words in text:
            cap, cmask = self._build_text_seq(text_feature, words)
            caps.append(cap)
            cmasks.append(cmask)
        caps = torch.stack(caps)
        cmasks = torch.stack(cmasks)
        # video of shape: (video_len)
        # text of shape (num_candidates, max_text_len)

        return {
            "caps": caps,
            "cmasks": cmasks,
            "vfeats": vfeats,
            "vmasks": vmasks,
            # "video_id": video_id,
        }

    def _get_video(self, video_feature, start, end):
        start_seek = random.randint(start, int(max(start, end - self.num_sec)))
        # duration = self.num_sec + 0.1
        return video_feature[start_seek : int(start_seek + self.num_sec)]

    def _get_text(self, cap):
        ind = random.randint(0, len(cap["start"]) - 1)
        if self.num_candidates == 1:
            words = [ind]
        else:
            words = []
            cap_start = self._find_nearest_candidates(cap, ind)
            for i in range(self.num_candidates):
                words.append([max(0, min(len(cap["cap"]) - 1, cap_start + i))])

        start, end = cap["start"][ind], cap["end"][ind]
        # TODO: May need to be improved for edge cases.
        # expand the min time.
        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time
        return words, int(start), int(end)

    def _find_nearest_candidates(self, caption, ind):
        """find the range of the clips."""
        start, end = ind, ind
        #diff = caption["end"][end] - caption["start"][start]
        n_candidate = 1
        while n_candidate < self.num_candidates:
            # the first clip
            if start == 0:
                return 0
            # we add () in the following condition to fix the bug.
            elif end == (len(caption["start"]) - 1):
                return start - (self.num_candidates - n_candidate)
            elif (caption["end"][end] - caption["start"][start - 1]) < (
                caption["end"][end + 1] - caption["start"][start]
            ):
                start -= 1
            else:
                end += 1
            n_candidate += 1
        return start


class PKLJSONStrTextProcessor(TextProcessor):
    """`caption.json` from howto100m are preprocessed as a
    dict `[video_id, json_str]`.
    Json parsing tokenization are conducted on-the-fly and cached into dict.
    """

    def __init__(self, config, max_clip_text_len=96):
        print("[Warning] PKLJSONStrTextProcessor is slow for num_workers > 0.")
        self.caption_pkl_path = str(config.caption_pkl_path)
        with open(self.caption_pkl_path, "rb") as fd:
            self.data = pickle.load(fd)
        self.max_clip_text_len = max_clip_text_len
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.bert_name), use_fast=config.use_fast
        )

    def __call__(self, video_id):
        caption = self.data[video_id]
        if isinstance(caption, str):
            import json
            caption = json.loads(caption)
            cap = []
            for clip_idx, text_clip in enumerate(caption["text"]):
                clip_ids = []
                if isinstance(text_clip, str):
                    clip_ids = self.tokenizer(
                        text_clip[: self.max_clip_text_len],
                        add_special_tokens=False
                    )["input_ids"]
                cap.append(clip_ids)
            caption["cap"] = cap
            caption.pop("text")  # save space.
            self.data[video_id] = caption
        return caption
