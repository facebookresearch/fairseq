# Copyright (c) Facebook, Inc. All Rights Reserved

import numpy as np
import os
import torch


class Processor(object):
    """
    A generic processor for video (codec, feature etc.) and text.
    """

    def __call__(self, **kwargs):
        raise NotImplementedError


class MetaProcessor(Processor):
    """
    A meta processor is expected to load the metadata of a dataset:
        (e.g., video_ids, or captions).
    You must implement the `__getitem__` (meta datasets are rather diverse.).
    """

    def __init__(self, config):
        self.split = config.split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_split_path(self, config):
        splits = {
            "train": config.train_path,
            "valid": config.val_path,
            "test": config.test_path,
        }
        if config.split is not None:
            return splits[config.split]
        return config.train_path


class TextProcessor(Processor):
    """
    A generic Text processor: rename this as `withTokenizer`.
    tokenize a string of text on-the-fly.
    Warning: mostly used for end tasks.
        (on-the-fly tokenization is slow for how2.)
    TODO(huxu): move this class as a subclass.
    """

    def __init__(self, config):
        self.bert_name = str(config.bert_name)
        self.use_fast = config.use_fast
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.bert_name, use_fast=self.use_fast
        )

    def __call__(self, text_id):
        caption = self.tokenizer(text_id, add_special_tokens=False)
        return caption["input_ids"]


class VideoProcessor(Processor):
    """
    A generic video processor: load a numpy video tokens by default.
    """

    def __init__(self, config):
        self.vfeat_dir = config.vfeat_dir

    def __call__(self, video_fn):
        if isinstance(video_fn, tuple):
            video_fn = video_fn[0]
        assert isinstance(video_fn, str)
        video_fn = os.path.join(self.vfeat_dir, video_fn + ".npy")
        feat = np.load(video_fn)
        return feat


class Aligner(object):
    """
    An alignprocessor align video and text and output a dict of tensors (for a model).
    """
    def __init__(self, config):
        """__init__ needs to be light weight for more workers/threads."""
        self.split = config.split
        self.max_video_len = config.max_video_len
        self.max_len = config.max_len
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(config.bert_name), use_fast=config.use_fast
        )
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id

    def __call__(self, video_id, video_feature, text_feature):
        raise NotImplementedError

    def _build_video_seq(self, video_feature, video_clips=None):
        """
        `video_feature`: available video tokens.
        `video_clips`: video clip sequence to build.
        """
        if not isinstance(video_feature, np.ndarray):
            raise ValueError(
                "unsupported type of video_feature", type(video_feature)
            )

        if video_clips is None:
            # this is borrowed from DSAligner
            video_start = 0
            video_end = min(len(video_feature), self.max_video_len)
            # the whole sequence is a single clip.
            video_clips = {"start": [video_start], "end": [video_end]}

        vfeats = np.zeros(
            (self.max_video_len, video_feature.shape[1]), dtype=np.float32
        )
        vmasks = torch.zeros((self.max_video_len,), dtype=torch.bool)
        video_len = 0
        for start, end in zip(video_clips["start"], video_clips["end"]):
            clip_len = min(self.max_video_len - video_len, (end - start))
            if clip_len > 0:
                vfeats[video_len: video_len + clip_len] = video_feature[
                    start: start + clip_len
                ]
                vmasks[video_len: video_len + clip_len] = 1
                video_len += clip_len
        vfeats = torch.from_numpy(vfeats)

        return vfeats, vmasks

    def _build_text_seq(self, text_feature, text_clip_indexs=None):
        """
        `text_feature`: all available clips.
        `text_clip_indexes`: clip sequence to build.
        """
        if text_clip_indexs is None:
            text_clip_indexs = [0]

        full_caps = []
        if isinstance(text_feature, dict):
            for clip_idx in text_clip_indexs:
                full_caps.extend(text_feature["cap"][clip_idx])
        else:
            full_caps = text_feature
        max_text_len = self.max_len - self.max_video_len - 3
        full_caps = full_caps[:max_text_len]
        full_caps = (
            [self.cls_token_id, self.sep_token_id] + full_caps + [self.sep_token_id]
        )
        text_pad_len = self.max_len - len(full_caps) - self.max_video_len
        padded_full_caps = full_caps + [self.pad_token_id] * text_pad_len
        caps = torch.LongTensor(padded_full_caps)
        cmasks = torch.zeros((len(padded_full_caps),), dtype=torch.bool)
        cmasks[: len(full_caps)] = 1

        return caps, cmasks

    def batch_post_processing(self, batch, video_feature):
        return batch


class MMAttentionMask2DProcessor(Processor):
    """text generation requires 2d mask
    that is harder to generate by GPU at this stage."""

    def __call__(self, vmask, cmask, mtype):
        if mtype == "textgen":
            return self._build_textgeneration_mask(vmask, cmask)
        elif mtype == "videogen":
            return self._build_videogeneration_mask(vmask, cmask)
        else:
            return self._build_mm_mask(vmask, cmask)

    def _build_mm_mask(self, vmask, cmask):
        mask_1d = torch.cat([cmask[:1], vmask, cmask[1:]], dim=0)
        return mask_1d[None, :].repeat(mask_1d.size(0), 1)

    def _build_videogeneration_mask(self, vmask, cmask):
        # cls_mask is only about text otherwise it will leak generation.
        cls_text_mask = torch.cat([
            # [CLS]
            torch.ones(
                (1,), dtype=torch.bool, device=cmask.device),
            # video tokens and [SEP] for video.
            torch.zeros(
                (vmask.size(0) + 1,), dtype=torch.bool, device=cmask.device),
            cmask[2:]
            ], dim=0)

        # concat horizontially.
        video_len = int(vmask.sum())
        video_masks = torch.cat([
            # [CLS]
            torch.ones(
                (video_len, 1), dtype=torch.bool, device=cmask.device
            ),
            torch.tril(
                torch.ones(
                    (video_len, video_len),
                    dtype=torch.bool, device=cmask.device)),
            # video_padding
            torch.zeros(
                (video_len, vmask.size(0) - video_len),
                dtype=torch.bool, device=cmask.device
            ),
            # [SEP] for video (unused).
            torch.zeros(
                (video_len, 1), dtype=torch.bool, device=cmask.device
            ),
            cmask[2:].unsqueeze(0).repeat(video_len, 1)
            ], dim=1)

        text_masks = cls_text_mask[None, :].repeat(
            cmask.size(0) - 2, 1)
        video_padding_masks = cls_text_mask[None, :].repeat(
            vmask.size(0) - video_len, 1)

        return torch.cat([
            cls_text_mask[None, :],
            video_masks,
            video_padding_masks,
            torch.cat([cmask[:1], vmask, cmask[1:]], dim=0)[None,:],
            text_masks
            ], dim=0)

    def _build_textgeneration_mask(self, vmask, cmask):
        # cls_mask is only about video otherwise it will leak generation.
        cls_video_mask = torch.cat([
            # [CLS]
            torch.ones(
                (1,), dtype=torch.bool, device=cmask.device),
            vmask,
            # [SEP]
            torch.ones((1,), dtype=torch.bool, device=cmask.device),
            torch.zeros(
                (cmask.size(0)-2,), dtype=torch.bool, device=cmask.device)
        ], dim=0)

        # concat horizontially.
        text_len = int(cmask[2:].sum())
        text_masks = torch.cat([
            # [CLS]
            torch.ones(
                (text_len, 1), dtype=torch.bool, device=cmask.device
            ),
            vmask.unsqueeze(0).repeat(text_len, 1),
            # [SEP] for video.
            torch.ones(
                (text_len, 1), dtype=torch.bool, device=cmask.device
            ),
            torch.tril(
                torch.ones(
                    (text_len, text_len),
                    dtype=torch.bool, device=cmask.device)),
            # padding.
            torch.zeros(
                (text_len, cmask.size(0) - text_len - 2),
                dtype=torch.bool, device=cmask.device
            )
        ], dim=1)

        cls_video_masks = cls_video_mask[None, :].repeat(
            vmask.size(0) + 2, 1)
        text_padding_masks = cls_video_mask[None, :].repeat(
            cmask.size(0) - text_len - 2, 1)
        return torch.cat([
            cls_video_masks, text_masks, text_padding_masks], dim=0)
