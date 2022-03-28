# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
softmax-based NCE loss, used by this project.
"""

import torch

from torch import nn

from .loss import Loss


class NCE(Loss):
    def __init__(self):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, align_scores, **kargs):
        # note: we reuse the same shape as cls head in BERT (batch_size, 2)
        # but NCE only needs one logits.
        # (so we drop all weights in the second neg logits.)
        align_scores = align_scores[:, :1]
        # duplicate negative examples
        batch_size = align_scores.size(0) // 2
        pos_scores = align_scores[:batch_size]
        neg_scores = align_scores[batch_size:].view(1, batch_size).repeat(
            batch_size, 1)
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        return self.loss(
            scores,
            torch.zeros(
                (batch_size,),
                dtype=torch.long,
                device=align_scores.device),
        )


class T2VContraLoss(Loss):
    """NCE for MM joint space, on softmax text2video matrix.
    """
    def __init__(self):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, pooled_video, pooled_text, **kargs):
        batch_size = pooled_video.size(0)
        logits = torch.mm(pooled_text, pooled_video.transpose(1, 0))
        targets = torch.arange(
            batch_size,
            dtype=torch.long,
            device=pooled_video.device)
        return self.loss(logits, targets)


class V2TContraLoss(Loss):
    """NCE for MM joint space, with softmax on video2text matrix."""

    def __init__(self):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, pooled_video, pooled_text, **kargs):
        batch_size = pooled_video.size(0)
        logits = torch.mm(pooled_video, pooled_text.transpose(1, 0))
        targets = torch.arange(
            batch_size,
            dtype=torch.long,
            device=pooled_video.device)
        return self.loss(logits, targets)


class MMContraLoss(Loss):
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, pooled_video, pooled_text, **kwargs):
        logits_per_video = pooled_video @ pooled_text.t()
        logits_per_text = pooled_text @ pooled_video.t()

        targets = torch.arange(
            pooled_video.size(0),
            dtype=torch.long,
            device=pooled_video.device)
        loss_video = self.loss(logits_per_video, targets)
        loss_text = self.loss(logits_per_text, targets)
        return loss_video + loss_text


class MTM(Loss):
    """Combination of MFM and MLM."""

    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        video_logits,
        text_logits,
        video_label,
        text_label,
        **kwargs
    ):
        text_logits = torch.cat([
            text_logits,
            torch.zeros(
                (text_logits.size(0), 1), device=text_logits.device)
        ], dim=1)
        vt_logits = torch.cat([video_logits, text_logits], dim=0)
        # loss for video.
        video_label = torch.zeros(
            (video_logits.size(0),),
            dtype=torch.long,
            device=video_logits.device
        )

        # loss for text.
        text_label = text_label.reshape(-1)
        labels_mask = text_label != -100
        selected_text_label = text_label[labels_mask]

        vt_label = torch.cat([video_label, selected_text_label], dim=0)
        return self.loss(vt_logits, vt_label)


class MFMMLM(Loss):
    """Combination of MFM and MLM."""

    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        video_logits,
        text_logits,
        video_label,
        text_label,
        **kwargs
    ):
        # loss for video.
        video_label = torch.zeros(
            (video_logits.size(0),),
            dtype=torch.long,
            device=video_logits.device
        )
        masked_frame_loss = self.loss(video_logits, video_label)

        # loss for text.
        text_label = text_label.reshape(-1)
        labels_mask = text_label != -100
        selected_text_label = text_label[labels_mask]
        masked_lm_loss = self.loss(text_logits, selected_text_label)
        return masked_frame_loss + masked_lm_loss
