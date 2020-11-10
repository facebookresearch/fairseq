# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
    This module contains collection of classes which implement
    collate functionalities for various tasks.

    Collaters should know what data to expect for each sample
    and they should pack / collate them into batches
"""


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
from fairseq.data import data_utils as fairseq_data_utils


class Seq2SeqCollater(object):
    """
    Implements collate function mainly for seq2seq tasks
    This expects each sample to contain feature (src_tokens) and
    targets.
    This collator is also used for aligned training task.
    """

    def __init__(
        self,
        feature_index=0,
        label_index=1,
        pad_index=1,
        eos_index=2,
        move_eos_to_beginning=True,
    ):
        self.feature_index = feature_index
        self.label_index = label_index
        self.pad_index = pad_index
        self.eos_index = eos_index
        self.move_eos_to_beginning = move_eos_to_beginning

    def _collate_frames(self, frames):
        """Convert a list of 2d frames into a padded 3d tensor
        Args:
            frames (list): list of 2d frames of size L[i]*f_dim. Where L[i] is
                length of i-th frame and f_dim is static dimension of features
        Returns:
            3d tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
        """
        len_max = max(frame.size(0) for frame in frames)
        f_dim = frames[0].size(1)
        res = frames[0].new(len(frames), len_max, f_dim).fill_(0.0)

        for i, v in enumerate(frames):
            res[i, : v.size(0)] = v

        return res

    def collate(self, samples):
        """
        utility function to collate samples into batch for speech recognition.
        """
        if len(samples) == 0:
            return {}

        # parse samples into torch tensors
        parsed_samples = []
        for s in samples:
            # skip invalid samples
            if s["data"][self.feature_index] is None:
                continue
            source = s["data"][self.feature_index]
            if isinstance(source, (np.ndarray, np.generic)):
                source = torch.from_numpy(source)
            target = s["data"][self.label_index]
            if isinstance(target, (np.ndarray, np.generic)):
                target = torch.from_numpy(target).long()
            elif isinstance(target, list):
                target = torch.LongTensor(target)

            parsed_sample = {"id": s["id"], "source": source, "target": target}
            parsed_samples.append(parsed_sample)
        samples = parsed_samples

        id = torch.LongTensor([s["id"] for s in samples])
        frames = self._collate_frames([s["source"] for s in samples])
        # sort samples by descending number of frames
        frames_lengths = torch.LongTensor([s["source"].size(0) for s in samples])
        frames_lengths, sort_order = frames_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        frames = frames.index_select(0, sort_order)

        target = None
        target_lengths = None
        prev_output_tokens = None
        if samples[0].get("target", None) is not None:
            ntokens = sum(len(s["target"]) for s in samples)
            target = fairseq_data_utils.collate_tokens(
                [s["target"] for s in samples],
                self.pad_index,
                self.eos_index,
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, sort_order)
            target_lengths = torch.LongTensor(
                [s["target"].size(0) for s in samples]
            ).index_select(0, sort_order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [s["target"] for s in samples],
                self.pad_index,
                self.eos_index,
                left_pad=False,
                move_eos_to_beginning=self.move_eos_to_beginning,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        else:
            ntokens = sum(len(s["source"]) for s in samples)

        batch = {
            "id": id,
            "ntokens": ntokens,
            "net_input": {"src_tokens": frames, "src_lengths": frames_lengths},
            "target": target,
            "target_lengths": target_lengths,
            "nsentences": len(samples),
        }
        if prev_output_tokens is not None:
            batch["net_input"]["prev_output_tokens"] = prev_output_tokens
        return batch
