#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import unittest

import numpy as np
import torch
from examples.speech_recognition.data.collaters import Seq2SeqCollater


class TestSeq2SeqCollator(unittest.TestCase):
    def test_collate(self):

        eos_idx = 1
        pad_idx = 0
        collater = Seq2SeqCollater(
            feature_index=0, label_index=1, pad_index=pad_idx, eos_index=eos_idx
        )

        # 2 frames in the first sample and 3 frames in the second one
        frames1 = np.array([[7, 8], [9, 10]])
        frames2 = np.array([[1, 2], [3, 4], [5, 6]])
        target1 = np.array([4, 2, 3, eos_idx])
        target2 = np.array([3, 2, eos_idx])
        sample1 = {"id": 0, "data": [frames1, target1]}
        sample2 = {"id": 1, "data": [frames2, target2]}
        batch = collater.collate([sample1, sample2])

        # collate sort inputs by frame's length before creating the batch
        self.assertTensorEqual(batch["id"], torch.tensor([1, 0]))
        self.assertEqual(batch["ntokens"], 7)
        self.assertTensorEqual(
            batch["net_input"]["src_tokens"],
            torch.tensor(
                [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [pad_idx, pad_idx]]]
            ),
        )
        self.assertTensorEqual(
            batch["net_input"]["prev_output_tokens"],
            torch.tensor([[eos_idx, 3, 2, pad_idx], [eos_idx, 4, 2, 3]]),
        )
        self.assertTensorEqual(batch["net_input"]["src_lengths"], torch.tensor([3, 2]))
        self.assertTensorEqual(
            batch["target"],
            torch.tensor([[3, 2, eos_idx, pad_idx], [4, 2, 3, eos_idx]]),
        )
        self.assertEqual(batch["nsentences"], 2)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)


if __name__ == "__main__":
    unittest.main()
