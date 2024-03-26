#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from examples.speech_recognition.data import data_utils


class DataUtilsTest(unittest.TestCase):
    def test_normalization(self):
        sample_len1 = torch.tensor(
            [
                [
                    -0.7661,
                    -1.3889,
                    -2.0972,
                    -0.9134,
                    -0.7071,
                    -0.9765,
                    -0.8700,
                    -0.8283,
                    0.7512,
                    1.3211,
                    2.1532,
                    2.1174,
                    1.2800,
                    1.2633,
                    1.6147,
                    1.6322,
                    2.0723,
                    3.1522,
                    3.2852,
                    2.2309,
                    2.5569,
                    2.2183,
                    2.2862,
                    1.5886,
                    0.8773,
                    0.8725,
                    1.2662,
                    0.9899,
                    1.1069,
                    1.3926,
                    1.2795,
                    1.1199,
                    1.1477,
                    1.2687,
                    1.3843,
                    1.1903,
                    0.8355,
                    1.1367,
                    1.2639,
                    1.4707,
                ]
            ]
        )
        out = data_utils.apply_mv_norm(sample_len1)
        assert not torch.isnan(out).any()
        assert (out == sample_len1).all()
