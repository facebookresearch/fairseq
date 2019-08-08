#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from examples.speech_recognition.criterions.cross_entropy_acc import CrossEntropyWithAccCriterion
from .asr_test_base import CrossEntropyCriterionTestBase


class CrossEntropyWithAccCriterionTest(CrossEntropyCriterionTestBase):
    def setUp(self):
        self.criterion_cls = CrossEntropyWithAccCriterion
        super().setUp()

    def test_cross_entropy_all_correct(self):
        sample = self.get_test_sample(correct=True, soft_target=False, aggregate=False)
        loss, sample_size, logging_output = self.criterion(
            self.model, sample, "sum", log_probs=True
        )
        assert logging_output["correct"] == 20
        assert logging_output["total"] == 20
        assert logging_output["sample_size"] == 20
        assert logging_output["ntokens"] == 20

    def test_cross_entropy_all_wrong(self):
        sample = self.get_test_sample(correct=False, soft_target=False, aggregate=False)
        loss, sample_size, logging_output = self.criterion(
            self.model, sample, "sum", log_probs=True
        )
        assert logging_output["correct"] == 0
        assert logging_output["total"] == 20
        assert logging_output["sample_size"] == 20
        assert logging_output["ntokens"] == 20
