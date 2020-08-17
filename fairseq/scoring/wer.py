# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import editdistance

from fairseq.scoring import register_scoring


@register_scoring("wer")
class WerScorer(object):
    def __init__(self, *unused):
        self.reset()

    def reset(self):
        self.distance = 0
        self.ref_length = 0

    def add_string(self, ref, pred):
        ref_items = ref.split()
        pred_items = pred.split()
        self.distance += editdistance.eval(ref_items, pred_items)
        self.ref_length += len(ref_items)

    def result_string(self):
        return f"WER: {self.score()}"

    def score(self):
        return (
            100.0 * self.distance / self.ref_length if self.ref_length > 0 else 0
        )
