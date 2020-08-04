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
        self.target_length = 0

    def add_string(self, ref, pred):
        pred_items = ref.split()
        targ_items = pred.split()
        self.distance += editdistance.eval(pred_items, targ_items)
        self.target_length += len(targ_items)

    def result_string(self):
        return f"WER: {self.score()}"

    def score(self):
        return (
            100.0 * self.distance / self.target_length if self.target_length > 0 else 0
        )
