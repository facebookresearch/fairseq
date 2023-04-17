# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from dataclasses import dataclass

from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer


@dataclass
class MeteorScorerConfig(FairseqDataclass):
    pass


@register_scorer("meteor", dataclass=MeteorScorerConfig)
class MeteorScorer(BaseScorer):
    def __init__(self, args):
        super(MeteorScorer, self).__init__(args)
        try:
            import nltk
        except ImportError:
            raise ImportError("Please install nltk to use METEOR scorer")

        self.nltk = nltk
        self.scores = []

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.pred.append(pred)

    def score(self, order=4):
        self.scores = [
            self.nltk.translate.meteor_score.single_meteor_score(r, p)
            for r, p in zip(self.ref, self.pred)
        ]
        return np.mean(self.scores)

    def result_string(self, order=4):
        return f"METEOR: {self.score():.4f}"
