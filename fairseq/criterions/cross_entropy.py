# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import math
import torch.nn.functional as F

from .fairseq_criterion import FairseqCriterion


class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, padding_idx):
        super().__init__()
        self.padding_idx = padding_idx

    def grad_denom(self, samples):
        return sum(s['ntokens'] if s else 0 for s in samples)

    def forward(self, model, sample, grad_denom):
        net_output = model(**sample['net_input'])
        input = net_output.view(-1, net_output.size(-1))
        target = sample['target'].view(-1)
        loss = F.cross_entropy(input, target, size_average=False, ignore_index=self.padding_idx)
        return {
            'loss': loss / grad_denom,
        }

    def aggregate(self, loss_dicts):
        return {
            'loss': sum(l['loss'].data[0] for l in loss_dicts if 'loss' in l) / math.log(2),
        }
