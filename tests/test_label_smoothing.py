# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import torch
import unittest
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedNLLLoss
from torch.autograd import Variable, gradcheck


torch.set_default_tensor_type('torch.DoubleTensor')


class TestLabelSmoothing(unittest.TestCase):

    def test_label_smoothing(self):
        input = Variable(torch.randn(3, 5), requires_grad=True)
        idx = torch.rand(3) * 4
        target = Variable(idx.long())
        criterion = LabelSmoothedNLLLoss()
        self.assertTrue(gradcheck(
            lambda x, y: criterion.apply(x, y, 0.1, 2, None), (input, target)
        ))
        weights = torch.ones(5)
        weights[2] = 0
        self.assertTrue(gradcheck(lambda x, y: criterion.apply(x, y, 0.1, None, weights), (input, target)))
        self.assertTrue(gradcheck(lambda x, y: criterion.apply(x, y, 0.1, None, None), (input, target)))


if __name__ == '__main__':
    unittest.main()
