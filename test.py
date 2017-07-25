import torch
import unittest
from modules import label_smoothed_cross_entropy
from torch.autograd import Variable, gradcheck


torch.set_default_tensor_type('torch.DoubleTensor')

class TestFConv(unittest.TestCase):

    def test_label_smoothing(self):
        input = Variable(torch.randn(3, 5), requires_grad=True)
        idx = torch.rand(3) * 4
        target = Variable(idx.long())
        self.assertTrue(gradcheck(lambda x, y: label_smoothed_cross_entropy(x, y, eps=0.1, padding_idx=2), (input, target)))
        weights = torch.ones(5)
        weights[2] = 0
        self.assertTrue(gradcheck(lambda x, y: label_smoothed_cross_entropy(x, y, weights=weights), (input, target)))
        self.assertTrue(gradcheck(lambda x, y: label_smoothed_cross_entropy(x, y), (input, target)))


if __name__ == '__main__':
    unittest.main()