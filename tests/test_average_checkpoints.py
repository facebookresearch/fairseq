# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import os
import tempfile
import unittest
import shutil

import numpy as np
import torch
from torch import nn


from scripts.average_checkpoints import average_checkpoints


class ModelWithSharedParameter(nn.Module):
    def __init__(self):
        super(ModelWithSharedParameter, self).__init__()
        self.embedding = nn.Embedding(1000, 200)
        self.FC1 = nn.Linear(200, 200)
        self.FC2 = nn.Linear(200, 200)
        # tie weight in FC2 to FC1
        self.FC2.weight = nn.Parameter(self.FC1.weight)
        self.FC2.bias = nn.Parameter(self.FC1.bias)

        self.relu = nn.ReLU()

    def forward(self, input):
        return self.FC2(self.ReLU(self.FC1(input))) + self.FC1(input)


class TestAverageCheckpoints(unittest.TestCase):
    def test_average_checkpoints(self):
        params_0 = collections.OrderedDict(
            [
                ('a', torch.DoubleTensor([100.0])),
                ('b', torch.FloatTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                ('c', torch.IntTensor([7, 8, 9])),
            ]
        )
        params_1 = collections.OrderedDict(
            [
                ('a', torch.DoubleTensor([1.0])),
                ('b', torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])),
                ('c', torch.IntTensor([2, 2, 2])),
            ]
        )
        params_avg = collections.OrderedDict(
            [
                ('a', torch.DoubleTensor([50.5])),
                ('b', torch.FloatTensor([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]])),
                # We expect truncation for integer division
                ('c', torch.IntTensor([4, 5, 5])),
            ]
        )

        fd_0, path_0 = tempfile.mkstemp()
        fd_1, path_1 = tempfile.mkstemp()
        torch.save(collections.OrderedDict([('model', params_0)]), path_0)
        torch.save(collections.OrderedDict([('model', params_1)]), path_1)

        output = average_checkpoints([path_0, path_1])['model']

        os.close(fd_0)
        os.remove(path_0)
        os.close(fd_1)
        os.remove(path_1)

        for (k_expected, v_expected), (k_out, v_out) in zip(
                params_avg.items(), output.items()):
            self.assertEqual(
                k_expected, k_out, 'Key mismatch - expected {} but found {}. '
                '(Expected list of keys: {} vs actual list of keys: {})'.format(
                    k_expected, k_out, params_avg.keys(), output.keys()
                )
            )
            np.testing.assert_allclose(
                v_expected.numpy(),
                v_out.numpy(),
                err_msg='Tensor value mismatch for key {}'.format(k_expected)
            )

    def test_average_checkpoints_with_shared_parameters(self):

        def _construct_model_with_shared_parameters(path, value):
            m = ModelWithSharedParameter()
            nn.init.constant_(m.FC1.weight, value)
            torch.save(
                {'model': m.state_dict()},
                path
            )
            return m

        tmpdir = tempfile.mkdtemp()
        paths = []
        path = os.path.join(tmpdir, "m1.pt")
        m1 = _construct_model_with_shared_parameters(path, 1.0)
        paths.append(path)

        path = os.path.join(tmpdir, "m2.pt")
        m2 = _construct_model_with_shared_parameters(path, 2.0)
        paths.append(path)

        path = os.path.join(tmpdir, "m3.pt")
        m3 = _construct_model_with_shared_parameters(path, 3.0)
        paths.append(path)

        new_model = average_checkpoints(paths)
        self.assertTrue(
            torch.equal(
                new_model['model']['embedding.weight'],
                (m1.embedding.weight +
                 m2.embedding.weight +
                 m3.embedding.weight) / 3.0
            )
        )

        self.assertTrue(
            torch.equal(
                new_model['model']['FC1.weight'],
                (m1.FC1.weight +
                 m2.FC1.weight +
                 m3.FC1.weight) / 3.0
            )
        )

        self.assertTrue(
            torch.equal(
                new_model['model']['FC2.weight'],
                (m1.FC2.weight +
                 m2.FC2.weight +
                 m3.FC2.weight) / 3.0
            )
        )
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    unittest.main()
