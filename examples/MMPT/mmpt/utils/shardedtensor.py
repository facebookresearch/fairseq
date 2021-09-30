# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import pickle
import numpy as np


class ShardedTensor(object):
    def __init__(self, data, starts):
        self.data = data
        self.starts = starts
        assert self.starts[0] == 0
        assert self.starts[-1] == len(self.data)
        assert (self.starts[1:] >= self.starts[:-1]).all()
        assert (self.starts > -1).all()

    @staticmethod
    def from_list(xs):
        starts = np.full((len(xs) + 1,), -1, dtype=np.long)
        data = np.concatenate(xs, axis=0)
        starts[0] = 0
        for i, x in enumerate(xs):
            starts[i + 1] = starts[i] + x.shape[0]
        assert (starts > -1).all()
        return ShardedTensor(data, starts)

    def __getitem__(self, i):
        return self.data[self.starts[i] : self.starts[i + 1]]

    def __len__(self):
        return len(self.starts) - 1

    def lengths(self):
        return self.starts[1:] - self.starts[:-1]

    def save(self, path):
        np.save(path + "_starts", self.starts)
        np.save(path + "_data", self.data)

    @staticmethod
    def load(path, mmap_mode=None):
        starts = np.load(path + "_starts.npy", mmap_mode)
        data = np.load(path + "_data.npy", mmap_mode)
        return ShardedTensor(data, starts)
