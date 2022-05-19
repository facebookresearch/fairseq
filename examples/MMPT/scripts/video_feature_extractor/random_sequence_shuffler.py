# Copyright (c) Facebook, Inc. All Rights Reserved

import numpy as np

from torch.utils.data.sampler import Sampler


class RandomSequenceSampler(Sampler):

    def __init__(self, n_sample, seq_len):
        self.n_sample = n_sample
        self.seq_len = seq_len

    def _pad_ind(self, ind):
        zeros = np.zeros(self.seq_len - self.n_sample % self.seq_len)
        ind = np.concatenate((ind, zeros))
        return ind

    def __iter__(self):
        idx = np.arange(self.n_sample)
        if self.n_sample % self.seq_len != 0:
            idx = self._pad_ind(idx)
        idx = np.reshape(idx, (-1, self.seq_len))
        np.random.shuffle(idx)
        idx = np.reshape(idx, (-1))
        return iter(idx.astype(int))

    def __len__(self):
        return self.n_sample + (self.seq_len - self.n_sample % self.seq_len)
