# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.utils.data


class FairseqDataset(torch.utils.data.Dataset):
    """A dataset that provides helpers for batching."""

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        raise NotImplementedError

    def get_dummy_batch(self, num_tokens, max_positions):
        """Return a dummy batch with a given number of tokens."""
        raise NotImplementedError

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        raise NotImplementedError

    def ordered_indices(self):
        """Ordered indices for batching."""
        raise NotImplementedError

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        raise NotImplementedError
