# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .task import Task


class MILNCETask(Task):
    def reshape_subsample(self, sample):
        if (
            hasattr(self.config.dataset, "subsampling")
            and self.config.dataset.subsampling is not None
            and self.config.dataset.subsampling > 1
        ):
            for key in sample:
                if torch.is_tensor(sample[key]):
                    tensor = self.flat_subsample(sample[key])
                    if key in ["caps", "cmasks"]:
                        size = tensor.size()
                        batch_size = size[0] * size[1]
                        expanded_size = (batch_size,) + size[2:]
                        tensor = tensor.view(expanded_size)
                    sample[key] = tensor
        return sample
