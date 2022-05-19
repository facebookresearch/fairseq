# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from .task import Task


class VLMTask(Task):
    """A VLM task for reproducibility.
    the collator split subsamples into two sub-batches.
    This has should have no logic changes.
    but changed the randomness in frame masking.
    """

    def flat_subsample(self, tensor):
        size = tensor.size()
        if len(size) >= 2:
            batch_size = size[0] * (size[1] // 2)
            expanded_size = (
                (batch_size, 2) + size[2:] if len(size) > 2
                else (batch_size, 2)
            )
            tensor = tensor.view(expanded_size)
            tensor = torch.cat([tensor[:, 0], tensor[:, 1]], dim=0)
        return tensor
