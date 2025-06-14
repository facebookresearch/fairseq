from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor

class AbsMask(torch.nn.Module, ABC):
    @property
    @abstractmethod
    def max_num_spk(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        input,
        ilens,
        bottleneck_feat,
        num_spk,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        raise NotImplementedError
    

# This is an implementation of the multiple 1x1 convolution layer architecture
# in https://arxiv.org/pdf/2203.17068.pdf

class MultiMask(AbsMask):
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 128,
        max_num_spk: int = 3,
        mask_nonlinear="relu",
    ):
        """Multiple 1x1 convolution layer Module.

        This module corresponds to the final 1x1 conv block and
        non-linear function in TCNSeparator.
        This module has multiple 1x1 conv blocks. One of them is selected
        according to the given num_spk to handle flexible num_spk.

        Args:
            input_dim: Number of filters in autoencoder
            bottleneck_dim: Number of channels in bottleneck 1 * 1-conv block
            max_num_spk: Number of mask_conv1x1 modules
                        (>= Max number of speakers in the dataset)
            mask_nonlinear: use which non-linear function to generate mask
        """
        super().__init__()
        # Hyper-parameter
        self._max_num_spk = max_num_spk
        self.mask_nonlinear = mask_nonlinear
        # [M, B, K] -> [M, C*N, K]
        self.mask_conv1x1 = nn.ModuleList()
        for z in range(1, max_num_spk + 1):
            self.mask_conv1x1.append(
                nn.Conv1d(bottleneck_dim, z * input_dim, 1, bias=False)
            )

    @property
    def max_num_spk(self) -> int:
        return self._max_num_spk

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        bottleneck_feat: torch.Tensor,
        num_spk: int,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Keep this API same with TasNet.

        Args:
            input: [M, K, N], M is batch size
            ilens (torch.Tensor): (M,)
            bottleneck_feat: [M, K, B]
            num_spk: number of speakers
            (Training: oracle,
            Inference: estimated by other module (e.g, EEND-EDA))

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(M, K, N), ...]
            ilens (torch.Tensor): (M,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]

        """
        M, K, N = input.size()
        bottleneck_feat = bottleneck_feat.transpose(1, 2)  # [M, B, K]
        score = self.mask_conv1x1[num_spk - 1](
            bottleneck_feat
        )  # [M, B, K] -> [M, num_spk*N, K]
        # add other outputs of the module list with factor 0.0
        # to enable distributed training
        for z in range(self._max_num_spk):
            if z != num_spk - 1:
                score += 0.0 * F.interpolate(
                    self.mask_conv1x1[z](bottleneck_feat).transpose(1, 2),
                    size=num_spk * N,
                ).transpose(1, 2)
        score = score.view(M, num_spk, N, K)  # [M, num_spk*N, K] -> [M, num_spk, N, K]
        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        elif self.mask_nonlinear == "sigmoid":
            est_mask = torch.sigmoid(score)
        elif self.mask_nonlinear == "tanh":
            est_mask = torch.tanh(score)
        else:
            raise ValueError("Unsupported mask non-linear function")

        masks = est_mask.transpose(2, 3)  # [M, num_spk, K, N]
        masks = masks.unbind(dim=1)  # List[M, K, N]

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return masked, ilens, others    