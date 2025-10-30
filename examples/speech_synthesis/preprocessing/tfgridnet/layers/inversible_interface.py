from abc import ABC, abstractmethod
from typing import Tuple

import torch


class InversibleInterface(ABC):
    @abstractmethod
    def inverse(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # return output, output_lengths
        raise NotImplementedError
