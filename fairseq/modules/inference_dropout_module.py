import torch.nn as nn


class InferenceDropoutModule(nn.Module):
    """Base class for modules that allow to apply dropout at inference time."""

    def __init__(self):
        super().__init__()
        self.retain_dropout = False

    def is_dropout_applied(self):
        return self.retain_dropout or self.training
