# Copyright (c) Facebook, Inc. All Rights Reserved

import torch

from torch import nn


class Loss(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


# Dummy Loss for testing.
class DummyLoss(Loss):
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, logits, targets, **kwargs):
        return self.loss(logits, targets)


class DummyK400Loss(Loss):
    """dummy k400 loss for MViT."""
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, logits, targets, **kwargs):
        return self.loss(
            logits, torch.randint(0, 400, (logits.size(0),), device=logits.device))


class CrossEntropy(Loss):
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, logits, targets, **kwargs):
        return self.loss(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


class ArgmaxCrossEntropy(Loss):
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, logits, targets, **kwargs):
        return self.loss(logits, targets.argmax(dim=1))


class BCE(Loss):
    def __init__(self):
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, logits, targets, **kwargs):
        targets = targets.squeeze(0)
        return self.loss(logits, targets)


class NLGLoss(Loss):
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, logits, text_label, **kwargs):
        targets = text_label[text_label != -100]
        return self.loss(logits, targets)


class MSE(Loss):
    def __init__(self):
        self.loss = nn.MSELoss()

    def __call__(self, logits, targets, **kwargs):
        return self.loss(logits, targets)


class L1(Loss):
    def __init__(self):
        self.loss = nn.L1Loss()

    def __call__(self, logits, targets, **kwargs):
        return self.loss(logits, targets)


class SmoothL1(Loss):
    def __init__(self):
        self.loss = nn.SmoothL1Loss()

    def __call__(self, logits, targets, **kwargs):
        return self.loss(logits, targets)
