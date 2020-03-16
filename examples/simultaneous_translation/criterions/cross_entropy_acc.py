# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion

@register_criterion("latency_augmented_cross_entropy_acc")
class LatencyAugmentedCrossEntropyWithAccCriterion(FairseqCriterion):
    pass
