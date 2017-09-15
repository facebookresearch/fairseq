# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from .cross_entropy import CrossEntropyCriterion
from .fairseq_criterion import FairseqCriterion
from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

__all__ = [
    'CrossEntropyCriterion',
    'LabelSmoothedCrossEntropyCriterion',
]
