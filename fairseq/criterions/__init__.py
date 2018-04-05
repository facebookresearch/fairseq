# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from .cross_entropy import CrossEntropyCriterion
from .combined_sequence_criterion import CombinedSequenceCriterion
from .fairseq_criterion import FairseqCriterion
from .fairseq_sequence_criterion import FairseqSequenceCriterion
from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from .sequence_cross_entropy import SequenceCrossEntropyCriterion
from .sequence_max_margin_criterion import SequenceMaxMarginCriterion
from .sequence_risk_criterion import SequenceRiskCriterion
from .sequence_soft_max_margin import SequenceSoftMaxMarginCriterion
from .sequence_multi_margin_criterion import SequenceMultiMarginCriterion

sequence_criterions = [
    'SequenceCrossEntropyCriterion',
    'SequenceMaxMarginCriterion',
    'SequenceRiskCriterion',
    'SequenceSoftMaxMarginCriterion',
    'SequenceMultiMarginCriterion'
]

__all__ = sequence_criterions + [
    'CrossEntropyCriterion',
    'LabelSmoothedCrossEntropyCriterion',
]
