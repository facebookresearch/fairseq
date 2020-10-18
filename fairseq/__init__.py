# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

__all__ = ['pdb']
__version__ = '1.0.0a0'

import sys

# backwards compatibility to support `from fairseq.meters import AverageMeter`
from fairseq.logging import meters, metrics, progress_bar  # noqa
sys.modules['fairseq.meters'] = meters
sys.modules['fairseq.metrics'] = metrics
sys.modules['fairseq.progress_bar'] = progress_bar

import fairseq.criterions  # noqa
import fairseq.models  # noqa
import fairseq.modules  # noqa
import fairseq.optim  # noqa
import fairseq.optim.lr_scheduler  # noqa
import fairseq.pdb  # noqa
import fairseq.scoring  # noqa
import fairseq.tasks  # noqa
import fairseq.token_generation_constraints  # noqa

import fairseq.benchmark  # noqa
import fairseq.model_parallel  # noqa
