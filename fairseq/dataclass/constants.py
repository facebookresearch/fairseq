# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.dataclass.utils import ChoiceEnum


LOG_FORMAT_CHOICES = ChoiceEnum(["json", "none", "simple", "tqdm"])
DDP_BACKEND_CHOICES = ChoiceEnum(["c10d", "no_c10d"])
DISTRIBUTED_WRAPPER_CHOICES = ChoiceEnum(["DDP", "SlowMo"])
ZERO_SHARDING_CHOICES = ChoiceEnum(["none", "os"])
PIPELINE_CHECKPOINT_CHOICES = ChoiceEnum(["always", "never", "except_last"])
