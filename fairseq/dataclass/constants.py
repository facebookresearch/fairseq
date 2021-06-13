# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum, EnumMeta
from typing import List


class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))


class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


LOG_FORMAT_CHOICES = ChoiceEnum(["json", "none", "simple", "tqdm"])
DDP_BACKEND_CHOICES = ChoiceEnum([
    "c10d",  # alias for pytorch_ddp
    "fully_sharded",  # FullyShardedDataParallel from fairscale
    "legacy_ddp",
    "no_c10d",  # alias for legacy_ddp
    "pytorch_ddp",
    "slow_mo",
])
DDP_COMM_HOOK_CHOICES = ChoiceEnum(["none", "fp16"])
DATASET_IMPL_CHOICES = ChoiceEnum(["raw", "lazy", "cached", "mmap", "fasta"])
GENERATION_CONSTRAINTS_CHOICES = ChoiceEnum(["ordered", "unordered"])
GENERATION_DECODING_FORMAT_CHOICES = ChoiceEnum(
    ["unigram", "ensemble", "vote", "dp", "bs"]
)
ZERO_SHARDING_CHOICES = ChoiceEnum(["none", "os"])
PIPELINE_CHECKPOINT_CHOICES = ChoiceEnum(["always", "never", "except_last"])
PRINT_ALIGNMENT_CHOICES = ChoiceEnum(["hard", "soft"])
