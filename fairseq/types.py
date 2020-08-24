# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Utility file containing some commonly used data types in fairseq."""

from typing import Dict, List, NamedTuple, Optional, Tuple

from torch import Tensor

IncrementalState = Dict[str, Dict[str, Optional[Tensor]]]
IncrementalBuffer = Dict[str, Optional[Tensor]]

class EncoderOut(NamedTuple):
    encoder_out: Tensor
    encoder_padding_mask: Optional[Tensor] = None
    encoder_embedding: Optional[Tensor] = None
    encoder_states: Optional[List[Tensor]] = None
    src_tokens: Optional[Tensor] = None
    src_lengths: Optional[Tensor] = None
