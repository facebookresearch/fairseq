# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import torch.nn as nn


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self):
        super().__init__()

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        raise NotImplementedError
