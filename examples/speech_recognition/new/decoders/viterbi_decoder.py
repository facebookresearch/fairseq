#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from typing import List, Dict

from .base_decoder import BaseDecoder


class ViterbiDecoder(BaseDecoder):
    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        def get_pred(e):
            toks = e.argmax(dim=-1).unique_consecutive()
            return toks[toks != self.blank]

        return [[{"tokens": get_pred(x), "score": 0}] for x in emissions]
