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
            score = e.log_softmax(dim=-1).max(dim=-1)[0].sum()
            toks = e.argmax(dim=-1).unique_consecutive()
            return {"tokens":toks[toks != self.blank], "score":score}
        return [[get_pred(x)] for x in emissions]
