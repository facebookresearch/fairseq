# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F


class FairseqDropout(nn.Module):

    def __init__(self, p, args=None, parent_module=None, apply_during_inference=False):
        super().__init__()
        if args is not None and hasattr(args, 'retain_dropout') and hasattr(args, 'exclude_dropout_modules'):
            retain = getattr(args, 'retain_dropout', False)
            exclude_modules = getattr(args, 'exclude_dropout_modules') or []
            self.apply_during_inference = retain and (type(parent_module).__name__ not in exclude_modules)
        else:
            self.apply_during_inference = apply_during_inference
        self.p = p

    def forward(self, x, inplace=False):
        if self.training or self.apply_during_inference:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x
