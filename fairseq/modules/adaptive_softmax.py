# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import torch.nn.functional as F
from torch import nn


class AdaptiveSoftmax(nn.Module):
    """This is an implementation of the efficient softmax approximation for graphical processing units (GPU),
    described in the paper "Efficient softmax approximation for GPUs" (http://arxiv.org/abs/1609.04309)."""

    def __init__(self, vocab_size, input_dim, cutoff, dropout):
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]

        output_dim = cutoff[0] + len(cutoff) - 1

        self.vocab_size = vocab_size
        self.cutoff = cutoff
        self.dropout = dropout

        self.lsm = nn.LogSoftmax(dim=1)
        self.head = nn.Linear(input_dim, output_dim, bias=False)
        self.tail = nn.ModuleList()

        for i in range(len(cutoff) - 1):
            self.tail.append(
                nn.Sequential(
                    nn.Linear(input_dim, input_dim // 4 ** i, bias=False),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim // 4 ** i, cutoff[i + 1] - cutoff[i], bias=False)
                )
            )

        def init_weights(m):
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform(m.weight)

        self.apply(init_weights)

    def adapt_target(self, target):
        """In order to be efficient, the AdaptiveSoftMax does not compute the scores for all the word of the
        vocabulary for all the examples.It is thus necessary to call the method adapt_target of the AdaptiveSoftMax
        layer inside each forward pass."""

        target = target.view(-1)
        new_target = [target.clone()]
        target_idxs = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i - 1

            if mask.any():
                target_idxs.append(mask.nonzero().squeeze(1))
                new_target.append(target[mask].add(-self.cutoff[i]))
            else:
                target_idxs.append(None)
                new_target.append(None)

        return new_target, target_idxs

    def forward(self, input, target):
        """ accepts input (b x t x d) and target (b x t) and returns
            2 lists: output for each cutoff section and new targets by cut off """

        input = input.contiguous().view(-1, input.size(-1))
        input = F.dropout(input, p=self.dropout, training=self.training)
        
        new_target, target_idxs = self.adapt_target(target)
        output = [self.head(input)]

        for i in range(len(target_idxs)):
            if target_idxs[i] is not None:
                output.append(self.tail[i](input.index_select(0, target_idxs[i])))
            else:
                output.append(None)

        return output, new_target

    def get_log_prob(self, input, target):
        """computes the log probabilities for all the words of the vocabulary, given a 2D tensor of hidden vectors"""

        bsz, length, dim = input.size()
        input = input.contiguous().view(-1, dim)

        if target is not None:
            _, target_idxs = self.adapt_target(target)
        else:
            target_idxs = None

        head_y = self.head(input)
        log_probs = head_y.new_zeros(input.size(0), self.vocab_size)

        head_sz = self.cutoff[0] + len(self.tail)
        log_probs[:, :head_sz] = self.lsm(head_y)
        tail_priors = log_probs[:, self.cutoff[0] - 1: head_sz - 1].clone()

        for i in range(len(self.tail)):
            start = self.cutoff[i]
            end = self.cutoff[i + 1]

            if target_idxs is None:
                tail_out = log_probs[:, start:end]
                tail_out.copy_(self.tail[i](input))
                log_probs[:, start:end] = self.lsm(tail_out).add_(tail_priors[:, i, None])
            elif target_idxs[i] is not None:
                idxs = target_idxs[i]
                tail_out = log_probs[idxs, start:end]
                tail_out.copy_(self.tail[i](input[idxs]))
                log_probs[idxs, start:end] = self.lsm(tail_out).add_(tail_priors[idxs, i, None])

        log_probs = log_probs.view(bsz, length, -1)
        return log_probs
