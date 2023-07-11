# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file is to re-implemented the low-rank and beam approximation of CRF layer
Proposed by:

Sun, Zhiqing, et al.
Fast Structured Decoding for Sequence Models
https://arxiv.org/abs/1910.11555

The CRF implementation is mainly borrowed from
https://github.com/kmkurn/pytorch-crf/blob/master/torchcrf/__init__.py

"""

import numpy as np
import torch
import torch.nn as nn


def logsumexp(x, dim=1):
    return torch.logsumexp(x.float(), dim=dim).type_as(x)


class DynamicCRF(nn.Module):
    """Dynamic CRF layer is used to approximate the traditional
    Conditional Random Fields (CRF)
    $P(y | x) = 1/Z(x) exp(sum_i s(y_i, x) + sum_i t(y_{i-1}, y_i, x))$

    where in this function, we assume the emition scores (s) are given,
    and the transition score is a |V| x |V| matrix $M$

    in the following two aspects:
     (1) it used a low-rank approximation for the transition matrix:
         $M = E_1 E_2^T$
     (2) it used a beam to estimate the normalizing factor Z(x)
    """

    def __init__(self, num_embedding, low_rank=32, beam_size=64):
        super().__init__()

        self.E1 = nn.Embedding(num_embedding, low_rank)
        self.E2 = nn.Embedding(num_embedding, low_rank)

        self.vocb = num_embedding
        self.rank = low_rank
        self.beam = beam_size

    def extra_repr(self):
        return "vocab_size={}, low_rank={}, beam_size={}".format(
            self.vocb, self.rank, self.beam
        )

    def forward(self, emissions, targets, masks, beam=None):
        """
        Compute the conditional log-likelihood of a sequence of target tokens given emission scores

        Args:
            emissions (`~torch.Tensor`): Emission score are usually the unnormalized decoder output
                ``(batch_size, seq_len, vocab_size)``. We assume batch-first
            targets (`~torch.LongTensor`): Sequence of target token indices
                ``(batch_size, seq_len)
            masks (`~torch.ByteTensor`): Mask tensor with the same size as targets

        Returns:
            `~torch.Tensor`: approximated log-likelihood
        """
        numerator = self._compute_score(emissions, targets, masks)
        denominator = self._compute_normalizer(emissions, targets, masks, beam)
        return numerator - denominator

    def forward_decoder(self, emissions, masks=None, beam=None):
        """
        Find the most likely output sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score are usually the unnormalized decoder output
                ``(batch_size, seq_len, vocab_size)``. We assume batch-first
            masks (`~torch.ByteTensor`): Mask tensor with the same size as targets

        Returns:
            `~torch.LongTensor`: decoded sequence from the CRF model
        """
        return self._viterbi_decode(emissions, masks, beam)

    def _compute_score(self, emissions, targets, masks=None):
        batch_size, seq_len = targets.size()
        emission_scores = emissions.gather(2, targets[:, :, None])[:, :, 0]  # B x T
        transition_scores = (self.E1(targets[:, :-1]) * self.E2(targets[:, 1:])).sum(2)

        scores = emission_scores
        scores[:, 1:] += transition_scores

        if masks is not None:
            scores = scores * masks.type_as(scores)
        return scores.sum(-1)

    def _compute_normalizer(self, emissions, targets=None, masks=None, beam=None):
        # HACK: we include "target" which is a hueristic for training
        # HACK: we use a beam of tokens to approximate the normalizing factor (which is bad?)

        beam = beam if beam is not None else self.beam
        batch_size, seq_len = emissions.size()[:2]
        if targets is not None:
            _emissions = emissions.scatter(2, targets[:, :, None], np.float("inf"))
            beam_targets = _emissions.topk(beam, 2)[1]
            beam_emission_scores = emissions.gather(2, beam_targets)
        else:
            beam_emission_scores, beam_targets = emissions.topk(beam, 2)
        beam_transition_score1 = self.E1(beam_targets[:, :-1])  # B x (T-1) x K x D
        beam_transition_score2 = self.E2(beam_targets[:, 1:])  # B x (T-1) x K x D
        beam_transition_matrix = torch.bmm(
            beam_transition_score1.view(-1, beam, self.rank),
            beam_transition_score2.view(-1, beam, self.rank).transpose(1, 2),
        )
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1, beam, beam)

        # compute the normalizer in the log-space
        score = beam_emission_scores[:, 0]  # B x K
        for i in range(1, seq_len):
            next_score = score[:, :, None] + beam_transition_matrix[:, i - 1]
            next_score = logsumexp(next_score, dim=1) + beam_emission_scores[:, i]

            if masks is not None:
                score = torch.where(masks[:, i : i + 1], next_score, score)
            else:
                score = next_score

        # Sum (log-sum-exp) over all possible tags
        return logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, masks=None, beam=None):
        # HACK: we use a beam of tokens to approximate the normalizing factor (which is bad?)

        beam = beam if beam is not None else self.beam
        batch_size, seq_len = emissions.size()[:2]
        beam_emission_scores, beam_targets = emissions.topk(beam, 2)
        beam_transition_score1 = self.E1(beam_targets[:, :-1])  # B x (T-1) x K x D
        beam_transition_score2 = self.E2(beam_targets[:, 1:])  # B x (T-1) x K x D
        beam_transition_matrix = torch.bmm(
            beam_transition_score1.view(-1, beam, self.rank),
            beam_transition_score2.view(-1, beam, self.rank).transpose(1, 2),
        )
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1, beam, beam)

        traj_tokens, traj_scores = [], []
        finalized_tokens, finalized_scores = [], []

        # compute the normalizer in the log-space
        score = beam_emission_scores[:, 0]  # B x K
        dummy = (
            torch.arange(beam, device=score.device).expand(*score.size()).contiguous()
        )

        for i in range(1, seq_len):
            traj_scores.append(score)
            _score = score[:, :, None] + beam_transition_matrix[:, i - 1]
            _score, _index = _score.max(dim=1)
            _score = _score + beam_emission_scores[:, i]

            if masks is not None:
                score = torch.where(masks[:, i : i + 1], _score, score)
                index = torch.where(masks[:, i : i + 1], _index, dummy)
            else:
                score, index = _score, _index
            traj_tokens.append(index)

        # now running the back-tracing and find the best
        best_score, best_index = score.max(dim=1)
        finalized_tokens.append(best_index[:, None])
        finalized_scores.append(best_score[:, None])

        for idx, scs in zip(reversed(traj_tokens), reversed(traj_scores)):
            previous_index = finalized_tokens[-1]
            finalized_tokens.append(idx.gather(1, previous_index))
            finalized_scores.append(scs.gather(1, previous_index))

        finalized_tokens.reverse()
        finalized_tokens = torch.cat(finalized_tokens, 1)
        finalized_tokens = beam_targets.gather(2, finalized_tokens[:, :, None])[:, :, 0]

        finalized_scores.reverse()
        finalized_scores = torch.cat(finalized_scores, 1)
        finalized_scores[:, 1:] = finalized_scores[:, 1:] - finalized_scores[:, :-1]

        return finalized_scores, finalized_tokens
