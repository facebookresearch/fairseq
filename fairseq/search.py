# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor


class Search(nn.Module):
    def __init__(self, tgt_dict):
        super().__init__()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.src_lengths = torch.tensor(-1)

    def step(self, step, lprobs, scores):
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            scores: (bsz x input_beam_size x step)
                the historical model scores of each hypothesis up to this point

        Return: A tuple of (scores, indices, beams) where:
            scores: (bsz x output_beam_size)
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: (bsz x output_beam_size)
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
        """
        raise NotImplementedError

    @torch.jit.export
    def set_src_lengths(self, src_lengths):
        self.src_lengths = src_lengths


class BeamSearch(Search):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    @torch.jit.export
    def step(self, step: int, lprobs, scores: Optional[Tensor]):
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        if torch.__version__ < '1.6.0':
            beams_buf = torch.div(indices_buf, vocab_size)
        else:
            beams_buf = torch.floor_divide(indices_buf, vocab_size)
        indices_buf = indices_buf.fmod(vocab_size)
        return scores_buf, indices_buf, beams_buf


class LengthConstrainedBeamSearch(Search):
    def __init__(self, tgt_dict, min_len_a, min_len_b, max_len_a, max_len_b):
        super().__init__(tgt_dict)
        self.min_len_a = min_len_a
        self.min_len_b = min_len_b
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.beam = BeamSearch(tgt_dict)

    def step(self, step: int, lprobs, scores):
        min_lens = self.min_len_a * self.src_lengths + self.min_len_b
        max_lens = self.max_len_a * self.src_lengths + self.max_len_b
        lprobs[step < min_lens, :, self.eos] = -math.inf
        lprobs[step == max_lens, :, self.eos] = 0
        lprobs[step > max_lens, :, self.eos] = -math.inf
        return self.beam.step(step, lprobs, scores)


class DiverseBeamSearch(Search):
    """Diverse Beam Search.

    See "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence
    Models" for details.

    We only implement the Hamming Diversity penalty here, which performed best
    in the original paper.
    """

    def __init__(self, tgt_dict, num_groups, diversity_strength):
        super().__init__(tgt_dict)
        self.num_groups = num_groups
        self.diversity_strength = -diversity_strength
        self.beam = BeamSearch(tgt_dict)

    @torch.jit.export
    def step(self, step: int, lprobs, scores):
        bsz, beam_size, vocab_size = lprobs.size()
        if beam_size % self.num_groups != 0:
            raise ValueError(
                "DiverseBeamSearch requires --beam to be divisible by the number of groups"
            )

        # initialize diversity penalty
        diversity_buf = torch.zeros(lprobs[:, 0, :].size()).to(lprobs)

        scores_G, indices_G, beams_G = [], [], []
        for g in range(self.num_groups):
            lprobs_g = lprobs[:, g :: self.num_groups, :]
            scores_g = scores[:, g :: self.num_groups, :] if step > 0 else None

            # apply diversity penalty
            if g > 0:
                lprobs_g = torch.add(
                    lprobs_g, self.diversity_strength, diversity_buf.unsqueeze(1)
                )
            else:
                lprobs_g = lprobs_g.contiguous()

            scores_buf, indices_buf, beams_buf = self.beam.step(
                step, lprobs_g, scores_g
            )
            beams_buf.mul_(self.num_groups).add_(g)

            scores_G.append(scores_buf.clone())
            indices_G.append(indices_buf.clone())
            beams_G.append(beams_buf.clone())

            # update diversity penalty
            diversity_buf.scatter_add_(
                1, indices_buf, torch.ones(indices_buf.size()).to(diversity_buf)
            )

        # interleave results from different groups
        scores_buf = torch.stack(scores_G, dim=2).view(bsz, -1)
        indices_buf = torch.stack(indices_G, dim=2).view(bsz, -1)
        beams_buf = torch.stack(beams_G, dim=2).view(bsz, -1)
        return scores_buf, indices_buf, beams_buf


class Sampling(Search):
    sampling_topk: int
    sampling_topp: float

    def __init__(self, tgt_dict, sampling_topk=-1, sampling_topp=-1.0):
        super().__init__(tgt_dict)
        self.sampling_topk = sampling_topk
        self.sampling_topp = sampling_topp

    def _sample_topp(self, lprobs):
        """Sample among the smallest set of elements whose cumulative probability mass exceeds p.

        See `"The Curious Case of Neural Text Degeneration"
        (Holtzman et al., 2019) <https://arxiv.org/abs/1904.09751>`_.

        Args:
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step

        Return: A tuple of (trimed_probs, truncated_indices) where:
            trimed_probs: (bsz x input_beam_size x ?)
                the model's probabilities over the elements selected to sample from. The
                width of the third dimension is determined by top-P.
            truncated_indices: (bsz x input_beam_size x ?)
                the indices of the chosen elements.
        """
        probs = lprobs.exp_()

        # sort the last dimension (vocab dimension) in descending order
        sorted_probs, sorted_indices = probs.sort(descending=True)

        # compute a mask to indicate the words to be included in the top-P set.
        cumsum_probs = sorted_probs.cumsum(dim=2)
        mask = cumsum_probs.lt(self.sampling_topp)

        # note that mask was computed by 'lt'. One more word needs to be included
        # so that the cumulative probability mass can exceed p.
        cumsum_mask = mask.cumsum(dim=2)
        last_included = cumsum_mask[:, :, -1:]
        last_included.clamp_(0, mask.size()[2] - 1)
        mask = mask.scatter_(2, last_included, 1)

        # truncate unnecessary dims.
        max_dim = last_included.max()
        truncated_mask = mask[:, :, : max_dim + 1]
        truncated_probs = sorted_probs[:, :, : max_dim + 1]
        truncated_indices = sorted_indices[:, :, : max_dim + 1]

        # trim the words that are not in top-P by setting their probabilities
        # to 0, so that they would not be sampled later.
        trim_mask = ~truncated_mask
        trimed_probs = truncated_probs.masked_fill_(trim_mask, 0)
        return trimed_probs, truncated_indices

    @torch.jit.export
    def step(self, step: int, lprobs, scores):
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()

        if self.sampling_topp > 0:
            # only sample from the smallest set of words whose cumulative probability mass exceeds p
            probs, top_indices = self._sample_topp(lprobs)
        elif self.sampling_topk > 0:
            # only sample from top-k candidates
            lprobs, top_indices = lprobs.topk(self.sampling_topk)
            probs = lprobs.exp_()
        else:
            probs = lprobs.exp_()

            # dummy data to be consistent with true branch for type check
            top_indices = torch.empty(0).to(probs)
        # sample
        if step == 0:
            indices_buf = torch.multinomial(
                probs.view(bsz, -1), beam_size, replacement=True,
            ).view(bsz, beam_size)
        else:
            indices_buf = torch.multinomial(
                probs.view(bsz * beam_size, -1),
                1,
                replacement=True,
            ).view(bsz, beam_size)

        if step == 0:
            # expand to beam size
            probs = probs.expand(bsz, beam_size, -1)

        # gather scores
        scores_buf = torch.gather(
            probs, dim=2, index=indices_buf.unsqueeze(-1)
        )
        scores_buf = scores_buf.log_().view(bsz, -1)

        # remap indices if using top-k or top-P sampling
        if self.sampling_topk > 0 or self.sampling_topp > 0:
            indices_buf = torch.gather(
                top_indices.expand(bsz, beam_size, -1),
                dim=2,
                index=indices_buf.unsqueeze(-1),
            ).squeeze(2)

        if step == 0:
            beams_buf = indices_buf.new_zeros(bsz, beam_size)
        else:
            beams_buf = torch.arange(0, beam_size).to(indices_buf).repeat(bsz, 1)
            # make scores cumulative
            scores_buf.add_(
                torch.gather(scores[:, :, step - 1], dim=1, index=beams_buf)
            )

        return scores_buf, indices_buf, beams_buf


class DiverseSiblingsSearch(Search):
    """
    Beam search with diverse siblings.

    See "A Simple, Fast Diverse Decoding Algorithm for Neural Generation" for details.
    https://arxiv.org/abs/1611.08562

    1/ Calculate hypotheses for each beam
    2/ Intra-sibling ordering
    3/ Rewrite scores
    4/ Choose top K hypotheses

    if diversity_rate == 0 is equivalent to BeamSearch
    """

    def __init__(self, tgt_dict, diversity_rate):
        super().__init__(tgt_dict)
        self.diversity_rate = diversity_rate
        self.beam = BeamSearch(tgt_dict)

    def step(self, step: int, lprobs, scores):
        bsz, beam_size, vocab_size = lprobs.size()
        k = min(
            # Take the best 2 x beam_size predictions. We'll choose the first
            # beam_size of these which don't predict eos to continue with.
            beam_size * 2,
            lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
        )
        s_list: List[Tensor]
        i_list: List[Tensor]
        s_list = [torch.empty(0).to(lprobs) for i in range(beam_size)]
        i_list = [torch.LongTensor().to(device=lprobs.device) for i in range(beam_size)]
        sibling_score = torch.arange(1, k + 1).to(lprobs) * self.diversity_rate

        if step == 0:
            return self.beam.step(step, lprobs, scores)
        lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

        # 1/ Calculate hypotheses for each beam
        for i in range(beam_size):
            torch.topk(lprobs[:, i, :].view(bsz, -1), k, out=(s_list[i], i_list[i]))
            i_list[i].fmod_(vocab_size)

            # 2/ Intra-sibling ordering by default from topk + 3/ Rewrite scores
            s_list[i].sub_(sibling_score)

        # 4/ Choose top K hypotheses
        indices = torch.stack(i_list, dim=1).view(bsz, -1)

        final_scores = torch.empty(0).to(lprobs)
        final_indices = torch.LongTensor().to(device=lprobs.device)
        final_beams = torch.LongTensor().to(device=lprobs.device)
        (final_scores, final_indices) = torch.topk(
            torch.stack(s_list, dim=1).view(bsz, -1),
            k,
        )

        final_beams = torch.div(final_indices, k)

        for i in range(bsz):
            final_indices[i] = indices[i][final_indices[i]]

        return final_scores, final_indices, final_beams
