# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


class MultichannelSearch(nn.Module):
    def __init__(self, tgt_dicts):
        super().__init__()
        tgt_dict = list(tgt_dicts.values())[0]
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        for tgt_dict in tgt_dicts.values():
            assert self.pad == tgt_dict.pad()
            assert self.unk == tgt_dict.unk()
            assert self.eos == tgt_dict.eos()
        self.vocab_sizes = {channel: len(tgt_dicts[channel]) for channel in tgt_dicts}
        self.src_lengths = torch.tensor(-1)
        self.supports_constraints = False
        self.stop_on_max_len = False

    def step(
        self, step, lprobs, scores, prev_output_tokens=None, original_batch_idxs=None
    ):
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            lprobs: dictionary of channels {channel : (bsz x input_beam_size x vocab_size_channel)}
                the model's log-probabilities over the vocabulary at the current step
            scores: {channel : (bsz x input_beam_size x step)}
                the historical model scores of each hypothesis up to this point
            prev_output_tokens: {channel : (bsz x step)}
                the previously generated oputput tokens
            original_batch_idxs: (bsz)
                the tensor with the batch indices, in the range [0, bsz)
                this is useful in case there has been applied a re-ordering
                and we need to know the orignal indices

        Return: A tuple of (scores, indices, beams) where:
            scores: {channel : (bsz x output_beam_size)}
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: {channel : (bsz x output_beam_size)}
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
        """
        raise NotImplementedError

    @torch.jit.export
    def set_src_lengths(self, src_lengths):
        self.src_lengths = src_lengths

    @torch.jit.export
    def init_constraints(self, batch_constraints: Optional[Tensor], beam_size: int):
        """Initialize constraint states for constrained decoding (if supported).

        Args:
            batch_constraints: (torch.Tensor, optional)
                the list of constraints, in packed form
            beam_size: (int)
                the beam size
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        pass

    def prune_sentences(self, batch_idxs: Tensor):
        """
        Removes constraint states for completed sentences (if supported).
        This is called from sequence_generator._generate() when sentences are
        deleted from the batch.

        Args:
            batch_idxs: Indices of *sentences* whose constraint state should be *kept*.
        """
        pass

    def update_constraints(self, active_hypos: Tensor):
        """
        Updates the constraint states by selecting the beam items that are retained.
        This is called at each time step of sequence_generator._generate() when
        the set of 2 * {beam_size} candidate hypotheses are reduced to the beam size.

        Args:
            active_hypos: (batch size, beam size)
              list of integers denoting, for each sentence, which beam candidate items
              should be kept.
        """
        pass


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return torch.stack(tuple(reversed(out)), dim=-1)


def topk_sum(lprobs_list, k):
    """
    lprobs_list = [lprobs_1,...,lprobs_n], where:
        lprobs_1 : (batch_size x beam_size x vocab_1)
        ...
        lprobs_n : (batch_size x beam_size x vocab_n)

    Return:
        - topk_values : (batch_size x k)
            values of the topk sum of the form :
                lprobs_1[bsz, beam_idx, vocab_1_idx] + ... + lprobs_n[bsz, beam_idx, vocab_n_idx]
        - topk_idxs : (batch_size x k x n+1)
            each (n+1)-tensor being [beam_idx, vocab_1_idx, ..., vocab_n_idx]
    """
    # Reduce all lprobs to k candidates first to reduce later complexity
    # We may assume that k << vocab
    lprobs_topk_list = []
    lprobs_topk_indices_list = []
    for lprobs in lprobs_list:
        k_i = min(k, lprobs.size(-1))
        topk_values, topk_indices = torch.topk(lprobs, k=k_i)
        # topk_values : (batch_size x beam_size x k_i)
        # topk_indices : (batch_size x beam_size x k_i)
        lprobs_topk_list.append(topk_values)
        lprobs_topk_indices_list.append(topk_indices)

    # Compute all possible sums
    sum_lprobs_topk = lprobs_topk_list[0]
    for i in range(1, len(lprobs_topk_list)):
        unsqueezed_lprobs = lprobs_topk_list[i]
        for _ in range(i):
            unsqueezed_lprobs = unsqueezed_lprobs.unsqueeze(-2)
        sum_lprobs_topk = sum_lprobs_topk.unsqueeze(-1) + unsqueezed_lprobs
    # sum_lprobs : (batch_size x beam_size x k_1 x ... x k_n)

    # Get the top k sums and the (transformed indices)
    topk_sum_values, topk_sum_indices = torch.topk(
        sum_lprobs_topk.view(sum_lprobs_topk.size(0), -1), k=k
    )
    # topk_sum_values : (batch_size x k)
    # topk_sum_indices : (batch_size x k)
    topk_sum_indices = unravel_index(topk_sum_indices, tuple(sum_lprobs_topk.shape[1:]))
    # topk_sum_indices : (batch_size x k x n+1)

    # Convert the transformed indices to the true indices
    for i_batch in range(topk_sum_indices.size(0)):
        for i_cand in range(topk_sum_indices.size(1)):
            i_beam, *transformed_vocab_indices = topk_sum_indices[i_batch, i_cand]
            true_vocab_indices = [i_beam]
            for j, transformed_vocab_j_idx in enumerate(transformed_vocab_indices):
                true_vocab_j_idx = lprobs_topk_indices_list[j][
                    i_batch, i_beam, transformed_vocab_j_idx
                ]
                true_vocab_indices.append(true_vocab_j_idx)
            topk_sum_indices[i_batch, i_cand] = torch.tensor(true_vocab_indices)

    topk_sum_beams = topk_sum_indices[:, :, 0]
    topk_sum_indices = topk_sum_indices[:, :, 1:]

    return topk_sum_values, topk_sum_indices, topk_sum_beams


class MultichannelBeamSearch(MultichannelSearch):
    def __init__(self, tgt_dicts):
        super().__init__(tgt_dicts)
        self.constraint_states = None

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs,
        scores: Optional[Dict[str, Tensor]],
        prev_output_tokens: Optional[Dict[str, Tensor]] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        channels = list(lprobs.keys())
        bsz, beam_size, _ = lprobs[channels[0]].size()

        lprobs_list = []
        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            for channel in channels:
                lprobs_list.append(lprobs[channel][:, ::beam_size, :].contiguous())
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            for channel in channels:
                lprobs_list.append(
                    lprobs[channel] + scores[channel][:, :, step - 1].unsqueeze(-1)
                )

        topk_sum_values, topk_sum_indices, topk_sum_beams = topk_sum(
            lprobs_list, k=beam_size * 2
        )

        beams_buf = topk_sum_beams
        scores_buf = {}
        indices_buf = {}
        for i, channel in enumerate(channels):
            indices_buf[channel] = topk_sum_indices[:, :, i]
            scores_buf[channel] = (
                torch.tensor(
                    [
                        lprobs_list[i][i_batch, i_beam, i_index]
                        for i_batch in range(bsz)
                        for i_beam, i_index in zip(
                            beams_buf[i_batch], indices_buf[channel][i_batch]
                        )
                    ]
                )
                .view(bsz, -1)
                .to(lprobs_list[i].device)
            )

        # At this point, beams_buf and indices_buf are single-dim and contain relative indices
        return scores_buf, indices_buf, beams_buf


class ContiguousMultichannelBeamSearch(MultichannelSearch):
    def __init__(self, tgt_dicts):
        super().__init__(tgt_dicts)
        self.constraint_states = None

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs,
        scores: Optional[Tensor],
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        n_channels = len(lprobs)
        bsz, beam_size, _ = lprobs[0].size()

        lprobs_list = []
        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            for i in range(n_channels):
                lprobs_list.append(lprobs[i][:, ::beam_size, :].contiguous())
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            for i in range(n_channels):
                lprobs_list.append(lprobs[i] + scores[:, :, step - 1, i].unsqueeze(-1))

        topk_sum_values, topk_sum_indices, topk_sum_beams = topk_sum(
            lprobs_list, k=beam_size * 2
        )

        beams_buf = topk_sum_beams
        indices_buf = topk_sum_indices
        scores_buf = (
            torch.tensor(
                [
                    lprobs_list[i][i_batch, i_beam, i_index]
                    for i in range(len(lprobs_list))
                    for i_batch in range(bsz)
                    for i_beam, i_index in zip(
                        beams_buf[i_batch], indices_buf[i_batch, :, i]
                    )
                ]
            )
            .view(len(lprobs_list), bsz, -1)
            .permute(1, 2, 0)
            .to(lprobs_list[0].device)
        )

        # At this point, beams_buf and indices_buf are single-dim and contain relative indices
        return scores_buf, indices_buf, beams_buf


class ContiguousMultichannelSampling(MultichannelSearch):
    sampling_topk: int
    sampling_topp: float

    def __init__(self, tgt_dicts, sampling_topk=-1, sampling_topp=-1.0):
        super().__init__(tgt_dicts)
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
    def step(
        self,
        step: int,
        lprobs,
        scores,
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        n_channels = len(lprobs)
        bsz, beam_size, vocab_size = lprobs[0].size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            for i in range(n_channels):
                lprobs[i] = lprobs[i][:, ::beam_size, :].contiguous()

        probs = []
        top_indices = []
        for i in range(n_channels):
            if self.sampling_topp > 0:
                # only sample from the smallest set of words whose cumulative probability mass exceeds p
                probs_i, top_indices_i = self._sample_topp(lprobs[i])
            elif self.sampling_topk > 0:
                # only sample from top-k candidates
                lprobs[i], top_indices_i = lprobs[i].topk(
                    min(self.sampling_topk, lprobs[i].size(-1))
                )
                probs_i = lprobs[i].exp_()
            else:
                probs_i = lprobs[i].exp_()

                # dummy data to be consistent with true branch for type check
                top_indices_i = torch.empty(0).to(probs_i)
            probs.append(probs_i)
            top_indices.append(top_indices_i)
        # sample
        indices_buf = []
        for i in range(n_channels):
            if step == 0:
                indices_buf.append(
                    torch.multinomial(
                        probs[i].view(bsz, -1),
                        beam_size,
                        replacement=True,
                    ).view(bsz, beam_size)
                )
            else:
                indices_buf.append(
                    torch.multinomial(
                        probs[i].view(bsz * beam_size, -1),
                        1,
                        replacement=True,
                    ).view(bsz, beam_size)
                )

        if step == 0:
            for i in range(n_channels):
                # expand to beam size
                probs[i] = probs[i].expand(bsz, beam_size, -1)

        # gather scores
        scores_buf = []
        for i in range(n_channels):
            scores_buf.append(
                torch.gather(probs[i], dim=2, index=indices_buf[i].unsqueeze(-1))
            )
            scores_buf[i] = scores_buf[i].log_().view(bsz, -1)

        # remap indices if using top-k or top-P sampling
        if self.sampling_topk > 0 or self.sampling_topp > 0:
            for i in range(n_channels):
                indices_buf[i] = torch.gather(
                    top_indices[i].expand(bsz, beam_size, -1),
                    dim=2,
                    index=indices_buf[i].unsqueeze(-1),
                ).squeeze(2)

        if step == 0:
            beams_buf = indices_buf[0].new_zeros(bsz, beam_size)
        else:
            beams_buf = torch.arange(0, beam_size).to(indices_buf[0]).repeat(bsz, 1)
            # make scores cumulative
            for i in range(n_channels):
                scores_buf[i].add_(
                    torch.gather(scores[:, :, step - 1, i], dim=1, index=beams_buf)
                )
        scores_buf = torch.stack(scores_buf, dim=-1)
        indices_buf = torch.stack(indices_buf, dim=-1)

        return scores_buf, indices_buf, beams_buf
