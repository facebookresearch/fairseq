# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

import torch
import torch.nn as nn
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from fairseq.ngram_repeat_block import NGramRepeatBlock
from .multichannel_search import ContiguousMultichannelBeamSearch
from fairseq.models.speech_dlm import SpeechDLM


class MultichannelSequenceGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dicts,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
        duration_temperature=1.0,
    ):
        """Generate multi-channel parallel units with the SpeechDLM model
        as described in the paper: https://arxiv.org/pdf/2203.16502.pdf;

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
            duration_temperature (float, optional): rate of the duration prediction,
                higher rate induces a faster generated wav (default: 1.0)
        """
        super().__init__()
        if isinstance(models, MultichannelEnsembleModel):
            self.model = models
        else:
            self.model = MultichannelEnsembleModel(models)
        self.tgt_dicts = tgt_dicts
        self.pad = list(tgt_dicts.values())[0].pad()
        self.unk = list(tgt_dicts.values())[0].unk()
        self.eos = list(tgt_dicts.values())[0].eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.channels = list(tgt_dicts.keys())
        self.n_channels = len(self.channels)
        self.vocab_sizes = [len(tgt_dicts[channel]) for channel in self.channels]
        # the max beam size is the dictionary size - 1, since we never select pad
        max_possible_beam_size = 1
        for i in self.vocab_sizes:
            max_possible_beam_size *= i - 1
        self.beam_size = min(beam_size, max_possible_beam_size)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        if isinstance(temperature, (int, float)):
            temperature = {channel: temperature for channel in self.channels}
        elif isinstance(temperature, ListConfig) or isinstance(temperature, list):
            temperature = {
                channel: temperature[i] for i, channel in enumerate(self.channels)
            }
        assert isinstance(temperature, DictConfig) or isinstance(
            temperature, dict
        ), f"temperature: expected dict, but found {type(temperature)}"
        self.temperature = temperature
        self.match_source_len = match_source_len

        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        for channel in temperature:
            assert temperature[channel] > 0, "--temperature must be greater than 0"

        if search_strategy is None:
            self.search = ContiguousMultichannelBeamSearch(tgt_dicts)
        else:
            self.search = search_strategy
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

        self.duration_prediction = bool(
            str(getattr(models[0].decoder.args, "duration_prediction", "false")).lower()
            == "true"
        )
        self.delayed_duration = bool(
            str(
                getattr(models[0].decoder.args, "delayed_duration_target", "false")
            ).lower()
            == "true"
        )
        self.duration_temperature = duration_temperature

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],  # TODO: Modify this
        prefix_tokens: Optional[Dict[str, Tensor]] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (dict of torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (dict of torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Dict[str, Tensor]] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """
        Here sample is expected to have the following form
            {
                'id': index,
                'net_input': {
                    'src_tokens': {
                        'channel1' : tensor((batch x src_length)),
                        'channel2' : tensor((batch x src_length)),
                    },
                    ...
                },
            }
        and prefix_tokens
            {
                'channel1' : tensor((batch x prefix_length)),
                'channel2' : tensor((batch x prefix_length)),
            }
        """
        if self.model.is_speech_dlm:
            incremental_states = torch.jit.annotate(
                List[Dict[str, Dict[str, Optional[Tensor]]]],
                [
                    torch.jit.annotate(
                        List[Dict[str, Dict[str, Optional[Tensor]]]],
                        [{} for _ in range(self.n_channels)],
                    )
                    for i in range(self.model.models_size)
                ],
            )
        else:
            incremental_states = torch.jit.annotate(
                List[Dict[str, Dict[str, Optional[Tensor]]]],
                [
                    torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                    for i in range(self.model.models_size)
                ],
            )
        net_input = sample["net_input"]
        # Convert from dict to tensor form
        # shape of src_tokens : (bsz x src_len x n_channels)
        src_tokens = torch.stack(
            [net_input["src_tokens"][channel] for channel in self.channels], dim=-1
        )
        prefix_tokens = torch.stack(
            [prefix_tokens[channel] for channel in self.channels], dim=-1
        )
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (
            (src_tokens[..., 0].ne(self.eos) & src_tokens[..., 0].ne(self.pad))
            .long()
            .sum(dim=1)
        )

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        # cumulative scores of hypotheses
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1, self.n_channels)
            .to(src_tokens)
            .float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2, self.n_channels)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        if self.duration_prediction:
            dur_counter = torch.ones(bsz * beam_size, self.n_channels).to(src_tokens)
            # save the indice where the dur_counter just copied from dur_pred
            dur_counter_jump_indices = None

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            input_tokens = {
                channel: tokens[:, : step + 1, i]
                for i, channel in enumerate(self.channels)
            }

            lprobs_dict, avg_attn_scores = self.model.forward_decoder(
                input_tokens,
                encoder_outs,
                incremental_states,
                self.temperature,
            )

            # Because the sizes of vocab is different, we cannot concat the lprobs to form a single tensor
            if not self.duration_prediction:
                lprobs_list = list(lprobs_dict.values())
            else:
                lprobs_list = [
                    net_output["pred_token"] for net_output in lprobs_dict.values()
                ]

                # non-positive predicted durations
                dur_preds = (
                    torch.stack(
                        [
                            net_output["pred_duration"]
                            for net_output in lprobs_dict.values()
                        ]
                    )
                    .squeeze(-1)
                    .T
                )
                dur_preds = dur_preds / self.duration_temperature
                dur_preds = dur_preds.round().long()
                dur_preds[dur_preds < 1] = 1

                # dur_preds & dur_counter needs to be modified when there isn't an edge
                if step > 0:
                    non_edge_indices = tokens[:, step, :] == tokens[:, step - 1, :]
                    if self.delayed_duration:
                        dur_preds[non_edge_indices] = 1
                    else:
                        if dur_counter_jump_indices is not None:
                            dur_counter[dur_counter_jump_indices & non_edge_indices] = 2

                # update dur_counter
                if step > 0:
                    if self.delayed_duration:
                        dur_counter -= (
                            (dur_counter == 1)
                            | (tokens[:, step, :] == tokens[:, step - 1, :])
                        ).int()
                        dur_counter[dur_counter < 0] = 0
                    else:
                        dur_counter -= (
                            tokens[:, step, :] == tokens[:, step - 1, :]
                        ).int()
                        dur_counter[dur_counter < 1] = 1

                # whether to copy previous token (ie. if the counter is still on)
                # and get get the new duration
                if self.delayed_duration:
                    dur_counter_jump_indices = dur_counter == 0
                    dur_counter[dur_counter_jump_indices] = dur_preds[
                        dur_counter_jump_indices
                    ]

                # whether to copy previous token in this step
                copy_prev_token = dur_counter != 1
                if self.delayed_duration is False:
                    dur_counter_jump_indices = dur_counter == 1
                    dur_counter[dur_counter_jump_indices] = dur_preds[
                        dur_counter_jump_indices
                    ]
                # else:
                # dur_counter[dur_counter==0] = dur_preds[dur_counter==0] - 1
                # copy_prev_token = (dur_counter > 0)

            if self.lm_model is not None:
                assert False, "Currently not supported in multichannelLM case"

            for i in range(self.n_channels):
                lprobs_list[i][lprobs_list[i] != lprobs_list[i]] = torch.tensor(
                    -math.inf
                ).to(lprobs_list[i])

                lprobs_list[i][:, self.pad] = -math.inf  # never select pad
                lprobs_list[i][:, self.unk] -= self.unk_penalty  # apply unk penalty

                # handle max length constraint
                if step >= max_len:
                    lprobs_list[i][:, : self.eos] = -math.inf
                    lprobs_list[i][:, self.eos + 1 :] = -math.inf
                else:
                    lprobs_list[i][
                        :, self.eos
                    ] = -math.inf  # quick fix for short generation

                # handle prefix tokens (possibly with different lengths)
                if (
                    prefix_tokens is not None
                    and step < prefix_tokens.size(1)
                    and step < max_len
                ):
                    (
                        lprobs_list[i],
                        tokens[..., i],
                        scores[..., i],
                    ) = self._prefix_tokens(
                        step,
                        lprobs_list[i],
                        scores[..., i],
                        tokens[..., i],
                        prefix_tokens[..., i],
                        beam_size,
                    )
                    if self.duration_prediction:
                        # Can copy previous token if the prefix token is padding or unk (1-channel conditionned case)
                        can_copy_mask = (
                            prefix_tokens[:, step, i].eq(self.pad)
                            | prefix_tokens[:, step, i].eq(self.unk)
                        ).repeat_interleave(beam_size)
                        copy_prev_token[:, i] &= can_copy_mask
                elif step < self.min_len:
                    # minimum length constraint (does not apply if using prefix_tokens)
                    lprobs_list[i][:, self.eos] = -math.inf

                if self.duration_prediction:
                    if step < max_len:
                        for j in range(copy_prev_token.size(0)):
                            if copy_prev_token[j, i]:
                                prev_token = tokens[j, step, i]
                                lprobs_list[i][j, :prev_token] = -math.inf
                                lprobs_list[i][j, prev_token + 1 :] = -math.inf
                                # lprobs_list[i][j, prev_token] = 0.
                                # dur_counter[j,i] -= 1
                            # else:
                            #     prev_token = tokens[j, step, i]
                            # if not (lprobs_list[i][j,:].ne(-math.inf).nonzero() == prev_token).all():
                            #     lprobs_list[i][j, prev_token] = -math.inf
                            #     dur_counter[j,i] = 0.

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs_list[0])
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                for i in range(self.n_channels):
                    lprobs_list[i] = self.repeat_ngram_blocker(
                        tokens, lprobs_list[i], bsz, beam_size, step
                    )

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                [
                    lprobs_list[i].view(bsz, -1, self.vocab_sizes[i])
                    for i in range(self.n_channels)
                ],
                scores.view(bsz, beam_size, -1, self.n_channels)[:, :, :step, :],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask = torch.any(eos_mask, dim=-1, keepdim=False)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.stack(
                    [
                        torch.masked_select(
                            cand_scores[:, :beam_size, i], mask=eos_mask[:, :beam_size]
                        )
                        for i in range(self.n_channels)
                    ],
                    dim=-1,
                )
                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(
                    new_bsz * beam_size, -1, self.n_channels
                )
                tokens = tokens.view(bsz, -1)[batch_idxs].view(
                    new_bsz * beam_size, -1, self.n_channels
                )
                if self.duration_prediction:
                    dur_counter = dur_counter.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, self.n_channels
                    )
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos
            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_bbsz_idx = active_bbsz_idx.view(-1)

            # active_scores = torch.stack([
            #     torch.gather(cand_scores[...,0], dim=1, index=active_hypos)
            #         for i in range(self.n_channels)
            #         ], dim = -1)
            # active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            for i in range(self.n_channels):
                tokens.view(bsz, beam_size, -1, self.n_channels)[
                    :, :, step + 1, i
                ] = torch.gather(cand_indices[..., i], dim=1, index=active_hypos)
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            for i in range(self.n_channels):
                scores.view(bsz, beam_size, -1, self.n_channels)[
                    :, :, step, i
                ] = torch.gather(cand_scores[..., i], dim=1, index=active_hypos)

            if self.duration_prediction:
                dur_counter = torch.index_select(
                    dur_counter, dim=0, index=active_bbsz_idx
                )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        # used for 1-channel generation, do not force the unk token (i.e. unk tokens are changed)
        prefix_mask &= prefix_toks.ne(self.unk)
        # zeroing the copying tokens
        # if step > 0:
        #     copy_mask = (prefix_tokens[:, step] == prefix_tokens[:, step-1]).unsqueeze(-1).repeat(1, beam_size).view(-1)
        #     prefix_lprobs[copy_mask & prefix_mask] = 0.
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # shouldn't stop at unk token
        unk_mask = prefix_toks.eq(self.unk)
        if len(lprobs[unk_mask]) > 0:
            # otherwise it won't assign to lprobs,
            # see: https://discuss.pytorch.org/t/how-to-mask-and-assign-a-value-to-tensor/18437
            copy_lprobs = lprobs[unk_mask][:, :]
            copy_lprobs[:, self.eos] = -math.inf
            lprobs[unk_mask] = copy_lprobs
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.size(0)

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step, :] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # The keys here are of the form "{sent}_{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # set() is not supported in script export
        sents_seen: Dict[str, Optional[Tensor]] = {}

        # For every finished beam item
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i].sum()
            # sentence index in the current (possibly reduced) batch
            unfin_idx = idx // beam_size
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf).to(score)

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished: List[int] = []

        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))

            if not finished[sent] and self.is_finished(
                step, unfin_idx, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False


class MultichannelEnsembleModel(nn.Module):
    """A wrapper around an ensemble of SpeechDLM models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.has_incremental: bool = False
        if all(
            hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
            for m in models
        ):
            self.has_incremental = True

        if isinstance(models[0], SpeechDLM):
            self.is_speech_dlm = True
        # Otherwise it's a multi-channel language model (without cross-prediction outputs)
        else:
            self.is_speech_dlm = False

        if getattr(models[0].decoder.args, "duration_prediction", False):
            self.is_duration_prediction = True
        else:
            self.is_duration_prediction = False

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models])

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: Dict[str, float] = 1.0,
    ):
        if isinstance(temperature, (float, int)):
            temperature = {channel: temperature for channel in tokens}
        log_probs = {channel: [] for channel in tokens}
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            if self.is_speech_dlm:
                if self.is_duration_prediction:
                    decoder_out_divided_by_temperature = {
                        channel_src: {
                            channel_pred: {
                                "pred_token": decoder_out[0][channel_src][channel_pred][
                                    "pred_token"
                                ][:, -1:, :].div_(temperature[channel_pred]),
                                "pred_duration": decoder_out[0][channel_src][
                                    channel_pred
                                ]["pred_duration"][:, -1:, :],
                            }
                            for channel_pred in decoder_out[0][channel_src]
                        }
                        for channel_src in decoder_out[0]
                    }
                else:
                    decoder_out_divided_by_temperature = {
                        channel_src: {
                            channel_pred: decoder_out[0][channel_src][channel_pred][
                                :, -1:, :
                            ].div_(temperature[channel_pred])
                            for channel_pred in decoder_out[0][channel_src]
                        }
                        for channel_src in decoder_out[0]
                    }
            else:
                decoder_out_divided_by_temperature = {
                    channel: decoder_out[0][channel][:, -1:, :].div_(
                        temperature[channel]
                    )
                    for channel in decoder_out[0]
                }
            decoder_out_tuple = (
                decoder_out_divided_by_temperature,
                None if decoder_len <= 1 else decoder_out[1],
            )

            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )

            if self.is_speech_dlm:
                if self.is_duration_prediction:
                    probs = {
                        channel: {
                            "pred_token": probs[channel][channel]["pred_token"][
                                :, -1, :
                            ],
                            "pred_duration": probs[channel][channel]["pred_duration"][
                                :, -1, :
                            ],
                        }
                        for channel in probs
                    }
                else:
                    probs = {
                        channel: probs[channel][channel][:, -1, :] for channel in probs
                    }
            else:
                probs = {channel: probs[channel][:, -1, :] for channel in probs}
            if self.models_size == 1:
                return probs, attn

            for channel in probs:
                log_probs[channel].append(probs[channel])
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = {}
        for channel in log_probs:
            avg_probs[channel] = torch.logsumexp(
                torch.stack(log_probs[channel], dim=0), dim=0
            ) - math.log(self.models_size)

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_encoder_out(
        self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order
    ):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )
