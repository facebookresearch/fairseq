# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor

from .noisy_channel_beam_search import NoisyChannelBeamSearch
from fairseq.sequence_generator import EnsembleModel


class NoisyChannelSequenceGenerator(object):
    def __init__(
        self,
        combine_method,
        tgt_dict,
        src_dict=None,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        len_penalty=1.0,
        unk_penalty=0.0,
        retain_dropout=False,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        normalize_scores=True,
        channel_models=None,
        k2=10,
        ch_weight=1.0,
        channel_scoring_type='log_norm',
        top_k_vocab=0,
        lm_models=None,
        lm_dict=None,
        lm_weight=1.0,
        normalize_lm_scores_by_tgt_len=False,
    ):
        """Generates translations of a given source sentence,
           using beam search with noisy channel decoding.

        Args:
            combine_method (string, optional): Method to combine direct, LM and
                channel model scores (default: None)
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            src_dict (~fairseq.data.Dictionary): source dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
            no_repeat_ngram_size (int, optional): Size of n-grams that we avoid
                repeating in the generation (default: 0)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            channel_models (List[~fairseq.models.FairseqModel]): ensemble of models
                translating from the target to the source
            k2 (int, optional): Top K2 candidates to score per beam at each step (default:10)
            ch_weight (int, optional): Weight associated with the channel model score
                assuming that the direct model score has weight 1.0 (default: 1.0)
            channel_scoring_type (str, optional): String specifying how to score
                the channel model (default: 'log_norm')
            top_k_vocab (int, optional): If `channel_scoring_type` is `'src_vocab'` or
                `'src_vocab_batched'`, then this parameter specifies the number of
                most frequent tokens to include in the channel model output vocabulary,
                in addition to the source tokens in the input batch (default: 0)
            lm_models (List[~fairseq.models.FairseqModel]): ensemble of models
                generating text in the target language
            lm_dict (~fairseq.data.Dictionary): LM Model dictionary
            lm_weight (int, optional): Weight associated with the LM model score
                assuming that the direct model score has weight 1.0 (default: 1.0)
            normalize_lm_scores_by_tgt_len (bool, optional): Should we normalize LM scores
                by the target length? By default, we normalize the combination of
                LM and channel model scores by the source length
        """
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.channel_models = channel_models
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.combine_method = combine_method
        self.k2 = k2
        self.ch_weight = ch_weight
        self.channel_scoring_type = channel_scoring_type
        self.top_k_vocab = top_k_vocab
        self.lm_models = lm_models
        self.lm_dict = lm_dict
        self.lm_weight = lm_weight
        self.log_softmax_fn = torch.nn.LogSoftmax(dim=1)
        self.normalize_lm_scores_by_tgt_len = normalize_lm_scores_by_tgt_len

        self.share_tgt_dict = (self.lm_dict == self.tgt_dict)
        self.tgt_to_lm = make_dict2dict(tgt_dict, lm_dict)

        self.ch_scoring_bsz = 3072

        assert temperature > 0, '--temperature must be greater than 0'

        self.search = NoisyChannelBeamSearch(tgt_dict)

    @torch.no_grad()
    def generate(
        self,
        models,
        sample,
        prefix_tokens=None,
        bos_token=None,
        **kwargs
    ):
        """Generate a batch of translations.
        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
        """
        model = EnsembleModel(models)
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(model.models_size)
            ],
        )
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }
        src_tokens = encoder_input['src_tokens']
        src_lengths_no_eos = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths_no_eos.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )

        # compute the encoder output for each beam
        encoder_outs = model.forward_encoder(encoder_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)

        src_lengths = encoder_input['src_lengths']
        # initialize buffers
        scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        lm_prefix_scores = src_tokens.new(bsz * beam_size).float().fill_(0)

        scores_buf = scores.clone()
        tokens = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos if bos_token is None else bos_token

        # reorder source tokens so they may be used as a reference in generating P(S|T)
        src_tokens = reorder_all_tokens(src_tokens, src_lengths, self.src_dict.eos_index)

        src_tokens = src_tokens.repeat(1, beam_size).view(-1, src_len)
        src_lengths = src_lengths.view(bsz, -1).repeat(1, beam_size).view(bsz*beam_size, -1)

        attn, attn_buf = None, None
        nonpad_idxs = None

        # The cands_to_ignore indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = src_tokens.new_zeros(bsz, beam_size).eq(-1)  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, combined_noisy_channel_eos_scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    fw scores for each hypothesis
                combined_noisy_channel_eos_scores: A vector of the same size as bbsz_idx containing
                    combined noisy channel scores for each hypothesis
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            assert not tokens_clone.eq(self.eos).any()
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                combined_noisy_channel_eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), combined_noisy_channel_eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths_no_eos[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i][nonpad_idxs[sent]]
                        _, alignment = hypo_attn.max(dim=0)
                    else:
                        hypo_attn = None
                        alignment = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': alignment,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        def noisy_channel_rescoring(lprobs, beam_size, bsz, src_tokens, tokens, k):
            """Rescore the top k hypothesis from each beam using noisy channel modeling
            Returns:
                new_fw_lprobs: the direct model probabilities after pruning the top k
                new_ch_lm_lprobs:  the combined channel and language model probabilities
                new_lm_lprobs: the language model probabilities after pruning the top k
            """
            with torch.no_grad():
                lprobs_size = lprobs.size()
                if prefix_tokens is not None and step < prefix_tokens.size(1):
                    probs_slice = lprobs.view(bsz, -1, lprobs.size(-1))[:, 0, :]
                    cand_scores = torch.gather(
                        probs_slice, dim=1,
                        index=prefix_tokens[:, step].view(-1, 1).data
                    ).expand(-1, beam_size).contiguous().view(bsz*beam_size, 1)
                    cand_indices = prefix_tokens[:, step].view(-1, 1).expand(bsz, beam_size).data.contiguous().view(bsz*beam_size, 1)

                    # need to calculate and save fw and lm probs for prefix tokens
                    fw_top_k = cand_scores
                    fw_top_k_idx = cand_indices
                    k = 1
                else:
                    # take the top k best words for every sentence in batch*beam
                    fw_top_k, fw_top_k_idx = torch.topk(lprobs.view(beam_size*bsz, -1), k=k)
                eos_idx = torch.nonzero(fw_top_k_idx.view(bsz*beam_size*k, -1) == self.eos)[:, 0]
                ch_scores = fw_top_k.new_full((beam_size*bsz*k, ), 0)
                src_size = torch.sum(src_tokens[:, :] != self.src_dict.pad_index, dim=1, keepdim=True, dtype=fw_top_k.dtype)

                if self.combine_method != "lm_only":
                    temp_src_tokens_full = src_tokens[:, :].repeat(1, k).view(bsz*beam_size*k, -1)
                    not_padding = temp_src_tokens_full[:, 1:] != self.src_dict.pad_index
                    cur_tgt_size = step+2

                    # add eos to all candidate sentences except those that already end in eos
                    eos_tokens = tokens[:, 0].repeat(1, k).view(-1, 1)
                    eos_tokens[eos_idx] = self.tgt_dict.pad_index

                    if step == 0:
                        channel_input = torch.cat((fw_top_k_idx.view(-1, 1), eos_tokens), 1)
                    else:
                        # move eos from beginning to end of target sentence
                        channel_input = torch.cat((tokens[:, 1:step + 1].repeat(1, k).view(-1, step), fw_top_k_idx.view(-1, 1), eos_tokens), 1)

                    ch_input_lengths = torch.tensor(np.full(channel_input.size(0), cur_tgt_size))
                    ch_input_lengths[eos_idx] = cur_tgt_size-1
                    if self.channel_scoring_type == "unnormalized":
                        ch_encoder_output = channel_model.encoder(channel_input, src_lengths=ch_input_lengths)
                        ch_decoder_output, _ = channel_model.decoder(temp_src_tokens_full, encoder_out=ch_encoder_output, features_only=True)
                        del ch_encoder_output
                        ch_intermed_scores = channel_model.decoder.unnormalized_scores_given_target(ch_decoder_output, target_ids=temp_src_tokens_full[:, 1:])
                        ch_intermed_scores = ch_intermed_scores.float()
                        ch_intermed_scores *= not_padding.float()
                        ch_scores = torch.sum(ch_intermed_scores, dim=1)
                    elif self.channel_scoring_type == "k2_separate":
                        for k_idx in range(k):
                            k_eos_tokens = eos_tokens[k_idx::k, :]
                            if step == 0:
                                k_ch_input = torch.cat((fw_top_k_idx[:, k_idx:k_idx+1], k_eos_tokens), 1)
                            else:
                                # move eos from beginning to end of target sentence
                                k_ch_input = torch.cat((tokens[:, 1:step + 1], fw_top_k_idx[:, k_idx:k_idx+1], k_eos_tokens), 1)
                            k_ch_input_lengths = ch_input_lengths[k_idx::k]
                            k_ch_output = channel_model(k_ch_input, k_ch_input_lengths, src_tokens)
                            k_ch_lprobs = channel_model.get_normalized_probs(k_ch_output, log_probs=True)
                            k_ch_intermed_scores = torch.gather(k_ch_lprobs[:, :-1, :], 2, src_tokens[:, 1:].unsqueeze(2)).squeeze(2)
                            k_ch_intermed_scores *= not_padding.float()
                            ch_scores[k_idx::k] = torch.sum(k_ch_intermed_scores, dim=1)
                    elif self.channel_scoring_type == "src_vocab":
                        ch_encoder_output = channel_model.encoder(channel_input, src_lengths=ch_input_lengths)
                        ch_decoder_output, _ = channel_model.decoder(temp_src_tokens_full, encoder_out=ch_encoder_output, features_only=True)

                        del ch_encoder_output
                        ch_lprobs = normalized_scores_with_batch_vocab(
                            channel_model.decoder,
                            ch_decoder_output, src_tokens, k, bsz, beam_size,
                            self.src_dict.pad_index, top_k=self.top_k_vocab)
                        ch_scores = torch.sum(ch_lprobs, dim=1)
                    elif self.channel_scoring_type == "src_vocab_batched":
                        ch_bsz_size = temp_src_tokens_full.shape[0]
                        ch_lprobs_list = [None] * len(range(0, ch_bsz_size, self.ch_scoring_bsz))
                        for i, start_idx in enumerate(range(0, ch_bsz_size, self.ch_scoring_bsz)):
                            end_idx = min(start_idx + self.ch_scoring_bsz, ch_bsz_size)
                            temp_src_tokens_full_batch = temp_src_tokens_full[start_idx:end_idx, :]
                            channel_input_batch = channel_input[start_idx:end_idx, :]
                            ch_input_lengths_batch = ch_input_lengths[start_idx:end_idx]
                            ch_encoder_output_batch = channel_model.encoder(channel_input_batch, src_lengths=ch_input_lengths_batch)
                            ch_decoder_output_batch, _ = channel_model.decoder(temp_src_tokens_full_batch, encoder_out=ch_encoder_output_batch, features_only=True)
                            ch_lprobs_list[i] = normalized_scores_with_batch_vocab(
                                channel_model.decoder,
                                ch_decoder_output_batch, src_tokens, k, bsz, beam_size,
                                self.src_dict.pad_index, top_k=self.top_k_vocab,
                                start_idx=start_idx, end_idx=end_idx)
                        ch_lprobs = torch.cat(ch_lprobs_list, dim=0)
                        ch_scores = torch.sum(ch_lprobs, dim=1)
                    else:
                        ch_output = channel_model(channel_input, ch_input_lengths, temp_src_tokens_full)
                        ch_lprobs = channel_model.get_normalized_probs(ch_output, log_probs=True)
                        ch_intermed_scores = torch.gather(ch_lprobs[:, :-1, :], 2, temp_src_tokens_full[:, 1:].unsqueeze(2)).squeeze().view(bsz*beam_size*k, -1)
                        ch_intermed_scores *= not_padding.float()
                        ch_scores = torch.sum(ch_intermed_scores, dim=1)

                else:
                    cur_tgt_size = 0
                ch_scores = ch_scores.view(bsz*beam_size, k)
                expanded_lm_prefix_scores = lm_prefix_scores.unsqueeze(1).expand(-1, k).flatten()

                if self.share_tgt_dict:
                    lm_scores = get_lm_scores(lm, tokens[:, :step + 1].view(-1, step+1), lm_incremental_states, fw_top_k_idx.view(-1, 1), torch.tensor(np.full(tokens.size(0), step+1)), k)
                else:
                    new_lm_input = dict2dict(tokens[:, :step + 1].view(-1, step+1), self.tgt_to_lm)
                    new_cands = dict2dict(fw_top_k_idx.view(-1, 1), self.tgt_to_lm)
                    lm_scores = get_lm_scores(lm, new_lm_input, lm_incremental_states, new_cands, torch.tensor(np.full(tokens.size(0), step+1)), k)

                lm_scores.add_(expanded_lm_prefix_scores)
                ch_lm_scores = combine_ch_lm(self.combine_method, ch_scores, lm_scores, src_size, cur_tgt_size)
                # initialize all as min value
                new_fw_lprobs = ch_scores.new(lprobs_size).fill_(-1e17).view(bsz*beam_size, -1)
                new_ch_lm_lprobs = ch_scores.new(lprobs_size).fill_(-1e17).view(bsz*beam_size, -1)
                new_lm_lprobs = ch_scores.new(lprobs_size).fill_(-1e17).view(bsz*beam_size, -1)
                new_fw_lprobs[:, self.pad] = -math.inf
                new_ch_lm_lprobs[:, self.pad] = -math.inf
                new_lm_lprobs[:, self.pad] = -math.inf

                new_fw_lprobs.scatter_(1, fw_top_k_idx, fw_top_k)
                new_ch_lm_lprobs.scatter_(1, fw_top_k_idx, ch_lm_scores)
                new_lm_lprobs.scatter_(1, fw_top_k_idx, lm_scores.view(-1, k))
                return new_fw_lprobs, new_ch_lm_lprobs, new_lm_lprobs

        def combine_ch_lm(combine_type, ch_scores, lm_scores1, src_size, tgt_size):
            if self.channel_scoring_type == "unnormalized":
                ch_scores = self.log_softmax_fn(
                    ch_scores.view(-1, self.beam_size * self.k2)
                ).view(ch_scores.shape)
            ch_scores = ch_scores * self.ch_weight
            lm_scores1 = lm_scores1 * self.lm_weight

            if combine_type == "lm_only":
                # log P(T|S) + log P(T)
                ch_scores = lm_scores1.view(ch_scores.size())
            elif combine_type == "noisy_channel":
                # 1/t log P(T|S) + 1/s log P(S|T) + 1/t log P(T)
                if self.normalize_lm_scores_by_tgt_len:
                    ch_scores.div_(src_size)
                    lm_scores_norm = lm_scores1.view(ch_scores.size()).div(tgt_size)
                    ch_scores.add_(lm_scores_norm)
                # 1/t log P(T|S) + 1/s log P(S|T) + 1/s log P(T)
                else:
                    ch_scores.add_(lm_scores1.view(ch_scores.size()))
                    ch_scores.div_(src_size)

            return ch_scores

        if self.channel_models is not None:
            channel_model = self.channel_models[0]  # assume only one channel_model model
        else:
            channel_model = None

        lm = EnsembleModel(self.lm_models)
        lm_incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(lm.models_size)
            ],
        )

        reorder_state = None
        batch_idxs = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)

                lm.reorder_incremental_state(lm_incremental_states, reorder_state)

            fw_lprobs, avg_attn_scores = model.forward_decoder(
                tokens[:, :step + 1], encoder_outs, incremental_states, temperature=self.temperature,
            )

            fw_lprobs[:, self.pad] = -math.inf  # never select pad
            fw_lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty
            fw_lprobs, ch_lm_lprobs, lm_lprobs = noisy_channel_rescoring(fw_lprobs, beam_size, bsz, src_tokens, tokens, self.k2)

            # handle min and max length constraints
            if step >= max_len:
                fw_lprobs[:, :self.eos] = -math.inf
                fw_lprobs[:, self.eos + 1:] = -math.inf
            elif step < self.min_len:
                fw_lprobs[:, self.eos] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if prefix_tokens is not None and step < prefix_tokens.size(1):
                prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
                prefix_mask = prefix_toks.ne(self.pad)

                prefix_fw_lprobs = fw_lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                fw_lprobs[prefix_mask] = -math.inf
                fw_lprobs[prefix_mask] = fw_lprobs[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_fw_lprobs
                )

                prefix_ch_lm_lprobs = ch_lm_lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                ch_lm_lprobs[prefix_mask] = -math.inf
                ch_lm_lprobs[prefix_mask] = ch_lm_lprobs[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_ch_lm_lprobs
                )

                prefix_lm_lprobs = lm_lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                lm_lprobs[prefix_mask] = -math.inf
                lm_lprobs[prefix_mask] = lm_lprobs[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lm_lprobs
                )

                # if prefix includes eos, then we should make sure tokens and
                # scores are the same across all beams
                eos_mask = prefix_toks.eq(self.eos)
                if eos_mask.any():
                    # validate that the first beam matches the prefix
                    first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
                    eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
                    target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
                    assert (first_beam == target_prefix).all()

                    def replicate_first_beam(tensor, mask):
                        tensor = tensor.view(-1, beam_size, tensor.size(-1))
                        tensor[mask] = tensor[mask][:, :1, :]
                        return tensor.view(-1, tensor.size(-1))

                    # copy tokens, scores and lprobs from the first beam to all beams
                    tokens = replicate_first_beam(tokens, eos_mask_batch_dim)
                    scores = replicate_first_beam(scores, eos_mask_batch_dim)

                    fw_lprobs = replicate_first_beam(fw_lprobs, eos_mask_batch_dim)
                    ch_lm_lprobs = replicate_first_beam(ch_lm_lprobs, eos_mask_batch_dim)
                    lm_lprobs = replicate_first_beam(lm_lprobs, eos_mask_batch_dim)

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                    nonpad_idxs = src_tokens.ne(self.pad)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(fw_lprobs)
            scores_buf = scores_buf.type_as(fw_lprobs)

            self.search.set_src_lengths(src_lengths_no_eos)

            if self.no_repeat_ngram_size > 0:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                    return gen_ngrams[bbsz_idx].get(ngram_index, [])

                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
                else:
                    banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                for bbsz_idx in range(bsz * beam_size):
                    fw_lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf

            combined_noisy_channel_scores, fw_lprobs_top_k, lm_lprobs_top_k, cand_indices, cand_beams = self.search.step(
                step,
                fw_lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step], ch_lm_lprobs.view(bsz, -1, self.vocab_size),
                lm_lprobs.view(bsz, -1, self.vocab_size), self.combine_method
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos (except for candidates to be ignored)
            eos_mask = cand_indices.eq(self.eos)
            eos_mask[:, :beam_size] &= ~cands_to_ignore

            # only consider eos when it's among the top beam_size indices
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents = set()
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    fw_lprobs_top_k[:, :beam_size], mask=eos_mask[:, :beam_size]
                )
                combined_noisy_channel_eos_scores = torch.masked_select(
                    combined_noisy_channel_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                )

                # finalize hypo using channel model score
                finalized_sents = finalize_hypos(
                    step, eos_bbsz_idx, eos_scores, combined_noisy_channel_eos_scores)

                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = torch.nonzero(batch_mask).squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)

                lm_lprobs_top_k = lm_lprobs_top_k[batch_idxs]

                fw_lprobs_top_k = fw_lprobs_top_k[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths_no_eos = src_lengths_no_eos[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                src_tokens = src_tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                src_lengths = src_lengths.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                lm_prefix_scores = lm_prefix_scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1).squeeze()

                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos or
            # ignored hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.
            eos_mask[:, :beam_size] |= cands_to_ignore
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_cands_to_ignore = buffer('active_hypos'), buffer('new_cands_to_ignore')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(new_cands_to_ignore, active_hypos)
            )

            # update cands_to_ignore to ignore any finalized hypos
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            assert (~cands_to_ignore).any(dim=1).all()

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                fw_lprobs_top_k, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                fw_lprobs_top_k, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )
            torch.gather(
                lm_lprobs_top_k, dim=1, index=active_hypos,
                out=lm_prefix_scores.view(bsz, beam_size)
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized


def get_lm_scores(model, input_tokens, incremental_states, cand_tokens, input_len, k):
    with torch.no_grad():
        lm_lprobs, avg_attn_scores = model.forward_decoder(
            input_tokens, encoder_outs=None, incremental_states=incremental_states,
        )

        lm_lprobs_size = lm_lprobs.size(0)
        probs_next_wrd = torch.gather(lm_lprobs.repeat(1, k).view(lm_lprobs_size*k, -1), 1, cand_tokens).squeeze().view(-1)

        return probs_next_wrd


def make_dict2dict(old_dict, new_dict):
    dict2dict_map = {}
    for sym in old_dict.symbols:
        dict2dict_map[old_dict.index(sym)] = new_dict.index(sym)
    return dict2dict_map


def dict2dict(tokens, dict2dict_map):
    if tokens.device == torch.device('cpu'):
        tokens_tmp = tokens
    else:
        tokens_tmp = tokens.cpu()
    return tokens_tmp.map_(
        tokens_tmp,
        lambda _, val, dict2dict_map=dict2dict_map : dict2dict_map[float(val)]
    ).to(tokens.device)


def reorder_tokens(tokens, lengths, eos):
    # reorder source tokens so they may be used as reference for P(S|T)
    return torch.cat((tokens.new([eos]), tokens[-lengths:-1], tokens[:-lengths]), 0)


def reorder_all_tokens(tokens, lengths, eos):
    # used to reorder src tokens from [<pad> <w1> <w2> .. <eos>] to [<eos> <w1> <w2>...<pad>]
    # so source tokens can be used to predict P(S|T)
    return torch.stack([reorder_tokens(token, length, eos) for token, length in zip(tokens, lengths)])


def normalized_scores_with_batch_vocab(
        model_decoder, features, target_ids, k, bsz, beam_size,
        pad_idx, top_k=0, vocab_size_meter=None, start_idx=None,
        end_idx=None, **kwargs):
    """
        Get normalized probabilities (or log probs) from a net's output
        w.r.t. vocab consisting of target IDs in the batch
    """
    if model_decoder.adaptive_softmax is None:
        weight = model_decoder.output_projection.weight
        vocab_ids = torch.unique(
            torch.cat(
                (torch.unique(target_ids), torch.arange(top_k, device=target_ids.device))
            )
        )
        id_map = dict(zip(vocab_ids.tolist(), range(len(vocab_ids))))
        mapped_target_ids = target_ids.cpu().apply_(
            lambda x, id_map=id_map: id_map[x]
        ).to(target_ids.device)
        expanded_target_ids = mapped_target_ids[:, :].repeat(1, k).view(bsz*beam_size*k, -1)
        if start_idx is not None and end_idx is not None:
            expanded_target_ids = expanded_target_ids[start_idx:end_idx, :]
        logits = F.linear(features, weight[vocab_ids, :])
        log_softmax = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        intermed_scores = torch.gather(
            log_softmax[:, :-1, :],
            2,
            expanded_target_ids[:, 1:].unsqueeze(2),
        ).squeeze()
        not_padding = expanded_target_ids[:, 1:] != pad_idx
        intermed_scores *= not_padding.float()
        return intermed_scores
    else:
        raise ValueError("adaptive softmax doesn't work with " +
                         "`normalized_scores_with_batch_vocab()`")
