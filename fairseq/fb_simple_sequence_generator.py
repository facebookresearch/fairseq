# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.models import FairseqIncrementalDecoder


class SimpleSequenceGenerator(nn.Module):
    def __init__(
        self,
        model,
        tgt_dict,
        beam_size=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
    ):
        """Generates translations of a given source sentence.
        Args:
            beam_size (int, optional): beam width (default: 1)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
        """
        super(SimpleSequenceGenerator, self).__init__()
        self.model = model
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        self.maxlen = self.model.max_decoder_positions() - 1
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty

        self.search = search.BeamSearch(tgt_dict)

    def cuda(self):
        self.model.cuda()
        return self

    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    def generate(self, encoder_input):
        """Generate translations."""
        with torch.no_grad():
            return self._generate(encoder_input)

    def _generate(self, encoder_input):
        src_tokens = encoder_input["src_tokens"]
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (
            (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        )
        # bsz: total number of sentences in beam
        bsz, srclen = src_tokens.size()

        # the max beam size is the dictionary size - 1, since we never select pad
        beam_size = min(self.beam_size, self.vocab_size - 1)

        incremental_states = {}
        self.model.eval()
        if isinstance(self.model.decoder, FairseqIncrementalDecoder):
            incremental_states[self.model] = {}
        else:
            incremental_states[self.model] = None

        # compute the encoder output for each beam
        encoder_out = self.model.encoder(**encoder_input)
        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_out = self.model.encoder.reorder_encoder_out(encoder_out, new_order)

        # initialize buffers
        scores = (
            src_tokens.data.new(bsz * beam_size, self.maxlen + 1).float().fill_(0)
        )  # +1 for eos; pad is never choosed for scoring
        tokens = src_tokens.data.new(bsz * beam_size, self.maxlen + 2).fill_(
            self.pad
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos

        # list of completed sentences
        finalized = [
            [] for i in range(bsz)
        ]  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step
        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        def finalize_hypos(step, bbsz_idx, eos_scores):
            """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
            Returns number of sentences being finalized.
            Args:
                bbsz_idx (Tensor):
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)[
                :, 1 : step + 2
            ]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            sents_seen = set()
            for i, (idx, score) in enumerate(
                zip(bbsz_idx.tolist(), eos_scores.tolist())
            ):
                sent = idx // beam_size
                sents_seen.add(sent)

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(
                        {
                            "tokens": tokens_clone[i],
                            "score": score,
                            "attention": None,  # src_len x tgt_len
                            "alignment": None,
                            "positional_scores": pos_scores[i],
                        }
                    )

            newly_finished = 0
            for sent in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and len(finalized[sent]) == beam_size:
                    finished[sent] = True
                    newly_finished += 1
            return newly_finished

        reorder_state = None
        for step in range(self.maxlen + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if isinstance(self.model.decoder, FairseqIncrementalDecoder):
                    self.model.decoder.reorder_incremental_state(
                        incremental_states[self.model], reorder_state
                    )
                encoder_out = self.model.encoder.reorder_encoder_out(
                    encoder_out, reorder_state
                )

            lprobs = self._decode(
                tokens[:, : step + 1], encoder_out, incremental_states
            )

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = (
                tokens.new()
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = (
                scores.new()
            )  # scores of hypothesis ending with eos (finished sentences)
            if step < self.maxlen:
                self.search.set_src_lengths(src_lengths)
                cand_scores, cand_indices, cand_beams = self.search.step(
                    step,
                    lprobs.view(bsz, -1, self.vocab_size),
                    scores.view(bsz, beam_size, -1)[:, :, :step],
                )
            else:
                # make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))

                # finalize all active hypotheses once we hit maxlen
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    lprobs[:, self.eos], descending=True, out=(eos_scores, eos_bbsz_idx)
                )
                num_remaining_sent -= finalize_hypos(step, eos_bbsz_idx, eos_scores)
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(
                (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
            )

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)

            # only consider eos when it's among the top beam_size indices
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )
            if eos_bbsz_idx.numel() > 0:
                torch.masked_select(
                    cand_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_scores,
                )
                num_remaining_sent -= finalize_hypos(step, eos_bbsz_idx, eos_scores)

            if num_remaining_sent == 0:
                break

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            _, active_hypos = torch.topk(active_mask, k=beam_size, dim=1, largest=False)

            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )

            scores[:, :step] = torch.index_select(
                scores[:, :step], dim=0, index=active_bbsz_idx
            )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(
                finalized[sent], key=lambda r: r["score"], reverse=True
            )

        return finalized

    def _decode(self, tokens, encoder_out, incremental_states):
        with torch.no_grad():
            if incremental_states[self.model] is not None:
                decoder_out = list(
                    self.model.decoder(
                        tokens,
                        encoder_out,
                        incremental_state=incremental_states[self.model],
                    )
                )
            else:
                decoder_out = list(self.model.decoder(tokens, encoder_out))
            decoder_out[0] = decoder_out[0][:, -1, :]
        probs = self.model.get_normalized_probs(decoder_out, log_probs=True)
        return probs
