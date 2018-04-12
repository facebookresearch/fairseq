# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch

from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder


class SequenceGenerator(object):
    def __init__(self, models, beam_size=1, minlen=1, maxlen=None,
                 stop_early=True, normalize_scores=True, len_penalty=1,
                 unk_penalty=0, retain_dropout=False, sampling=False):
        """Generates translations of a given source sentence.

        Args:
            min/maxlen: The length of the generated output will be bounded by
                minlen and maxlen (not including the end-of-sentence marker).
            stop_early: Stop generation immediately after we finalize beam_size
                hypotheses, even though longer hypotheses might have better
                normalized scores.
            normalize_scores: Normalize scores by the length of the output.
        """
        self.models = models
        self.pad = models[0].dst_dict.pad()
        self.unk = models[0].dst_dict.unk()
        self.eos = models[0].dst_dict.eos()
        assert all(m.dst_dict.pad() == self.pad for m in self.models[1:])
        assert all(m.dst_dict.unk() == self.unk for m in self.models[1:])
        assert all(m.dst_dict.eos() == self.eos for m in self.models[1:])
        self.vocab_size = len(models[0].dst_dict)
        self.beam_size = beam_size
        self.minlen = minlen
        max_decoder_len = min(m.max_decoder_positions() for m in self.models)
        max_decoder_len -= 1  # we define maxlen not including the EOS marker
        self.maxlen = max_decoder_len if maxlen is None else min(maxlen, max_decoder_len)
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.sampling = sampling

    def cuda(self):
        for model in self.models:
            model.cuda()
        return self

    def generate_batched_itr(self, data_itr, beam_size=None, maxlen_a=0.0, maxlen_b=None,
                             cuda=False, timer=None, prefix_size=0):
        """Iterate over a batched dataset and yield individual translations.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """
        if maxlen_b is None:
            maxlen_b = self.maxlen

        for sample in data_itr:
            s = utils.make_variable(sample, volatile=True, cuda=cuda)
            input = s['net_input']
            srclen = input['src_tokens'].size(1)
            if timer is not None:
                timer.start()
            with utils.maybe_no_grad():
                hypos = self.generate(
                    input['src_tokens'],
                    input['src_lengths'],
                    beam_size=beam_size,
                    maxlen=int(maxlen_a*srclen + maxlen_b),
                    prefix_tokens=s['target'][:, :prefix_size] if prefix_size > 0 else None,
                )
            if timer is not None:
                timer.stop(sum(len(h[0]['tokens']) for h in hypos))
            for i, id in enumerate(s['id'].data):
                src = input['src_tokens'].data[i, :]
                # remove padding from ref
                ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                yield id, src, ref, hypos[i]

    def generate(self, src_tokens, src_lengths, beam_size=None, maxlen=None, prefix_tokens=None):
        """Generate a batch of translations."""
        with utils.maybe_no_grad():
            return self._generate(src_tokens, src_lengths, beam_size, maxlen, prefix_tokens)

    def _generate(self, src_tokens, src_lengths, beam_size=None, maxlen=None, prefix_tokens=None):
        bsz, srclen = src_tokens.size()
        maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen

        # the max beam size is the dictionary size - 1, since we never select pad
        beam_size = beam_size if beam_size is not None else self.beam_size
        beam_size = min(beam_size, self.vocab_size - 1)

        encoder_outs = []
        incremental_states = {}
        for model in self.models:
            if not self.retain_dropout:
                model.eval()
            if isinstance(model.decoder, FairseqIncrementalDecoder):
                incremental_states[model] = {}
            else:
                incremental_states[model] = None

            # compute the encoder output for each beam
            encoder_out = model.encoder(
                src_tokens.repeat(1, beam_size).view(-1, srclen),
                src_lengths.expand(beam_size, src_lengths.numel()).t().contiguous().view(-1),
            )
            encoder_outs.append(encoder_out)

        # initialize buffers
        scores = src_tokens.data.new(bsz * beam_size, maxlen + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.data.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos
        attn = scores.new(bsz * beam_size, src_tokens.size(1), maxlen + 2)
        attn_buf = attn.clone()

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz)*beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}
        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == maxlen or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= maxlen
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
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
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step+2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2]

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step+1)**self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                def get_hypo():
                    _, alignment = attn_clone[i].max(dim=0)
                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': attn_clone[i],  # src_len x tgt_len
                        'alignment': alignment,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        for step in range(maxlen + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                for i, model in enumerate(self.models):
                    if isinstance(model.decoder, FairseqIncrementalDecoder):
                        model.decoder.reorder_incremental_state(incremental_states[model], reorder_state)
                        encoder_outs[i] = model.decoder.reorder_encoder_out(encoder_outs[i], reorder_state)

            probs, avg_attn_scores = self._decode(
                tokens[:, :step+1], encoder_outs, incremental_states)
            if step == 0:
                # at the first step all hypotheses are equally likely, so use
                # only the first beam
                probs = probs.unfold(0, 1, beam_size).squeeze(2).contiguous()
                scores = scores.type_as(probs)
                scores_buf = scores_buf.type_as(probs)
            elif not self.sampling:
                # make probs contain cumulative scores for each hypothesis
                probs.add_(scores[:, step-1].view(-1, 1))

            probs[:, self.pad] = -math.inf  # never select pad
            probs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # Record attention scores
            attn[:, :, step+1].copy_(avg_attn_scores)

            cand_scores = buffer('cand_scores', type_of=scores)
            cand_indices = buffer('cand_indices')
            cand_beams = buffer('cand_beams')
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < maxlen:
                if prefix_tokens is not None and step < prefix_tokens.size(1):
                    probs_slice = probs.view(bsz, -1, probs.size(-1))[:, 0, :]
                    cand_scores = torch.gather(
                        probs_slice, dim=1,
                        index=prefix_tokens[:, step].view(-1, 1).data
                    ).expand(-1, cand_size)
                    cand_indices = prefix_tokens[:, step].view(-1, 1).expand(bsz, cand_size).data
                    cand_beams.resize_as_(cand_indices).fill_(0)
                elif self.sampling:
                    assert self.pad == 1, 'sampling assumes the first two symbols can be ignored'
                    exp_probs = probs.exp_().view(-1, self.vocab_size)
                    if step == 0:
                        # we exclude the first two vocab items, one of which is pad
                        torch.multinomial(exp_probs[:, 2:], beam_size, replacement=True, out=cand_indices)
                        cand_indices.add_(2)
                    else:
                        torch.multinomial(exp_probs[:, 2:], 1, replacement=True, out=cand_indices)
                        cand_indices.add_(2)
                    torch.gather(exp_probs, dim=1, index=cand_indices, out=cand_scores)
                    cand_scores.log_()
                    cand_indices = cand_indices.view(bsz, -1).repeat(1, 2)
                    cand_scores = cand_scores.view(bsz, -1).repeat(1, 2)
                    if step == 0:
                        cand_beams = torch.zeros(bsz, cand_size).type_as(cand_indices)
                    else:
                        cand_beams = torch.arange(0, beam_size).repeat(bsz, 2).type_as(cand_indices)
                        # make scores cumulative
                        cand_scores.add_(
                            torch.gather(
                                scores[:, step-1].view(bsz, beam_size), dim=1,
                                index=cand_beams,
                            )
                        )
                else:
                    # take the best 2 x beam_size predictions. We'll choose the first
                    # beam_size of these which don't predict eos to continue with.

                    torch.topk(
                        probs.view(bsz, -1),
                        k=min(cand_size, probs.view(bsz, -1).size(1) - 1),  # -1 so we never select pad
                        out=(cand_scores, cand_indices),
                    )
                    torch.div(cand_indices, self.vocab_size, out=cand_beams)
                    cand_indices.fmod_(self.vocab_size)
            else:
                # finalize all active hypotheses once we hit maxlen
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    probs[:, self.eos],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= len(finalize_hypos(
                    step, eos_bbsz_idx, eos_scores))
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)

            finalized_sents = set()
            if step >= self.minlen:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,
                )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_scores,
                    )
                    finalized_sents = finalize_hypos(
                        step, eos_bbsz_idx, eos_scores, cand_scores)
                    num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < maxlen

            if len(finalized_sents) > 0:
                # construct batch_idxs which holds indices of batches to keep for the next pass

                new_bsz = bsz - len(finalized_sents)

                batch_mask = torch.ones(bsz).type_as(cand_indices)
                batch_mask[torch.LongTensor(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)

                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer('active_mask')
            torch.add(
                eos_mask.type_as(cand_offsets)*cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(_ignore, active_hypos)
            )
            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step+1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step+1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step+1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            torch.index_select(
                attn[:, :, :step+2], dim=0, index=active_bbsz_idx,
                out=attn_buf[:, :, :step+2],
            )

            # swap buffers
            old_tokens = tokens
            tokens = tokens_buf
            tokens_buf = old_tokens
            old_scores = scores
            scores = scores_buf
            scores_buf = old_scores
            old_attn = attn
            attn = attn_buf
            attn_buf = old_attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized

    def _decode(self, tokens, encoder_outs, incremental_states):
        # wrap in Variable
        tokens = utils.volatile_variable(tokens)

        avg_probs = None
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            with utils.maybe_no_grad():
                if incremental_states[model] is not None:
                    decoder_out = list(model.decoder(tokens, encoder_out, incremental_states[model]))
                else:
                    decoder_out = list(model.decoder(tokens, encoder_out))
                decoder_out[0] = decoder_out[0][:, -1, :]
                attn = decoder_out[1]
            probs = model.get_normalized_probs(decoder_out, log_probs=False).data
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None:
                attn = attn[:, -1, :].data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs.div_(len(self.models))
        avg_probs.log_()
        if avg_attn is not None:
            avg_attn.div_(len(self.models))

        return avg_probs, avg_attn
