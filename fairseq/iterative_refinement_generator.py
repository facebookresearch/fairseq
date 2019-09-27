# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.models.model_utils import skip_tensors as _skip


class IterativeRefinementGenerator(object):
    def __init__(self,
                 tgt_dict,
                 eos_penalty=0.,
                 max_iter=10,
                 max_ratio=2,
                 decoding_format=None,
                 retain_dropout=False,
                 adaptive=True):
        """
        Generates translations based on iterative refinement.

        Args:
            tgt_dict: target dictionary
            eos_penalty: if > 0.0, it penalized early-stopping in decoding
            max_iter: maximum number of refinement iterations
            max_ratio: generate sequences of maximum length ax, where x is the source length
            decoding_format: decoding mode in {'unigram', 'ensemble', 'vote', 'dp', 'bs'}
            retain_dropout: retaining dropout in the inference
            adaptive: decoding with early stop
        """
        self.bos = tgt_dict.bos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.eos_penalty = eos_penalty
        self.max_iter = max_iter
        self.max_ratio = max_ratio
        self.decoding_format = decoding_format
        self.retain_dropout = retain_dropout
        self.adaptive = adaptive

    @torch.no_grad()
    def generate(self, models, sample, prefix_tokens=None):

        # TODO: model ensemble
        assert len(models) == 1, 'only support single model'
        model = models[0]
        if not self.retain_dropout:
            model.eval()

        # TODO: better encoder inputs?
        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']
        bsz, src_len = src_tokens.size()
        sent_idxs = torch.arange(bsz, device=src_tokens.device)

        # encoding
        encoder_out = model.forward_encoder([src_tokens, src_lengths])

        # initialize buffers (very model specific, with length prediction or not)
        prev_decoder_out = model.initialize_output_tokens(
            encoder_out, src_tokens)
        prev_out_tokens = prev_decoder_out['output_tokens'].clone()

        finalized = [[] for _ in range(bsz)]

        def is_a_loop(x, y, s, a):
            b, l_x, l_y = x.size(0), x.size(1), y.size(1)
            if l_x > l_y:
                y = torch.cat([y, x.new_zeros(b, l_x - l_y).fill_(self.pad)], 1)
                s = torch.cat([s, s.new_zeros(b, l_x - l_y)], 1)
                if a is not None:
                    a = torch.cat([a, a.new_zeros(b, l_x - l_y, a.size(2))], 1)
            elif l_x < l_y:
                x = torch.cat([x, y.new_zeros(b, l_y - l_x).fill_(self.pad)], 1)
            return (x == y).all(1), y, s, a

        def finalized_hypos(step, prev_out_token, prev_out_score, prev_out_attn):
            cutoff = prev_out_token.ne(self.pad)
            tokens = prev_out_token[cutoff]
            scores = prev_out_score[cutoff]
            if prev_out_attn is None:
                hypo_attn, alignment = None, None
            else:
                hypo_attn = prev_out_attn[cutoff]
                alignment = hypo_attn.max(dim=1)[1]
            return {
                'steps': step,
                'tokens': tokens,
                'positional_scores': scores,
                'score': scores.mean(),
                'hypo_attn': hypo_attn,
                'alignment': alignment,
            }

        for step in range(self.max_iter + 1):

            decoder_options = {
                'eos_penalty': self.eos_penalty,
                'max_ratio': self.max_ratio,
                'decoding_format': self.decoding_format
            }
            prev_decoder_out['step'] = step
            prev_decoder_out['max_step'] = self.max_iter + 1

            decoder_out = model.forward_decoder(
                prev_decoder_out, encoder_out, **decoder_options
            )

            if self.adaptive:
                # terminate if there is a loop
                terminated, out_tokens, out_scores, out_attn = is_a_loop(
                    prev_out_tokens, decoder_out['output_tokens'],
                    decoder_out['output_scores'], decoder_out['attn'])
                decoder_out['output_tokens'] = out_tokens
                decoder_out['output_scores'] = out_scores
                decoder_out['attn'] = out_attn

            else:
                terminated = decoder_out['output_tokens'].new_zeros(
                    decoder_out['output_tokens'].size(0)).bool()

            if step == self.max_iter:  # reach last iteration, terminate
                terminated.fill_(1)

            # collect finalized sentences
            finalized_idxs = sent_idxs[terminated]
            finalized_tokens = decoder_out['output_tokens'][terminated]
            finalized_scores = decoder_out['output_scores'][terminated]
            finalized_attn = None if decoder_out['attn'] is None else decoder_out['attn'][terminated]

            for i in range(finalized_idxs.size(0)):
                finalized[finalized_idxs[i]] = [
                    finalized_hypos(
                        step,
                        finalized_tokens[i],
                        finalized_scores[i],
                        None if finalized_attn is None else finalized_attn[i]
                    )
                ]
            # check if all terminated
            if terminated.sum() == terminated.size(0):
                break

            # for next step
            prev_decoder_out = _skip(decoder_out, ~terminated)
            encoder_out = _skip(encoder_out, ~terminated)
            sent_idxs = _skip(sent_idxs, ~terminated)

            prev_out_tokens = prev_decoder_out['output_tokens'].clone()

        return finalized
