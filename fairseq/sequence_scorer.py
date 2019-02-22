# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from fairseq import utils


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict):
        self.pad = tgt_dict.pad()

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model.forward(**net_input)
            attn = decoder_out[1]

            probs = model.get_normalized_probs(decoder_out, log_probs=len(models) == 1, sample=sample)
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))
        avg_probs = avg_probs.gather(
            dim=2,
            index=sample['target'].unsqueeze(-1),
        ).squeeze(2)

        hypos = []
        for i in range(avg_probs.size(0)):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, :], self.pad) if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][:tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                _, alignment = avg_attn_i.max(dim=0)
            else:
                avg_attn_i = alignment = None
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
            }])
        return hypos
