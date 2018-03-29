# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq import utils


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, models):
        self.models = models
        self.pad = models[0].dst_dict.pad()
        assert all(m.dst_dict.pad() == self.pad for m in self.models[1:])

    def cuda(self):
        for model in self.models:
            model.cuda()
        return self

    def score_batched_itr(self, data_itr, cuda=False, timer=None):
        """Iterate over a batched dataset and yield scored translations."""
        for sample in data_itr:
            s = utils.make_variable(sample, volatile=True, cuda=cuda)
            if timer is not None:
                timer.start()
            pos_scores, attn = self.score(s)
            if timer is not None:
                timer.stop(s['ntokens'])
            for i, id in enumerate(s['id'].data):
                src = s['net_input']['src_tokens'].data[i, :]
                # remove padding from ref
                ref = utils.strip_pad(s['target'].data[i, :], self.pad)
                tgt_len = ref.numel()
                pos_scores_i = pos_scores[i][:tgt_len]
                score_i = pos_scores_i.sum() / tgt_len
                attn_i = attn[i]
                _, alignment = attn_i.max(dim=0)
                hypos = [{
                    'tokens': ref,
                    'score': score_i,
                    'attention': attn_i,
                    'alignment': alignment,
                    'positional_scores': pos_scores_i,
                }]
                # return results in the same format as SequenceGenerator
                yield id, src, ref, hypos

    def score(self, sample):
        """Score a batch of translations."""
        net_input = sample['net_input']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in self.models:
            with utils.maybe_no_grad():
                model.eval()
                encoder_out = model.encoder(
                    net_input['src_tokens'],
                    net_input['src_lengths'],
                )
                decoder_out = model.decoder(
                    net_input['prev_output_tokens'],
                    encoder_out,
                )
                attn = decoder_out[1]
            probs = model.get_normalized_probs(decoder_out, log_probs=False).data
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None:
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs.div_(len(self.models))
        avg_probs.log_()
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        avg_probs = avg_probs.gather(
            dim=2,
            index=sample['target'].data.unsqueeze(-1),
        )
        return avg_probs.squeeze(2), avg_attn
