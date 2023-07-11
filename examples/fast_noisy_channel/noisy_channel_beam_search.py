# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.search import Search


class NoisyChannelBeamSearch(Search):

    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)
        self.fw_scores_buf = None
        self.lm_scores_buf = None

    def _init_buffers(self, t):
        # super()._init_buffers(t)
        if self.fw_scores_buf is None:
            self.scores_buf = t.new()
            self.indices_buf = torch.LongTensor().to(device=t.device)
            self.beams_buf = torch.LongTensor().to(device=t.device)
            self.fw_scores_buf = t.new()
            self.lm_scores_buf = t.new()

    def combine_fw_bw(self, combine_method, fw_cum, bw, step):
        if combine_method == "noisy_channel":
            fw_norm = fw_cum.div(step + 1)
            lprobs = bw + fw_norm
        elif combine_method == "lm_only":
            lprobs = bw + fw_cum

        return lprobs

    def step(self, step, fw_lprobs, scores, bw_lprobs, lm_lprobs, combine_method):
        self._init_buffers(fw_lprobs)
        bsz, beam_size, vocab_size = fw_lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            fw_lprobs = fw_lprobs[:, ::beam_size, :].contiguous()
            bw_lprobs = bw_lprobs[:, ::beam_size, :].contiguous()
            # nothing to add since we are at the first step
            fw_lprobs_cum = fw_lprobs

        else:
            # make probs contain cumulative scores for each hypothesis
            raw_scores = (scores[:, :, step - 1].unsqueeze(-1))
            fw_lprobs_cum = (fw_lprobs.add(raw_scores))

        combined_lprobs = self.combine_fw_bw(combine_method, fw_lprobs_cum, bw_lprobs, step)

        # choose the top k according to the combined noisy channel model score
        torch.topk(
            combined_lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                combined_lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.scores_buf, self.indices_buf),
        )
        # save corresponding fw and lm scores
        self.fw_scores_buf = torch.gather(fw_lprobs_cum.view(bsz, -1), 1, self.indices_buf)
        self.lm_scores_buf = torch.gather(lm_lprobs.view(bsz, -1), 1, self.indices_buf)
        # Project back into relative indices and beams
        self.beams_buf = self.indices_buf // vocab_size
        self.indices_buf.fmod_(vocab_size)
        return self.scores_buf, self.fw_scores_buf, self.lm_scores_buf, self.indices_buf, self.beams_buf
