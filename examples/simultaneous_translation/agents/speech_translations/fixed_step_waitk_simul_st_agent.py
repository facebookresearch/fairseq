import os
import sys
import json
import torch
from simuleval import READ_ACTION, WRITE_ACTION

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from fairseq_simul_st_agent import FairseqSimulSTAgent  # noqa: E402


class FixedStepWaitkSimulSTAgent(FairseqSimulSTAgent):
    def __init__(self, args):
        super().__init__(args)
        self.waitk =
        self.speech_segment_size = 40 * self.model.decoder.layers[0].encoder_attn.pooling_ratio

    @staticmethod
    def add_args(parser):
        super(
            FixedStepWaitkSimulSTAgent,
            FixedStepWaitkSimulSTAgent
        ).add_args(parser)
        parser.add_argument("--waitk", type=int, default=3)
        #parser.add_argument("--segment-size", type=int, default=40)

    def policy(self, states):

        if (
            not states.finish_read()
            and states.num_milliseconds() / self.speech_segment_size - len(states.target) < self.waitk
        ):
            return READ_ACTION

        if not getattr(states, "encoder_states", None):
            return READ_ACTION

        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.model.decoder.dictionary.eos()] + states.units.target.value
            ).unsqueeze(0)
        )

        states.incremental_states["online"] = (
            False
        )

        x, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=states.encoder_states,
            incremental_state=states.incremental_states,
            features_only=True,
        )

        states.decoder_out = x

        torch.cuda.empty_cache()

        return WRITE_ACTION
