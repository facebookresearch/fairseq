import os
import sys
import torch
import logging
from simuleval import READ_ACTION, WRITE_ACTION

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from amctf_agent import AMCTFSimulSTAgent  # noqa: E402

logger = logging.getLogger('simuleval.agent')


class AMCTFMMASimulSTAgent(AMCTFSimulSTAgent):

    def policy(self, states):
        result = self.pre_policy_check(states)
        if result is not None:
            return result

        # Current tgt_indices in tensor
        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.model.decoder.dictionary.eos()]
                + states.units.target.value
            ).unsqueeze(0)
        )

        # Current encoder states
        encoder_states = states.encoder_states[
            : max(
                self.fixed_pooling_ratio,
                states.incremental_states['steps']['src']
            )  # Read at least one fixed pooling chunk of encoder states
        ]

        while True:
            # Run decoder once to get actions and decoder states
            decoder_outputs, extra_outputs = self.decoder(
                states, tgt_indices, encoder_states
            )

            if extra_outputs["action"] == 1 or states.finish_read():
                # WRITE a token
                break

            if (
                states.incremental_states['steps']['src']
                + self.fixed_pooling_ratio
                > states.encoder_states.size(0)
            ):
                # If there is no more encoder states in the buffer
                return READ_ACTION
            else:
                # If there is new encoder states already
                states.incremental_states['steps']['src'] \
                    += self.fixed_pooling_ratio
                encoder_states = states.encoder_states[
                    : states.incremental_states['steps']['src']
                ]

        # Decoder out will be used for prediction
        assert decoder_outputs is not None
        states.decoder_out = decoder_outputs

        torch.cuda.empty_cache()

        return WRITE_ACTION
