import os
import sys
import torch
import logging
from simuleval import READ_ACTION
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.speech_to_text.utils import pad_sequence

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from fairseq_simul_st_agent import FairseqSimulSTAgent  # noqa: E402

logger = logging.getLogger('simuleval.agent')


class UnidirectionalConvTransformerSimulSTAgent(FairseqSimulSTAgent):
    '''
    This class is a template for augmented memory transformer,
    policy() needs to be implemented for each model
    '''
    def __init__(self, args):
        super().__init__(args)

        # Fixed predicion size
        self.fixed_pooling_ratio = getattr(
            self.model.decoder.layers[0].encoder_attn, "pooling_ratio", None
        )

        self.speech_segment_size_base = self.fixed_pooling_ratio * 40

        if self.fixed_pooling_ratio is None:
            logger.warn("No pooling ratio is provided, use 1.")
            self.fixed_pooling_ratio = 1

    def initialize_states(self, states):
        super().initialize_states(states)
        # To store memory banks and encoder hidden states
        states.encoder_states = None
        states.incremental_states = dict()
        states.incremental_states['steps'] = {'src': 0, 'tgt': 1}

    def update_model_encoder(self, states):
        """
        Incremental encoding happens here.
        """
        # if len(states.units.source) == 1:
            # input_seq = states.units.source.value[-1]
        # else:
            # states.incremental_states['prev_residual'] = (
                # states.units.source.value[-2][-8:]
            # )
            # input_seq = torch.cat(
                # [
                    # states.incremental_states['prev_residual'],
                    # states.units.source.value[-1]
                # ],
                # dim=0
            # )

        # encoder_out = self.model.encoder.incremental_forward(
            # self.to_device(input_seq.unsqueeze(0)),
            # states.incremental_states
        # )

        # if states.encoder_states is None:
            # states.encoder_states = encoder_out.encoder_out
        # else:
            # states.encoder_states = torch.cat(
                # [
                    # states.encoder_states,
                    # encoder_out.encoder_out,
                # ],
                # dim=0
            # )

        input_seq_full = self.to_device(torch.cat(
            states.units.source.value[:],
            dim=0
        )).unsqueeze(0)
        input_seq_full_src_lengths = self.to_device(
            torch.LongTensor([
                input_seq_full.size(1)
            ])
        )
        states.encoder_states = self.model.encoder(
            input_seq_full,
            input_seq_full_src_lengths
        ).encoder_out

        torch.cuda.empty_cache()

    def pre_policy_check(self, states):
        # Run before excuting the policy
        if len(states.units.target) >= self.max_len:
            states.status["write"] = False

        if getattr(states, "encoder_states", None) is None:
            # Reading the first frames
            if len(states.units.source) == 0:
                # Add the look ahead here
                # The first frame needs to be larger
                # 15 ms is the look ahead for feature extraction
                # 40 ms is for the conv layers
                self.speech_segment_size = (
                    self.speech_segment_size_base + 15 + 40 * 2
                )
                #self.speech_segment_size = 10000
                return READ_ACTION

        self.speech_segment_size = self.speech_segment_size_base

        return None

    def decoder(self, states, tgt_indices, encoder_states):

        encoder_out = EncoderOut(
            encoder_out=encoder_states,
            encoder_padding_mask=None,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

        states.incremental_states['steps'] = {
            'src': encoder_states.size(0),
            'tgt': len(states.units.target.value) + 1
        }

        states.incremental_states['online'] = not states.finish_read()

        x, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=encoder_out,
            incremental_state=states.incremental_states,
            features_only=False,
        )

        return x, outputs

    def policy(self, states):
        raise NotImplementedError