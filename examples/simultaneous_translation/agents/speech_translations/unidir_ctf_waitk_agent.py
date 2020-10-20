import os
import sys
import torch
import logging
from simuleval import READ_ACTION, WRITE_ACTION
from fairseq.models.fairseq_encoder import EncoderOut

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from unidir_ctf_agent import UnidirectionalConvTransformerSimulSTAgent  # noqa: E402

logger = logging.getLogger('simuleval.agent')


class UnidirectionalConvTransformerWaitkSimulSTAgent(UnidirectionalConvTransformerSimulSTAgent):
    def __init__(self, args):
        super().__init__(args)

        self.waitk = getattr(
            self.model.decoder.layers[0].encoder_attn, "waitk_lagging", None
        )

        if args.waitk is not None:
            logger.warning(
                "Usaing a different lagging from loaded model"
                f"({args.waitk}, {self.waitk})"
            )
            self.waitk = args.waitk

        if self.waitk is None:
            logger.error(
                "Wait K lagging can't be found from the model"
                "Please use '--waitk' argument."
                )

    @staticmethod
    def add_args(parser):
        super(
            UnidirectionalConvTransformerWaitkSimulSTAgent,
            UnidirectionalConvTransformerWaitkSimulSTAgent
        ).add_args(parser)
        parser.add_argument("--waitk", type=int, default=None,
                            help="Use a different k for inference")

    def policy(self, states):

        result = self.pre_policy_check(states)
        if result is not None:
            return result

        encoder_states = states.encoder_states

        while(
            not states.finish_read()
            and states.incremental_states['steps']['src']
            / self.fixed_pooling_ratio
            - len(states.target) + 1 < self.waitk
        ):
            if (
                states.incremental_states['steps']['src']
                + self.fixed_pooling_ratio
                > states.encoder_states.size(0)
            ):
                # TODO: remove previous states
                return READ_ACTION
            else:
                # If there is new encoder states already
                states.incremental_states['steps']['src'] \
                    += self.fixed_pooling_ratio
                encoder_states = states.encoder_states[
                    : states.incremental_states['steps']['src']
                ]

        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.model.decoder.dictionary.eos()]
                + states.units.target.value
            ).unsqueeze(0)
        )

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

        states.incremental_states['online'] = False

        x, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=encoder_out,
            incremental_state=states.incremental_states,
            features_only=False,
        )

        states.decoder_out = x

        torch.cuda.empty_cache()

        return WRITE_ACTION
