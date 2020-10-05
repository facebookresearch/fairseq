import os
import sys
import torch
import logging
from simuleval import READ_ACTION, WRITE_ACTION
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.speech_to_text.utils import pad_sequence

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from fairseq_simul_st_agent import FairseqSimulSTAgent  # noqa: E402

logger = logging.getLogger('simuleval.agent')


class AMCTFWaitkSimulSTAgent(FairseqSimulSTAgent):
    def __init__(self, args):
        super().__init__(args)

        # Segment size on features
        self.segment_size = self.model.encoder.segment_size
        self.left_context = self.model.encoder.left_context
        self.right_context = self.model.encoder.right_context
        self.full_context = (
            self.left_context + self.segment_size + self.right_context
        )

        # Segment size on samples
        self.speech_segment_size = int(
            self.segment_size * self.feature_extractor.shift_size
        )
        self.speech_left_context = int(
            self.left_context * self.feature_extractor.shift_size
        )
        self.speech_right_context = int(
            self.right_context * self.feature_extractor.shift_size
        )

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

        # Fixed predicion size
        self.fixed_pooling_ratio = getattr(
            self.model.decoder.layers[0].encoder_attn, "pooling_ratio", None
        )

        if self.fixed_pooling_ratio is None:
            logger.warn("No pooling ratio is provided, use 1.")
            self.fixed_pooling_ratio = 1

    def initialize_states(self, states):
        super().initialize_states(states)
        # To store memory banks and encoder hidden states
        states.encoder_states_tracker = None
        states.encoder_states = None
        states.incremental_states = dict()
        states.incremental_states['steps'] = {'src': 0, 'tgt': 1}

    @staticmethod
    def add_args(parser):
        super(
            AMCTFWaitkSimulSTAgent,
            AMCTFWaitkSimulSTAgent
        ).add_args(parser)
        parser.add_argument("--waitk", type=int, default=None,
                            help="Use a different k for inference")

    def update_model_encoder(self, states):
        """
        Incremental encoding happens here.
        """
        if len(states.units.source) == 1:
            # First segment. No previous history
            seg_src_indices = self.to_device(
                pad_sequence(
                    states.units.source.value[-1][
                        0: self.segment_size + self.right_context
                    ].unsqueeze(0),
                    time_axis=1,
                    extra_left_context=self.left_context,
                    extra_right_context=0,
                )
            )
        else:
            # Construct frames with look ahead
            assert (
                states.units.source.value[-2].size(0) >=
                self.left_context
            )
            # TODO: try to refactor here later
            seg_src_indices = self.to_device(
                torch.cat(
                    [
                        torch.cat(states.units.source.value[-3:-1], dim=0)[
                            - (self.left_context + self.right_context):
                        ],
                        states.units.source.value[-1]
                    ]
                ).unsqueeze(0)
            )

        if states.finish_read():
            seg_src_indices = pad_sequence(
                seg_src_indices,
                time_axis=1,
                extra_left_context=0,
                extra_right_context=(
                    self.right_context
                )
            )

        seg_src_lengths = self.to_device(
            torch.LongTensor([
                seg_src_indices.size(1)
            ])
        )


        (
            seg_encoder_states, seg_src_lengths, states.encoder_states_tracker
        ) = self.model.encoder.incremental_encode(
            seg_src_indices, seg_src_lengths, states.encoder_states_tracker
        )
        if states.encoder_states is None:
            states.encoder_states = seg_encoder_states
        else:
            states.encoder_states = torch.cat(
                [
                    states.encoder_states,
                    seg_encoder_states,
                ],
                dim=0
            )

        torch.cuda.empty_cache()
    def policy(self, states):

        if len(states.units.target) >= self.max_len:
            states.status["write"] = False

        if getattr(states, "encoder_states", None) is None:
            # Reading the first frames
            if len(states.units.source) == 0:
                # Add the look ahead here
                # The first frame needs to be larger
                # 15 ms is the look ahead for feature extraction
                self.speech_segment_size = (
                    self.speech_segment_size
                    + self.speech_right_context + 15
                )
                return READ_ACTION

        self.speech_segment_size = int(
            self.segment_size
            * self.feature_extractor.shift_size
        )

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
