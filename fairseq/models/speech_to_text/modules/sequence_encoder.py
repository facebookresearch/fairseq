#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


from typing import Dict, List, Optional, Tuple

from fairseq.models.speech_to_text.utils import (
    segments_to_sequence,
    sequence_to_segments,
)

from torch import Tensor

from fairseq.models import FairseqEncoder

from fairseq.models.fairseq_encoder import EncoderOut

from fairseq.models.speech_to_text.utils import (
    lengths_to_encoder_padding_mask
)


# ------------------------------------------------------------------------------
#   SequenceEncoder
# ------------------------------------------------------------------------------
class SequenceEncoder(FairseqEncoder):
    """
    SequenceEncoder encodes sequences.

    More specifically, `src_tokens` and `src_lengths` in `forward()` should
    describe a batch of "complete" sequences rather than segments.

    Segment-by-segment inference can be triggered by `segment_size`:
    1) `segment_size` is None:
        SequenceEncoder treats the input sequence as one single segment.
    2) `segment_size` is not None (some int instead):
        SequenceEncoder does the following:
            1. breaks the input sequence into several segments
            2. inference on each segment and collect the outputs
            3. concatanete segment outputs into the output sequence.
    Note that `segment_size` here shouldn't include additional left/right
    contexts needed, for example if we wish to infer with LC-BLSTM where the
    middle chunk size is 100 and right context is 20, `segment_size` should be
    100.
    """

    def __init__(self, args, module):
        super().__init__(None)

        self.module = module
        self.input_time_axis = 1
        self.output_time_axis = 0
        self.segment_size = args.segment_size
        self.left_context = args.left_context
        self.right_context = args.right_context

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        states=None,
    ):

        seg_src_tokens_lengths = sequence_to_segments(
            sequence=src_tokens,
            time_axis=self.input_time_axis,
            lengths=src_lengths,
            segment_size=self.segment_size,
            extra_left_context=self.left_context,
            extra_right_context=self.right_context,
        )

        seg_encoder_states_lengths: List[Tuple[Tensor, Tensor]] = []

        for seg_src_tokens, seg_src_lengths in seg_src_tokens_lengths:
            (
                seg_encoder_states, seg_enc_lengths, states
            ) = self.module(
                seg_src_tokens, seg_src_lengths, states=states,
            )

            seg_encoder_states_lengths.append(
                (seg_encoder_states, seg_enc_lengths)
            )

        encoder_out, enc_lengths = segments_to_sequence(
            segments=seg_encoder_states_lengths,
            time_axis=self.output_time_axis
        )

        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            enc_lengths, batch_first=True
        )

        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        return EncoderOut(
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask,
            encoder_embedding=None,
            encoder_states=states,
            src_tokens=None,
            src_lengths=None,
        )

    def incremental_encode(
        self,
        seg_src_tokens: Tensor,
        seg_src_lengths: Tensor,
        states=None,
    ):
        """
        Different from forward function, this function takes segmented speech
        as input, and append encoder states to previous states
        """
        (
            seg_encoder_states, seg_enc_lengths, states
        ) = self.module(
            seg_src_tokens, seg_src_lengths, states=states,
        )
        return seg_encoder_states, seg_enc_lengths, states
