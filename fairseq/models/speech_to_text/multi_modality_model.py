# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import FairseqDecoder, FairseqEncoder


# a container for different encoders with training samples from  different modality
# each time, only one encoder is selected
class MultiModalityEncoder(FairseqEncoder):
    def __init__(self, dictionary):
        super().__init__(dictionary)

    def select_encoder(self, mode, **kwargs):
        raise NotImplementedError("Model must implement the select_encoder method")
        return None, kwargs

    # def post_encoder(self, encoder_out, src_tokens, src_lengths, mode, **kwargs):
    #    # Default do nothing
    #    return encoder_out

    # get sample data from JointSpeechTextDataset
    def forward(self, src_tokens, src_lengths=None, mode="", **kwargs):
        encoder, kwargs = self.select_encoder(mode, **kwargs)
        # return self.post_encoder(encoder(src_tokens, src_lengths, **kwargs), src_tokens, src_lengths, mode, **kwargs)
        return encoder(src_tokens, src_lengths, **kwargs)


# a container for different decoders with training samples from  different modality
# each time, only one decoder is selected
class MultiInputDecoder(FairseqDecoder):
    def __init__(self, dictionary):
        super().__init__(dictionary)

    def select_decoder(self, mode, **kwargs):
        raise NotImplementedError("Model must implement the select_decoder method")
        return None, kwargs

    def forward(
        self, prev_output_tokens, encoder_out, incremental_state=None, mode="", **kwargs
    ):
        decoder, kwargs = self.select_decoder(mode, **kwargs)
        return decoder(
            prev_output_tokens,
            encoder_out,
            incremental_state=incremental_state,
            **kwargs
        )
