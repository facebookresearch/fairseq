import copy
import logging

import torch.nn.functional as F

from fairseq.models import FairseqIncrementalDecoder
from .t5_modules import T5Stack

logger = logging.getLogger(__name__)


class T5Decoder(FairseqIncrementalDecoder):
    """Decoder to T5 model."""

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)

        decoder_config = copy.deepcopy(args)
        decoder_config.is_decoder = True
        self.config = decoder_config
        self.t5_stack = T5Stack(decoder_config, embed_tokens)

        self.embed_tokens = embed_tokens

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            **extra_args
        )
        if not features_only:
            x = x * (self.config.d_model ** -0.5)
            x = self.output_layer(x)
        return x, extra

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        return F.linear(features, self.embed_tokens.weight)

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        full_context_alignment=False,
        **unused,
    ):
        decoder_attention_mask = ~ prev_output_tokens.eq(0)
        decoder_attention_mask[:, 0] = True
        hidden_states = self.t5_stack.forward(
            input_ids=prev_output_tokens,
            attention_mask=decoder_attention_mask,
            inputs_embeds=None,
            encoder_hidden_states=encoder_out.encoder_out.transpose(0, 1),
            encoder_attention_mask=encoder_out.encoder_padding_mask,
            head_mask=None,
        )[0]
        return hidden_states, None
