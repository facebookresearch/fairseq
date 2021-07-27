# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    FairseqEncoderDecoderModel,
    TransformerDecoder, 
    FairseqEncoder, 
    FairseqIncrementalDecoder
) 
try: 
    from transformers.models.marian.modeling_marian import (
        MarianEncoder, 
        MarianDecoder, 
        MarianConfig, 
        MarianMTModel
    )
except ImportError:
    raise ImportError(
        "\n\nPlease install huggingface/transformers with:"
        "\n\n  pip install transformers"
    )

logger = logging.getLogger(__name__)


DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("hf_marian")
class HuggingFaceMarianNMT(FairseqEncoderDecoderModel):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.hf_config = MarianConfig.from_pretrained(cfg.common_eval.path)


    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        logger.info(cfg.common)
        encoder = HuggingFaceMarianEncoder(cfg, task.dictionary)
        decoder = HuggingFaceMarianDecoder(cfg, task.dictionary)
        return cls(cfg, encoder, decoder)


    def max_positions(self):
        return 512

    def max_source_positions(self):
        return 512

    def max_target_positions(self):
        return 512


class HuggingFaceMarianEncoder(FairseqEncoder):
    def __init__(self, cfg, dictionary):
        super().__init__(dictionary)
        config = MarianConfig.from_pretrained(cfg.common_eval.path)
        self.model = MarianMTModel.from_pretrained(cfg.common_eval.path).get_encoder()
        self.dictionary = dictionary
        self.config = config

    
    def forward(self, src_tokens, src_lengths=None, return_all_hiddens=False, ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states = self.model(**src_tokens)
        features = inner_states[0].float()
        return features, {'inner_states': inner_states[2] if return_all_hiddens else None}


        

class HuggingFaceMarianDecoder(FairseqIncrementalDecoder):
    def __init__(self, cfg, dictionary):
        super().__init__(dictionary)
        config = MarianConfig.from_pretrained(cfg.common_eval.path)
        self.model = MarianMTModel.from_pretrained(cfg.common_eval.path).get_decoder()
        self.dictionary = dictionary
        self.config = config


    def forward(
        self,
        prev_output_tokens,
        src_lengths=None,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
    ):
        features = self.extract_features(prev_output_tokens, incremental_state)
        lm_logits = self.model.lm_head(features)
        return (lm_logits,)

    def extract_features(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
    ):
        if incremental_state:
            past = self.get_incremental_state("past")
        else:
            past = None

        # don't attend to padding symbols
        attention_mask = prev_output_tokens.ne(self.pad_idx).int()

        # set position ids to exclude padding symbols
        position_ids = attention_mask * (
            torch.arange(1, 1 + prev_output_tokens.size(1))
            .to(prev_output_tokens)
            .repeat(prev_output_tokens.size(0), 1)
        )

        outputs = self.model.transformer(
            input_ids=prev_output_tokens,
            past=past,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        last_hidden_states = outputs[0]

        if incremental_state:
            self.set_incremental_state(incremental_state, "past", outputs[1])

        return last_hidden_states

    


@register_model_architecture('hf_marian', 'hf_marian')
def default_architecture(args):
    args.max_target_positions = getattr(args, 'max_target_positions', 512)
    args.max_source_positions = getattr(args, 'max_source_positions', 512)
